# finetuning_agent.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import random, numpy as np, torch
import unsloth  # MUST be imported before transformers/peft
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from config import FINETUNING_PARAMS, ADAPTER_DIRS, MODEL_ID, RANDOM_SEED

# --- 논문(Bug_Report_2504.18804v1.pdf)의 Listing 2를 반영한 시스템 프롬프트 ---
PAPER_SFT_SYSTEM_PROMPT = (
    "You are a senior software engineer specialized in generating detailed bug reports.\n\n"
    "### Instruction:\n"
    "Please create a bug report that includes the following sections:\n"
    "1. Steps to Reproduce (S2R): Detailed steps to replicate the issue.\n"
    "2. Expected Result (ER): What you expected to happen.\n"
    "3. Actual Result (AR): What actually happened.\n"
    "4. Additional Information: Include relevant details such as software version, build number, environment, etc.\n\n"
    "If any of these sections are missing from the provided report, explicitly notify the user which information is missing."
)

# --- 논문 프리셋 하이퍼파라미터 ---
PAPER_FINETUNING_PARAMS = {
    "max_seq_len": FINETUNING_PARAMS["max_seq_len"],
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "train_batch_size": 1,                                # 논문: per-device batch 8
    "grad_accum_steps": FINETUNING_PARAMS.get("grad_accum_steps", 1),
    "epochs": 3,
    "lr_qwen_mistral": 2e-4,                              # Qwen/Mistral
    "lr_llama": 3e-3,                                     # Llama-3.2-3B
    "grad_ckpt": FINETUNING_PARAMS.get("grad_ckpt", "unsloth"),
}

# --- 기존 qlora4 프리셋용 시스템 프롬프트 ---
DEFAULT_SFT_SYSTEM_PROMPT = (
    "You are a helpful QA assistant. Rewrite the user's bug summary into a clear, concise, and complete bug report.\n\n"
    "## Structured Output\n"
    "[Summary]\n<one sentence>\n\n"
    "[Steps to Reproduce]\n1. <step>\n2. <step>\n3. <step>\n\n"
    "[Expected Behavior]\n<text>\n\n"
    "[Actual Behavior]\n<text>\n\n"
    "[Environment]\n- OS: <>\n- Device/Browser: <>\n- App Version/Build: <>\n- Network: <>\n- Locale/Region: <>\n\n"
    "[Evidence]\n<one item>\n\n"
    "[Additional Info]\nFrequency: <>\nWorkaround: <>\n"
)

def set_seed(seed:int=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class PairCausalDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len:int, system_prompt:str):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.system_prompt = system_prompt

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        inp = str(r["NEW_llama_output"]).strip()
        tgt = str(r["text"]).strip()

        # Alpaca 형식: ### Input / ### Response
        prompt_text = self.system_prompt + "\n\n### Input:\n" + inp + "\n\n### Response:\n"
        prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tok(tgt, add_special_tokens=False).input_ids
        eos_id = self.tok.eos_token_id

        input_ids = prompt_ids + target_ids
        if eos_id is not None:
            input_ids = input_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + target_ids
        if eos_id is not None:
            labels = labels + [eos_id]

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels    = labels[:self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }

# -------- 커스텀 collator: 배치 내 최대 길이에 맞춰 pad --------
def make_causal_collator(tokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # pad 토큰이 없으면 eos로 설정 (causal LM 관례)
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id

    def collate(features):
        # features: list of dicts with tensors
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            L = f["input_ids"].size(0)
            pad_len = max_len - L
            input_ids.append(torch.nn.functional.pad(f["input_ids"], (0, pad_len), value=pad_id))
            attention_mask.append(torch.nn.functional.pad(f["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(f["labels"], (0, pad_len), value=-100))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
    return collate
# -------------------------------------------------------------

class UnslothFinetuner:
    # tuning_preset: "qlora4"(기본) | "paper"(논문)
    def __init__(self, model_name:str, seed:int=RANDOM_SEED, tuning_preset:str = "qlora4"):
        assert model_name in MODEL_ID, f"Unknown model_name {model_name}"
        assert tuning_preset in ["qlora4", "paper"], f"Unknown tuning_preset {tuning_preset}"
        self.model_name = model_name
        self.model_id = MODEL_ID[model_name]
        self.seed = seed
        self.tuning_preset = tuning_preset
        self.adapter_root = ADAPTER_DIRS[model_name] / self.tuning_preset

    def run(self, train_df) -> str:
        set_seed(self.seed)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.adapter_root / f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 프리셋 선택
        if self.tuning_preset == "paper":
            params = PAPER_FINETUNING_PARAMS
            system_prompt = PAPER_SFT_SYSTEM_PROMPT
            if "llama" in self.model_name.lower():
                lr = params["lr_llama"]
            else:
                lr = params["lr_qwen_mistral"]
        else:  # "qlora4"
            params = FINETUNING_PARAMS
            system_prompt = DEFAULT_SFT_SYSTEM_PROMPT
            lr = params["lr"]

        # Load base
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=params["max_seq_len"],
            load_in_4bit=True,
            dtype=None,
            device_map="auto",
        )

        # pad_token 보정 (없으면 eos로)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA with latest Unsloth API
        model = FastLanguageModel.get_peft_model(
            model,
            r=params["lora_r"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            bias="none",
            use_gradient_checkpointing=params.get("grad_ckpt", "unsloth"),
        )

        # Dataset & Trainer
        ds = PairCausalDataset(train_df, tokenizer, params["max_seq_len"], system_prompt=system_prompt)
        data_collator = make_causal_collator(tokenizer)

        args = TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=params.get("train_batch_size", FINETUNING_PARAMS.get("batch_size", 1)),
            gradient_accumulation_steps=params.get("grad_accum_steps", FINETUNING_PARAMS.get("grad_accum_steps", 1)),
            num_train_epochs=params["epochs"],
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=torch.cuda.is_available(),
            logging_steps=params.get("logging_steps", 10),
            save_steps=params.get("save_steps", 200),
            save_total_limit=params.get("save_total_limit", 2),
            report_to="none",
            # 길이 버킷팅을 켜면 같은 길이끼리 묶여 효율 ↑ (옵션)
            group_by_length=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=data_collator,   # ← 핵심: 패딩 포함 collator
        )
        trainer.train()

        # Save adapter
        model.save_pretrained(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))
        (self.adapter_root / "LATEST_ADAPTER.txt").write_text(str(run_dir), encoding="utf-8")
        return str(run_dir)
