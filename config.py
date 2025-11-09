from __future__ import annotations
from pathlib import Path
import os

# ---- Paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "Plus14_filtered_bug_report_scores_Summary.xlsx"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
INDIVIDUAL_DIR = RESULTS_DIR / "individual"
SUMMARY_DIR = RESULTS_DIR / "summary"
REPORTS_DIR = RESULTS_DIR / "reports"
CACHE_DIR = RESULTS_DIR / "cache"

# ---- Base models (WSL local)
DEFAULT_MODELS = {
    "qwen":   "/mnt/c/Users/selab/LLM_Models/models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit",
    "llama":  "/mnt/c/Users/selab/LLM_Models/models--unsloth--llama-3.2-3b-instruct-unsloth-bnb-4bit",
    "mistral":"/mnt/c/Users/selab/LLM_Models/models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit",
}
# Allow env override: QWEN_MODEL_ID, LLAMA_MODEL_ID, MISTRAL_MODEL_ID
MODEL_ID = {
    "qwen": os.getenv("QWEN_MODEL_ID", DEFAULT_MODELS["qwen"]),
    "llama": os.getenv("LLAMA_MODEL_ID", DEFAULT_MODELS["llama"]),
    "mistral": os.getenv("MISTRAL_MODEL_ID", DEFAULT_MODELS["mistral"]),
}

# ---- Adapters
ADAPTER_ROOT = Path(os.getenv("ADAPTER_ROOT", PROJECT_ROOT / "adapters"))
ADAPTER_DIRS = {
    "qwen": ADAPTER_ROOT / "qwen2.5-7b",
    "llama": ADAPTER_ROOT / "llama-3.2-3b",
    "mistral": ADAPTER_ROOT / "mistral-7b",
}
for p in ADAPTER_DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

# ---- FT Params
FINETUNING_PARAMS = {
    "max_epochs": 3,
    "batch_size": 1,
    # 통일: 트레이너가 사용하는 키 이름
    "grad_accum_steps": 8,
    "max_seq_len": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lr": 2e-4,
}

# ---- Eval/Prompt
SBERT_MODEL = "all-mpnet-base-v2"
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"

# ---- Reporting profile (unified_multi)
INDIVIDUAL_COLUMNS = [
    "bug_id", "tuning", "model", "prompt", "decode_mode",
    "gen", "gen_ctqrs", "gen_rouge1_r", "gen_rouge1_f1", "gen_sbert", "score_v2"
]
SUMMARY_COLUMNS = ["bug_id", "model", "best_report", "ctqrs", "rouge1_r", "rouge1_f1", "sbert", "score_v2"]

# ---- Misc
RANDOM_SEED = 42
