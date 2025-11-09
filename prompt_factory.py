from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import BGE_MODEL_ID

# =========================
# Base System Prompts (no CoT inside)
# =========================
CTQRS_BASE_PROMPT = (
    "You are an expert QA engineer. Convert the user's unstructured input into a CTQRS-maximized bug report.\n\n"
    "## Rules\n"
    "- Include EXACTLY these sections: [Summary], [Steps to Reproduce], [Expected Behavior], [Actual Behavior], "
    "[Environment], [Evidence], [Additional Info].\n"
    "- Steps to Reproduce: CLEAR, NUMBERED, ≥3 steps; each line = ONE user action - ONE observable result.\n"
    "- Expected vs Actual: directly comparable; reuse the same key nouns.\n"
    "- Environment: include ≥4 details among {OS, Device/Browser, App Version/Build, Network, Locale/Region}.\n"
    "- Evidence: provide EXACTLY ONE realistic item (short log line OR concrete file path OR screenshot filename).\n"
    "- If a detail is missing, infer a conservative, plausible value and prefix with '[inferred]'.\n\n"
    "## Structured Output\n"
    "[Summary]\n"
    "<one sentence: component/module + trigger condition + failure symptom>\n\n"
    "[Steps to Reproduce]\n"
    "1. <user action> - <immediate result/observation>\n"
    "2. <user action> - <immediate result/observation>\n"
    "3. <user action> - <immediate result/observation>\n"
    "4. <optional extra user action> - <result>\n\n"
    "[Expected Behavior]\n"
    "<precise, testable outcome using the same key nouns as in Actual>\n\n"
    "[Actual Behavior]\n"
    "<what actually happens; include any error text or visible symptom; reuse the same key nouns>\n\n"
    "[Environment]\n"
    "- OS: <name version>\n"
    "- Device/Browser: <model or browser name+version>\n"
    "- App Version/Build: <semver or build tag>\n"
    "- Network: <Wi-Fi/LTE/VPN/Proxy>\n"
    "- Locale/Region: <e.g., en-US>\n\n"
    "[Evidence]\n"
    "<ONE of: short log line with error code | concrete file path | screenshot filename>\n"
    "e.g., \"ERROR 503 at startup\", \"/var/log/app/error_2025-09-07.log\", \"screenshot_2025-09-07_102314.png\"\n\n"
    "[Additional Info]\n"
    "Frequency: always / often / sometimes\n"
    "Workaround: <if any>\n"
)

BASELINE_BASE_PROMPT = (
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

# =========================
# Unified CoT Block (single)
# =========================
COT_BLOCK = (
    "## Internal Review (silent)\n"
    "- Before finalizing, self-check and FIX silently if missing: S2R ≥ 3 and numbered; Expected–Actual comparable (shared nouns); Environment ≥ 4 details; exactly ONE Evidence present; Summary has component + trigger + symptom.\n"
    "- Do NOT print this review.\n\n"
)

def _assemble_system_prompt(
    use_ctqrs: bool = True,
    enable_cot: bool = True,
    cot_custom_block: Optional[str] = None,   # if given, overrides the default COT_BLOCK
) -> str:
    base = CTQRS_BASE_PROMPT if use_ctqrs else BASELINE_BASE_PROMPT
    if not enable_cot:
        return base

    # choose the single CoT block (custom wins)
    cot_block = (cot_custom_block.strip() + "\n\n") if cot_custom_block else COT_BLOCK

    # insert CoT block right before the structured output marker
    marker = "## Structured Output"
    pos = base.find(marker)
    if pos != -1:
        return base[:pos] + cot_block + base[pos:]
    return cot_block + base  # fallback if marker is missing

class QABugPromptFactory:
    """
    Builds prompts and (optionally) attaches k example pairs via dense retrieval.

    Args:
        train_df: training dataframe for retrieval examples.
        k: number of retrieved examples (0..3).
        use_ctqrs: use CTQRS base prompt (if False, baseline prompt).
        enable_cot: whether to insert the unified CoT block.
        cot_style: (ignored; kept for backward-compatibility)
        cot_custom_text: replace CoT block with this text if provided.
        cot_custom_file: path to a file that contains the CoT block (UTF-8). Overrides cot_custom_text.
    """
    def __init__(
        self,
        train_df: pd.DataFrame,
        k: int = 0,
        *,
        use_ctqrs: bool = True,
        enable_cot: bool = True,
        cot_style: str = "internal",           # kept but ignored
        cot_custom_text: Optional[str] = None,
        cot_custom_file: Optional[str] = None,
    ):
        self.k = max(0, min(3, int(k)))
        self.train_df = train_df.reset_index(drop=True) if train_df is not None else pd.DataFrame()
        self.use_few_shot = (self.k > 0 and len(self.train_df) > 0)

        # load custom CoT if provided
        cot_custom_block = None
        if cot_custom_file:
            try:
                with open(cot_custom_file, "r", encoding="utf-8") as f:
                    cot_custom_block = f.read()
            except Exception as e:
                print(f"[WARN] Failed to read cot_custom_file: {e}")
        if cot_custom_block is None:
            cot_custom_block = cot_custom_text

        self.system_prompt = _assemble_system_prompt(
            use_ctqrs=use_ctqrs,
            enable_cot=enable_cot,
            cot_custom_block=cot_custom_block,
        )

        self.index = None
        self.search_model = None
        if self.use_few_shot:
            self._build_faiss()

    # ---- Retrieval index
    def _build_faiss(self):
        self.search_model = SentenceTransformer(BGE_MODEL_ID, device="cuda" if self._has_cuda() else "cpu")
        corpus_texts = [str(x).strip() for x in self.train_df["NEW_llama_output"].tolist()]
        embs = self.search_model.encode(
            corpus_texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        embs = embs.astype(np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ---- Few-shot block
    def _few_shot_block(self, input_summary: str) -> str:
        if not self.use_few_shot or self.index is None or self.search_model is None:
            return ""
        k = min(self.k, len(self.train_df))
        if k <= 0:
            return ""
        q = self.search_model.encode([input_summary], show_progress_bar=False, normalize_embeddings=True)
        q = q.astype(np.float32)
        _, idxs = self.index.search(q, k)
        exs = []
        for idx in idxs[0]:
            row = self.train_df.iloc[int(idx)]
            ex_in = str(row.get("NEW_llama_output", "")).strip()
            ex_out = str(row.get("text", "")).strip()
            exs.append(f"[Example Input]\n{ex_in}\n\n[Example Output]\n{ex_out}\n")
        return "--- Relevant Examples ---\n" + "\n".join(exs) + "--- End of Examples ---\n\n"

    # ---- Final prompt
    def create_prompt(self, input_summary: str) -> str:
        return (
            self.system_prompt
            + self._few_shot_block(input_summary)
            + f"[Input to Convert]\n{input_summary}\n\n[Output Bug Report]\n"
        )
