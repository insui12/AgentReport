from __future__ import annotations
from datetime import datetime
import pandas as pd
from config import INDIVIDUAL_COLUMNS, SUMMARY_COLUMNS, INDIVIDUAL_DIR, SUMMARY_DIR, REPORTS_DIR

def save_individual(results: list[dict], model_name: str, few_shot_k:int, run_id:str, tuning_tag:str="qlora4"):
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    out = pd.DataFrame(columns=INDIVIDUAL_COLUMNS)
    for c in INDIVIDUAL_COLUMNS:
        if c in df.columns: out[c] = df[c]
    path = INDIVIDUAL_DIR / f"{model_name}_{tuning_tag}_k{few_shot_k}_{run_id}.xlsx"
    out.to_excel(path, index=False)
    return path

def save_summary(model_rows: list[dict], run_id:str, tuning_tag:str="qlora4"):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(model_rows)
    best_rows = []
    for bug_id, gdf in df.groupby("bug_id"):
        g = gdf.sort_values(
            by=["gen_ctqrs","gen_rouge1_f1","gen_rouge1_r","gen_sbert","decode_mode","model"],
            ascending=[False, False, False, False, True, True]
        ).iloc[0]
        best_rows.append({
            "bug_id": int(bug_id),
            "model": g["model"],
            "best_report": g["gen"],
            "ctqrs": float(g["gen_ctqrs"]),
            "rouge1_r": float(g["gen_rouge1_r"]),
            "rouge1_f1": float(g["gen_rouge1_f1"]),
            "sbert": float(g["gen_sbert"]),
            "score_v2": float(g["score_v2"]),
        })
    out = pd.DataFrame(best_rows, columns=SUMMARY_COLUMNS)
    path = SUMMARY_DIR / f"summary_{tuning_tag}_{run_id}.xlsx"
    out.to_excel(path, index=False)
    return path

def save_best_of_id(all_results:list[list[dict]], run_id:str, tuning_tag:str="qlora4"):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.DataFrame(x) for x in all_results if len(x) > 0], ignore_index=True)
    rows = []
    for bug_id, group in df.groupby("bug_id"):
        g = group.sort_values(
            by=["gen_ctqrs","gen_rouge1_f1","gen_rouge1_r","gen_sbert","decode_mode","model"],
            ascending=[False, False, False, False, True, True]
        ).iloc[0]
        rows.append({
            "bug_id": int(bug_id),
            "best_model": g["model"],
            "best_decode_mode": g["decode_mode"],
            "best_report": g["gen"],
            "ctqrs": float(g["gen_ctqrs"]),
            "rouge1_r": float(g["gen_rouge1_r"]),
            "rouge1_f1": float(g["gen_rouge1_f1"]),
            "sbert": float(g["gen_sbert"]),
            "score_v2": float(g["score_v2"]),
        })
    out = pd.DataFrame(rows, columns=["bug_id","best_model","best_decode_mode","best_report",
                                      "ctqrs","rouge1_r","rouge1_f1","sbert","score_v2"])
    path = SUMMARY_DIR / f"summary_best_of_id_{tuning_tag}_{run_id}.xlsx"
    out.to_excel(path, index=False)
    return path

def save_consolidated(all_results:list[list[dict]], total_sec:float, run_id:str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, "elapsed_sec": float(total_sec)}
    for model_name, rows in _iter_by_model(all_results):
        df = pd.DataFrame(rows)
        if len(df) == 0: continue
        row[f"{model_name}_avg_ctqrs"] = float(df["gen_ctqrs"].mean())
        row[f"{model_name}_avg_rouge1_r"] = float(df["gen_rouge1_r"].mean())
        row[f"{model_name}_avg_rouge1_f1"] = float(df["gen_rouge1_f1"].mean())
        row[f"{model_name}_avg_sbert"] = float(df["gen_sbert"].mean())
        row[f"{model_name}_avg_score_v2"] = float(df["score_v2"].mean())
    out = pd.DataFrame([row])
    path = REPORTS_DIR / f"consolidated_report_{run_id}.xlsx"
    out.to_excel(path, index=False)
    return path

def _iter_by_model(all_results):
    by = {}
    for lst in all_results:
        for r in lst:
            by.setdefault(r["model"], []).append(r)
    return by.items()
