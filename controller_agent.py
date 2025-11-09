from __future__ import annotations
import argparse, time, gc, sys, os, subprocess
from datetime import datetime

from config import RANDOM_SEED  # (유지)
from data_agent import load_and_split_data
from prompt_factory import QABugPromptFactory
from evaluation_agent import EvaluationAgent
from reporting import save_individual, save_summary, save_best_of_id, save_consolidated
from config import ADAPTER_DIRS, MODEL_ID


def _latest_adapter_or_fail(model_name: str, tuning_preset: str) -> str:
    """
    지정 preset(paper|qlora4)의 최신 어댑터 경로를 반환.
    """
    d = ADAPTER_DIRS[model_name] / tuning_preset
    latest_file = d / "LATEST_ADAPTER.txt"
    if latest_file.exists():
        p = latest_file.read_text(encoding="utf-8").strip()
        if p:
            return p
    cands = [x for x in d.iterdir() if x.is_dir()]
    if cands:
        latest = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        return str(latest)
    print(f"[ERROR] No adapter found for {model_name} (preset={tuning_preset}). Run with --train first.")
    raise SystemExit(1)


def _spawn_infer_subprocess(base_args: argparse.Namespace, mode: str):
    """
    학습 후 또는 별도 실행에서, 특정 'mode'로 추론을 '별도 프로세스(HF-only)'에서 수행.
    - 환경변수 UNSLOTH_DISABLE_BACKEND_PATCHING=1 강제
    - 동일 스크립트(controller_agent.py)를 --infer_only 로 다시 실행
    """
    cmd = [sys.executable, os.path.abspath(__file__),
           "--models", *base_args.models,
           "--batch_size", str(base_args.batch_size),
           "--mode", mode,
           "--tuning_preset", base_args.tuning_preset,
           "--infer_only"]

    # 사용자가 명시한 플래그는 그대로 전달(모드 기본보다 우선)
    if any(x in sys.argv for x in ["--few_shot_k"]):
        cmd.extend(["--few_shot_k", str(base_args.few_shot_k)])
    if any(x in sys.argv for x in ["--prompt"]):
        cmd.extend(["--prompt", base_args.prompt])
    if base_args.cot:
        cmd.append("--cot")
    elif any(x in sys.argv for x in ["--no_cot"]):
        cmd.append("--no_cot")
    if any(x in sys.argv for x in ["--adapter"]):
        cmd.extend(["--adapter", base_args.adapter])
    if base_args.greedy:
        cmd.append("--greedy")
    if base_args.test_run:
        cmd.append("--test_run")
    if base_args.cot_style:
        cmd.extend(["--cot_style", base_args.cot_style])
    if base_args.cot_file:
        cmd.extend(["--cot_file", base_args.cot_file])
    if base_args.force_resplit:
        cmd.append("--force_resplit")

    env = os.environ.copy()
    env["UNSLOTH_DISABLE_BACKEND_PATCHING"] = "1"  # HF-only 추론 보장

    print("\n[INFO] Spawning inference subprocess (HF-only)…")
    print("[INFO] CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print("[ERROR] Inference subprocess failed.")
        raise SystemExit(proc.returncode)


def _resolve_preset(mode: str, explicit_prompt: bool, explicit_cot: bool,
                    explicit_k: bool, explicit_adapter: bool):
    """
    모드별 기본값을 반환. 사용자가 해당 항목을 명시했다면(None 반환하여 기존 값을 유지하게 함).
    """
    if mode == "ctqrs":
        return dict(
            prompt=None if explicit_prompt else "ctqrs",
            cot=None if explicit_cot else False,
            k=None if explicit_k else 0,
            adapter=None if explicit_adapter else "base",
        )
    if mode == "cot":
        return dict(
            prompt=None if explicit_prompt else "baseline",
            cot=None if explicit_cot else True,
            k=None if explicit_k else 0,
            adapter=None if explicit_adapter else "base",
        )
    if mode == "retrieval1":
        return dict(
            prompt=None if explicit_prompt else "baseline",
            cot=None if explicit_cot else False,
            k=None if explicit_k else 1,
            adapter=None if explicit_adapter else "base",
        )
    if mode == "qlora4":
        return dict(
            prompt=None if explicit_prompt else "baseline",
            cot=None if explicit_cot else False,
            k=None if explicit_k else 0,
            adapter=None if explicit_adapter else "qlora4",
        )
    if mode == "all":
        return dict(
            prompt=None if explicit_prompt else "ctqrs",
            cot=None if explicit_cot else True,
            k=None if explicit_k else 1,
            adapter=None if explicit_adapter else "qlora4",
        )
    # fallback: 변경 없음
    return dict(prompt=None, cot=None, k=None, adapter=None)


def main():
    ap = argparse.ArgumentParser(description="Unified single-try pipeline (Qwen/Llama/Mistral)")
    ap.add_argument("--train", action="store_true", help="Run Unsloth QLoRA finetuning per model.")
    ap.add_argument("--few_shot_k", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--test_run", action="store_true")
    ap.add_argument("--force_resplit", action="store_true")
    ap.add_argument("--models", nargs="+", default=["qwen", "llama", "mistral"],
                    choices=["qwen", "llama", "mistral"])

    # 논문 파인튜닝 프리셋 선택
    ap.add_argument("--tuning_preset", choices=["qlora4", "paper"], default="qlora4",
                    help="Fine-tuning preset: 'qlora4'(default) or 'paper' (Acharya & Ginde 2025).")

    # Prompt/CoT/Adapter fine flags
    ap.add_argument("--prompt", choices=["ctqrs", "baseline"], default="ctqrs", help="Prompt style.")
    cot_grp = ap.add_mutually_exclusive_group()
    cot_grp.add_argument("--cot", dest="cot", action="store_true",
                         help="Enable CoT-like internal self-check.")
    cot_grp.add_argument("--no_cot", dest="cot", action="store_false",
                         help="Disable CoT-like internal self-check.")
    ap.set_defaults(cot=True)
    ap.add_argument("--cot_style", choices=["internal", "step", "deliberate", "custom"],
                    default="internal", help="CoT style block to inject when --cot.")
    ap.add_argument("--cot_file", type=str, default=None,
                    help="Path to custom CoT block text (UTF-8). Used when --cot_style=custom.")
    ap.add_argument("--adapter", choices=["qlora4", "base"], default="qlora4",
                    help="Use QLoRA adapter or base model for generation.")

    # High-level mode (single) and multi-mode
    ap.add_argument("--mode",
                    choices=["ctqrs", "cot", "retrieval1", "qlora4", "all"],
                    default=None,
                    help=("Shortcut modes:\n"
                          "  ctqrs      -> prompt=ctqrs,   no_cot, k=0, adapter=base\n"
                          "  cot        -> prompt=baseline, cot,   k=0, adapter=base\n"
                          "  retrieval1 -> prompt=baseline, no_cot,k=1, adapter=base\n"
                          "  qlora4     -> prompt=baseline, no_cot,k=0, adapter=qlora4\n"
                          "  all        -> prompt=ctqrs,    cot,   k=1, adapter=qlora4"))
    ap.add_argument("--modes",
                    nargs="+",
                    choices=["ctqrs", "cot", "retrieval1", "qlora4", "all"],
                    help="Run multiple modes sequentially (inference loops or subprocesses). "
                         "If provided, takes precedence over --mode.")

    # 내부용: 추론 전용 하위 프로세스 플래그(사용자 표시는 안 함)
    ap.add_argument("--infer_only", action="store_true", help=argparse.SUPPRESS)

    args = ap.parse_args()

    start = time.time()
    base_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 사용자가 명시했는지 여부(모드 기본값보다 우선 적용)
    explicit_prompt = any(x in sys.argv for x in ["--prompt"])
    explicit_cot = any(x in sys.argv for x in ["--cot", "--no_cot"])
    explicit_k = any(x in sys.argv for x in ["--few_shot_k"])
    explicit_adapter = any(x in sys.argv for x in ["--adapter"])

    # 1) data
    train_df, val_df, _ = load_and_split_data(force_resplit=args.force_resplit, test_run=args.test_run)

    # === 학습 분기 ===
    if args.train and not args.infer_only:
        # 학습(현재 프로세스에서 Unsloth 사용)
        from finetuning_agent import UnslothFinetuner  # ★ 지연 임포트
        # 학습 프리셋: --modes가 있으면 첫 모드 기준, 없으면 --mode/기본
        training_mode = (args.modes[0] if args.modes else (args.mode or "qlora4"))
        preset = _resolve_preset(training_mode, explicit_prompt, explicit_cot, explicit_k, explicit_adapter)

        # 논문 프리셋(paper)은 finetuning_agent 내부의 시스템 프롬프트를 사용
        # (기존 프롬프트 팩토리는 추론용에 주로 사용)
        train_k = args.few_shot_k if preset["k"] is None else preset["k"]
        use_ctqrs = (args.prompt if preset["prompt"] is None else preset["prompt"]) == "ctqrs"
        train_cot = args.cot if preset["cot"] is None else preset["cot"]
        _ = QABugPromptFactory(  # 생성만 해서 로그/설정 확인 용도
            train_df=train_df,
            k=train_k,
            use_ctqrs=use_ctqrs,
            enable_cot=train_cot,
            cot_style=args.cot_style,
            cot_custom_file=args.cot_file,
        )

        for model_name in args.models:
            print(f"\n=== [TRAIN] Model: {model_name} | mode={training_mode} | "
                  f"prompt={'ctqrs' if use_ctqrs else 'baseline'} | cot={train_cot} | k={train_k} | preset={args.tuning_preset} ===")
            if args.adapter != "qlora4":
                print("[WARN] --train is meaningful only with --adapter qlora4; proceeding with qlora4 finetune.")
            # 논문 프리셋이면 논문 하이퍼파라미터 및 Alpaca-역할 프롬프트 사용
            from finetuning_agent import UnslothFinetuner as _FT
            _ = _FT(model_name, tuning_preset=args.tuning_preset).run(train_df)
            gc.collect()

        # 학습 완료 후: 추론을 HF-only 별도 프로세스에서 모드별 실행
        modes_to_run = args.modes if args.modes else ([args.mode] if args.mode else ["ctqrs"])
        for m in modes_to_run:
            _spawn_infer_subprocess(args, m)

        total_sec = time.time() - start
        print(f"\n[INFO] Train finished. Inference triggered via subprocess(es). Elapsed: {total_sec:.1f}s")
        return

    # === 추론 전용(메인 실행 or 하위 프로세스 --infer_only) ===
    # 추론 프로세스에서 HF-only 강제
    os.environ.setdefault("UNSLOTH_DISABLE_BACKEND_PATCHING", "1")
    from generation_agent import UnifiedGenAgent  # ★ 지연 임포트

    all_results = []
    modes_to_run = args.modes if args.modes else ([args.mode] if args.mode else ["ctqrs"])

    # 2) per mode × per model (INFER)
    for mode in modes_to_run:
        # 모드별 프리셋 적용(명시 인자 있으면 보존)
        preset = _resolve_preset(mode, explicit_prompt, explicit_cot, explicit_k, explicit_adapter)
        eff_prompt = args.prompt if preset["prompt"] is None else preset["prompt"]
        eff_cot = args.cot if preset["cot"] is None else preset["cot"]
        eff_k = args.few_shot_k if preset["k"] is None else preset["k"]
        eff_adapter = args.adapter if preset["adapter"] is None else preset["adapter"]

        # 모드별 run_id
        run_id = f"{base_run_id}_{mode}"

        # 모드별 프롬프트/평가 객체
        eval_agent = EvaluationAgent()
        prompt_factory = QABugPromptFactory(
            train_df=train_df,
            k=eff_k,
            use_ctqrs=(eff_prompt == "ctqrs"),
            enable_cot=eff_cot,
            cot_style=args.cot_style,
            cot_custom_file=args.cot_file,
        )

        for model_name in args.models:
            print(f"\n=== [INFER] Model: {model_name} | mode={mode} | "
                  f"prompt={eff_prompt} | cot={eff_cot} | k={eff_k} | adapter={eff_adapter} | preset={args.tuning_preset} ===")

            tuning_tag = "qlora4" if eff_adapter == "qlora4" else "base"
            if eff_adapter == "qlora4":
                adapter_path = _latest_adapter_or_fail(model_name, args.tuning_preset)
            else:
                adapter_path = None  # base

            gen = UnifiedGenAgent(model_name=model_name, adapter_path=adapter_path,
                                  prompt_factory=prompt_factory, eval_agent=eval_agent,
                                  batch_size=args.batch_size, greedy=args.greedy,
                                  tuning_label=tuning_tag)
            results = gen.run(val_df)
            all_results.append(results)

            save_individual(results, model_name, eff_k, run_id, tuning_tag=tuning_tag)

            if results:
                import pandas as pd
                df = pd.DataFrame(results)
                print("Final Averages -",
                      model_name,
                      f"CTQRS={df['gen_ctqrs'].mean():.3f}",
                      f"R1-R={df['gen_rouge1_r'].mean():.3f}",
                      f"R1-F1={df['gen_rouge1_f1'].mean():.3f}",
                      f"SBERT={df['gen_sbert'].mean():.3f}",
                      f"SCORE={df['score_v2'].mean():.3f}",
                      sep=" | ")

            import torch
            del gen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 모드별 요약/베스트
        if any(len(r) > 0 for r in all_results):
            flat = [x for x in all_results for x in x]  # 간단히 누적 기반 사용
            save_summary(flat, run_id, tuning_tag=f"{tuning_tag}-{args.tuning_preset}")
            save_best_of_id(all_results, run_id, tuning_tag=f"{tuning_tag}-{args.tuning_preset}")

    # 3) 전체 consolidated (모드 전부 포함)
    total_sec = time.time() - start
    save_consolidated(all_results, total_sec, f"{base_run_id}_multi")


if __name__ == "__main__":
    main()
