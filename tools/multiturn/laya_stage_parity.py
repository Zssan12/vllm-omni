"""A LAYA prompt variant through the stage's own path vs the laya package's scores.

Builds each prompt with response_judge._laya_prompt from the response_judge options this
variant would get in a deploy file, runs it through LayaDecisionModel (vLLM pooling, CUDA
graphs with the LAYA overlay's capture sizes), turns the logits into P(reply) the same way
the stage does, and compares with the laya package's P(reply) from a laya_prompt_tuning.py
raw.jsonl. Also prints the response_judge block to paste into a deploy file.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=<vllm-omni tree> python laya_stage_parity.py \
        --vllm-model-dir <prepare_laya_dir.py output> --variants tools/multiturn/variants/laya-best.json \
        --variant en-choice-4way-heard --threshold 0.0173 --raw multiturn-zh/results/judge-text/t4-laya-qtts/raw.jsonl --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

CAPTURE = [16, 32, 48, 64, 96, 128, 192, 256]  # deploy/aura_omni_judged_laya.yaml


def judge_options(variant: dict, threshold: float) -> dict:
    # laya_prompt_tuning uses "{t}" and "reply"; the stage uses "{transcript}" and "reply_option".
    return {"format": "laya", "question_type": variant["type"], "instructions": variant["ins"],
            "options": variant["crit"], "reply_option": variant["reply"], "threshold": threshold,
            "state_template": variant["state"].replace("{t}", "{transcript}")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-model-dir", required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    import torch
    from vllm import LLM, PoolingParams
    from vllm.plugins import load_general_plugins

    from vllm_omni.model_executor.stage_input_processors import response_judge as rj

    variant = json.loads(a.variants.read_text())[a.variant]
    if variant["type"] != "choice":
        raise SystemExit("only choice variants map to the stage's option scores")
    options = judge_options(variant, a.threshold)
    spec = rj.JudgeSpec("laya", {k: v for k, v in options.items() if k != "format"})
    load_general_plugins()
    llm = LLM(model=a.vllm_model_dir, runner="pooling", max_num_seqs=8, max_model_len=1024,
              max_num_batched_tokens=1024, enable_prefix_caching=False, dtype="float32",
              gpu_memory_utilization=a.gpu_memory_utilization,
              compilation_config={"cudagraph_capture_sizes": CAPTURE})
    model_config = llm.llm_engine.model_config
    keys = list(options["options"])
    params = PoolingParams(task="classify")

    ref = [json.loads(line) for line in a.raw.read_text().splitlines()]
    ref = [r for r in ref if r["variant"] == a.variant]
    for r in ref[:5]:
        llm.encode([rj._laya_prompt(spec, r["transcript"], model_config)], pooling_params=params,
                   pooling_task="classify", use_tqdm=False)
    rows = []
    for r in ref:
        prompt = rj._laya_prompt(spec, r["transcript"], model_config)
        t0 = time.perf_counter()
        out = llm.encode([prompt], pooling_params=params, pooling_task="classify", use_tqdm=False)
        torch.cuda.synchronize()
        call_ms = (time.perf_counter() - t0) * 1000
        logits = out[0].outputs.data.float().cpu().flatten()
        probs = torch.softmax(logits, dim=-1).tolist()
        p_stage = probs[keys.index(options["reply_option"])] if len(probs) == len(keys) else None
        rejected = rj._option_scores_reject(spec, out[0])  # the stage's own decision
        rows.append({"split": r["split"], "id": r["id"], "transcript": r["transcript"], "reply": r["reply"],
                     "p_laya": r["p_reply"], "p_stage": p_stage, "stage_rejects": rejected,
                     "laya_rejects": r["p_reply"] < a.threshold, "prompt_tokens": len(prompt["prompt_token_ids"]),
                     "call_ms": round(call_ms, 3)})
    diffs = [abs(r["p_stage"] - r["p_laya"]) for r in rows if r["p_stage"] is not None]
    summary = {
        "variant": a.variant, "threshold": a.threshold, "n": len(rows),
        "p_missing": sum(r["p_stage"] is None for r in rows),
        "decision_mismatch": sum(r["stage_rejects"] != r["laya_rejects"] for r in rows),
        "p_abs_diff_max": max(diffs) if diffs else None,
        "p_abs_diff_median": statistics.median(diffs) if diffs else None,
        "near_threshold_x2": sum(a.threshold / 2 <= r["p_laya"] <= a.threshold * 2 for r in rows),
        "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
        "call_ms_p50_back_to_back": round(statistics.median(r["call_ms"] for r in rows), 2),
        "response_judge": options,
    }
    (a.out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
