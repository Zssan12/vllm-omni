"""CLM judge: CUDA graph capture sizes vs judge-call latency (never part of the PR).

Loads a CLM model directory prepared by clm_tune_and_prepare.py as the stage's
ClmDecisionModel, in-process, with the judge-stage engine arguments of
deploy/aura_omni_judged_clm.yaml (max_num_seqs 8 inherited from the base file,
no prefix cache, CUDA graphs on). Prompts come from the stage's own builder
(response_judge._clm_prompt with the directory's response_judge options). Run
once without and once with --capture-sizes; each run times back-to-back calls
and calls after an idle gap, and records P(reply) so runs can be compared.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=<vllm-omni tree> python clm_capture_check.py \
        --model-dir <prepared>/model --cases judge-heldout-zh.json --out <new dir> [--capture-sizes 16 32 ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--capture-sizes", type=int, nargs="*")
    p.add_argument("--gaps", type=float, nargs="+", default=[0.0, 2.0])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    import torch
    from vllm import LLM, PoolingParams
    from vllm.plugins import load_general_plugins

    from vllm_omni.model_executor.stage_input_processors import response_judge as rj

    load_general_plugins()
    kwargs = {}
    if a.capture_sizes:
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": a.capture_sizes}
    llm = LLM(
        model=a.model_dir,
        runner="pooling",
        max_num_seqs=8,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        gpu_memory_utilization=a.gpu_memory_utilization,
        **kwargs,
    )
    core = llm.llm_engine.engine_core.engine_core
    wrapper = core.model_executor.driver_worker
    runner = getattr(wrapper, "worker", wrapper).model_runner
    exec_ms: list[float] = []
    orig = runner.execute_model

    def execute_model(*args, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*args, **kw)
        torch.cuda.synchronize()
        exec_ms.append((time.perf_counter() - t0) * 1000)
        return out

    runner.execute_model = execute_model
    captured = sorted(getattr(llm.llm_engine.vllm_config.compilation_config, "cudagraph_capture_sizes", None) or [])

    model_config = llm.llm_engine.model_config
    spec = rj.JudgeSpec.from_hf_config(model_config.hf_config)
    assert spec.format == "clm", spec
    keys = list(spec.options.get("option_keys") or [])
    reply = spec.options.get("reply_option")
    reply = [reply] if isinstance(reply, str) else list(reply or [])
    cases = [c for c in json.loads(a.cases.read_text()) if c["expected"].get("response_needed")][: a.n]
    prompts = [(c["id"], rj._clm_prompt(spec, c["state"]["transcript"], model_config)) for c in cases]
    params = PoolingParams(task="classify")
    for _, prompt in prompts[:5]:
        llm.encode([prompt], pooling_params=params, pooling_task="classify", use_tqdm=False)

    rows = []
    summary = {}
    for gap in a.gaps:
        label = "burst" if gap == 0 else f"gap{gap:g}"
        for case_id, prompt in prompts:
            if gap:
                time.sleep(gap)
            exec_ms.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = llm.encode([prompt], pooling_params=params, pooling_task="classify", use_tqdm=False)
            torch.cuda.synchronize()
            logits = out[0].outputs.data.float().cpu()
            probs = torch.softmax(logits, dim=-1).tolist()
            p_reply = sum(probs[keys.index(k)] for k in reply if k in keys) if keys else None
            rows.append({"schedule": label, "id": case_id, "call_ms": (time.perf_counter() - t0) * 1000,
                         "execute_model_ms": sum(exec_ms), "p_reply": p_reply,
                         "prompt_tokens": len(out[0].prompt_token_ids or [])})
        mine = [r for r in rows if r["schedule"] == label]
        summary[label] = {
            "n": len(mine),
            "call_ms_p50": round(statistics.median(r["call_ms"] for r in mine), 2),
            "execute_model_ms_p50": round(statistics.median(r["execute_model_ms"] for r in mine), 2),
            "prompt_tokens_p50": statistics.median(r["prompt_tokens"] for r in mine),
        }
    meta = {"capture_sizes_requested": a.capture_sizes, "capture_sizes_effective": captured, "n": len(prompts),
            "threshold": spec.options.get("threshold"), "reply_option": reply}
    (a.out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    (a.out / "summary.json").write_text(json.dumps({"meta": meta, "median": summary}, indent=2))
    print(json.dumps({"meta": meta, "median": summary}, indent=1))


if __name__ == "__main__":
    main()
