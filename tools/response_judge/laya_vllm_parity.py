"""Parity check: laya package vs LayaDecisionModel served by vLLM (pooling runner).

Phase ``laya``: run the reference ``laya`` agent on the prompt-variant cases,
capture every question row's exact input ids and raw option logits.
Phase ``vllm``: feed the same ids to vLLM and compare logits / decisions.
Both phases write into ``--out`` (a new directory); nothing is overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def phase_laya(a) -> None:
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, a.laya_pkg)
    import laya
    import torch

    agent = laya.load(a.model_dir, device="cuda")
    captured: list[dict] = []
    original = agent._infer

    def infer(b):
        logits, act = original(b)
        for r in range(b["input_ids"].shape[0]):
            mask = b["attention_mask"][r].bool()
            k = int(b["marker_mask"][r].sum())
            captured.append(
                {
                    "ids": b["input_ids"][r][mask].tolist(),
                    "logits": logits[r, :k].float().cpu().tolist(),
                }
            )
        return logits, act

    agent._infer = infer
    rows = []
    for path in a.cases:
        for case in json.loads(Path(path).read_text()):
            for qid, q in case["questions"].items():
                start = len(captured)
                result = agent.predict(case["state"], {qid: q})
                assert len(captured) == start + 1, "one question -> one row"
                row = captured[start]
                row.update(file=Path(path).name, id=case["id"], question=qid, answer=result[qid] if qid in result else result)
                rows.append(row)
    torch.cuda.synchronize()
    (a.out / "laya.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(json.dumps({"phase": "laya", "rows": len(rows), "torch": torch.__version__}))


def phase_vllm(a) -> None:
    import torch
    from vllm import LLM, PoolingParams
    from vllm.plugins import load_general_plugins

    # Registers vllm-omni's LayaDecisionModel (the class the stage uses);
    # run with PYTHONPATH pointing at the vllm-omni tree under test.
    load_general_plugins()
    rows = [json.loads(line) for line in (a.out / "laya.jsonl").read_text().splitlines() if line]
    llm = LLM(
        model=str(a.vllm_model_dir),
        runner="pooling",
        skip_tokenizer_init=True,
        dtype=a.dtype,
        max_model_len=1024,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enforce_eager=a.enforce_eager,
    )
    params = PoolingParams(task="classify")
    prompts = [{"prompt_token_ids": r["ids"]} for r in rows]
    # warmup, then one request at a time for latency, then batched for parity
    for p in prompts[:3]:
        llm.encode([p], pooling_params=params, pooling_task="classify", use_tqdm=False)
    lat = []
    single = []
    for p in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.encode([p], pooling_params=params, pooling_task="classify", use_tqdm=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        single.append(out[0].outputs.data.float().cpu().tolist())
    max_abs = 0.0
    agree = 0
    report = []
    for r, got in zip(rows, single):
        ref = r["logits"]
        diff = max(abs(x - y) for x, y in zip(ref, got)) if len(ref) == len(got) else float("inf")
        max_abs = max(max_abs, diff)
        same = len(ref) == len(got) and ref.index(max(ref)) == got.index(max(got))
        agree += same
        report.append({"file": r["file"], "id": r["id"], "question": r["question"], "ref": ref, "vllm": got, "max_abs_diff": diff, "argmax_agree": same})
    (a.out / "vllm.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in report) + "\n")
    lat_sorted = sorted(lat)
    summary = {
        "phase": "vllm",
        "rows": len(rows),
        "argmax_agree": agree,
        "max_abs_logit_diff": max_abs,
        "dtype": a.dtype,
        "enforce_eager": a.enforce_eager,
        "latency_ms_p50": lat_sorted[len(lat) // 2],
        "latency_ms_p95": lat_sorted[min(len(lat) - 1, int(0.95 * len(lat) + 0.999) - 1)],
        "note": "LLM.encode one prompt at a time, offline API, includes engine step overhead",
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("laya", "vllm"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", nargs="+", default=[])
    p.add_argument("--model-dir", help="original laya checkpoint dir")
    p.add_argument("--laya-pkg")
    p.add_argument("--vllm-model-dir", type=Path)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--enforce-eager", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    phase_laya(a) if a.phase == "laya" else phase_vllm(a)


if __name__ == "__main__":
    main()
