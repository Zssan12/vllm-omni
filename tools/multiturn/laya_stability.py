"""Stability of one LAYA judge config through the stage's own path (standalone, no main model).

Same engine settings as the judge stage in deploy/aura_omni_judged_laya.yaml (pooling, fp32,
max_num_seqs 8, CUDA graphs with the overlay's capture sizes). Prompts come from
response_judge._laya_prompt and decisions from response_judge._option_scores_reject.

1. flip:  every transcript scored alone, then in batches of 2, 4 and 8 (one llm.encode call per
          batch, so the scheduler runs them together). Reports P(reply) differences against the
          batch-1 run and decisions that flip, plus a second batch-1 pass for determinism.
2. soak:  for --minutes, bursts of --burst back-to-back calls separated by random idle gaps
          (--gap-min-s .. --gap-max-s), cycling through the transcripts. Per call: latency and
          decision; every --sample-s: CUDA allocated/reserved, process RSS. Reports latency per
          minute (first call after a gap vs in a burst), memory drift and decision consistency.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=<vllm-omni tree> python laya_stability.py \
        --vllm-model-dir <laya dir> --judge-json multiturn-zh/results/parity/summary.json \
        --cases multiturn-zh/results/cases/tune-asr-qtts.json multiturn-zh/results/cases/heldout-asr-qtts.json --out <new dir> --minutes 12
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path

CAPTURE = [16, 32, 48, 64, 96, 128, 192, 256]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-model-dir", required=True)
    p.add_argument("--judge-json", type=Path, required=True, help="JSON with a 'response_judge' block")
    p.add_argument("--cases", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--minutes", type=float, default=12)
    p.add_argument("--burst", type=int, default=20)
    p.add_argument("--gap-min-s", type=float, default=0.2)
    p.add_argument("--gap-max-s", type=float, default=1.5)
    p.add_argument("--sample-s", type=float, default=30)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    import torch
    from vllm import LLM, PoolingParams
    from vllm.plugins import load_general_plugins

    from vllm_omni.model_executor.stage_input_processors import response_judge as rj

    options = dict(json.loads(a.judge_json.read_text())["response_judge"])
    options.pop("format", None)
    spec = rj.JudgeSpec("laya", options)
    keys = list(options["options"])
    reply_key = options["reply_option"]
    threshold = float(options["threshold"])
    load_general_plugins()
    llm = LLM(model=a.vllm_model_dir, runner="pooling", max_num_seqs=8, max_model_len=1024,
              max_num_batched_tokens=8192, enable_prefix_caching=False, dtype="float32",
              gpu_memory_utilization=a.gpu_memory_utilization,
              compilation_config={"cudagraph_capture_sizes": CAPTURE})
    mc = llm.llm_engine.model_config
    params = PoolingParams(task="classify")
    cases = [c for path in a.cases for c in json.loads(path.read_text())]
    prompts = [rj._laya_prompt(spec, c["state"]["transcript"], mc) for c in cases]

    def encode(batch):
        outs = llm.encode(batch, pooling_params=params, pooling_task="classify", use_tqdm=False)
        res = []
        for o in outs:
            probs = torch.softmax(o.outputs.data.float().cpu().flatten(), dim=-1).tolist()
            res.append((probs[keys.index(reply_key)], rj._option_scores_reject(spec, o)))
        return res

    for i in range(0, 16, 8):  # warm up graphs of several sizes
        encode(prompts[i:i + 8])

    # 1. flip
    runs = {}
    for label, size in (("b1", 1), ("b2", 2), ("b4", 4), ("b8", 8), ("b1_repeat", 1)):
        res = []
        for i in range(0, len(prompts), size):
            res.extend(encode(prompts[i:i + size]))
        runs[label] = res
    base = runs["b1"]
    flip = {}
    for label, res in runs.items():
        if label == "b1":
            continue
        diffs = [abs(x[0] - y[0]) for x, y in zip(res, base)]
        flips = [i for i, (x, y) in enumerate(zip(res, base)) if x[1] != y[1]]
        flip[label] = {"max_abs_diff": max(diffs), "median_abs_diff": statistics.median(diffs),
                       "decision_flips": len(flips),
                       "flipped": [{"id": cases[i]["id"], "transcript": cases[i]["state"]["transcript"],
                                    "p_b1": base[i][0], "p": res[i][0]} for i in flips]}
    near = sum(threshold / 2 <= p <= threshold * 2 for p, _ in base)
    flip_summary = {"n": len(prompts), "threshold": threshold, "near_threshold_x2": near, "vs_b1": flip}
    (a.out / "flip.json").write_text(json.dumps(flip_summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: (v if k != "vs_b1" else {l: {kk: vv for kk, vv in d.items() if kk != "flipped"}
                                                     for l, d in v.items()}) for k, v in flip_summary.items()}),
          flush=True)

    # 2. soak
    rng = random.Random(0)
    ref = {i: base[i] for i in range(len(prompts))}
    calls, samples = [], []
    decision_changes = 0
    p_dev = 0.0
    t_start = time.monotonic()
    next_sample = t_start
    i = 0
    while time.monotonic() - t_start < a.minutes * 60:
        for k in range(a.burst):
            idx = i % len(prompts)
            i += 1
            t0 = time.perf_counter()
            (pr, rej), = encode([prompts[idx]])
            ms = (time.perf_counter() - t0) * 1000
            decision_changes += rej != ref[idx][1]
            p_dev = max(p_dev, abs(pr - ref[idx][0]))
            calls.append((time.monotonic() - t_start, ms, k == 0))
        now = time.monotonic()
        if now >= next_sample:
            samples.append({"t_s": round(now - t_start, 1), "calls": len(calls),
                            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
                            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)})
            next_sample = now + a.sample_s
        time.sleep(rng.uniform(a.gap_min_s, a.gap_max_s))
    per_min = defaultdict(lambda: {"after_gap": [], "burst": []})
    for t, ms, first in calls:
        per_min[int(t // 60)]["after_gap" if first else "burst"].append(ms)

    def pct(v, q):
        v = sorted(v)
        return round(v[min(len(v) - 1, int(q * len(v)))], 2) if v else None

    minutes = {m: {"n": len(d["after_gap"]) + len(d["burst"]),
                   "after_gap_p50": pct(d["after_gap"], 0.5), "after_gap_p99": pct(d["after_gap"], 0.99),
                   "burst_p50": pct(d["burst"], 0.5), "burst_p99": pct(d["burst"], 0.99)}
               for m, d in sorted(per_min.items())}
    soak = {"minutes": a.minutes, "calls": len(calls), "decision_changes_vs_b1": decision_changes,
            "max_p_deviation_vs_b1": p_dev,
            "after_gap_ms": {"p50": pct([c[1] for c in calls if c[2]], 0.5), "p99": pct([c[1] for c in calls if c[2]], 0.99),
                             "max": round(max(c[1] for c in calls if c[2]), 2)},
            "burst_ms": {"p50": pct([c[1] for c in calls if not c[2]], 0.5),
                         "p99": pct([c[1] for c in calls if not c[2]], 0.99),
                         "max": round(max(c[1] for c in calls if not c[2]), 2)},
            "per_minute": minutes, "memory_samples": samples}
    (a.out / "soak.json").write_text(json.dumps(soak, indent=1))
    (a.out / "soak-calls.jsonl").write_text("\n".join(json.dumps({"t_s": round(t, 3), "ms": round(ms, 3), "after_gap": f})
                                                      for t, ms, f in calls) + "\n")
    print(json.dumps({k: v for k, v in soak.items() if k not in ("per_minute", "memory_samples")}), flush=True)
    print(json.dumps({"memory_first": samples[0] if samples else None, "memory_last": samples[-1] if samples else None}))


if __name__ == "__main__":
    main()
