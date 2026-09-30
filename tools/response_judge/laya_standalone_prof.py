"""Standalone LAYA judge timing with and without idle gaps (never part of the PR).

Loads LayaDecisionModel in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) with the
same engine arguments as the pipeline's judge stage, times the runner's
_model_forward / _pool with a CUDA sync before and after (like rj_prof.py), and
reads the GPU SM clock and P-state just before and after every request.

Schedules, run in order after a warmup:
  burst      requests back to back
  gap<S>     one request every S seconds (the pipeline sends one per voice turn)

    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=<vllm-omni tree> python laya_standalone_prof.py \
        --model-dir <prepared laya dir> --inputs laya.jsonl --out <new dir> [--enforce-eager] [--gaps 0 2 5]
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
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--gaps", type=float, nargs="+", default=[0.0, 2.0, 5.0])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-len", type=int, default=80, help="use inputs with at most this many ids")
    p.add_argument("--wait", choices=("sleep", "spin"), default="sleep",
                   help="how the gap is waited: sleep (CPU idles) or busy-spin (this thread keeps a core busy)")
    p.add_argument("--gpu-keepalive", action="store_true",
                   help="a separate process launches a small GPU matmul every 2 ms during the whole run")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    import pynvml
    import torch
    from vllm import LLM, PoolingParams
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    keepalive = None
    if a.gpu_keepalive:
        import subprocess
        import sys

        code = ("import time, torch\nx = torch.randn(256, 256, device='cuda')\n"
                "while True:\n    x @ x\n    torch.cuda.synchronize()\n    time.sleep(0.002)\n")
        keepalive = subprocess.Popen([sys.executable, "-c", code])
    rows = [json.loads(line) for line in a.inputs.read_text().splitlines() if line]
    ids = [r["ids"] for r in rows if len(r["ids"]) <= a.max_len][: a.n]
    llm = LLM(
        model=a.model_dir,
        runner="pooling",
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=1024,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enforce_eager=a.enforce_eager,
        enable_prefix_caching=False,
    )
    core = llm.llm_engine.engine_core.engine_core
    wrapper = core.model_executor.driver_worker
    runner = getattr(wrapper, "worker", wrapper).model_runner
    steps: dict[str, list[float]] = {}

    def timed(name, fn):
        def inner(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            steps.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
            return result

        return inner

    for name in ("execute_model", "_prepare_inputs", "_model_forward", "_pool"):
        if callable(getattr(runner, name, None)):
            setattr(runner, name, timed(name, getattr(runner, name)))

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def cpu_state():
        from pathlib import Path as _P

        cpu = int(_P("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[36])
        try:
            freq = int(_P(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text()) / 1000
        except OSError:
            freq = None
        return cpu, freq

    def gpu_state():
        return {
            "sm_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "pstate": pynvml.nvmlDeviceGetPerformanceState(handle),
        }

    params = PoolingParams(task="classify")
    for x in ids[:5]:
        llm.encode([{"prompt_token_ids": x}], pooling_params=params, pooling_task="classify", use_tqdm=False)

    results = []
    summary = {}
    for gap in a.gaps:
        label = "burst" if gap == 0 else f"gap{gap:g}"
        for x in ids:
            if gap and a.wait == "sleep":
                time.sleep(gap)
            elif gap:
                end = time.perf_counter() + gap
                while time.perf_counter() < end:
                    pass
            steps.clear()
            cpu, cpu_mhz = cpu_state()
            before = gpu_state()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.encode([{"prompt_token_ids": x}], pooling_params=params, pooling_task="classify", use_tqdm=False)
            torch.cuda.synchronize()
            total = (time.perf_counter() - t0) * 1000
            after = gpu_state()
            results.append({
                "schedule": label, "n_ids": len(x), "encode_ms": total, "cpu": cpu, "cpu_cur_mhz_before": cpu_mhz,
                **{f"{k}_ms": sum(v) for k, v in steps.items()},
                "sm_mhz_before": before["sm_mhz"], "pstate_before": before["pstate"],
                "sm_mhz_after": after["sm_mhz"], "pstate_after": after["pstate"],
            })
        mine = [r for r in results if r["schedule"] == label]
        summary[label] = {
            key: round(statistics.median(r[key] for r in mine), 2)
            for key in ("encode_ms", "execute_model_ms", "_prepare_inputs_ms", "_model_forward_ms", "_pool_ms",
                        "sm_mhz_before", "sm_mhz_after", "cpu_cur_mhz_before")
            if all(r.get(key) is not None for r in mine)
        }
        summary[label]["n"] = len(mine)
    if keepalive is not None:
        keepalive.terminate()
        keepalive.wait(timeout=30)
    meta = {"enforce_eager": a.enforce_eager, "gaps": a.gaps, "n": len(ids), "wait": a.wait,
            "gpu_keepalive": a.gpu_keepalive, "torch": torch.__version__}
    (a.out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    (a.out / "summary.json").write_text(json.dumps({"meta": meta, "median": summary}, indent=2))
    print(json.dumps({"meta": meta, "median": summary}, indent=1))


if __name__ == "__main__":
    main()
