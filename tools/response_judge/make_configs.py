"""Write the deploy configs for tomorrow's GPU runs into a new directory.

Starts from the official AURA duplex smoke profile of the source tree
(examples/online_serving/aura_omni/aura_omni_duplex_smoke.yaml), sets local
model paths, and derives:

  smoke-off.yaml            aura_omni, no judge
  smoke-on-qwen3.yaml       aura_omni_judged, Qwen3-1.7B judge (chat_yes_no)
  smoke-on-laya.yaml        aura_omni_judged, LAYA judge (pooling), if --laya-dir
  smoke-on-clm.yaml         aura_omni_judged, CLM judge (pooling), if --clm-dir
  conc-*.yaml               the same with max_sessions / every max_num_seqs = 8

--judge-device / --asr-device put those stages on another GPU (two 32 GB cards:
e.g. --asr-device 1 --judge-device 1, AURA + TTS stay on "0").

    python make_configs.py --source <tree> --models <dir> [--laya-dir <prepared>] --out <new dir>
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

SMOKE = "examples/online_serving/aura_omni/aura_omni_duplex_smoke.yaml"
MODEL_NAMES = ["Qwen3-ASR-1.7B", "AURA-88ee5506", "Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-CustomVoice"]


def qwen3_judge(models: Path) -> dict:
    # Same values as the earlier GPU evidence (pr-evidence-aura-judge-final-20260929/config/smoke-on.yaml).
    return {
        "stage_id": 1,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.08,
        "enforce_eager": False,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
        "async_scheduling": True,
        "max_model_len": 2048,
        "max_num_batched_tokens": 2048,
        "devices": "0",
        "model": str(models / "Qwen3-1.7B"),
        "hf_overrides": {"response_judge": {"format": "chat_yes_no"}},
        "default_sampling_params": {"temperature": 0.0, "max_tokens": 1},
    }


def laya_judge(laya_dir: Path, response_judge: dict) -> dict:
    return {
        "stage_id": 1,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.05,
        "enforce_eager": False,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "max_model_len": 1024,
        "max_num_batched_tokens": 8192,
        "devices": "0",
        "model": str(laya_dir),
        "model_arch": "LayaDecisionModel",
        "runner": "pooling",
        "default_pooling_params": {"task": "classify"},
        "hf_overrides": {"laya_question_type": response_judge.get("question_type", "choice"), "response_judge": response_judge},
    }


def clm_judge(clm_dir: Path, gpu_memory_utilization: float) -> dict:
    # Judge options (incl. option_keys / threshold) come from the prepared dir's config.json.
    return {
        "stage_id": 1,
        "max_num_seqs": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": False,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "max_model_len": 2048,
        "max_num_batched_tokens": 2048,
        "devices": "0",
        "model": str(clm_dir),
        "model_arch": "ClmDecisionModel",
        "runner": "pooling",
        "default_pooling_params": {"task": "classify"},
    }


def judged(off: dict, judge: dict) -> dict:
    on = copy.deepcopy(off)
    on["pipeline"] = "aura_omni_judged"
    for stage in on["stages"][1:]:
        stage["stage_id"] += 1
        for field, old, new in (("input_connectors", "from_stage_2", "from_stage_3"), ("output_connectors", "to_stage_3", "to_stage_4")):
            if field in stage:
                stage[field] = {new if k == old else k: v for k, v in stage[field].items()}
    on["stages"].insert(1, judge)
    return on


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--base-profile", type=Path, help="4-stage aura_omni profile to start from (default: the smoke profile)")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--laya-dir", type=Path)
    p.add_argument("--laya-judge-json", type=Path, help="response_judge options for LAYA (default: the overlay's)")
    p.add_argument("--clm-dir", type=Path, help="model dir written by clm_tune_and_prepare.py (<out>/model)")
    p.add_argument("--clm-gpu-util", type=float, default=0.25)
    p.add_argument("--aura-gpu-util", type=float, help="override AURA's gpu_memory_utilization (KV size only)")
    p.add_argument("--judge-gpu-util", type=float, help="override the judge stage's gpu_memory_utilization")
    p.add_argument("--judge-eager", action="store_true", help="enforce_eager on the judge stage (32 GB profiles)")
    p.add_argument("--judge-device")
    p.add_argument("--asr-device")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; choose a new directory")
    off = yaml.safe_load((a.base_profile or a.source / SMOKE).read_text())
    if off.get("pipeline") != "aura_omni" or [s["stage_id"] for s in off["stages"]] != [0, 1, 2, 3]:
        raise SystemExit("unexpected baseline smoke topology")
    for stage, name in zip(off["stages"], MODEL_NAMES):
        stage["model"] = str(a.models / name)
    if a.asr_device:
        off["stages"][0]["devices"] = a.asr_device
    if a.aura_gpu_util:
        off["stages"][1]["gpu_memory_utilization"] = a.aura_gpu_util
    profiles = {"smoke-off": off, "smoke-on-qwen3": judged(off, qwen3_judge(a.models))}
    if a.laya_dir:
        overlay = yaml.safe_load((a.source / "vllm_omni/deploy/aura_omni_judged_laya.yaml").read_text())
        options = next(s for s in overlay["stages"] if s["stage_id"] == 1)["hf_overrides"]["response_judge"]
        if a.laya_judge_json:
            options = json.loads(a.laya_judge_json.read_text())
        profiles["smoke-on-laya"] = judged(off, laya_judge(a.laya_dir.resolve(), options))
    if a.clm_dir:
        profiles["smoke-on-clm"] = judged(off, clm_judge(a.clm_dir.resolve(), a.clm_gpu_util))
    for name, cfg in profiles.items():
        if name != "smoke-off":
            if a.judge_device:
                cfg["stages"][1]["devices"] = a.judge_device
            if a.judge_gpu_util:
                cfg["stages"][1]["gpu_memory_utilization"] = a.judge_gpu_util
            if a.judge_eager:
                cfg["stages"][1]["enforce_eager"] = True
    for name in list(profiles):
        conc = copy.deepcopy(profiles[name])
        conc["duplex_session"]["max_sessions"] = 8
        for stage in conc["stages"]:
            stage["max_num_seqs"] = 8
        profiles[name.replace("smoke-", "conc-")] = conc
    a.out.mkdir(parents=True)
    budget = {}
    for name, cfg in profiles.items():
        per_device: dict[str, float] = {}
        for stage in cfg["stages"]:
            dev = str(stage.get("devices", "0"))
            per_device[dev] = round(per_device.get(dev, 0.0) + float(stage.get("gpu_memory_utilization", 0.0)), 3)
        budget[name] = per_device
        if any(v > 0.95 for v in per_device.values()):
            print(f"WARNING {name}: gpu_memory_utilization per device {per_device} exceeds 0.95", flush=True)
    for name, cfg in profiles.items():
        if cfg.get("async_chunk") is not True or cfg.get("session_mode") != "duplex":
            raise SystemExit(f"not a duplex profile: {name}")
        (a.out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (a.out / "memory-budget.json").write_text(json.dumps(budget, indent=1))
    print(json.dumps({"out": str(a.out), "profiles": sorted(profiles), "gpu_util_per_device": budget}))


if __name__ == "__main__":
    main()
