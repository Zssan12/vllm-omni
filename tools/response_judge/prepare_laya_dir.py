"""Build a vLLM-loadable directory for a LAYA checkpoint (response-judge stage).

LAYA snapshots keep the encoder config in ``encoder/`` and the tokenizer in
``tokenizer/``; recent transformers also cannot read their tokenizer_config
(``TokenizersBackend`` class, list-valued extra_special_tokens). This writes a
new directory and never modifies the snapshot:

    <out>/config.json            encoder config + architectures=[LayaDecisionModel]
    <out>/model.safetensors      relative symlink to the snapshot weights
    <out>/tokenizer.json         copy
    <out>/tokenizer_config.json  copy, patched the same way the laya package does
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", type=Path, help="local laya checkpoint (e.g. laya-multilingual snapshot)")
    p.add_argument("out", type=Path, help="new directory to create")
    p.add_argument("--question-type", default="choice", choices=("choice", "score", "noul"))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; choose a new directory")
    agent_cfg = json.loads((a.snapshot / "rl_agent_config.json").read_text())
    cfg = json.loads((a.snapshot / "encoder" / "config.json").read_text())
    cfg["architectures"] = ["LayaDecisionModel"]
    cfg["laya_head_layers"] = int(agent_cfg.get("head_layers", 2))
    cfg["laya_question_type"] = a.question_type
    a.out.mkdir(parents=True)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    weights = (a.snapshot / "model.safetensors").resolve()
    # Relative, so the link survives mounting the models directory elsewhere (e.g. Docker),
    # as long as <out> and the resolved weights (HF blobs included) are mounted together.
    (a.out / "model.safetensors").symlink_to(os.path.relpath(weights, a.out.resolve()))
    shutil.copyfile(a.snapshot / "tokenizer" / "tokenizer.json", a.out / "tokenizer.json")
    tcfg = json.loads((a.snapshot / "tokenizer" / "tokenizer_config.json").read_text())
    if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tcfg.pop("backend", None)
        tcfg.pop("is_local", None)
    extra = tcfg.get("extra_special_tokens")
    if isinstance(extra, list):
        tcfg["extra_special_tokens"] = {f"extra_{i}": t for i, t in enumerate(extra)}
    (a.out / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2))
    print(json.dumps({"out": str(a.out), "architectures": cfg["architectures"], "max_len": agent_cfg.get("max_len")}))


if __name__ == "__main__":
    main()
