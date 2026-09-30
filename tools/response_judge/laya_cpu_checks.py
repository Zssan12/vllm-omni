"""CPU checks for the LAYA judge (no GPU engine needed).

1. The tokenizer vLLM loads from a prepare_laya_dir.py directory has the same
   special ids as the laya package tokenizer.
2. response_judge's LAYA bridge builds exactly the token ids the laya package
   fed its model (rows captured in laya.jsonl by laya_vllm_parity.py).
3. vLLM resolves the prepared directory to LayaDecisionModel (pooling runner)
   through vllm-omni's model registry.

Run from a vllm-omni checkout that contains the response_judge changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--laya-dir", type=Path, required=True, help="output of prepare_laya_dir.py")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--laya-pkg", required=True)
    p.add_argument("--laya-jsonl", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    report: dict[str, object] = {}

    # 1. tokenizers
    sys.path.insert(0, a.laya_pkg)
    import laya
    from vllm.tokenizers import get_tokenizer

    agent = laya.load(a.snapshot, device="cpu")
    ref_tok = agent.tok
    tok = get_tokenizer(str(a.laya_dir))
    special = ("mask_token_id", "cls_token_id", "sep_token_id", "pad_token_id")
    report["special_ids"] = {k: [getattr(ref_tok, k), getattr(tok, k)] for k in special}
    report["special_ids_equal"] = all(getattr(ref_tok, k) == getattr(tok, k) for k in special)

    # 2. bridge ids vs laya package ids
    import vllm.tokenizers

    from vllm_omni.model_executor.stage_input_processors import response_judge as rj

    vllm.tokenizers.cached_tokenizer_from_config = lambda model_config: tok
    captured = {}
    for line in a.laya_jsonl.read_text().splitlines():
        row = json.loads(line)
        captured[(row["file"], row["id"], row["question"])] = row["ids"]
    checked = mismatched = 0
    examples = []
    for path in a.cases:
        for case in json.loads(Path(path).read_text()):
            for qid, q in case["questions"].items():
                key = (Path(path).name, case["id"], qid)
                if key not in captured:
                    continue
                spec = rj.JudgeSpec(
                    "laya",
                    {
                        "question_type": q["type"],
                        "instructions": q["instructions"],
                        "options": q["criteria"],
                        "state_template": "{transcript}",
                    },
                )
                ids = rj._laya_prompt(spec, case["state"], SimpleNamespace())["prompt_token_ids"]
                checked += 1
                if ids != captured[key]:
                    mismatched += 1
                    if len(examples) < 3:
                        examples.append({"key": key, "ours": ids[:40], "laya": captured[key][:40]})
    report["bridge_ids_checked"] = checked
    report["bridge_ids_mismatched"] = mismatched
    report["bridge_ids_examples"] = examples

    # 3. vLLM model resolution
    try:
        from vllm.config import ModelConfig
        from vllm.plugins import load_general_plugins

        # What engine start-up does: runs vllm-omni's vllm.general_plugins entry point.
        load_general_plugins()

        mc = ModelConfig(model=str(a.laya_dir), runner="pooling", max_model_len=1024, dtype="float32")
        report["vllm_architecture"] = mc.architecture if hasattr(mc, "architecture") else mc.architectures
        report["vllm_runner"] = getattr(mc, "runner_type", None)
        report["vllm_supported_tasks"] = list(getattr(mc, "supported_tasks", []) or [])
    except Exception as exc:  # recorded, not hidden
        report["vllm_model_config_error"] = f"{type(exc).__name__}: {exc}"[:2000]

    (a.out / "laya-cpu-checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
