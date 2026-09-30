"""Head-port check without the vLLM engine (no Triton/C compiler needed).

Feeds the reference laya encoder's hidden states, packed like a vLLM batch,
into LayaDecisionPooler and compares the option logits with the logits the
laya package produced (laya.jsonl from laya_vllm_parity.py). This validates
the pooler (per-request split, marker lookup, head, weight names); the vLLM
ModernBERT encoder itself is checked by laya_vllm_parity.py on a GPU host.
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
    p.add_argument("--laya-jsonl", type=Path, required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--laya-pkg", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, a.laya_pkg)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import laya
    import torch

    from laya_vllm_model import LayaDecisionPooler

    agent = laya.load(a.model_dir, device="cuda")
    ref_model = agent.model.float().eval()
    enc = ref_model.encoder
    cfg = enc.config
    pooler = LayaDecisionPooler(cfg.hidden_size, 2, int(cfg.mask_token_id), 0).cuda().float().eval()
    state = {k: v for k, v in ref_model.state_dict().items() if k.startswith(("head.", "type_emb.", "scorer."))}
    missing = pooler.load_state_dict(state, strict=True)
    rows = [json.loads(line) for line in a.laya_jsonl.read_text().splitlines() if line]
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), a.batch):
            chunk = rows[start : start + a.batch]
            hs, lens = [], []
            for r in chunk:
                ids = torch.tensor([r["ids"]], device="cuda")
                h = enc(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
                hs.append(h)
                lens.append(len(r["ids"]))
            width = max(lens)
            token_ids = torch.zeros(len(chunk), width, dtype=torch.long, device="cuda")
            for i, r in enumerate(chunk):
                token_ids[i, : lens[i]] = torch.tensor(r["ids"], device="cuda")
            cursor = SimpleNamespace(
                num_scheduled_tokens_cpu=torch.tensor(lens), is_partial_prefill=lambda: False
            )
            md = SimpleNamespace(get_pooling_cursor=lambda: cursor, prompt_token_ids=token_ids)
            outs = pooler(torch.cat(hs, 0), md)
            for r, o in zip(chunk, outs):
                got = o.float().cpu().tolist()
                ref = r["logits"]
                diff = max(abs(x - y) for x, y in zip(ref, got)) if len(ref) == len(got) else float("inf")
                results.append(
                    {"id": r["id"], "question": r["question"], "file": r["file"], "max_abs_diff": diff,
                     "argmax_agree": len(ref) == len(got) and ref.index(max(ref)) == got.index(max(got))}
                )
    summary = {
        "rows": len(results),
        "argmax_agree": sum(x["argmax_agree"] for x in results),
        "max_abs_logit_diff": max(x["max_abs_diff"] for x in results),
        "strict_load": str(missing),
        "note": "reference encoder hidden states (laya package, fp32) packed as a vLLM batch into LayaDecisionPooler",
    }
    (a.out / "head-parity.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n")
    (a.out / "head-parity-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
