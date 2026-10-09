"""Report every timing_patch / prof_patch anchor against a target tree, without applying.

    python3 h200/check_anchors.py <tree of the PR head>

A patch anchor is a literal source snippet that `timing_patch.py` / `prof_patch.py`
look for before rewriting it. When the upstream tree moves, an anchor stops matching
and the patch would either skip that edit silently or fail halfway. Run this after
checking out a new PR head: every anchor must report count=1.

    -> 0 bad anchors   both patches apply cleanly to this tree
    -> n bad anchors   n anchors no longer match; fix them before running an arm
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RJ = HERE.parent / "tools" / "response_judge"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, RJ / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(mod, tree: Path) -> int:
    bad = 0
    # An anchor may be a tuple of alternative forms covering upstream revisions that
    # reformatted the same code. Resolution is delegated to the patcher itself, so this
    # check cannot disagree with what applying would do.
    resolve = getattr(mod, "resolve_anchor", None)
    for rel, edits in mod.PATCHES.items():
        f = tree / rel
        if not f.exists():
            print(f"  MISSING FILE {rel}")
            bad += 1
            continue
        text = f.read_text()
        for i, (old, _new) in enumerate(edits):
            if resolve is not None:
                anchor = resolve(text, old)
                if anchor is not None:
                    continue
                forms = (old,) if isinstance(old, str) else old
                bad += 1
                counts = ", ".join(str(text.count(c)) for c in forms)
                first = forms[0].splitlines()[0] if forms[0].splitlines() else forms[0]
                print(f"  {rel} edit[{i}]: counts=[{counts}]  anchor starts: {first[:100]!r}")
                continue
            n = text.count(old)
            if n != 1:
                bad += 1
                first = old.splitlines()[0] if old.splitlines() else old
                print(f"  {rel} edit[{i}]: count={n}  anchor starts: {first[:100]!r}")
    return bad


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 h200/check_anchors.py <tree of the PR head>")
    tree = Path(sys.argv[1]).resolve()
    print(f"=== timing_patch vs {tree} ===")
    n1 = check(load("timing_patch"), tree)
    print(f"  -> {n1} bad anchors")
    print(f"=== prof_patch vs {tree} ===")
    n2 = check(load("prof_patch"), tree)
    print(f"  -> {n2} bad anchors")
