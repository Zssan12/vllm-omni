"""Copy a timing-patched vllm-omni tree and add the rj_prof hooks (never part of the PR).

Run timing_patch.py first (OMNI_TIMING / OMNI_HOP lines), then:

    python prof_patch.py <timing-patched tree> <new destination>

Adds rj_prof.py at the tree root (on PYTHONPATH for the server and every
StageEngineCoreProc) and two call sites: install_engine() right before each
stage's busy loop, install_client() when the orchestrator is built.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

PATCHES = {
    "vllm_omni/engine/stage_engine_core_proc.py": [
        (
            "            engine_core.run_busy_loop()\n",
            "            import rj_prof\n\n            rj_prof.install_engine(engine_core)\n            engine_core.run_busy_loop()\n",
        ),
    ],
    "vllm_omni/engine/orchestrator.py": [
        (
            "        self._event_driven_orch = _event_driven_orch_enabled(default=event_driven_orch_default)\n",
            "        self._event_driven_orch = _event_driven_orch_enabled(default=event_driven_orch_default)\n"
            "        import rj_prof\n\n"
            "        rj_prof.install_client()\n"
            '        logger.warning("RJ_PROF orchestrator event_driven=%s", self._event_driven_orch)\n',
        ),
    ],
}


def main() -> None:
    source, dest = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if dest.exists():
        raise SystemExit(f"{dest} exists; choose a new directory")
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"), symlinks=True)
    applied = {}
    for rel, patches in PATCHES.items():
        path = dest / rel
        text = path.read_text()
        for before, after in patches:
            if text.count(before) != 1:
                raise SystemExit(f"anchor not found exactly once in {rel}: {before[:80]!r}")
            text = text.replace(before, after)
        path.write_text(text)
        applied[rel] = hashlib.sha256(text.encode()).hexdigest()
    shutil.copyfile(Path(__file__).with_name("rj_prof.py"), dest / "rj_prof.py")
    (dest / "PROF-PATCH.json").write_text(json.dumps({"source": str(source), "patched_sha256": applied}, indent=2))
    print(json.dumps({"dest": str(dest), "patched": sorted(applied)}))


if __name__ == "__main__":
    main()
