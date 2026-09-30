"""Copy a vllm-omni tree and add temporary timing log lines (never part of the PR).

OMNI_TIMING lines are byte-identical to the earlier suite, so
tools/rfc_bench/analyze_onoff.py keeps working. OMNI_HOP lines are new and
split the judge's added time into hops (hop_analysis.py):

  OMNI_HOP output stage=S req=R finished=F t=T   stage output reached the orchestrator
  OMNI_HOP bridge stage=S req=R t0=A t1=B        a stage input processor ran from A to B
  OMNI_HOP stats stage=S req=R <stage metrics>   engine-side stage metrics (if any)
  OMNI_HOP judge_rejected stage=S req=R t=T      the engine saw a response_judge "no reply"
  OMNI_HOP listen req=R reason=... abort=... t=T the session's existing _on_model_listen ran
  OMNI_HOP aborted req=R t=T                     its abort of the prewarmed downstream returned
  OMNI_HOP released req=R,... t=T                the turn's stage requests were released

    python timing_patch.py <source tree> <new destination>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

PATCHES = {
    "vllm_omni/engine/duplex/session/model_channel.py": [
        (
            "        data_plane = self._ctx.plugin.data_plane\n"
            "        auto_response = self._out.auto_responds()\n"
            "        close_reason: str | None = None\n"
            "        emitted_response = False\n"
            "        self._end_active_response_before_future_model_turn(model_turn_id=model_turn_id)\n",
            "        data_plane = self._ctx.plugin.data_plane\n"
            "        auto_response = self._out.auto_responds()\n"
            "        close_reason: str | None = None\n"
            "        emitted_response = False\n"
            '        logger.warning("OMNI_HOP listen req=%s reason=%s abort=%s t=%.6f", data_plane_request_id, '
            'model_result.get("reason"), model_result.get("abort_data_plane_request"), time.time())\n'
            "        self._end_active_response_before_future_model_turn(model_turn_id=model_turn_id)\n",
        ),
        (
            "            await self._abort_request([data_plane_request_id], notify=False)\n",
            "            await self._abort_request([data_plane_request_id], notify=False)\n"
            '            logger.warning("OMNI_HOP aborted req=%s t=%.6f", data_plane_request_id, time.time())\n',
        ),
        (
            "        session.release_resources_for_request_ids(release_ids)\n"
            "        await self._ctx.stage_port.cleanup(release_ids)\n",
            "        session.release_resources_for_request_ids(release_ids)\n"
            "        await self._ctx.stage_port.cleanup(release_ids)\n"
            '        logger.warning("OMNI_HOP released req=%s t=%.6f", ",".join(release_ids), time.time())\n',
        ),
    ],
    "vllm_omni/engine/duplex_orchestrator.py": [
        (
            "            response_judge_rejected=finished and self._is_response_judge_stage(stage_id) and judge_rejects(output),\n"
            "        )\n",
            "            response_judge_rejected=finished and self._is_response_judge_stage(stage_id) and judge_rejects(output),\n"
            "        )\n"
            "        if context.response_judge_rejected:\n"
            '            logger.warning("OMNI_HOP judge_rejected stage=%d req=%s t=%.6f", stage_id, request_id, _time.time())\n',
        ),
        (
            "        del replica_id\n        if not isinstance(req_state, DuplexOrchestratorRequestState) or req_state.fence is None:\n",
            "        del replica_id\n"
            '        logger.warning("OMNI_TIMING submit stage=%d req=%s t=%.6f", stage_id, request_id, _time.time())\n'
            "        if not isinstance(req_state, DuplexOrchestratorRequestState) or req_state.fence is None:\n",
        ),
    ],
    "vllm_omni/engine/orchestrator.py": [
        (
            '        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)\n'
            "        _t_submit_start = _time.perf_counter()\n",
            '        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)\n'
            "        _t_submit_start = _time.perf_counter()\n"
            '        logger.warning("OMNI_TIMING forward src=%d dst=%d req=%s t=%.6f", src_stage_id, next_logical, req_id, _time.time())\n',
        ),
        (
            "        if await self._intercept_stage_output(stage_id, replica_id, output, req_state, stage_metrics, submit_ts):\n",
            '        logger.warning("OMNI_HOP output stage=%d req=%s finished=%d t=%.6f", stage_id, req_id, int(bool(finished)), _time.time())\n'
            "        if stage_metrics is not None:\n"
            '            logger.warning("OMNI_HOP stats stage=%d req=%s %s", stage_id, req_id, repr(stage_metrics)[:600])\n'
            "        if await self._intercept_stage_output(stage_id, replica_id, output, req_state, stage_metrics, submit_ts):\n",
        ),
    ],
    "vllm_omni/engine/stage_engine_core_client.py": [
        (
            "        if self.custom_process_input_func is not None:\n"
            "            return self._call_custom_process_input(source_outputs, prompt, streaming_context)\n",
            "        if self.custom_process_input_func is not None:\n"
            '            _hop_t0 = __import__("time").time()\n'
            "            _hop_result = self._call_custom_process_input(source_outputs, prompt, streaming_context)\n"
            "            logger.warning(\n"
            '                "OMNI_HOP bridge stage=%d req=%s t0=%.6f t1=%.6f",\n'
            "                self.stage_id,\n"
            '                ",".join(str(getattr(o, "request_id", "?")) for o in source_outputs),\n'
            "                _hop_t0,\n"
            '                __import__("time").time(),\n'
            "            )\n"
            "            return _hop_result\n",
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
    (dest / "TIMING-PATCH.json").write_text(json.dumps({"source": str(source), "patched_sha256": applied}, indent=2))
    print(json.dumps({"dest": str(dest), "patched": sorted(applied)}))


if __name__ == "__main__":
    main()
