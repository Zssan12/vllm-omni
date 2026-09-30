"""Restricted auto-response prototype; never patches the PR checkout in place.

The suite imports transform() and writes into a new disposable source copy.
CLI defaults to a diff; --output creates a NEW file, refusing overwrites.
"""

from __future__ import annotations

import argparse
import ast
import difflib
from pathlib import Path

EDITS = [
    (
        """        if not auto_response and data_plane_request_id == session.active_request_id:
            session.clear_request()
        model_listen = model_result.get("model_listen")""",
        """        if not auto_response and data_plane_request_id == session.active_request_id:
            session.clear_request()
        terminal_listen = model_result.get("end_of_turn") is True
        model_listen = model_result.get("model_listen")""",
    ),
    (
        """        self._attach_runtime_metadata(payload, model_result)
        self._out.emit(payload)
        if model_result.get("abort_data_plane_request") is True and isinstance(data_plane_request_id, str):""",
        """        self._attach_runtime_metadata(payload, model_result)
        self._out.emit(payload)
        if terminal_listen and auto_response:
            # PROTOTYPE: in an auto-response session a terminal listen completes the
            # model turn even with no active response or with continuation budget
            # left, and drops a stale turn binding so the next append opens T+1.
            # Manual sessions keep their existing continuation semantics.
            completed_turn_id = model_turn_id if model_turn_id is not None else session.active_response_turn_id
            if completed_turn_id is not None:
                session.complete_model_turn(completed_turn_id)
                bound_turn_id = session.active_response_turn_id
                if session.active_response_id is None and bound_turn_id is not None and bound_turn_id <= completed_turn_id:
                    session.bind_response_turn(None)
        if model_result.get("abort_data_plane_request") is True and isinstance(data_plane_request_id, str):""",
    ),
]


def transform(source):
    for old, new in EDITS:
        if source.count(old) != 1:
            raise ValueError(f"prototype anchor must occur once: {old[:60]!r}")
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.source_root / "vllm_omni/engine/duplex/session/model_channel.py"
    before = path.read_text()
    after = transform(before)
    if args.output:
        with args.output.open("x") as out:
            out.write(after)
    else:
        print(
            "".join(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile=str(path),
                    tofile="prototype-only/model_channel.py",
                )
            )
        )


if __name__ == "__main__":
    main()
