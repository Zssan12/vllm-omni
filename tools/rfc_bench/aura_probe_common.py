"""Shared experiment client: bounded sessions and response-owned measurements.

No vLLM/Pillow imports until a live client is opened; the scoring functions are
also used by offline regression tests. This module never opens a remote host.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

TEXT = {
    "response.output_text.delta",
    "response.text.delta",
    "response.output_audio_transcript.delta",
}
AUDIO = {"response.audio.delta", "response.output_audio.delta"}
TEXT_DONE = {
    "response.output_text.done",
    "response.text.done",
    "response.output_audio_transcript.done",
}
BAD_STATUS = {"failed", "cancelled", "canceled", "incomplete", "error"}


def response_id(event):
    return event.get("response_id") or (event.get("response") or {}).get("id")


def model_turn(event):
    for source in (
        event,
        event.get("metadata") or {},
        (event.get("response") or {}).get("metadata") or {},
    ):
        value = source.get(
            "model_turn_id", (source.get("vllm_omni") or {}).get("model_turn_id")
        )
        if isinstance(value, int):
            return value
    return None


def is_model_listen(event):
    sources = (
        event,
        event.get("metadata") or {},
        (event.get("response") or {}).get("metadata") or {},
    )
    return event.get("type") == "response.listen" and any(
        source.get("model_listen") is True for source in sources
    )


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


def distribution(values):
    values = [v for v in values if v is not None]
    return {
        "n": len(values),
        "p50": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def safe_event(event):
    # Keep identities and measurements, not resume credentials or PCM payloads.
    if event.get("type") in {"session.created", "session.resumed"}:
        return {"type": event["type"], "session_id": event.get("session_id")}
    clean = {
        k: v
        for k, v in event.items()
        if k not in {"resume_token", "token", "api_key", "_bench_audio_bytes"}
    }
    if event.get("type") in AUDIO:
        clean = {k: v for k, v in clean.items() if k not in {"delta", "audio"}}
        clean["audio_bytes"] = audio_size(event)
    return clean


def audio_size(event):
    if event.get("type") not in AUDIO:
        return 0
    if "_bench_audio_bytes" in event:
        return event["_bench_audio_bytes"]
    try:
        size = len(
            base64.b64decode(
                event.get("delta") or event.get("audio") or "", validate=True
            )
        )
    except (ValueError, TypeError):
        size = 0
    # Collector events are immutable observations. Decode each PCM payload
    # once, not on every polling pass for all eight concurrent clients.
    event["_bench_audio_bytes"] = size
    return size


class TraceReader:
    """Tail only our single-line server diagnostics; never parse other log payloads."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.offset = self.path.stat().st_size if self.path else 0
        self.rows = []

    def poll(self):
        if self.path is None:
            return
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while True:
                pos = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    self.offset = pos
                    return
                self.offset = stream.tell()
                if b"AURA_BENCH " in line:
                    payload = line.split(b"AURA_BENCH ", 1)[1].decode("utf-8")
                    record, _ = json.JSONDecoder().raw_decode(payload)
                    self.rows.append(record)


@dataclass
class Turn:
    start: int
    committed_at: float
    commit_wall: float
    known_responses: set[str]
    trace_start: int = 0
    request_id: str | None = None
    turn_id: int | None = None
    epoch: int | None = None
    response: str | None = None
    finished: bool = False
    ambiguous: bool = False
    traces: list[dict] = field(default_factory=list)
    created_end: int | None = None
    trace_end: int | None = None


def measure(turn, events, now, *, single_turn=False):
    window = events[turn.start :]
    creation_window = events[turn.start : turn.created_end]
    created = {
        response_id(e) for e in creation_window if e.get("type") == "response.created"
    }
    created.discard(None)
    new = created - turn.known_responses
    if turn.response is None and len(new) == 1:
        turn.response = next(iter(new))
    if len(new) > 1:
        turn.ambiguous = True
    owned = []
    for event in window:
        rid = response_id(event)
        if rid and turn.response and rid == turn.response:
            owned.append(event)
        elif event.get("type") == "response.listen" and rid is None:
            # An auto-response NO has no response.created/done. Match its model
            # turn to the server's append identity, or the sole fresh-session turn.
            if (
                turn.turn_id is not None and model_turn(event) == turn.turn_id
            ) or single_turn:
                owned.append(event)
    errors = [safe_event(e) for e in window if e.get("type") == "error"]
    done = next((e for e in owned if e.get("type") == "response.done"), None)
    listen = next((e for e in owned if is_model_listen(e)), None)
    audio_bytes = sum(audio_size(e) for e in owned)
    text = "".join(str(e.get("delta") or "") for e in owned if e.get("type") in TEXT)
    status = (done or {}).get("status") or ((done or {}).get("response") or {}).get(
        "status"
    )
    failed = bool(errors or status in BAD_STATUS or turn.ambiguous)
    if (
        any(e.get("type") == "session.closed" for e in window)
        and done is None
        and listen is None
    ):
        failed = True
        errors.append(
            {"type": "session.closed", "message": "closed before target turn completed"}
        )
    terminal_event = (
        listen
        if listen is not None and not text and not audio_bytes
        else done or listen
    )
    ended_at = (terminal_event or {}).get("_client_received_at_s", now)

    def first(types):
        points = [
            e.get("_client_received_at_s", now)
            for e in owned
            if e.get("type") in types
            and (audio_size(e) > 0 if types == AUDIO else bool(e.get("delta")))
        ]
        return round((min(points) - turn.committed_at) * 1000, 3) if points else None

    no = listen is not None and not text and audio_bytes == 0 and not failed
    answered = done is not None and audio_bytes > 0 and bool(text) and not failed
    return {
        "request_id": turn.request_id,
        "model_turn_id": turn.turn_id,
        "epoch": turn.epoch,
        "response_id": turn.response,
        "terminal": terminal_event.get("type") if terminal_event else None,
        "status": status,
        "terminal_ms": round((ended_at - turn.committed_at) * 1000, 3),
        "first_text_ms": first(TEXT),
        "first_audio_ms": first(AUDIO),
        "text": text,
        "audio_bytes": audio_bytes,
        "has_text": bool(text),
        "has_audio": audio_bytes > 0,
        "blocked": no,
        "answered": answered,
        "errors": errors,
        "ambiguous_response": turn.ambiguous,
        "failed": failed,
        "text_done": any(e.get("type") in TEXT_DONE for e in owned),
        "transcripts": [
            e.get("transcript")
            for e in window
            if e.get("type") == "conversation.item.input_audio_transcription.completed"
        ],
        "event_types": sorted({e.get("type", "") for e in window}),
        # response.listen carries the deciding component (e.g. "response_judge", "aura_silent")
        # in response.metadata.vllm_omni.listen_source once the server forwards it.
        "listen_sources": [
            ((e.get("response") or {}).get("metadata") or {}).get("vllm_omni", {}).get("listen_source")
            for e in window
            if e.get("type") == "response.listen"
        ],
    }


class ProbeSession:
    def __init__(self, client, collector, consume, trace, *, single_turn=False):
        self.client, self.collector, self.consume, self.trace = (
            client,
            collector,
            consume,
            trace,
        )
        self.single_turn = single_turn
        self.turns = []

    def session_id(self):
        return self.client.session_id

    def refresh(self, turn):
        self.trace.poll()
        rows = [
            r
            for r in self.trace.rows[turn.trace_start :]
            if r.get("session_id") == self.session_id()
        ]
        appends = [
            r
            for r in self.trace.rows[turn.trace_start : turn.trace_end]
            if r.get("session_id") == self.session_id() and r.get("kind") == "append"
        ]
        ids = {r["request_id"] for r in appends}
        if len(ids) > 1:
            turn.ambiguous = True
        if appends and turn.request_id is None:
            first = appends[0]
            turn.request_id, turn.turn_id, turn.epoch = (
                first["request_id"],
                first["turn_id"],
                first["epoch"],
            )
        turn.traces = [
            r
            for r in rows
            if r.get("request_id") == turn.request_id or r.get("kind") == "history"
        ]

    async def start(self, pcm, frame, *, speech=True, chunk_ms=None):
        self.trace.poll()
        for previous in self.turns:
            if previous.created_end is None:
                previous.created_end = len(self.collector.events)
                previous.trace_end = len(self.trace.rows)
        known = {response_id(e) for e in self.collector.events}
        known.discard(None)
        turn = Turn(
            len(self.collector.events),
            time.monotonic(),
            time.time(),
            known,
            len(self.trace.rows),
        )
        self.turns.append(turn)
        chunk = 16000 * 2 * chunk_ms // 1000 if chunk_ms else len(pcm)
        for offset in range(0, len(pcm), max(1, chunk)):
            await asyncio.wait_for(
                self.client.append_audio(
                    pcm[offset : offset + chunk],
                    is_speech=speech,
                    video_frames=[frame, frame] if not speech else [frame],
                ),
                15,
            )
            if chunk_ms:
                await asyncio.sleep(chunk_ms / 1000)
        turn.committed_at, turn.commit_wall = time.monotonic(), time.time()
        await asyncio.wait_for(self.client.commit(create_response=True), 15)
        return turn

    async def wait(self, turn, *, timeout, release_on_text=False, listen_wait=1.5):
        deadline = time.monotonic() + timeout
        while True:
            self.refresh(turn)
            now = time.monotonic()
            row = measure(
                turn, self.collector.events, now, single_turn=self.single_turn
            )
            if self.consume.done() and not self.consume.cancelled():
                error = self.consume.exception()
                if error:
                    row["failed"] = True
                    row["errors"].append(
                        {"type": type(error).__name__, "message": str(error)}
                    )
                elif row["terminal"] is None:
                    row["failed"] = True
                    row["errors"].append(
                        {
                            "type": "consumer_stopped",
                            "message": "event stream ended before target response",
                        }
                    )
            listen_ready = (
                row["blocked"]
                and now - turn.committed_at - row["terminal_ms"] / 1000 >= listen_wait
            )
            if row["failed"] or row["terminal"] == "response.done" or listen_ready:
                turn.finished = True
                break
            if release_on_text and row["text_done"] and turn.response:
                break
            if now >= deadline:
                row["failed"] = True
                row["errors"].append(
                    {"type": "timeout", "message": "target response did not finish"}
                )
                break
            await asyncio.sleep(0.01)
        row.update(
            commit_wall=turn.commit_wall,
            session_id=self.session_id(),
            turn_ms=round((time.monotonic() - turn.committed_at) * 1000, 3),
            timed_out=any(e.get("type") == "timeout" for e in row["errors"]),
            diagnostics=turn.traces,
        )
        return row


@contextlib.asynccontextmanager
async def session(a, *, single_turn=False):
    from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig

    if urlsplit(a.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("experiment clients are loopback-only")
    config = SessionConfig(
        modalities=("text", "audio"),
        auto_response=True,
        instructions="You are AURA. Reply briefly in Chinese.",
        extra_body={
            "aura_system_prompt": "You are AURA. Reply briefly in Chinese.",
            "tts_task_type": "CustomVoice",
            "tts_language": "Chinese",
            "tts_speaker": "Vivian",
        },
    )
    deadline = time.monotonic() + a.open_timeout
    client = None
    while time.monotonic() < deadline:
        candidate = DuplexClient(
            a.url,
            model=a.model,
            config=config,
            reconnect=None,
            heartbeat_interval_s=None,
            handshake_timeout_s=min(30, a.open_timeout),
        )
        try:
            await asyncio.wait_for(
                candidate.__aenter__(), max(0.1, min(35, deadline - time.monotonic()))
            )
            client = candidate
            break
        except Exception as exc:
            if "capacity_exhausted" not in str(exc) and "resource_exhausted" not in str(
                exc
            ):
                raise
            await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))
    if client is None:
        raise TimeoutError("duplex admission deadline exhausted")
    collector = EventCollector()
    consumer = asyncio.create_task(collector.consume(client))
    try:
        await asyncio.sleep(0)  # install subscriber before sending any audio
        yield ProbeSession(
            client,
            collector,
            consumer,
            TraceReader(getattr(a, "server_log", None)),
            single_turn=single_turn,
        )
    finally:
        try:
            await asyncio.wait_for(client.close(timeout_s=3), 5)
        finally:
            try:
                await asyncio.wait_for(client.__aexit__(None, None, None), 5)
            finally:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)


def frame_b64():
    from io import BytesIO
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (64, 64), (200, 40, 40)).save(output, format="JPEG", quality=85)
    return base64.b64encode(output.getvalue()).decode("ascii")


def load_clips(paths):
    from vllm_omni.clients.duplex import read_pcm16_wav

    return {Path(p).name: read_pcm16_wav(Path(p)) for p in paths}


def add_client_arguments(parser):
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime?duplex=1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--open-timeout", type=float, default=45)
    parser.add_argument("--server-log")


def failure_row(exc, **identity):
    return {
        **identity,
        "failed": True,
        "answered": False,
        "blocked": False,
        "errors": [{"type": type(exc).__name__, "message": str(exc)}],
    }
