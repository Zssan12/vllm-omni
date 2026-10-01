# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA's own silence must not swallow the next turn of the same auto-response session.

AURA's data plane projects a finished ``<|silent|>`` as a terminal listen. The
model channel has to complete that turn; otherwise the next append reuses the
silent turn's request id, the data plane still holds that request as silent,
and the next answer is swallowed instead of opening a response.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.model_channel import ModelChannel
from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin
from vllm_omni.model_executor.stage_input_processors.aura_omni import SILENT_TEXT

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AURA_STAGE = 1


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate, fmt, speed
    return "ZmFrZQ=="


def _thinker_output(request_id: str, text: str, *, finished: bool) -> SimpleNamespace:
    completion = SimpleNamespace(text=text, cumulative_text=text, finished=finished)
    return SimpleNamespace(request_id=request_id, finished=finished, stage_id=AURA_STAGE, outputs=[completion])


class _Out:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def auto_responds(self) -> bool:
        return True

    def emit(self, payload: dict[str, object]) -> None:
        self.events.append(payload)


class _Port:
    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        del request_ids, abort


async def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _next_request_id(session: DuplexEngineSession) -> str:
    fence = helpers.append_fence(session, None)
    return DuplexSessionManager.stage_request_id(fence, stage_id=AURA_STAGE, resumable=False)


def test_aura_silence_lets_the_next_turn_of_the_session_speak() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    session = DuplexEngineSession(
        session_id="s-aura-silent",
        config=DuplexSessionConfig(model="aura", modalities=["text", "audio"]),
        capabilities=DuplexCapabilities(supports_core_resumable_request=False),
    )
    out = _Out()
    channel = ModelChannel(
        SimpleNamespace(
            session=session,
            plugin=plugin,
            stage_port=_Port(),
            model_state=SimpleNamespace(
                continuation_owner_id=None, continuation_units=0, clear_continuation=lambda: None
            ),
            services=SimpleNamespace(spawn=lambda coro, **_k: coro.close()),
            run=SimpleNamespace(closing=False),
        ),
        out,
        close_from_runtime=_noop,
        schedule_silence_continuation=_noop,
        abort_request=_noop,
    )

    async def deliver(output: SimpleNamespace) -> None:
        for event in plugin.data_plane.project({"data_plane_outputs": [output]}):
            await channel._send_one_model_output_event(event)

    silent_id = _next_request_id(session)
    plugin.data_plane.begin_request(silent_id)
    session.bind_request(silent_id)
    asyncio.run(deliver(_thinker_output(silent_id, SILENT_TEXT, finished=True)))
    assert [event.get("type") for event in out.events] == ["response.listen"]

    next_id = _next_request_id(session)
    plugin.data_plane.begin_request(next_id)
    session.bind_request(next_id)
    out.events.clear()
    asyncio.run(deliver(_thinker_output(next_id, "上海明天多云。", finished=False)))
    assert "response.listen" not in [event.get("type") for event in out.events]
    assert session.active_response_id is not None
    assert next_id != silent_id
