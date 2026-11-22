# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import _should_emit_engine_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_ASYNC_CHUNK_SENDER = "vllm_omni.model_executor.stage_input_processors.qwen3_tts.talker2code2wav_async_chunk"


@pytest.mark.parametrize(
    ("async_chunk", "final_output", "use_v2", "next_stage_func", "stopped", "has_control", "expected"),
    [
        # async+v2 connector sender suppresses intermediate engine outputs
        (True, False, True, _ASYNC_CHUNK_SENDER, False, False, False),
        # orchestrator-forwarded intermediate stages (no connector sender hook, e.g. ASR/judge) must still emit
        (True, False, True, None, False, False, True),
        (True, False, True, "", False, False, True),
        # final stages always emit
        (True, True, True, _ASYNC_CHUNK_SENDER, False, False, True),
        # v1 continues to emit
        (True, False, False, _ASYNC_CHUNK_SENDER, False, False, True),
        # stopped always emits
        (True, False, True, _ASYNC_CHUNK_SENDER, True, False, True),
        # control-only still emits
        (True, False, True, _ASYNC_CHUNK_SENDER, False, True, True),
    ],
)
def test_async_chunk_intermediate_stage_emits_only_control_outputs(
    async_chunk: bool,
    final_output: bool,
    use_v2: bool,
    next_stage_func: str | None,
    stopped: bool,
    has_control: bool,
    expected: bool,
) -> None:
    model_config = SimpleNamespace(
        async_chunk=async_chunk,
        final_output=final_output,
        use_v2_model_runner=use_v2,
        custom_process_next_stage_input_func=next_stage_func,
    )

    assert (
        _should_emit_engine_output(
            model_config,
            stopped=stopped,
            has_control=has_control,
        )
        is expected
    )
