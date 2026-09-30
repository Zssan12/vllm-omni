"""Temporary per-request timing hooks for the judge-stage profiling runs (never part of the PR).

prof_patch.py copies a vllm-omni tree, drops this file next to ``vllm_omni/`` and
calls two hooks:

* ``install_engine(engine_core)`` in each StageEngineCoreProc, right before
  ``run_busy_loop()``. Active only for stages listed in ``RJ_PROF_STAGES``
  (comma-separated stage ids, default ``1``).
* ``install_client()`` once in the API-server process (orchestrator side).

Every line is ``RJ_PROF <event> stage=S req=R t=<epoch seconds> [k=v ...]`` so
all processes on the host share one clock. Events, in request order:

  client_send0 / client_send1   StageEngineCoreClient.add_request_async entered / returned
  eng_recv                      input thread decoded the request (preprocess_add_request)
  eng_add                       busy loop added it to the scheduler
  eng_sched                     scheduler.schedule() returned with it scheduled (t0/t1 = schedule call)
  eng_exec                      model_executor.execute_model(...) called (t) and returned (t1)
  eng_update                    scheduler.update_from_output() ran (t0/t1); t0 ~ model output ready
  eng_outq                      its EngineCoreOutputs went into the output queue
  eng_encode                    output IO thread encoded it for the socket (t0/t1)
  client_decode                 API-server side decoded the EngineCoreOutputs (t0/t1)
  run step=<name>               (RJ_PROF_DEEP_STAGES) one model-runner step, CUDA-synced before and after
  other_exec                    (RJ_PROF_EXEC_STAGES) execute_model of another stage, to see GPU overlap

The orchestrator's own "output reached the orchestrator" line is OMNI_HOP output
(timing_patch.py).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("vllm.rj_prof")


def _stages() -> set[int]:
    raw = os.environ.get("RJ_PROF_STAGES", "1")
    return {int(x) for x in raw.split(",") if x.strip()}


def _log(event: str, stage: Any, req: Any, **kv: Any) -> None:
    extra = " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items())
    logger.warning("RJ_PROF %s stage=%s req=%s %s", event, stage, req, extra)


def _output_ids(outputs: Any) -> list[tuple[str, int]]:
    ids = []
    for eco in getattr(outputs, "outputs", None) or ():
        ids.append((str(getattr(eco, "request_id", "?")), int(getattr(eco, "finish_reason", None) is not None)))
    return ids


def _env_stages(name: str, default: str) -> set[int]:
    raw = os.environ.get(name, default)
    return {int(x) for x in raw.split(",") if x.strip()}


RUNNER_STEPS = (
    "_prefix_cache_step_begin",
    "register_chunk_recv",
    "recv_full_payload_inputs",
    "_update_states",
    "_prepare_inputs",
    "_preprocess",
    "_model_forward",
    "_prefix_cache_save_step",
    "_pool",
    "attach_omni_connector_output",
)


def _install_runner(engine_core: Any, stage: Any) -> None:
    """Time the runner's steps with a CUDA sync before and after each (attribution, not throughput)."""
    import torch

    executor = engine_core.model_executor
    wrapper = getattr(executor, "driver_worker", None)
    worker = getattr(wrapper, "worker", wrapper)
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        logger.warning("RJ_PROF deep stage=%s runner not found (executor=%s)", stage, type(executor).__name__)
        return
    logger.warning("RJ_PROF deep stage=%s runner=%s", stage, type(runner).__name__)

    def timed(name: str, fn: Any) -> Any:
        def inner(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            t0 = time.time()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            _log("run", stage, "-", step=name, t0=t0, t1=time.time())
            return result

        return inner

    for name in ("execute_model", *RUNNER_STEPS):
        fn = getattr(runner, name, None)
        if callable(fn):
            setattr(runner, name, timed(name, fn))
    kv = getattr(runner, "kv_transfer_manager", None)
    if kv is not None and hasattr(kv, "handle_finished_requests_kv_transfer"):
        kv.handle_finished_requests_kv_transfer = timed("kv_transfer", kv.handle_finished_requests_kv_transfer)


def _install_exec_only(engine_core: Any, stage: Any) -> None:
    executor = engine_core.model_executor
    orig_exec = executor.execute_model

    def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        out = orig_exec(scheduler_output, *args, **kwargs)
        n = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        if n:
            _log("other_exec", stage, "-", t=t0, t1=time.time(), tokens=n)
        return out

    executor.execute_model = execute_model


def _install_loop(engine_core: Any, stage: Any) -> None:
    """Busy-loop steps between add_request and schedule(), logged only right after an add."""
    state = {"armed": False}

    orig_add = engine_core.add_request

    def add_request(*args: Any, **kwargs: Any) -> Any:
        result = orig_add(*args, **kwargs)
        state["armed"] = True
        return result

    engine_core.add_request = add_request

    def wrap(name: str) -> None:
        fn = getattr(engine_core, name, None)
        if not callable(fn):
            return

        def inner(*args: Any, **kwargs: Any) -> Any:
            t0 = time.time()
            result = fn(*args, **kwargs)
            if state["armed"]:
                _log("loop", stage, "-", step=name, t0=t0, t1=time.time())
            return result

        setattr(engine_core, name, inner)

    for name in ("_process_input_queue", "_maybe_publish_request_counts", "_process_engine_step", "post_step"):
        wrap(name)

    orig_step = engine_core.step_fn

    def step_fn(*args: Any, **kwargs: Any) -> Any:
        if state["armed"]:
            _log("loop", stage, "-", step="step_fn_enter", t0=time.time(), t1=time.time())
            state["armed"] = False
        return orig_step(*args, **kwargs)

    engine_core.step_fn = step_fn
    import sys as _sys

    logger.warning("RJ_PROF loop stage=%s switchinterval=%s", stage, _sys.getswitchinterval())


def install_engine(engine_core: Any) -> None:
    model_config = engine_core.vllm_config.model_config
    stage = getattr(model_config, "stage_id", None)
    if stage not in _stages():
        if stage in _env_stages("RJ_PROF_EXEC_STAGES", ""):
            _install_exec_only(engine_core, stage)
        return
    if stage in _env_stages("RJ_PROF_DEEP_STAGES", ""):
        _install_runner(engine_core, stage)
    switch = os.environ.get("RJ_PROF_SWITCHINTERVAL")
    if switch:
        import sys as _sys

        _sys.setswitchinterval(float(switch))
    if os.environ.get("RJ_PROF_LOOP") == "1":
        _install_loop(engine_core, stage)
    scheduler = engine_core.scheduler
    adapter = getattr(scheduler, "chunk_transfer_adapter", None)
    logger.warning(
        "RJ_PROF install stage=%s pid=%d scheduler=%s async_chunk=%s chunk_adapter=%s receives_chunks=%s "
        "input_coordinator=%s runner=%s enforce_eager=%s async_scheduling=%s batch_queue=%s block=%s",
        stage,
        os.getpid(),
        type(scheduler).__name__,
        getattr(model_config, "async_chunk", None),
        adapter is not None,
        getattr(adapter, "receives_chunks", None),
        type(getattr(scheduler, "input_coordinator", None)).__name__,
        getattr(model_config, "runner_type", None),
        getattr(model_config, "enforce_eager", None),
        getattr(engine_core.vllm_config.scheduler_config, "async_scheduling", None),
        getattr(engine_core, "batch_queue", None) is not None,
        getattr(engine_core, "process_input_queue_block", None),
    )
    import sys
    import threading

    logger.warning(
        "RJ_PROF threads stage=%s switchinterval=%s threads=%s",
        stage,
        sys.getswitchinterval(),
        ",".join(sorted(t.name for t in threading.enumerate())),
    )

    orig_pre = engine_core.preprocess_add_request

    def preprocess_add_request(request: Any) -> Any:
        _log("eng_recv", stage, getattr(request, "request_id", "?"), t=time.time())
        return orig_pre(request)

    engine_core.preprocess_add_request = preprocess_add_request

    orig_add = engine_core.add_request

    def add_request(request: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        result = orig_add(request, *args, **kwargs)
        _log("eng_add", stage, getattr(request, "request_id", "?"), t=t0, t1=time.time(),
             status=getattr(getattr(request, "status", None), "name", None))
        return result

    engine_core.add_request = add_request

    orig_schedule = scheduler.schedule

    def schedule(*args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        out = orig_schedule(*args, **kwargs)
        t1 = time.time()
        new_ids = [getattr(r, "req_id", "?") for r in getattr(out, "scheduled_new_reqs", ()) or ()]
        cached = getattr(getattr(out, "scheduled_cached_reqs", None), "req_ids", ()) or ()
        if new_ids or cached:
            _log("eng_sched", stage, ",".join([*new_ids, *cached]), t0=t0, t1=t1,
                 tokens=getattr(out, "total_num_scheduled_tokens", "?"))
        return out

    scheduler.schedule = schedule

    executor = engine_core.model_executor
    orig_exec = executor.execute_model

    def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        out = orig_exec(scheduler_output, *args, **kwargs)
        if getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            _log("eng_exec", stage, "-", t=t0, t1=time.time(), non_block=kwargs.get("non_block"))
        return out

    executor.execute_model = execute_model

    orig_update = scheduler.update_from_output

    def update_from_output(scheduler_output: Any, model_output: Any) -> Any:
        t0 = time.time()
        out = orig_update(scheduler_output, model_output)
        t1 = time.time()
        if getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            ids = [rid for outs in (out or {}).values() for rid, _ in _output_ids(outs)]
            _log("eng_update", stage, ",".join(ids) or "-", t0=t0, t1=t1)
        return out

    scheduler.update_from_output = update_from_output

    output_queue = engine_core.output_queue
    orig_put = output_queue.put_nowait

    def put_nowait(item: Any) -> Any:
        if isinstance(item, tuple) and len(item) == 2:
            ids = _output_ids(item[1])
            if ids:
                _log("eng_outq", stage, ",".join(f"{r}:{f}" for r, f in ids), t=time.time())
        return orig_put(item)

    output_queue.put_nowait = put_nowait

    from vllm.v1.serial_utils import MsgpackEncoder

    orig_encode_into = MsgpackEncoder.encode_into

    def encode_into(self: Any, obj: Any, buf: Any) -> Any:
        ids = _output_ids(obj) if hasattr(obj, "outputs") else []
        t0 = time.time()
        result = orig_encode_into(self, obj, buf)
        if ids:
            _log("eng_encode", stage, ",".join(f"{r}:{f}" for r, f in ids), t0=t0, t1=time.time())
        return result

    MsgpackEncoder.encode_into = encode_into


def install_client() -> None:
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.serial_utils import MsgpackDecoder

    from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

    stages = _stages()
    orig_decode = MsgpackDecoder.decode

    def decode(self: Any, bufs: Any) -> Any:
        t0 = time.time()
        obj = orig_decode(self, bufs)
        if isinstance(obj, EngineCoreOutputs) and obj.outputs:
            pooled = any(getattr(eco, "pooling_output", None) is not None for eco in obj.outputs)
            finished = [r for r, f in _output_ids(obj) if f]
            if pooled or finished:
                _log("client_decode", "?", ",".join(finished) or "-", t0=t0, t1=time.time(),
                     engine=getattr(obj, "engine_index", "?"), pooled=int(pooled), n=len(obj.outputs))
        return obj

    MsgpackDecoder.decode = decode

    orig_add = StageEngineCoreClient.add_request_async

    async def add_request_async(self: Any, request: Any) -> None:
        stage = getattr(self, "stage_id", None)
        if stage not in stages:
            return await orig_add(self, request)
        rid = getattr(request, "request_id", "?")
        _log("client_send0", stage, rid, t=time.time())
        await orig_add(self, request)
        _log("client_send1", stage, rid, t=time.time())

    StageEngineCoreClient.add_request_async = add_request_async
    logger.warning("RJ_PROF install_client pid=%d stages=%s", os.getpid(), sorted(stages))
