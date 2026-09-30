"""Final AURA GPU suite. Default: local preflight only; --execute starts services.

Freeze a manifest locally before transferring the approved source and these tools.
The server creates two NEW experiment copies; it never edits the approved source.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import yaml

from apply_silent_turn_fix_prototype import transform as prototype

TOOLS = (
    "aura_final_gpu_suite.py",
    "run_final_gpu_suite.sh",
    "aura_probe_common.py",
    "aura_bench_diagnostics.py",
    "aura_concurrency_real.py",
    "aura_judge_onoff.py",
    "aura_reject_probe.py",
    "aura_ptt_probe.py",
    "apply_silent_turn_fix_prototype.py",
    "analyze_onoff.py",
)
SMOKE = "examples/online_serving/aura_omni/aura_omni_duplex_smoke.yaml"
AUDIO_SHA256 = {
    "a0.wav": "b7c7ae1478d70e6df78c1424b23e3aedbfaa199a842e3ec486a637ff658d23d1",
    "a1.wav": "0314e924bd9f0fb8a51632cbbcaac8f3bd25da5e953916cd6d12cac2b891765b",
    "a3.wav": "11aee8c032314dc855a0b7c8bd4bb5c135f7d4a2b6f7675cbce829693117f9ed",
    "a4.wav": "d916c6c098b27b5d580ce759bb654ff507aa75fe635432f8988ee663700c1cae",
}
DECORATORS = {
    "vllm_omni/model_executor/models/aura_omni/duplex/plugin.py": {
        "AuraDuplexPlugin.plan_append": "trace_append",
        "AuraDuplexPlugin.commit_model_context": "trace_history",
        "AuraJudgedDuplexPlugin._judge_rejects": "trace_judge",
    },
    "vllm_omni/model_executor/stage_input_processors/aura_omni.py": {
        "asr2judge": "trace_asr"
    },
    "vllm_omni/engine/duplex/session/model_channel.py": {
        "_on_model_listen": "trace_listen"
    },
}
TIMING = {
    "vllm_omni/engine/duplex_orchestrator.py": (
        "        del replica_id\n",
        '        logger.warning("OMNI_TIMING submit stage=%d req=%s t=%.6f", stage_id, request_id, _time.time())\n',
    ),
    "vllm_omni/engine/orchestrator.py": (
        '        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)\n        _t_submit_start = _time.perf_counter()\n',
        '        logger.warning("OMNI_TIMING forward src=%d dst=%d req=%s t=%.6f", src_stage_id, next_logical, req_id, _time.time())\n',
    ),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def runtime_files(root):
    # Runtime source/resources only. Never traverse .git, credentials, model caches,
    # or developer home directories. All entries must be regular files inside root.
    files = []
    for path in sorted((root / "vllm_omni").rglob("*")):
        if (
            "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
            or not path.is_file()
        ):
            continue
        if path.is_symlink() or any(
            s in path.name.lower() for s in ("credential", ".env", "secret")
        ):
            raise ValueError(f"unexpected runtime input (not read): {path}")
        files.append(path.relative_to(root).as_posix())
    files += [SMOKE, "pyproject.toml"]
    for name in files:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing/non-regular source: {name}")
    return sorted(files)


def manifest(root, toolroot):
    return {
        "schema": 1,
        "source": {name: digest(root / name) for name in runtime_files(root)},
        "tools": {name: digest(toolroot / name) for name in TOOLS},
    }


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"instrumentation anchor must occur once: {before[:60]!r}")
    return source.replace(before, after, 1)


def decorate(source, selectors):
    tree = ast.parse(source)
    matches = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                key = f"{node.name}.{getattr(child, 'name', '')}"
                if key in selectors:
                    matches[key] = child
    for key in selectors:
        if "." not in key:
            nodes = [
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == key
            ]
            if len(nodes) != 1:
                raise ValueError(f"decorator target must occur once: {key}")
            matches[key] = nodes[0]
    if set(matches) != set(selectors):
        raise ValueError("missing decorator target")
    lines = source.splitlines(True)
    for key, node in sorted(
        matches.items(), key=lambda pair: pair[1].lineno, reverse=True
    ):
        if node.decorator_list:
            raise ValueError(f"unexpected existing decorators: {key}")
        lines.insert(
            node.lineno - 1, " " * node.col_offset + "@" + selectors[key] + "\n"
        )
    futures = [
        n.end_lineno
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    doc_end = (
        tree.body[0].end_lineno
        if tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
        else 0
    )
    lines.insert(
        max(futures or [doc_end]),
        "from aura_bench_diagnostics import "
        + ", ".join(sorted(set(selectors.values())))
        + "\n",
    )
    result = "".join(lines)
    ast.parse(result)
    return result


def transformed(root, *, fix):
    result = {}
    for name in set(TIMING) | set(DECORATORS):
        text = (root / name).read_text()
        if fix and name.endswith("/model_channel.py"):
            text = prototype(text)
        if name in TIMING:
            anchor, hook = TIMING[name]
            text = once(text, anchor, anchor + hook)
        if name in DECORATORS:
            text = decorate(text, DECORATORS[name])
        ast.parse(text)
        result[name] = text
    return result


def configs(root, models):
    off = yaml.safe_load((root / SMOKE).read_text())
    if off.get("pipeline") != "aura_omni" or [s["stage_id"] for s in off["stages"]] != [
        0,
        1,
        2,
        3,
    ]:
        raise ValueError("unexpected baseline smoke topology")
    model_names = [
        "Qwen3-ASR-1.7B",
        "AURA-88ee5506",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ]
    for stage, name in zip(off["stages"], model_names):
        stage["model"] = str(models / name)
    on = copy.deepcopy(off)
    on["pipeline"] = "aura_omni_judged"
    for stage in on["stages"][1:]:
        stage["stage_id"] += 1
        for field, old, new in [
            ("input_connectors", "from_stage_2", "from_stage_3"),
            ("output_connectors", "to_stage_3", "to_stage_4"),
        ]:
            if field in stage:
                stage[field] = {
                    new if k == old else k: v for k, v in stage[field].items()
                }
    on["stages"].insert(
        1,
        {
            "stage_id": 1,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.08,
            "enforce_eager": False,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "async_scheduling": True,
            "max_model_len": 2048,
            "max_num_batched_tokens": 2048,
            "devices": "0",
            "model": str(models / "Qwen3-1.7B"),
            "default_sampling_params": {"temperature": 0.0, "max_tokens": 1},
        },
    )
    result = {"smoke-off": off, "smoke-on": on}
    for mode in ("off", "on"):
        cfg = copy.deepcopy(result[f"smoke-{mode}"])
        cfg["duplex_session"]["max_sessions"] = 8
        for stage in cfg["stages"]:
            stage["max_num_seqs"] = 8
        result[f"conc-{mode}"] = cfg
    for name, cfg in result.items():
        if cfg.get("async_chunk") is not True or cfg.get("session_mode") != "duplex":
            raise ValueError(f"not a duplex profile: {name}")
    return result


def save_json(path, value):
    with Path(path).open("x") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, default=str)
        out.write("\n")


def make_copy(root, destination, frozen, changes, toolroot):
    destination.mkdir()  # exclusive; never reuse/overwrite an earlier experiment
    for name in frozen["source"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (
            changes[name].encode() if name in changes else (root / name).read_bytes()
        )
        with target.open("xb") as out:
            out.write(content)
    shutil.copyfile(
        toolroot / "aura_bench_diagnostics.py",
        destination / "aura_bench_diagnostics.py",
    )
    save_json(
        destination / "EXPERIMENT-SOURCE.json",
        {
            "base_manifest": frozen,
            "changed": {name: digest(destination / name) for name in changes},
            "prototype": any("PROTOTYPE:" in content for content in changes.values()),
        },
    )


class SuiteStopped(RuntimeError):
    pass


class OwnedProcess:
    """Track owned identities; never kill a process just because its name matches."""

    def __init__(self, argv, *, cwd, env, log):
        import psutil

        self.psutil = psutil
        self.stream = Path(log).open("xb")
        try:
            self.process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=self.stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.stream.close()
            raise
        self.pid = self.process.pid
        self.owned = {}
        self.refresh()

    def refresh(self):
        try:
            root = self.psutil.Process(self.pid)
            # Root PID reuse cannot extend ownership to a foreign process.
            if self.pid in self.owned and root.create_time() != self.owned[self.pid]:
                return
            for p in [root, *root.children(recursive=True)]:
                self.owned.setdefault(p.pid, p.create_time())
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            pass

    def living(self):
        found = []
        for pid, created in self.owned.items():
            try:
                process = self.psutil.Process(pid)
                if (
                    process.create_time() == created
                    and process.status() != self.psutil.STATUS_ZOMBIE
                ):
                    found.append(process)
            except self.psutil.NoSuchProcess:
                pass
        return found

    def stop(self):
        self.refresh()
        for sig, grace in [(signal.SIGTERM, 12), (signal.SIGKILL, 4)]:
            living = self.living()
            if not living:
                break
            # The leader and children inherit this new PGID. Only signal it
            # while a verified owned member still belongs to it.
            group_owned = False
            for p in living:
                try:
                    group_owned |= os.getpgid(p.pid) == self.pid
                except ProcessLookupError:
                    pass
            if group_owned:
                try:
                    os.killpg(self.pid, sig)
                except ProcessLookupError:
                    pass
            for p in living:
                try:
                    if p.create_time() == self.owned[p.pid]:
                        p.send_signal(
                            sig
                        )  # also covers an owned child that changed PGID
                except self.psutil.NoSuchProcess:
                    pass
            until = time.monotonic() + grace
            while self.living() and time.monotonic() < until:
                time.sleep(0.2)
        self.process.wait(timeout=3)
        self.stream.close()
        if self.living():
            raise SuiteStopped("owned processes survived cleanup")


def gpu_query(fields, *, compute=False):
    query = "--query-compute-apps=" if compute else "--query-gpu="
    return subprocess.check_output(
        ["nvidia-smi", "--id=0", query + fields, "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )


def gpu_pids():
    return {
        int(line.strip())
        for line in gpu_query("pid", compute=True).splitlines()
        if line.strip().isdigit()
    }


def port_free(port):
    with socket.socket() as sock:
        # SO_REUSEADDR: closed client connections leave the port in TIME_WAIT for
        # ~60 s after the server stops; that is not a listener. An actual
        # listening socket still makes bind() fail.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def warmup_state(text):
    if "Duplex warmup failed;" in text or "aborting warmup" in text:
        return "failed"
    if "Duplex warmup finished" in text:
        return "ready"
    return "waiting"


class Suite:
    def __init__(self, args, run):
        self.a, self.run = args, run
        self.start = time.monotonic()
        self.deadline = self.start + args.budget_minutes * 60
        self.service = None
        self.client = None
        self.results = []
        self.baseline_memory = float(gpu_query("memory.used").strip())
        self.phase_file = run / "phase.control"
        self.phase("timing")
        self.toolroot = Path(__file__).resolve().parent
        self.env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_OMNI_VERSION_OVERRIDE": "0.25.0.dev0",
            "VLLM_OMNI_TARGET_DEVICE": "cuda",
            "VLLM_AURA_SILENT_TOKEN_ID": "151669",
            "VLLM_AURA_IM_END_TOKEN_ID": "151645",
            "VLLM_AURA_IM_START_TOKEN_ID": "151644",
            "VLLM_AURA_ASSISTANT_TOKEN_ID": "77091",
            "AURA_BENCH_PHASE_FILE": str(self.phase_file),
        }

    def phase(self, value):
        # Mutable run control, not an experimental result. All logs/results use x/xb.
        if self.phase_file.exists() and self.phase_file.read_text().strip() == value:
            return
        self.phase_file.write_text(value + "\n")
        # Let the diagnostic cache expire before starting a new client. Avoid
        # a file read on every judge callback inside the latency interval.
        time.sleep(2.1)

    def work_seconds(self):
        return max(0, self.deadline - time.monotonic() - 180)

    def tick(self):
        if self.work_seconds() <= 0:
            raise SuiteStopped(
                "suite deadline reached; remaining budget reserved for cleanup"
            )
        if self.service:
            self.service.refresh()

    def telemetry(self, name):
        value = {
            "gpu": gpu_query(
                "timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem"
            ),
            "processes": gpu_query("pid,process_name,used_gpu_memory", compute=True),
        }
        save_json(self.run / (name + ".gpu.json"), value)

    def stop_service(self, name):
        if self.service:
            current = self.service
            try:
                current.stop()
            finally:
                save_json(
                    self.run / (name + ".cleanup.json"),
                    {
                        "pid": current.pid,
                        "owned": current.owned,
                        "remaining": [p.pid for p in current.living()],
                    },
                )
            self.service = None
        until = min(self.deadline, time.monotonic() + 15)

        def idle():
            return (
                not gpu_pids()
                and port_free(self.a.port)
                and float(gpu_query("memory.used").strip()) <= self.baseline_memory + 64
            )

        while not idle() and time.monotonic() < until:
            time.sleep(0.5)
        self.telemetry(name + "-after-stop")
        if not idle():
            raise SuiteStopped("GPU process or listener remains; refusing next service")

    def start_service(self, code, config, name):
        self.tick()
        if not port_free(self.a.port) or gpu_pids():
            raise SuiteStopped(
                "GPU/port occupied before startup; no foreign process will be killed"
            )
        self.telemetry(name + "-before")
        self.code, self.log = code, self.run / (name + ".server.log")
        self.active_env = {**self.env, "PYTHONPATH": str(code)}
        import_program = (
            "import importlib,importlib.metadata,json,pathlib; "
            f"root=pathlib.Path({str(code)!r}).resolve(); "
            "names=['vllm_omni.clients.duplex','vllm_omni.model_executor.models.aura_omni.duplex.plugin',"
            "'vllm_omni.model_executor.models.aura_omni.judge','vllm_omni.engine.duplex.session.model_channel']; "
            "paths={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names}; "
            "assert all(pathlib.Path(x).is_relative_to(root) for x in paths.values()), paths; "
            "versions={n:importlib.metadata.version(n) for n in ['vllm','torch','transformers','psutil','Pillow']}; "
            "import torch; versions['torch_cuda']=torch.version.cuda; "
            "print(json.dumps({'paths':paths,'versions':versions}))"
        )
        checked = subprocess.run(
            [sys.executable, "-B", "-c", import_program],
            cwd=code,
            env=self.active_env,
            capture_output=True,
            text=True,
            timeout=min(45, self.work_seconds()),
        )
        save_json(
            self.run / (name + ".imports.json"),
            {
                "exit_code": checked.returncode,
                "stdout": checked.stdout,
                "stderr": checked.stderr,
            },
        )
        if checked.returncode:
            raise SuiteStopped("runtime import path/dependency preflight failed")
        self.tick()
        argv = [
            str(Path(sys.executable).parent / "vllm-omni"),
            "serve",
            str(self.a.models / "AURA-88ee5506"),
            "--omni",
            "--deploy-config",
            str(config),
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.a.port),
        ]
        save_json(
            self.run / (name + ".command.json"),
            {"argv": argv, "cwd": str(code), "PYTHONPATH": str(code)},
        )
        self.service = OwnedProcess(argv, cwd=code, env=self.active_env, log=self.log)
        save_json(
            self.run / (name + ".process.json"),
            {"pid": self.service.pid, "owned": self.service.owned},
        )
        until = time.monotonic() + min(300, self.work_seconds())
        while time.monotonic() < until:
            self.tick()
            if self.service.process.poll() is not None:
                raise SuiteStopped(f"server exited: {name}")
            state = warmup_state(self.log.read_text(errors="replace"))
            if state == "failed":
                raise SuiteStopped(f"warmup failed: {name}")
            if state == "ready":
                owned_listener = False
                for p in self.service.living():
                    try:
                        connections = (
                            p.net_connections
                            if hasattr(p, "net_connections")
                            else p.connections
                        )
                        owned_listener |= any(
                            c.status == "LISTEN" and c.laddr.port == self.a.port
                            for c in connections(kind="inet")
                        )
                    except self.service.psutil.Error:
                        pass
                if owned_listener:
                    try:
                        with build_opener(ProxyHandler({})).open(
                            f"http://127.0.0.1:{self.a.port}/health", timeout=2
                        ) as response:
                            if response.status == 200:
                                self.telemetry(name + "-warm")
                                return
                    except OSError:
                        pass
            time.sleep(0.5)
        raise SuiteStopped(f"startup deadline exhausted: {name}")

    def run_client(
        self, script, name, arguments, *, seconds=240, accepted=(0,), diagnostic=False
    ):
        self.tick()
        self.phase("diagnostic" if diagnostic else "timing")
        self.tick()
        log_start = self.log.stat().st_size
        argv = [
            sys.executable,
            "-B",
            str(self.toolroot / script),
            "--model",
            str(self.a.models / "AURA-88ee5506"),
            "--url",
            f"ws://127.0.0.1:{self.a.port}/v1/realtime?duplex=1",
            "--out",
            str(self.run / (name + ".jsonl")),
            *arguments,
        ]
        if diagnostic:
            argv += ["--server-log", str(self.log)]
        save_json(
            self.run / (name + ".client-command.json"),
            {"argv": argv, "PYTHONPATH": str(self.code)},
        )
        until = time.monotonic() + min(seconds, self.work_seconds())
        self.client = OwnedProcess(
            argv,
            cwd=self.code,
            env=self.active_env,
            log=self.run / (name + ".client.log"),
        )
        try:
            while self.client.process.poll() is None:
                self.tick()
                self.client.refresh()
                if time.monotonic() >= until:
                    raise SuiteStopped(f"client deadline: {name}")
                if self.service.process.poll() is not None:
                    raise SuiteStopped(f"server died during {name}")
                time.sleep(0.25)
            code = self.client.process.returncode
            self.results.append(
                {"name": name, "exit_code": code, "accepted": code in accepted}
            )
            save_json(self.run / (name + ".exit.json"), self.results[-1])
            if code not in accepted:
                raise SuiteStopped(
                    f"{name}: exit {code}; see raw results, no continuation"
                )
            if name in ("onoff-on", "onoff-off"):
                mode = name.removeprefix("onoff-")
                with (
                    self.log.open("rb") as source,
                    (self.run / ("server-" + mode + ".log")).open("xb") as target,
                ):
                    source.seek(log_start)
                    target.write(source.read())
                save_json(
                    self.run / (name + ".server-slice.json"),
                    {
                        "source": str(self.log),
                        "start_byte": log_start,
                        "slice_sha256": digest(self.run / ("server-" + mode + ".log")),
                    },
                )
            return code
        finally:
            current = self.client
            try:
                current.stop()
            finally:
                save_json(
                    self.run / (name + ".client-cleanup.json"),
                    {
                        "pid": current.pid,
                        "owned": current.owned,
                        "remaining": [p.pid for p in current.living()],
                    },
                )
            self.client = None
            self.telemetry(name + "-after-client")

    def cleanup(self):
        errors = []
        if self.client:
            try:
                self.client.stop()
                self.client = None
            except BaseException as exc:
                errors.append("client: " + str(exc))
        try:
            self.stop_service("final")
        except BaseException as exc:
            errors.append("server: " + str(exc))
        return errors

    def execute(self, frozen, cfg):
        code_timing, code_fix = self.run / "code-timing", self.run / "code-fix"
        for target, fix in [(code_timing, False), (code_fix, True)]:
            make_copy(
                self.a.source,
                target,
                frozen,
                transformed(self.a.source, fix=fix),
                self.toolroot,
            )
        config_paths = {}
        for key, value in cfg.items():
            path = self.run / (key + ".yaml")
            with path.open("x") as out:
                yaml.safe_dump(value, out, allow_unicode=True, sort_keys=False)
            config_paths[key] = path
        probe_args = [
            "--audio-dir",
            str(self.a.audio_dir),
            "--sessions",
            "1",
            "--timeout",
            "35",
        ]
        clips = [
            str(self.a.audio_dir / (name + ".wav")) + "=" + str(n)
            for name, n in [("a3", 8), ("a4", 8), ("a0", 3), ("a1", 3)]
        ]
        # Recovery first; the nofix timing server also supplies the final on arm.
        # The recovery probe sends each turn as one append + commit, which carries the
        # previous turn's audio into ASR unless input_audio_buffer.clear is sent first
        # (seen in final-suite-20260929T120859Z: the judge saw "...天气怎么样？嗯嗯" and
        # never answered NO). The streaming PTT probe does not need it.
        self.start_service(code_timing, config_paths["smoke-on"], "nofix-on")
        self.run_client(
            "aura_reject_probe.py",
            "recovery-nofix",
            probe_args + ["--expect", "broken", "--clear-before-append"],
            diagnostic=True,
        )
        self.run_client(
            "aura_judge_onoff.py",
            "fresh-judge-tokens",
            [
                "--mode",
                "on",
                "--warmup",
                "0",
                "--clips",
                str(self.a.audio_dir / "a0.wav") + "=1",
                str(self.a.audio_dir / "a1.wav") + "=1",
            ],
            diagnostic=True,
            seconds=90,
        )
        self.run_client(
            "aura_judge_onoff.py",
            "onoff-on",
            ["--mode", "on", "--clips", *clips],
            seconds=360,
        )
        self.stop_service("nofix-on")
        self.start_service(code_fix, config_paths["smoke-on"], "fix-on")
        self.run_client(
            "aura_reject_probe.py",
            "recovery-fix",
            probe_args + ["--expect", "fixed", "--clear-before-append"],
            diagnostic=True,
        )
        self.run_client("aura_ptt_probe.py", "ptt-fix-on", probe_args, diagnostic=True)
        self.stop_service("fix-on")
        self.start_service(code_timing, config_paths["smoke-off"], "nofix-off")
        self.run_client(
            "aura_judge_onoff.py",
            "onoff-off",
            ["--mode", "off", "--clips", *clips],
            seconds=360,
        )
        from analyze_onoff import analyze

        analysis = analyze(self.run)
        save_json(self.run / "timing-analysis.json", analysis)
        if analysis["status"] != "valid":
            raise SuiteStopped("latency audit invalid; do not quote the aggregates")
        self.run_client(
            "aura_ptt_probe.py",
            "ptt-nofix-off",
            probe_args,
            diagnostic=True,
            accepted=(0, 4),
        )
        self.stop_service("nofix-off")
        self.start_service(code_fix, config_paths["smoke-off"], "fix-off")
        self.run_client("aura_ptt_probe.py", "ptt-fix-off", probe_args, diagnostic=True)
        self.stop_service("fix-off")
        if self.work_seconds() < 12 * 60:
            raise SuiteStopped(
                "not enough budget for BOTH concurrency arms; neither started"
            )
        for mode in ("off", "on"):
            self.start_service(
                self.a.source, config_paths["conc-" + mode], "concurrency-" + mode
            )
            self.run_client(
                "aura_concurrency_real.py",
                "concurrency-" + mode,
                [
                    "--mode",
                    mode,
                    "--users",
                    "4",
                    "8",
                    "--rounds",
                    "3",
                    "--clips",
                    *[
                        str(self.a.audio_dir / (n + ".wav"))
                        for n in ("a3", "a4", "a0", "a1")
                    ],
                ],
                seconds=360,
            )
            self.stop_service("concurrency-" + mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=os.environ.get("FINAL"))
    p.add_argument("--models", type=Path, default=Path("/root/autodl-tmp/models"))
    p.add_argument(
        "--audio-dir", type=Path, default=Path("/root/autodl-tmp/rfc/turn-audio")
    )
    p.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/rfc/runs"))
    p.add_argument("--manifest", type=Path)
    p.add_argument(
        "--freeze",
        type=Path,
        help="write a NEW frozen source/tool manifest locally; no GPU activity",
    )
    p.add_argument("--execute", action="store_true")
    p.add_argument("--budget-minutes", type=float, default=55)
    p.add_argument("--port", type=int, default=8099)
    a = p.parse_args()
    if not 1 <= a.port <= 65535:
        p.error("port must be between 1 and 65535")
    if a.source is None:
        p.error("--source or FINAL is required")
    a.source = a.source.resolve()
    a.models, a.audio_dir, a.runs_root = (
        a.models.resolve(),
        a.audio_dir.resolve(),
        a.runs_root.resolve(),
    )
    toolroot = Path(__file__).resolve().parent
    frozen = manifest(a.source, toolroot)
    cfg = configs(a.source, a.models)
    transformed(a.source, fix=False)
    transformed(a.source, fix=True)
    if a.manifest and json.loads(a.manifest.read_text()) != frozen:
        p.error(
            "frozen source/tools do not match; rebuild/review before spending GPU time"
        )
    if a.freeze:
        if a.execute:
            p.error("freeze and execute must be separate steps")
        save_json(a.freeze, frozen)
    if not a.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "source_files": len(frozen["source"]),
                    "tools": len(frozen["tools"]),
                    "server_starts": 6,
                    "budget_minutes": a.budget_minutes,
                    "concurrency": {
                        k: [s["max_num_seqs"] for s in v["stages"]]
                        for k, v in cfg.items()
                        if k.startswith("conc")
                    },
                },
                indent=2,
            )
        )
        return 0
    if not a.manifest or not sys.platform.startswith("linux") or a.budget_minutes < 10:
        p.error(
            "execution needs Linux, a frozen --manifest and at least 10 budget minutes"
        )
    import psutil  # dependency preflight, before any GPU service

    del psutil
    for model in (
        "AURA-88ee5506",
        "Qwen3-ASR-1.7B",
        "Qwen3-1.7B",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ):
        if not (a.models / model / "config.json").is_file():
            p.error(f"missing local model {model}; no download will be attempted")
    for name in ("a0", "a1", "a3", "a4"):
        path = a.audio_dir / (name + ".wav")
        if not path.is_file() or digest(path) != AUDIO_SHA256[path.name]:
            p.error("missing/changed fixed audio clip; do not mix workloads")
    if not a.runs_root.is_dir() or shutil.disk_usage(a.runs_root).free < 4 * 1024**3:
        p.error("runs-root must exist with at least 4 GiB free; no automatic cleanup")
    if gpu_pids() or not port_free(a.port):
        p.error("GPU or port already occupied")
    run = a.runs_root / (
        "final-suite-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run.mkdir()
    save_json(run / "frozen-manifest.json", frozen)
    suite = Suite(a, run)

    def interrupted(signum, frame):
        raise SuiteStopped(f"interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    status, reason = "complete", None
    try:
        suite.execute(frozen, cfg)
    except BaseException as exc:
        status, reason = "incomplete", f"{type(exc).__name__}: {exc}"
    finally:
        # Do not interrupt cleanup a second time; only owned processes are touched.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        cleanup_errors = suite.cleanup()
        if cleanup_errors:
            status, reason = "cleanup_failed", f"{reason}; {cleanup_errors}"
        try:
            unchanged = manifest(a.source, toolroot) == frozen
        except Exception:
            unchanged = False
        if not unchanged and status != "cleanup_failed":
            status, reason = (
                "incomplete",
                "approved source/tools changed during the run",
            )
        summary = {
            "status": status,
            "reason": reason,
            "results": suite.results,
            "elapsed_s": time.monotonic() - suite.start,
            "run": str(run),
            "source_unchanged": unchanged,
        }
        save_json(run / "SUITE-RESULT.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
