"""Run one arm on the GPU host: start the server, wait for duplex warmup, run a client, stop.

Reuses tools/rfc_bench/aura_final_gpu_suite.py's OwnedProcess (only processes this
script started are ever signalled), port check and warmup detection. Every file
is created exclusively in <out>; nothing is overwritten.

    python run_arm.py --code <timing-patched tree> --config <deploy yaml> --name on-qwen3 \
        --mode on --client onoff --models <dir> --audio-dir <dir> --out <arm dir>

Outputs (arm dir):
  <name>.server.log, <name>.client.log, <name>.jsonl      raw server / client output
  server-<mode>.log, onoff-<mode>.jsonl                    onoff arms: the client window only,
                                                           the layout tools/rfc_bench/analyze_onoff.py reads
  <name>.hops.json                                         hop_analysis.py on the client window
  gpu-<name>-{before,ready,after}.txt                      nvidia-smi snapshots
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rfc_bench"))
from aura_final_gpu_suite import OwnedProcess, gpu_pids, port_free, warmup_state  # noqa: E402

CLIPS_ONOFF = [("a3", 8), ("a4", 8), ("a0", 3), ("a1", 3)]


def snapshot(path: Path) -> None:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
    path.open("x").write(out.stdout + out.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=("on", "off"), required=True)
    p.add_argument("--client", choices=("onoff", "conc"), required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--serve-bin", type=Path, help="default: the vllm-omni next to this Python")
    p.add_argument("--startup-timeout", type=float, default=420)
    p.add_argument("--client-timeout", type=float, default=900)
    p.add_argument("--after-failed-warmup-s", type=float, default=90)
    p.add_argument("--skip", type=int, default=8, help="conc arms only: hop stats skip the first N requests")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not port_free(a.port) or gpu_pids():
        raise SystemExit("GPU or port busy before start; nothing is killed, stop here")
    env = {
        **os.environ,
        "PYTHONPATH": str(a.code),
        # Same as the earlier GPU suite: no Hugging Face network lookups (they
        # retry with back-off and block the server's event loop), and the AURA
        # bridge reads the Qwen3-TTS tokenizer from the local model directory.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "VLLM_AURA_TTS_TOKENIZER": str(a.models / "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    }
    snapshot(a.out / f"gpu-{a.name}-before.txt")
    model = a.models / "AURA-88ee5506"
    argv = [str(a.serve_bin or Path(sys.executable).parent / "vllm-omni"), "serve", str(model), "--omni", "--deploy-config",
            str(a.config), "--trust-remote-code", "--host", "127.0.0.1", "--port", str(a.port)]
    shutil.copyfile(a.config, a.out / f"{a.name}.deploy.yaml")
    (a.out / f"{a.name}.command.json").open("x").write(json.dumps({"argv": argv, "PYTHONPATH": str(a.code)}, indent=1))
    log = a.out / f"{a.name}.server.log"
    server = OwnedProcess(argv, cwd=a.code, env=env, log=log)
    result = {"name": a.name, "server_pid": server.pid}
    try:
        until = time.monotonic() + a.startup_timeout
        state = "waiting"
        while time.monotonic() < until:
            server.refresh()
            if server.process.poll() is not None:
                raise SystemExit(f"server exited during startup; see {log}")
            state = warmup_state(log.read_text(errors="replace"))
            if state != "waiting":
                break
            time.sleep(2)
        result["warmup"] = state
        if state == "waiting":
            raise SystemExit(f"warmup did not finish in {a.startup_timeout:.0f}s; see {log}")
        if state == "failed":
            # The server logs "Duplex warmup failed; continuing to serve" when the
            # cold start (first vision/TTS pass) outlasts the warmup socket. It is
            # still serving; the client's own warmup rows absorb the cold start
            # and are excluded from all statistics.
            server.refresh()
            if server.process.poll() is not None:
                raise SystemExit(f"warmup failed and the server exited; see {log}")
            result["warmup"] = "failed_server_still_serving"
            # The broken warmup session holds the session slot until its idle
            # timeout (60 s, entrypoints/duplex/warmup.py) is reaped.
            time.sleep(a.after_failed_warmup_s)
        snapshot(a.out / f"gpu-{a.name}-ready.txt")
        start_byte = log.stat().st_size
        if a.client == "onoff":
            script = HERE.parent / "rfc_bench" / "aura_judge_onoff.py"
            extra = ["--mode", a.mode, "--clips", *[f"{a.audio_dir / (c + '.wav')}={n}" for c, n in CLIPS_ONOFF]]
        else:
            script = HERE.parent / "rfc_bench" / "aura_concurrency_real.py"
            extra = ["--mode", a.mode, "--users", "4", "8", "--rounds", "3",
                     "--clips", *[str(a.audio_dir / (c + ".wav")) for c in ("a3", "a4", "a0", "a1")]]
        client_argv = [sys.executable, "-B", str(script), "--model", str(model),
                       "--url", f"ws://127.0.0.1:{a.port}/v1/realtime?duplex=1",
                       "--out", str(a.out / f"{a.name}.jsonl"), *extra]
        client = OwnedProcess(client_argv, cwd=a.code, env=env, log=a.out / f"{a.name}.client.log")
        try:
            client.process.wait(timeout=a.client_timeout)
        finally:
            client.stop()
        result["client_exit"] = client.process.returncode
        snapshot(a.out / f"gpu-{a.name}-after.txt")
        with log.open("rb") as src:
            src.seek(start_byte)
            window = src.read()
        (a.out / f"{a.name}.server-window.log").open("xb").write(window)
        client_rows = a.out / f"{a.name}.jsonl"
        result["client_output"] = client_rows.exists()
        if a.client == "onoff" and client_rows.exists():
            (a.out / f"server-{a.mode}.log").open("xb").write(window)
            shutil.copyfile(client_rows, a.out / f"onoff-{a.mode}.jsonl")
        hop_argv = [sys.executable, str(HERE / "hop_analysis.py"), str(a.out / f"{a.name}.server-window.log"),
                    "--mode", a.mode, "--json", str(a.out / f"{a.name}.hops.json")]
        if a.client == "onoff" and client_rows.exists():
            hop_argv += ["--client-jsonl", str(client_rows), "--hop-clips", "a3.wav", "a4.wav"]
        else:
            hop_argv += ["--skip", str(a.skip)]
        hops = subprocess.run(hop_argv, capture_output=True, text=True)
        (a.out / f"{a.name}.hops.stderr").open("x").write(hops.stderr)
        result["hop_analysis_exit"] = hops.returncode
    finally:
        server.stop()
        result["server_stopped"] = not server.living()
        (a.out / f"{a.name}.result.json").open("x").write(json.dumps(result, indent=1))
        print(json.dumps(result))
    # Non-zero unless the arm produced usable data: do not start the next arm.
    ok = result.get("client_exit") == 0 and result.get("client_output") and result.get("hop_analysis_exit") == 0
    if not ok:
        raise SystemExit(f"arm {a.name} did not complete cleanly: {json.dumps(result)}")


if __name__ == "__main__":
    main()
