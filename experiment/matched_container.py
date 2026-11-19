"""Container-only standard-library helpers; no image builders or cloud objects."""
import os
from pathlib import Path
import subprocess
import time

STATE = Path('/data')
BUNDLE = Path('/opt/agent-lightning-lab/experiment')
PYTHON = Path('/opt/agent-lightning/.venv/bin/python')

def environment(req):
    env = dict(os.environ, HF_HOME=str(STATE / "cache/huggingface"), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1")
    # PATH is changed only for subprocesses, never for the Modal worker's interpreter.
    # The upstream agent launches calculator and other console scripts from this venv.
    env["PATH"] = f"{PYTHON.parent}:{env.get('PATH', '')}"
    env.pop("AGL_WEAVE_PROJECT", None)
    if req["wandb_entity"]:
        env["WANDB_ENTITY"] = req["wandb_entity"]
    if req["weave"]:
        env["AGL_WEAVE_PROJECT"] = req["wandb_entity"] + "/cs329z-agent-lightning"
    return env


def run_command(command, log, env, timeout, follow_files=(), cancelled=lambda:False):
    """Close subprocesses/logs before Volume commit, including on command timeout."""
    command = [str(arg) for arg in command]
    print(f"Running {Path(command[0]).name}; log: {log}", flush=True)
    with Path(log).open("w") as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env,
                                stdin=subprocess.DEVNULL)
        tails = {}

        def stream():
            # Poll existing log files on this thread; no extra worker or tracing service.
            for path in (Path(log), *follow_files):
                if path not in tails and path.is_file():
                    tails[path] = path.open(errors="replace")
                if path in tails:
                    chunk = tails[path].read(262144)
                    if chunk:
                        print(f"[{path.name}] {chunk}", end="" if chunk.endswith("\n") else "\n", flush=True)

        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if cancelled():
                    raise RuntimeError('Supervisor lease lost or owner stop requested')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                stream()
                try:
                    proc.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            code = proc.returncode
        except BaseException:
            # launch.py handles SIGTERM and stops its own gateway/controller/trainer groups.
            proc.terminate()
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        finally:
            stream()
            for handle in tails.values():
                handle.close()
    if code:
        raise RuntimeError(f"Command exited {code}; inspect {log}")

