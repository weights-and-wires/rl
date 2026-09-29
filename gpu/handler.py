"""RunPod serverless handler for the exercises that want a GPU.

The worker runs the same harness the app runs locally, over the same exercise
directories, so a lesson behaves identically on either backend. The only
difference the reader sees is the badge above the console, and that `torch.cuda`
is available.

Build and push the image from `gpu/Dockerfile`, then point a RunPod serverless
endpoint at it and set `RUNPOD_ENDPOINT_ID` in the app's `.env`.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import runpod

REPO = Path(os.environ.get("ML_REPO", "/repo"))
HARNESS = REPO / "backend" / "app" / "runner" / "harness.py"
EXERCISES = REPO / "content" / "exercises"

MAX_TIMEOUT = int(os.environ.get("ML_MAX_TIMEOUT", "900"))


def _gpu_banner() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
            return {"type": "status", "text": f"{name}, {total:.0f} GiB, torch {torch.__version__}"}
        return {"type": "status", "text": "No GPU is visible to this worker"}
    except Exception as exc:  # noqa: BLE001 - a banner must never fail the job
        return {"type": "status", "text": f"Could not query the GPU: {exc}"}


def run_exercise(
    exercise_id: str, mode: str, code: str, timeout: int
) -> Iterator[dict[str, Any]]:
    exercise = EXERCISES / exercise_id
    if not exercise.is_dir():
        yield {"type": "error", "message": f"This worker has no exercise {exercise_id!r}"}
        yield {"type": "done", "ok": False, "seconds": 0.0}
        return

    yield _gpu_banner()

    with tempfile.TemporaryDirectory(prefix="rl-gpu-") as scratch:
        work = Path(scratch)
        if mode == "check":
            tests = exercise / "tests.py"
            if not tests.is_file():
                yield {"type": "error", "message": f"{exercise_id} has no checks"}
                yield {"type": "done", "ok": False, "seconds": 0.0}
                return
            shutil.copy(tests, work / "tests.py")

        submission = work / "submission.py"
        submission.write_text(code)

        started = time.perf_counter()
        process = subprocess.Popen(
            [sys.executable, "-I", str(HARNESS), mode, str(work), str(submission)],
            cwd=str(work),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env={
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "HOME": scratch,
                "TMPDIR": scratch,
                "HF_HOME": os.environ.get("HF_HOME", "/models"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "TOKENIZERS_PARALLELISM": "false",
            },
        )

        saw_done = False
        deadline = time.monotonic() + timeout
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if time.monotonic() > deadline:
                    raise TimeoutError
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    yield {"type": "stderr", "text": line}
                    continue
                if event.get("type") == "done":
                    saw_done = True
                yield event
        except TimeoutError:
            _kill(process)
            yield {
                "type": "error",
                "message": f"The run passed its {timeout} second limit and was stopped.",
            }
            yield {
                "type": "done",
                "ok": False,
                "seconds": time.perf_counter() - started,
                "timeout": True,
            }
            return
        finally:
            _kill(process)

        if not saw_done:
            stderr = process.stderr.read() if process.stderr else ""
            yield {
                "type": "error",
                "message": "The worker stopped before finishing.",
                "detail": stderr[-4000:] or None,
            }
            yield {
                "type": "done",
                "ok": False,
                "seconds": time.perf_counter() - started,
            }


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def handler(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    payload = job.get("input", {}) or {}
    exercise_id = payload.get("exercise_id")
    mode = payload.get("mode", "check")
    code = payload.get("code", "")
    timeout = min(int(payload.get("timeout", 300)), MAX_TIMEOUT)

    if not exercise_id:
        yield {"type": "error", "message": "The job carried no exercise_id"}
        yield {"type": "done", "ok": False}
        return
    if mode not in ("run", "check"):
        yield {"type": "error", "message": f"Unknown mode {mode!r}"}
        yield {"type": "done", "ok": False}
        return

    yield from run_exercise(exercise_id, mode, code, timeout)


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
