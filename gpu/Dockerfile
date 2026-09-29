# The RunPod serverless worker for the exercises marked `gpu: true`.
#
# It carries the same harness and the same exercise directories the app runs
# locally, so a lesson behaves identically on either backend.
#
# Build from the repository root, so the context includes `content/` and
# `backend/`:
#
#   docker build -f gpu/Dockerfile -t ghcr.io/weights-and-wires/rl-gpu-worker:latest .
#
# The publish workflow does this on every push that touches the worker.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false \
    ML_REPO=/repo

RUN pip install --no-cache-dir \
        numpy==2.3.3 \
        scipy==1.16.2 \
        runpod==1.7.7

WORKDIR /repo

# Only the two modules the harness needs, not the whole web application.
COPY backend/app/runner/harness.py /repo/backend/app/runner/harness.py
COPY backend/app/runner/rlcheck.py /repo/backend/app/runner/rlcheck.py
COPY backend/app/runner/rlenvs.py /repo/backend/app/runner/rlenvs.py
COPY content/exercises /repo/content/exercises
COPY gpu/handler.py /repo/gpu/handler.py

CMD ["python", "-u", "/repo/gpu/handler.py"]
