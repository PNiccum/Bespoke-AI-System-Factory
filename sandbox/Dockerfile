# The sandbox image: Python, pytest and the small pinned set in requirements.txt, nothing else. Generated code
# runs in it with no network, no keys, a read-only root, and caps on memory, CPU and processes.
#   docker build -t aspire-sandbox sandbox/          (podman build works the same)
#   apptainer build aspire-sandbox.sif docker-daemon://aspire-sandbox:latest   (on ARC; or from a registry copy)
ARG BASE=python:3.11-slim
FROM ${BASE}
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --create-home --uid 1000 runner
USER runner
WORKDIR /repo
