# The sandbox image: Python and pytest, nothing else. Generated code runs in it with no network, no keys, a
# read-only root, and caps on memory, CPU and processes.
#   docker build -t simple-sandbox sandbox/          (podman build works the same)
#   apptainer build simple-sandbox.sif docker-daemon://simple-sandbox:latest   (on ARC; or from a registry copy)
ARG BASE=python:3.11-slim
FROM ${BASE}
RUN pip install --no-cache-dir "pytest==8.3.5" \
    && useradd --create-home --uid 1000 runner
USER runner
WORKDIR /repo
