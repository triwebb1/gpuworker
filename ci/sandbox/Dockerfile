# A small image for ci/sandbox/run.py: the echo module and the sandbox probe on a plain
# Python base, with no CUDA. The broker launches it with the same flags as the shared
# sandbox image (images/Dockerfile.sandbox), which differs only in what it installs.
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 worker \
    && mkdir -p /opt/venvs /job/bundle /job/out /job/weights \
    && chown -R worker:worker /opt/venvs /job

COPY modules/echo/ /opt/modules/echo/
RUN bash /opt/modules/echo/provision.sh
COPY ci/sandbox/probe/src/probe.py /opt/probe/probe.py

USER worker
WORKDIR /job
