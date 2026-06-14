# The thin, per-commit layer: drop the freshly-built wheel + tests onto the cached model
# image. Rebuilds in seconds because the heavy layers below are cached (testing-strategy §4.2).
# Used by compose.integration.yml and compose.airgap.yml to run the test suite in-container.
ARG BASE=indx-core:latest
FROM ${BASE}

# Extras installed alongside the wheel; compose files override per suite
# (integration adds the real-backend clients, airgap keeps the default).
ARG EXTRAS=dev

WORKDIR /app
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir "/tmp/$(ls /tmp | grep '.whl')[${EXTRAS}]"

COPY tests/ /app/tests/
COPY pyproject.toml /app/pyproject.toml

ENTRYPOINT []
CMD ["python", "-m", "pytest", "-q"]
