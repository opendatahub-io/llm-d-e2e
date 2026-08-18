FROM python:3.11-slim

ARG MANIFEST_REF=main
ARG MANIFEST_REPO=https://github.com/opendatahub-io/llm-d-conformance-manifests.git

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install kubectl /usr/local/bin/kubectl && rm kubectl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .
RUN rm -rf .venv && uv sync --frozen

RUN git clone --depth 1 --branch ${MANIFEST_REF} ${MANIFEST_REPO} /tmp/manifests && \
    mkdir -p deploy/manifests && \
    cp /tmp/manifests/*.yaml deploy/manifests/ && \
    printf 'branch: %s\nrepo: %s\n' "${MANIFEST_REF}" "${MANIFEST_REPO}" > deploy/manifests/.manifest-ref && \
    rm -rf /tmp/manifests

ENTRYPOINT ["uv", "run", "llm-d-e2e"]
