# llm-d-e2e

End-to-end conformance tests for [llm-d](https://github.com/llm-d) / KServe `LLMInferenceService` deployments on Kubernetes.

Python + pytest rewrite of the [Go/Ginkgo conformance framework](https://github.com/aneeshkp/llm-d-conformance-test). See that project's [architecture docs](https://github.com/aneeshkp/llm-d-conformance-test/blob/main/docs/architecture.md) for detailed diagrams, test topologies, and metrics coverage.

**Guides:** [Adding a Test Case](docs/adding-a-test-case.md)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- `kubectl` configured with cluster access
- Cluster with `LLMInferenceService` CRD installed (RHAI or KServe)

Install uv if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/aneeshkp/llm-d-e2e.git
cd llm-d-e2e

# 2. Install dependencies
uv sync

# 3. Clone test manifests — interactive (lists branches)
uv run llm-d-e2e --setup

# 3. Or clone a specific branch directly
uv run llm-d-e2e --setup main          # latest
uv run llm-d-e2e --setup 3.5-GA        # 3.5 GA manifests
uv run llm-d-e2e --setup 3.4-stable    # 3.4 stable manifests

# 4. (Optional) Set up a shortcut
alias e2e='uv run llm-d-e2e'
```

You can also use `make setup` which provides the same interactive branch selection:
```bash
make setup                      # interactive — lists branches, pick one
make setup BRANCH=3.5-GA        # direct — specific branch
```

## Quick Start

```bash
# List available tests and profiles
e2e --list-testcases
e2e --list-profiles

# Run smoke test
e2e -t single-gpu-smoke

# Run all conformance tests
e2e -p configs/profiles/all.yaml
```

## Usage

```bash
# Single test case
e2e -t single-gpu

# Multiple test cases
e2e -t single-gpu,cache-aware

# Setup manifests and run tests in one command
e2e --setup 3.5-GA -t single-gpu,cache-aware --mock -v

# Keep resources after test (for debugging)
e2e -t single-gpu --nocleanup

# Simulate vLLM with llm-d-inference-sim (no GPU needed)
e2e -t single-gpu --mock

# Run test require GPU (requiresGpu) — skipped on CPU only clsuter by default; opt in on a GPU cluster
e2e -t kv-offloading-cpu --need-gpu

# Verbose output, stop on first failure
e2e -t single-gpu -v -x

# Generate HTML report
e2e -t single-gpu --html report.html

# Target a specific cluster
e2e -t single-gpu --kubeconfig ~/.kube/config --namespace my-test-ns

# Validate an existing deployment (no deploy/cleanup)
e2e -t single-gpu --mode discover --endpoint http://my-service:8000
```

## Testing an Existing Deployment

If you already have an LLMInferenceService running and just want to validate it (health, inference, metrics) without deploying or cleaning up:

```bash
# Get the service URL
kubectl get llminferenceservice -n my-namespace
# NAME         URL                                    READY
# my-model     http://gateway.example.com/ns/model    True

# Run validation against it
e2e -t single-gpu --mode discover --namespace my-namespace

# Or specify the endpoint directly
e2e -t single-gpu --mode discover --endpoint http://gateway.example.com/ns/model
```

This skips the deploy and cleanup phases — only runs health, models, inference, and metrics checks.

## Container Image

The test suite is available as a container image at `quay.io/aneeshkp/llm-d-e2e`.

### Build

```bash
# Default (main manifests baked in)
docker build -t llm-d-e2e .

# Specific manifest branch baked in
docker build --build-arg MANIFEST_REF=3.5-GA -t llm-d-e2e:3.5 .
```

### Run tests

```bash
# Run with baked-in manifests
docker run --rm \
  -v ~/.kube:/root/.kube:z \
  quay.io/aneeshkp/llm-d-e2e \
  -t single-gpu-smoke --mock -v

# Use a non-default kubeconfig
docker run --rm \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  quay.io/aneeshkp/llm-d-e2e \
  -t single-gpu-smoke,single-gpu,cache-aware --mock -v

# Setup different manifests and run tests in one command
docker run --rm \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  quay.io/aneeshkp/llm-d-e2e \
  --setup 3.5-GA \
  -t cache-aware,flow-control,flow-control-tokens --mock -v

# Generate HTML report (mount reports directory)
docker run --rm \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  -v $(pwd)/reports:/app/reports:z \
  quay.io/aneeshkp/llm-d-e2e \
  -t single-gpu-smoke --mock --html reports/mock-ci.html -v
```

### Switch manifest branch

```bash
# List available branches and pick one (requires -it for interactive prompt)
docker run --rm -it quay.io/aneeshkp/llm-d-e2e --setup

# Direct — no prompt
docker run --rm quay.io/aneeshkp/llm-d-e2e --setup 3.5-GA
docker run --rm quay.io/aneeshkp/llm-d-e2e --setup 3.4-stable
```

### Interactive mode

```bash
# Interactive shell — switch branches, run multiple tests
docker run --rm -it \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  -v $(pwd)/reports:/app/reports:z \
  --entrypoint bash quay.io/aneeshkp/llm-d-e2e

# Inside the container:
uv run llm-d-e2e --setup              # interactive branch selection
uv run llm-d-e2e --setup 3.5-GA       # or direct
uv run llm-d-e2e -t single-gpu --mock -v
uv run llm-d-e2e -t cache-aware --mock --html reports/cache.html -v
```

### Utility commands

```bash
# List test cases
docker run --rm quay.io/aneeshkp/llm-d-e2e --list-testcases

# List profiles
docker run --rm quay.io/aneeshkp/llm-d-e2e --list-profiles

# Setup only (clone manifests, show test case mapping)
docker run --rm quay.io/aneeshkp/llm-d-e2e --setup 3.5-GA
```

### How manifests work in the container

- **Build time**: `main` branch manifests are baked into the image (configurable via `--build-arg MANIFEST_REF=`)
- **Runtime `--setup <branch>`**: clones the specified branch from GitHub, replaces baked-in manifests
- **Runtime `--setup`** (interactive, requires `-it`): lists all branches from GitHub, prompts to pick
- Network access to GitHub is required at runtime for `--setup`; without it, the baked-in manifests are used

## Test Cases

| Name | GPUs | What it tests |
|------|------|---------------|
| single-gpu-smoke | 1 | Fast baseline (no metrics) |
| single-gpu | 1 | Scheduler + metrics |
| single-gpu-no-scheduler | 3 | K8s native round-robin |
| cache-aware | 2 | Prefix KV cache routing |
| pd | 3 | Prefill/Decode disaggregation |
| pd-cache-aware | 3 | P/D + prefix cache |
| moe | 8 | MoE, RDMA, expert parallelism |
| multi-pool | 2 | Multiple InferencePools |
| flow-control | 1 | Flow control with utilization-based saturation detector |
| flow-control-tokens | 1 | Flow control with token-based concurrency detector |
| pd-performance | 16 | P/D benchmark with GuideLLM (4 prefill + 2 decode, NIXL, RDMA) |

## Test Phases

Each test case runs through ordered phases:

1. **Prereq** — CRD exists
2. **Deploy** — apply LLMInferenceService manifest
3. **Service** — wait for Service
4. **Gateway** — wait for Gateway programmed
5. **Pods** — wait for pods Running
6. **Ready** — wait for Ready=True
7. **Health** — GET /health
8. **Models** — GET /v1/models
9. **Inference** — POST /v1/chat/completions
10. **Metrics** — scrape and validate Prometheus metrics
11. **Cleanup** — delete resources

If deploy fails (e.g., manifest missing for the selected branch), all subsequent phases for that test case are automatically skipped. CrashLoopBackOff is detected within ~45 seconds instead of waiting the full timeout.

## P/D Performance Benchmark

The `pd-performance` test case runs a [GuideLLM](https://github.com/vllm-project/guidellm) benchmark against a P/D disaggregated deployment (gpt-oss-120b) and validates performance thresholds and NIXL transfer metrics.

### Requirements

- 16 GPUs: 2 decode nodes (4 GPU each, TP=4) + 4 prefill nodes (2 GPU each, TP=2)
- RDMA/InfiniBand networking (`rdma/ib` resource)
- Pre-created PVC `model-cache-pvc` with the model downloaded (500Gi)

### Pre-cache the model

```bash
e2e -t pd-performance --mode cache
```

This creates the `model-cache-pvc` PVC and downloads `openai/gpt-oss-120b` from HuggingFace.

### Run the benchmark

```bash
# Full run: deploy → conformance phases → benchmark → post-benchmark metrics → cleanup
e2e -t pd-performance

# With node placement control
e2e -t pd-performance \
  --decode-node-selector gpu-type=a100-80g \
  --prefill-node-selector gpu-type=a100-40g

# Override the GuideLLM image
e2e -t pd-performance --guidellm-image ghcr.io/vllm-project/guidellm:v0.7.0

# Keep resources for debugging
e2e -t pd-performance --nocleanup
```

### What the benchmark validates

| Check | Threshold |
|-------|-----------|
| Output tokens/s | >= 8000 |
| TTFT median | <= 2000 ms |
| TTFT p95 | <= 5000 ms |
| ITL median | <= 50 ms |
| ITL p95 | <= 100 ms |
| Failed request ratio | <= 5% |
| NIXL transfers | > 0 (KV transfer happened) |
| NIXL failed transfers | == 0 |
| Decode KV transfer > local compute | P/D topology is working |

Thresholds are configurable in `configs/testcases/pd-performance.yaml` under `validation.benchmark.thresholds`.

### Benchmark phases

The benchmark adds three phases after the standard conformance checks:

- **test_12** — Pre-benchmark P/D metrics (raw metric dump for baseline)
- **test_20** — GuideLLM benchmark (warmup + main run + threshold assertions)
- **test_21** — Post-benchmark P/D metrics (validates NIXL transfers after load)

## Development

```bash
# Run unit tests (no cluster needed)
uv run pytest tests/test_smoke.py -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

CI runs lint, format, and smoke tests automatically on every PR via GitHub Actions.
