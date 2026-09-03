# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end conformance test suite for [llm-d](https://github.com/llm-d) / KServe `LLMInferenceService` deployments on Kubernetes. Python + pytest framework that deploys LLMInferenceService resources and validates them through ordered phases (CRD check, deploy, service/gateway/pod readiness, health, inference, metrics, cleanup).

## Prerequisites

- Python 3.11+, [uv](https://docs.astral.sh/uv/) package manager
- `kubectl` configured with cluster access (for conformance tests, not unit tests)
- Cluster with `LLMInferenceService` CRD installed (RHAI or KServe)
- Manifests from [llm-d-conformance-manifests](https://github.com/opendatahub-io/llm-d-conformance-manifests) (cloned via `--setup`)

## Common Commands

```bash
uv sync                                              # install dependencies
uv run llm-d-e2e --setup main                        # clone test manifests (latest)
uv run llm-d-e2e --setup 3.5-GA                      # clone manifests (specific branch)

uv run pytest tests/test_smoke.py -v                  # unit tests (no cluster needed)
uv run ruff check src/ tests/                         # lint
uv run ruff format src/ tests/                        # format

uv run llm-d-e2e -t single-gpu-smoke                  # run single conformance test case
uv run llm-d-e2e -t single-gpu,cache-aware            # run multiple test cases
uv run llm-d-e2e -t single-gpu --mock                 # simulate vLLM (no GPU)
uv run llm-d-e2e -t single-gpu --mode discover --endpoint http://svc:8000  # validate existing deployment
uv run llm-d-e2e -t single-gpu --mode cache           # pre-cache model into PVC then exit
uv run llm-d-e2e -p configs/profiles/smoke.yaml       # run a profile
uv run llm-d-e2e -p configs/profiles/3.5.yaml         # run version-specific conformance profile
uv run llm-d-e2e -t single-gpu --nocleanup            # keep resources after test
uv run llm-d-e2e -t single-gpu --html report.html     # generate HTML report
uv run llm-d-e2e -t single-gpu -x                     # stop on first failure
uv run llm-d-e2e --list-testcases                     # list available test cases
uv run llm-d-e2e --list-profiles                      # list available profiles

# Run a single conformance phase (by method name prefix)
uv run pytest tests/test_conformance.py -k "test_09a_inference" --testcase single-gpu
```

Makefile targets mirror CLI: `make test TESTCASE=single-gpu`, `make unittest`, `make lint`, `make format`, `make setup`.

## Architecture

### CLI → pytest delegation

`cli.py:main()` parses user flags and translates them to pytest options, then runs `pytest tests/test_conformance.py` as a subprocess. **Every CLI flag in `cli.py` must have a matching `conftest.py:pytest_addoption()` entry** — when adding a new flag, update both files and the `flag_map` dict in `cli.py:main()`. Boolean flags (`--nocleanup`, `--disable-auth`) are handled separately after `flag_map` iteration since they use `store_true` rather than values.

### Test case parametrization

`conftest.py:pytest_generate_tests()` resolves which test cases to run (from `--testcase` names or `--profile` YAML) and parametrizes the `tc` fixture. Each `tc` is a `TestCase` dataclass loaded from `configs/testcases/*.yaml`. `TestConformance` methods run once per test case.

### Run modes

The `--mode` flag controls which phases execute:
- **`deploy`** (default) — full lifecycle: deploy → validate → cleanup.
- **`discover`** — skip deploy/cleanup, validate an existing deployment (requires `--endpoint` or auto-detected). Phases call `_require_deployed()` which returns early in discover mode.
- **`cache`** — run only the model download phase (create PVC + download Job), then exit.

### Ordered conformance phases

`test_conformance.py:TestConformance` uses numeric method name prefixes (`test_01_` through `test_99_`) for phase ordering:

| Phase | Method | What it validates |
|-------|--------|-------------------|
| 01 | `test_01_prereq` | CRD exists, manifest present |
| 02 | `test_02_deploy` | Apply LLMInferenceService manifest |
| 03 | `test_03_service` | Service creation |
| 04 | `test_04_gateway` | Gateway programmed with address |
| 05 | `test_05_pods` | Pods running without crashes |
| 06 | `test_06_ready` | LLMInferenceService Ready=True |
| 07 | `test_07_health` | GET /health (direct pod, bypasses EPP) |
| 08 | `test_08_models` | GET /v1/models (direct pod, + LoRA adapters if configured) |
| 09a | `test_09a_inference` | Chat completions + completions (+ LoRA adapter inference) |
| 09b | `test_09b_messages_responses` | Anthropic /v1/messages + OpenAI /v1/responses |
| 10 | `test_10_metrics_vllm` | Basic vLLM request success metrics |
| 11 | `test_11_metrics_cache` | Prefix KV cache hit metrics |
| 12 | `test_12_metrics_pd` | P/D token distribution + NIXL transfer metrics |
| 13 | `test_13_metrics_scheduler` | EPP/scheduler processed request metrics |
| 14 | `test_14_metrics_flow_control` | Flow control dispatch activity metrics |
| 15 | `test_15_metrics_lora` | LoRA adapter state metrics (`vllm:lora_requests_info`) |
| 20 | `test_20_benchmark` | GuideLLM benchmark with performance thresholds |
| 21 | `test_21_metrics_post_benchmark` | P/D metrics after benchmark load |
| 99 | `test_99_cleanup` | Delete LLMInferenceService |

Phases skip themselves based on `tc` config flags or `--mode discover`.

### Skip propagation

Two helpers control cascading skips across phases:
- `_require_manifest(tc)` — skips if the manifest file doesn't exist for the current branch.
- `_require_deployed(deployer, tc, test_mode)` — skips if deploy failed or was skipped. In discover mode, this check is bypassed.

### Fast-fail behaviors

- **CrashLoopBackOff**: `wait_for_pods()` and `wait_for_ready()` poll every 15s. After 3 consecutive detections (~45s), the deploy fails immediately instead of waiting the full timeout (30+ minutes).
- **Persistent controller error**: `wait_for_ready()` polls the `Ready` condition's `reason` and `message`. If the same reason+message pair persists across 3 consecutive polls, it raises immediately. Changing reason/message indicates controller progress and polling continues.
- **Operator image pull detection**: Both `test_01_prereq` and `wait_for_ready()` call `_check_operator_image_issues()`, scanning pods in `OPERATOR_NAMESPACES` for `ImagePullBackOff`/`ErrImagePull` states.

### Webhook and CRD transient error retry

`deployer.py:_apply_with_webhook_retry()` wraps `kubectl apply` with retry logic for transient post-upgrade errors (webhook not ready, CRD not registered). Non-transient errors (bad fields, RBAC) are re-raised immediately.

### Endpoint routing (gateway vs pod)

Health and models endpoints return 503 through the Gateway API + EPP because EPP only handles inference. The suite uses two separate port-forwards:
- **Gateway** (`client` fixture): `svc/inference-gateway-istio:80` (HTTP) in `redhat-ods-applications` — for `/v1/chat/completions`, `/v1/messages`, `/v1/responses`.
- **Pod** (`pod_client` fixture): `workload-pod:8000` in the test namespace — for `/health` and `/v1/models`. HTTPS (self-signed).

The gateway service name and namespace are RHOAI-specific hardcoded values in `deployer.py:_ensure_port_forward()`.

### Fixture scoping

- **Session-scoped**: `deployer`, `report`, `no_cleanup`, `test_mode`, `mock_mode`, `guidellm_image`
- **Class-scoped**: `endpoint`, `client`, `pod_endpoint`, `pod_client`, `scraper`

### Source modules (`src/conformance/`)

- **config.py** — Dataclass config types and YAML loaders. YAML keys are camelCase, Python fields are snake_case; `_build()` handles recursive conversion.
- **deployer.py** — `Deployer`: manages LLMInferenceService lifecycle via `kubectl` subprocess calls. Handles deploy, wait-for-ready, port-forwarding, manifest patching (mock image, pull secrets, auth disable, LoRA spec injection, node selectors, env overrides), EPP metrics RBAC, pull secret propagation, namespace labeling for gateway access, and cleanup.
- **client.py** — `LLMClient`: HTTP client (httpx) for `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`.
- **metrics.py** — `Scraper`: scrapes Prometheus metrics from pods via `kubectl exec` (python3/wget), falling back to port-forward + httpx for minimal containers. Per-topology validators: `validate_vllm_basic`, `validate_cache_aware`, `validate_pd`, `validate_scheduler`, `validate_flow_control`, `validate_lora`. `parse_prometheus()` parses text exposition format.
- **model.py** — `ModelDownloader`: creates PVCs and download Jobs for pre-caching models from HuggingFace.
- **report.py** — JSON report generation with pass/fail/skip summary.
- **benchmark.py** — `run_benchmark()`: creates a GuideLLM K8s Job, parses JSON results delimited by `---GUIDELLM_JSON_START---` marker.

### Metrics validation topology

| Validator | Scrape target | Phase | Topology | Gate flag |
|-----------|--------------|-------|----------|-----------|
| `validate_vllm_basic` | workload pods | test_10 | All | `checkVLLM` |
| `validate_cache_aware` | workload + EPP | test_11 | Prefix KV cache | `checkPrefixCache` |
| `validate_pd` | workload + prefill | test_12 | P/D disaggregation | `checkPD` |
| `validate_scheduler` | EPP pods | test_13 | Scheduler/EPP | `checkScheduler` |
| `validate_flow_control` | EPP pods | test_14 | Flow control | `checkFlowControl` |
| `validate_lora` | workload pods | test_15 | LoRA adapters | `checkLora` |

`MetricsCheck.check_nixl` exists in the dataclass but has no validator method yet.

EPP pod discovery tries multiple label patterns (`EPP_LABELS` list in `metrics.py`) because the component label varies across llm-d versions.

### EPP metrics auth

The EPP's `--metrics-endpoint-auth=true` flag (default in RHOAI 3.5+) requires bearer token auth to scrape `/metrics` on port 9090. `Deployer.ensure_metrics_rbac()` creates a `ClusterRoleBinding` granting the EPP's service account access to `kserve-metrics-reader-cluster-role`. The scraper generates a token via `kubectl create token`.

## Adding a New CLI Flag

1. Add `parser.add_argument()` in `cli.py:main()`.
2. Add matching `parser.addoption()` in `conftest.py:pytest_addoption()`.
3. Add the mapping in `cli.py:flag_map` dict (or handle boolean flags separately after the `flag_map` loop).
4. Access via `request.config.getoption("--flag-name")` in fixtures or test methods.

## Adding a New Test Case

Use `scripts/new-testcase.sh <name>` to generate stubs, then customize:

1. Create `configs/testcases/<name>.yaml` using camelCase keys matching the `TestCase` dataclass hierarchy in `config.py`.
2. Add the corresponding manifest to the manifest repo (or `deploy/manifests/` for local testing). Set `deployment.manifestPath` to the filename.
3. Enable the appropriate `metricsCheck` flags based on the deployment topology.
4. Add the test case name to relevant profiles in `configs/profiles/*.yaml`.

## Adding a New Config Field

1. Add the dataclass field in `config.py` (snake_case).
2. If it's a new nested type, add a `_build()` branch for it (matching on the type name string in hints).
3. Use camelCase for the key in YAML files.
4. Duration fields (`timeout`, `ready_timeout`, `retry_interval`) are auto-parsed from strings like `"15m"`, `"2h"`, `"300s"`.

## Adding a New Conformance Phase

1. Add a method `test_NN_<name>` to `TestConformance` in `test_conformance.py`. Pick a number between existing phases.
2. Use `pytest.skip()` for conditions where the phase doesn't apply.
3. Phases receive fixtures via parameter names: `deployer`, `tc`, `client`, `endpoint`, `scraper`, `test_mode`, `no_cleanup`, `request`.

## Adding a New Metrics Validator

1. Define metric constants at the top of `metrics.py`.
2. Add a `validate_<topology>(results) -> list[CheckResult]` function.
3. Add a `check_<flag>: bool` field to `MetricsCheck` in `config.py`.
4. Wire it up in `test_conformance.py` as a new `test_NN_metrics_<name>` method gated by the new flag.
5. Import the validator in `test_conformance.py`.

## Key Design Decisions

- All cluster interaction goes through `kubectl` subprocess calls (no Python K8s client library).
- Test cases are data-driven via YAML configs, not hardcoded in test files.
- The `--mock` flag swaps the vLLM container with llm-d-inference-sim, injects simulator args, and strips GPU resource requests, enabling full e2e flow without GPUs.
- camelCase in YAML, snake_case in Python — `_snake()` and `_build()` in `config.py` bridge the two.
- Health/models go directly to pods; inference goes through the gateway — the EPP only routes inference requests.
- Metrics scraping tries `kubectl exec` first (python3, wget), falls back to port-forward + httpx for minimal container images.
- Global pytest timeout is 21600s (6 hours) to accommodate slow model downloads and pod startup.

### Gateway namespace access (xKS / rhoai-3.5+)

As of [odh-gitops PR#156](https://github.com/opendatahub-io/odh-gitops/pull/156), the inference gateway uses a label selector instead of `allowedRoutes.namespaces.from: All`. The helm chart install must include:

```bash
--set-json 'components.kserve.gateway.allowedRoutes.namespaces={"from":"Selector","selector":{"matchLabels":{"inference-gateway-access":"true"}}}'
```

`ensure_namespace()` labels the test namespace with `inference-gateway-access=true` (idempotent, applied on every run) so HTTPRoutes from `llm-conformance-test` are accepted by the gateway. No cluster-admin gateway patching is required or performed.

## Config files

- **configs/testcases/*.yaml** — Each file maps to one `TestCase` dataclass. Contains model info (including LoRA adapters), deployment spec, validation criteria, and metrics check flags.
- **configs/profiles/*.yaml** — Named groups of test case names. Includes version-specific profiles (`3.4.yaml`, `3.5.yaml`, `3.5-gpu.yaml`) and topology profiles (`smoke`, `pd`, `cache-aware`, `flow-control`, `lora`).
- **deploy/manifests/*.yaml** — LLMInferenceService manifests, cloned from the manifest repo via `--setup`. Gitignored.
- **deploy/manifests/.manifest-ref** — Tracks the active manifest branch, repo URL, commit SHA, and clone timestamp.

## CI

GitHub Actions (`.github/workflows/ci.yaml`) runs on push/PR to `main`:
1. **lint-and-format** — `ruff check` + `ruff format --check`
2. **smoke-tests** — clones manifests, runs `pytest tests/test_smoke.py`

No cluster integration tests run in CI.

## Code Style

- Ruff for linting and formatting, line length 120, target Python 3.11.
- Uses `from __future__ import annotations` throughout.
- Config types are plain dataclasses (no Pydantic).
