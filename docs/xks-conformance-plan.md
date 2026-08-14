# llm-d xKS Conformance Tests — Integration Plan for opendatahub-tests

## Goal

Add llm-d / LLMInferenceService conformance tests to `opendatahub-io/opendatahub-tests` that work on both **RHOAI (OCP with GPUs)** and **RHAII on xKS (AKS/EKS with or without GPUs)**. Tests should run with real vLLM on GPU clusters and with llm-d-inference-sim (simulator) on CPU-only clusters.

## Current State

### Existing tests in opendatahub-tests (`tests/model_serving/model_server/llmd/`)

| Test | What it covers | Platform |
|------|---------------|----------|
| `test_llmd_smoke.py` | TinyLlama OCI deploy + chat completions | OCP (GPU) |
| `test_llmd_connection_cpu.py` | CPU-only deploy | OCP |
| `test_llmd_connection_gpu.py` | GPU deploy | OCP |
| `test_llmd_no_scheduler.py` | No scheduler (K8s round-robin) | OCP |
| `test_llmd_singlenode_estimated_prefix_cache.py` | Estimated prefix cache | OCP (GPU) |
| `test_llmd_singlenode_precise_prefix_cache.py` | Precise prefix cache | OCP (GPU) |
| `test_llmd_singlenode_prefill_decode.py` | P/D disaggregation | OCP (GPU) |
| `test_llmd_auth.py` | Authentication | OCP |
| `test_llmd_kueue_integration.py` | Kueue scheduling | OCP |

These use `ocp_resources` Python K8s client, OCP Routes, DSC resource, and RHOAI-specific fixtures.

### Our standalone suite (`opendatahub-io/llm-d-e2e`)

- Python + pytest, kubectl subprocess, data-driven YAML configs
- Platform-agnostic (OCP, AKS, EKS, GKE)
- `--mock` flag for simulator mode (no GPU)
- Ordered phases: CRD → deploy → service → gateway → pods → ready → health → models → inference → metrics → cleanup

## Design

### Test Location

```
tests/model_serving/model_server/llmd/
├── conftest.py                                 # existing — add xKS fixtures
├── constants.py                                # existing — add xKS constants
├── utils.py                                    # existing — add xKS helpers
├── llmd_configs/
│   ├── config_base.py                          # existing
│   ├── config_xks_smoke.py                     # NEW: xKS smoke config
│   ├── config_xks_cache_aware.py               # NEW: xKS cache-aware config
│   └── config_xks_pd.py                        # NEW: xKS P/D config
│
│   # --- xKS conformance tests ---
├── test_llmd_xks_smoke.py                      # NEW: smoke (simulator or GPU)
├── test_llmd_xks_single_gpu.py                 # NEW: single GPU with scheduler
├── test_llmd_xks_cache_aware.py                # NEW: prefix cache routing
├── test_llmd_xks_prefill_decode.py             # NEW: P/D disaggregation
├── test_llmd_xks_metrics.py                    # NEW: Prometheus metrics validation
├── test_llmd_xks_mock.py                       # NEW: simulator-specific tests
│
│   # --- shared utilities ---
├── xks_utils.py                                # NEW: xKS-specific helpers
└── xks_fixtures.py                             # NEW: xKS fixtures (gateway, simulator)
```

### Platform Modes

Tests run in three modes controlled by pytest markers and CLI flags:

| Mode | Platform | GPU | Model | Use case |
|------|----------|-----|-------|----------|
| `gpu` | OCP / xKS with GPU | Real | Real vLLM | Full validation |
| `mock` | xKS (CPU-only) | None | llm-d-inference-sim | CI, no-GPU validation |
| `discover` | Any | N/A | Pre-deployed | Validate existing deployment |

```python
# pytest markers
pytestmark = [
    pytest.mark.llmd,
    pytest.mark.xks,
]

# Conditional skip
@pytest.mark.skipif(not has_gpus(), reason="No GPUs available")
def test_single_gpu_real():
    ...

@pytest.mark.mock
def test_single_gpu_mock():
    ...
```

### Key Fixtures

```python
# xks_fixtures.py

@pytest.fixture(scope="session")
def xks_platform(admin_client):
    """Detect platform: ocp, aks, eks, gke."""
    ...

@pytest.fixture(scope="session")
def gpu_available(admin_client):
    """Detect if GPU nodes exist on the cluster."""
    ...

@pytest.fixture(scope="session")
def simulator_image():
    """Return simulator image: ghcr.io/llm-d/llm-d-inference-sim:latest."""
    ...

@pytest.fixture(scope="class")
def llmisvc_with_simulator(admin_client, simulator_image, unprivileged_model_namespace):
    """Deploy LLMInferenceService with simulator image patched in."""
    ...

@pytest.fixture(scope="class")
def llmisvc_with_gpu(admin_client, unprivileged_model_namespace):
    """Deploy LLMInferenceService with real vLLM (requires GPU)."""
    ...

@pytest.fixture(scope="session")
def xks_gateway(admin_client):
    """Ensure gateway allows routes from all namespaces (xKS only)."""
    ...

@pytest.fixture(scope="session")
def xks_pull_secrets(admin_client, unprivileged_model_namespace):
    """Copy pull secrets to test namespace (xKS only)."""
    ...
```

### Test Structure — Conformance Phases

Each test class follows ordered phases matching the existing `test_llmd_smoke.py` pattern:

```python
# test_llmd_xks_smoke.py

@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, XksSmokeConfig)],
    indirect=True,
)
class TestLLMDXksSmoke:
    """xKS smoke: deploy LLMInferenceService and verify inference."""

    def test_health(self, llmisvc):
        """Health endpoint returns 200."""
        status, _ = send_health_check(llmisvc=llmisvc)
        assert status == 200

    def test_models(self, llmisvc):
        """Model is listed in /v1/models."""
        status, body = send_list_models(llmisvc=llmisvc)
        assert status == 200
        assert body["data"], "No models returned"

    def test_inference(self, llmisvc):
        """Chat completion returns tokens."""
        status, body = send_chat_completions(llmisvc=llmisvc, prompt="What is 2+2?")
        assert status == 200
        text = parse_completion_text(response_body=body)
        assert text, "Empty response"

    def test_metrics(self, llmisvc, admin_client):
        """vLLM Prometheus metrics show successful requests."""
        metrics = scrape_vllm_metrics(llmisvc=llmisvc, client=admin_client)
        assert metrics.get("vllm:request_success_total", 0) > 0
```

### Config Classes

Follow existing `config_base.py` pattern:

```python
# config_xks_smoke.py

from tests.model_serving.model_server.llmd.llmd_configs.config_base import LLMDConfigBase

class XksSmokeConfig(LLMDConfigBase):
    MANIFEST = "single-gpu-smoke.yaml"
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    REPLICAS = 1
    GPU_COUNT = 1  # 0 for mock mode
    METRICS_CHECK = False
    TIMEOUT = "10m"
```

### Simulator Integration

When `--mock` is passed or no GPUs detected:

1. Patch the LLMInferenceService manifest:
   - Replace main container image with `ghcr.io/llm-d/llm-d-inference-sim:latest`
   - Set `command: ["/app/llm-d-inference-sim"]`
   - Set `args: ["--model", "sim-model", "--served-model-name", "<real-model-name>", "--port", "8000", "--self-signed-certs", "--mode", "random"]`
   - Strip GPU resource requests
2. No render sidecar needed (simulated tokenizer)
3. Model download still happens via storage-initializer (CRD requires `uri`)

### Gateway and Namespace Handling (xKS)

On non-OCP clusters:
- Patch inference gateway `allowedRoutes.namespaces.from: All`
- Copy pull secrets from `rhaii` namespace to test namespace
- Use Gateway API HTTPRoute (not OCP Route)

These are handled by `xks_gateway` and `xks_pull_secrets` fixtures.

### Metrics Validation

Reuse `utilities/monitoring.py` for Prometheus scraping. Add llm-d specific validators:

| Topology | Metrics checked |
|----------|----------------|
| Basic | `vllm:request_success_total > 0` |
| Cache-aware | `vllm:prefix_cache_queries > 0`, `vllm:prefix_cache_hits > 0`, hit rate > 0% |
| P/D | `vllm:prompt_tokens_total > 0`, `vllm:generation_tokens_total > 0` |
| Scheduler | `inference_extension_scheduler_e2e_duration_seconds_count > 0`, `inference_pool_ready_pods > 0` |

### CI Pipeline

```
┌─────────────────────────────────────────────────┐
│  CI Pipeline (GitHub Actions / Jenkins)          │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. Provision cluster (AKS/EKS/OCP)             │
│  2. Helm install RHAII (xKS) or RHOAI (OCP)     │
│  3. Wait for operators ready                     │
│  4. Run tests:                                   │
│     ├── pytest tests/.../llmd/ -m xks --mock    │ ← CPU-only CI
│     ├── pytest tests/.../llmd/ -m xks           │ ← GPU CI
│     └── pytest tests/.../llmd/ -m ocp           │ ← OCP CI
│  5. Collect HTML + JSON reports                  │
│  6. Teardown cluster                             │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Implementation Order

### Phase 1: Smoke test (week 1)
- [ ] Add `xks_fixtures.py` with platform detection, gateway patch, pull secret copy
- [ ] Add `config_xks_smoke.py` with Qwen3-0.6B config
- [ ] Add `test_llmd_xks_smoke.py` with health, models, inference phases
- [ ] Add simulator patching logic in fixtures
- [ ] Verify on AKS with `--mock` and on OCP with real GPU

### Phase 2: Metrics + topologies (week 2)
- [ ] Add `test_llmd_xks_metrics.py` with vLLM basic metrics validation
- [ ] Add `test_llmd_xks_cache_aware.py` with prefix cache config + metrics
- [ ] Add `test_llmd_xks_prefill_decode.py` with P/D config + metrics
- [ ] Add metrics scraping utilities to `xks_utils.py`

### Phase 3: CI pipeline (week 3)
- [ ] Add GitHub Actions workflow for xKS (AKS + mock)
- [ ] Add GPU CI workflow (OCP or GPU xKS)
- [ ] HTML report generation and artifact upload
- [ ] Pre-flight health check (verify RHAII/RHOAI install before tests)

### Phase 4: Platform hardening (week 4)
- [ ] EKS platform support
- [ ] Discover mode (validate pre-deployed services)
- [ ] Upgrade tests (deploy on old version, upgrade, verify)
- [ ] Multi-pool / MoE test cases

## Extension: Dashboard, MaaS, and Cross-Platform Testing

### Dashboard Integration Tests

Validate that the Dashboard UI correctly manages LLMInferenceService on both platforms:

```
tests/model_serving/model_server/llmd/
├── test_llmd_dashboard_deploy.py           # Deploy model via Dashboard API
├── test_llmd_dashboard_status.py           # Verify model status in Dashboard
├── test_llmd_dashboard_inference.py        # Run inference via Dashboard endpoint
└── test_llmd_dashboard_cleanup.py          # Delete model via Dashboard
```

| Test | RHOAI (OCP) | RHAII (xKS) | Notes |
|------|:-----------:|:-----------:|-------|
| Deploy model via Dashboard | Yes | Yes | Dashboard API, not kubectl |
| Model catalog listing | Yes | Yes | Verify model appears in catalog |
| Configure serving runtime | Yes | Yes | vLLM runtime selection |
| Set GPU/resource requests | Yes | Yes | Dashboard may lack fsGroup config on xKS |
| Inference via Dashboard endpoint | Yes | Yes | Uses Route (OCP) or Gateway (xKS) |

```python
# test_llmd_dashboard_deploy.py

class TestDashboardDeploy:
    """Deploy LLMInferenceService through Dashboard API."""

    def test_create_model_serving(self, dashboard_client, model_config):
        """Create model serving via Dashboard REST API."""
        resp = dashboard_client.create_inference_service(model_config)
        assert resp.status_code == 200

    def test_model_appears_in_catalog(self, dashboard_client, model_config):
        """Model appears in Dashboard model catalog."""
        models = dashboard_client.list_models()
        assert model_config.name in [m["name"] for m in models]

    def test_inference_via_dashboard(self, dashboard_client, model_config):
        """Inference works through Dashboard-managed endpoint."""
        resp = dashboard_client.chat_completion(
            model=model_config.name, prompt="What is 2+2?"
        )
        assert resp.status_code == 200
        assert resp.json()["choices"]
```

### MaaS (Models as a Service) Tests

Test the MaaS billing, subscription, and multi-tenant inference flow:

```
tests/model_serving/maas_billing/
├── test_maas_llmd_deploy.py                # Deploy model with MaaS billing
├── test_maas_llmd_subscription.py          # API key / subscription validation
├── test_maas_llmd_inference.py             # Metered inference
├── test_maas_llmd_multitenancy.py          # Tenant isolation
└── test_maas_llmd_billing_metrics.py       # Billing Prometheus metrics
```

| Test | RHOAI (OCP) | RHAII (xKS) | Notes |
|------|:-----------:|:-----------:|-------|
| MaaS model deploy | Yes | Yes | MaaS CR + LLMInferenceService |
| API key auth | Yes | Yes | Kuadrant / AI Gateway |
| Metered inference | Yes | Yes | Token usage tracking |
| Tenant isolation | Yes | Yes | Namespace-level isolation |
| Billing metrics | Yes | Yes | Prometheus counters |
| Rate limiting | Yes | Yes | Per-tenant limits |

### Cross-Platform Test Matrix

```
                    ┌──────────────────────────────────────────┐
                    │         Test Categories                   │
                    ├──────────┬──────────┬──────────┬─────────┤
                    │ Conform. │ Dashboard│  MaaS    │ Upgrade │
  ──────────────────┼──────────┼──────────┼──────────┼─────────┤
  OCP + GPU         │    ✓     │    ✓     │    ✓     │    ✓    │
  OCP + Mock        │    ✓     │    ✓     │    ✓     │    ─    │
  AKS + GPU         │    ✓     │    ✓     │    ✓     │    ✓    │
  AKS + Mock        │    ✓     │    ✓     │    ✓     │    ─    │
  EKS + GPU         │    ✓     │    ✓     │    ✓     │    ✓    │
  EKS + Mock        │    ✓     │    ✓     │    ✓     │    ─    │
  ──────────────────┴──────────┴──────────┴──────────┴─────────┘
```

### Shared Abstractions

To avoid duplicating platform logic, add a platform abstraction layer:

```python
# utilities/platform.py

class PlatformAdapter:
    """Abstract platform differences between OCP and xKS."""

    def get_inference_endpoint(self, llmisvc) -> str:
        """Return the inference URL (Route on OCP, Gateway on xKS)."""
        ...

    def ensure_pull_secrets(self, namespace: str):
        """Copy pull secrets to test namespace (no-op on OCP)."""
        ...

    def ensure_gateway_access(self):
        """Patch gateway allowedRoutes (no-op on OCP)."""
        ...

    def get_model_image(self, mode: str) -> str:
        """Return vLLM image (real) or simulator image (mock)."""
        ...

    def get_dashboard_url(self) -> str:
        """Return Dashboard URL (Route on OCP, Ingress on xKS)."""
        ...

class OCPPlatform(PlatformAdapter): ...
class AKSPlatform(PlatformAdapter): ...
class EKSPlatform(PlatformAdapter): ...
```

This lets every test call `platform.get_inference_endpoint(llmisvc)` without caring whether it's an OCP Route or a Gateway API HTTPRoute.

## Open Questions

1. **Helm install in CI**: Should tests assume RHAII is pre-installed, or should the test pipeline do the helm install? The existing llmd tests assume RHOAI is already deployed via OLM.
2. **Simulator image versioning**: Pin to a specific tag or use `latest`?
3. **Model download in mock mode**: CRD requires `spec.model.uri`. Can we use a no-op URI scheme to skip the storage-initializer download?
4. **Test ownership**: Who owns the xKS tests in `opendatahub-tests` — xKS team or llm-d team?
