from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import conformance.maas as maas
from conformance.maas import (
    MaaSClient,
    MaaSResource,
    MaaSSubscription,
    exhaust_token_quota,
    manifest_subscription,
    wait_for_maas_ready,
)

_MODEL_REF = MaaSResource("MaaSModelRef", "maas-single-gpu", "llm-conformance-test")
_POLICY = MaaSResource("MaaSAuthPolicy", "maas-single-gpu-auth", "models-as-a-service")


def _kubectl_returning(phases: dict[str, list[str]]):
    """Fake kubectl: each ``get <kind>`` returns the next phase queued for that kind (last one repeats)."""

    def kubectl(*args):
        queue = phases[args[1]]
        phase = queue.pop(0) if len(queue) > 1 else queue[0]
        status = {"phase": phase, "endpoint": "https://203.0.113.10/", "resolvedModelAlias": "publishers/x/models/q"}
        return json.dumps({"status": status})

    return kubectl


def test_waits_through_pending_and_returns_model_ref_status():
    kubectl = _kubectl_returning({"maasmodelref": ["Pending", "Pending", "Ready"], "maasauthpolicy": ["Active"]})

    status = wait_for_maas_ready(kubectl, [_MODEL_REF, _POLICY], timeout=5, poll_interval=0)

    assert status["endpoint"] == "https://203.0.113.10/"


def test_transient_failed_phase_is_tolerated():
    kubectl = _kubectl_returning({"maasmodelref": ["Failed", "Failed", "Ready"]})

    assert wait_for_maas_ready(kubectl, [_MODEL_REF], timeout=5, poll_interval=0)["phase"] == "Ready"


def test_persistent_failed_phase_fails_fast():
    kubectl = _kubectl_returning({"maasmodelref": ["Failed"]})

    with pytest.raises(RuntimeError, match="MaaSModelRef/maas-single-gpu is Failed"):
        wait_for_maas_ready(kubectl, [_MODEL_REF], timeout=60, poll_interval=0)


def test_manifest_subscription_reads_name_limit_and_window(tmp_path):
    manifest = tmp_path / "maas.yaml"
    manifest.write_text(
        "kind: MaaSSubscription\nmetadata:\n  name: maas-single-gpu-subscription\nspec:\n  modelRefs:\n"
        "  - name: maas-single-gpu\n    tokenRateLimits:\n    - limit: 100\n      window: 1m\n"
    )

    assert manifest_subscription(manifest) == MaaSSubscription("maas-single-gpu-subscription", 100, 60.0)


class _TokenQuotaGateway:
    """Fake MaaS gateway: admits requests while the window has quota left, then charges usage."""

    def __init__(self, limit: int, tokens_per_request: int):
        self.limit = limit
        self.tokens_per_request = tokens_per_request
        self.used = 0

    def handle(self, request):
        if self.used >= self.limit:
            return httpx.Response(429)
        self.used += self.tokens_per_request
        return httpx.Response(200, json={"choices": [{}]})


def test_quota_spent_by_earlier_inference_is_waited_out_before_measuring(monkeypatch):
    """A preceding authenticated request can exhaust the quota; 30e must still see 200 then 429."""
    gateway = _TokenQuotaGateway(limit=100, tokens_per_request=60)
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        gateway.used = 0  # the window rolls over

    monkeypatch.setattr(maas.time, "sleep", sleep)
    subscription = MaaSSubscription("maas-single-gpu-subscription", 100, 60.0)

    with closing(MaaSClient("http://gw", "http://gw", transport=httpx.MockTransport(gateway.handle))) as client:
        client.chat_completion("model", "hello", api_key="sk-oai-test")  # 30d
        client.chat_completion("model", "hello", api_key="sk-oai-test")  # spends the rest of the window
        statuses = exhaust_token_quota(client, "model", "sk-oai-test", subscription, max_tokens=16)

    assert slept == [61.0]
    assert statuses == [200, 200, 429]
