from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from conformance.maas import MaaSClient, maas_urls

API_URL = "http://maas.example"
MODEL_URL = "http://maas.example"


def test_create_api_key_sends_bearer_user_token_name_and_subscription():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201, json={"key": "sk-oai-test"})

    with closing(MaaSClient(API_URL, MODEL_URL, transport=httpx.MockTransport(handler))) as client:
        response = client.create_api_key("k8s-user-token", "llm-d-e2e", subscription="maas-single-gpu-subscription")

    assert response.status_code == 201
    assert requests[0].url.path == "/v1/api-keys"
    assert requests[0].headers["authorization"] == "Bearer k8s-user-token"
    assert json.loads(requests[0].read()) == {
        "name": "llm-d-e2e",
        "expiresIn": "7d",
        "subscription": "maas-single-gpu-subscription",
    }


def test_api_key_binds_requested_subscription_when_another_is_accessible():
    """MaaS binds the highest-priority accessible subscription unless one is requested."""

    def handler(request):
        requested = json.loads(request.read()).get("subscription")
        return httpx.Response(201, json={"key": "sk-oai-test", "subscription": requested or "premium-subscription"})

    with closing(MaaSClient(API_URL, MODEL_URL, transport=httpx.MockTransport(handler))) as client:
        response = client.create_api_key("token", "llm-d-e2e", subscription="maas-single-gpu-subscription")

    assert response.json()["subscription"] == "maas-single-gpu-subscription"


def test_path_based_model_endpoint_keeps_management_api_at_gateway_root():
    model_url, api_url = maas_urls("https://gateway.example/llm-conformance-test/maas-single-gpu/", "http")
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={})

    with closing(MaaSClient(api_url, model_url, transport=httpx.MockTransport(handler))) as client:
        client.create_api_key("token", "llm-d-e2e")
        client.chat_completion("model-alias", "hello", api_key="sk-oai-test")

    assert (model_url, api_url) == (
        "http://gateway.example/llm-conformance-test/maas-single-gpu",
        "http://gateway.example",
    )
    assert paths == ["/v1/api-keys", "/llm-conformance-test/maas-single-gpu/v1/chat/completions"]


def test_chat_completion_uses_api_key_bearer_and_returns_error_responses():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, json={"error": "rate limit"})

    with closing(MaaSClient(API_URL, MODEL_URL, transport=httpx.MockTransport(handler))) as client:
        response = client.chat_completion("publishers/llm/models/qwen", "hello", api_key="sk-oai-test")

    assert response.status_code == 429
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer sk-oai-test"
    assert (
        requests[0].read() == b'{"model":"publishers/llm/models/qwen","messages":[{"role":"user","content":"hello"}]}'
    )


def test_chat_completion_can_send_unauthenticated_request():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(401)

    with closing(MaaSClient(API_URL, MODEL_URL, transport=httpx.MockTransport(handler))) as client:
        response = client.chat_completion("model-alias", "hello")

    assert response.status_code == 401
    assert "authorization" not in requests[0].headers
