"""MaaS (Models-as-a-Service) helpers for testcases whose manifest declares MaaS resources.

A MaaS testcase manifest deploys an LLMInferenceService together with its
``MaaSModelRef``, ``MaaSAuthPolicy`` and ``MaaSSubscription``. The conformance
phases validate the model like any other testcase, then exercise the governed
MaaS route: unauthenticated rejection, API-key creation, authenticated
inference, and the subscription's token rate limit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx

from conformance.config import load_manifest_documents, parse_duration

# Namespace holding MaaS tenant resources; API-key creation accepts tokens for its service accounts.
MAAS_NAMESPACE = "models-as-a-service"
TOKEN_AUDIENCE = "https://kubernetes.default.svc"

_ACTIVE_PHASES = {
    "MaaSModelRef": "Ready",
    "MaaSAuthPolicy": "Active",
    "MaaSSubscription": "Active",
}
_FAILED_PHASES = {"Failed", "Invalid"}
# A failed phase can be a reconcile race; only fail once it persists this many polls.
_FAILED_POLLS = 3


@dataclass(frozen=True)
class MaaSResource:
    kind: str
    name: str
    namespace: str


@dataclass(frozen=True)
class MaaSSubscription:
    """The testcase's MaaSSubscription and the token rate limit it grants."""

    name: str
    token_limit: int | None = None
    window_seconds: float = 0


@dataclass
class MaaSTarget:
    """MaaS route state shared by the MaaS phases of one testcase."""

    resources: list[MaaSResource]
    subscription: MaaSSubscription | None
    model_url: str = ""
    api_url: str = ""
    model_alias: str = ""
    api_key: str = ""


def manifest_maas_resources(manifest_path: str | Path, default_namespace: str) -> list[MaaSResource]:
    """MaaS resources a manifest declares, in manifest order."""
    path = Path(manifest_path)
    if not path.is_file():
        return []
    resources = []
    for document in load_manifest_documents(path):
        if not isinstance(document, dict) or document.get("kind") not in _ACTIVE_PHASES:
            continue
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        if metadata.get("name"):
            resources.append(
                MaaSResource(document["kind"], metadata["name"], metadata.get("namespace") or default_namespace)
            )
    return resources


def manifest_subscription(manifest_path: str | Path) -> MaaSSubscription | None:
    """The manifest's first MaaSSubscription with its first token rate limit, if any."""
    path = Path(manifest_path)
    if not path.is_file():
        return None
    for document in load_manifest_documents(path):
        if not isinstance(document, dict) or document.get("kind") != "MaaSSubscription":
            continue
        name = (document.get("metadata") or {}).get("name", "")
        for model_ref in (document.get("spec") or {}).get("modelRefs") or []:
            for rate_limit in model_ref.get("tokenRateLimits") or []:
                window = parse_duration(str(rate_limit["window"])).total_seconds()
                return MaaSSubscription(name, int(rate_limit["limit"]), window)
        return MaaSSubscription(name)
    return None


def wait_for_maas_ready(
    kubectl: Callable[..., str],
    resources: list[MaaSResource],
    timeout: float,
    poll_interval: float = 5,
) -> dict:
    """Wait until every MaaS resource reaches its active phase; return the MaaSModelRef status.

    The MaaSModelRef only becomes Ready once a policy and subscription pair with
    it, so every resource must already be applied when this is called.
    """
    model_ref_status: dict = {}
    deadline = time.monotonic() + timeout
    for resource in resources:
        target = _ACTIVE_PHASES[resource.kind]
        failed_polls = 0
        while True:
            output = kubectl("get", resource.kind.lower(), resource.name, "-n", resource.namespace, "-o", "json")
            status = json.loads(output).get("status", {})
            phase = status.get("phase", "Pending")
            if phase == target:
                if resource.kind == "MaaSModelRef":
                    model_ref_status = status
                break
            detail = "; ".join(
                f"{c.get('type')}={c.get('status')} {c.get('message', '')}".strip()
                for c in status.get("conditions", [])
            )
            failed_polls = failed_polls + 1 if phase in _FAILED_PHASES else 0
            if failed_polls >= _FAILED_POLLS:
                raise RuntimeError(f"{resource.kind}/{resource.name} is {phase}: {detail}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout:g}s waiting for {resource.kind}/{resource.name} "
                    f"to become {target} (phase={phase}; {detail})"
                )
            time.sleep(poll_interval)
    return model_ref_status


def maas_urls(endpoint: str, scheme: str) -> tuple[str, str]:
    """``(model URL, management API URL)`` for a MaaSModelRef endpoint.

    The endpoint may carry a model path, but the MaaS management API
    (``/v1/api-keys``) is routed at the gateway root, so it uses only the origin.
    ``scheme`` overrides the endpoint's (e.g. ``http`` where port 443 is blocked).
    """
    parsed = urlsplit(endpoint)
    parsed = parsed._replace(scheme=scheme or parsed.scheme)
    return parsed.geturl().rstrip("/"), parsed._replace(path="", query="", fragment="").geturl()


def patch_maas_refs(documents: list, primary_name: str | None, service_name: str, namespace: str) -> None:
    """Keep MaaS resources pointing at the deployed service.

    The deployer renames the primary LLMInferenceService to the testcase name and
    deploys into an overridable test namespace; model refs, policies and
    subscriptions follow both.
    """
    for document in documents:
        spec = document.get("spec") if isinstance(document, dict) else None
        if not isinstance(spec, dict):
            continue
        kind = document.get("kind")
        if kind == "MaaSModelRef":
            ref = spec.get("modelRef")
            if isinstance(ref, dict) and ref.get("kind") == "LLMInferenceService" and ref.get("name") == primary_name:
                ref["name"] = service_name
        elif kind in ("MaaSAuthPolicy", "MaaSSubscription"):
            for ref in spec.get("modelRefs") or []:
                if isinstance(ref, dict):
                    ref["namespace"] = namespace


def create_user_token(kubectl: Callable[..., str]) -> str:
    """Kubernetes token accepted by the MaaS API for API-key creation."""
    return kubectl(
        "create",
        "token",
        "default",
        "-n",
        MAAS_NAMESPACE,
        "--duration=1h",
        f"--audience={TOKEN_AUDIENCE}",
    ).strip()


def exhaust_token_quota(
    client: MaaSClient, model: str, api_key: str, subscription: MaaSSubscription, max_tokens: int
) -> list[int]:
    """Statuses of output-capped requests sent until the first non-200.

    Each admitted request spends at least one token, so ``token_limit + 1``
    requests always exhaust the quota. If earlier traffic already spent it (the
    first request is rejected), wait one window for a fresh quota and retry once.
    """

    def send_until_rejected() -> list[int]:
        statuses: list[int] = []
        for _ in range(subscription.token_limit + 1):
            response = client.chat_completion(
                model, "Write a long essay about AI", api_key=api_key, max_tokens=max_tokens
            )
            statuses.append(response.status_code)
            if response.status_code != 200:
                break
        return statuses

    statuses = send_until_rejected()
    if statuses == [429]:
        time.sleep(subscription.window_seconds + 1)
        statuses = send_until_rejected()
    return statuses


class MaaSClient:
    """Call MaaS routes while preserving non-success responses for assertions."""

    def __init__(
        self, api_url: str, model_url: str, timeout: float = 120, transport: httpx.BaseTransport | None = None
    ):
        self._api_url = api_url.rstrip("/")
        self._model_url = model_url.rstrip("/")
        self._client = httpx.Client(verify=False, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def create_api_key(
        self, user_token: str, name: str, subscription: str = "", expires_in: str = "7d"
    ) -> httpx.Response:
        """Create an API key with a Kubernetes user token; return status/body intact.

        Without ``subscription`` MaaS binds the user's highest-priority accessible one.
        """
        body = {"name": name, "expiresIn": expires_in}
        if subscription:
            body["subscription"] = subscription
        return self._client.post(
            f"{self._api_url}/v1/api-keys",
            headers={"Authorization": f"Bearer {user_token}"},
            json=body,
        )

    def chat_completion(
        self,
        model: str,
        prompt: str,
        *,
        api_key: str = "",
        max_tokens: int | None = None,
    ) -> httpx.Response:
        """Send one OpenAI-compatible MaaS request without raising for HTTP errors."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        body: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return self._client.post(f"{self._model_url}/v1/chat/completions", headers=headers, json=body)
