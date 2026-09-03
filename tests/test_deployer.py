"""Unit tests for the Deployer: kubectl wrappers, manifest parsing, retry logic.

Run with: ``uv run pytest tests/test_deployer.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.config import load_testcase


def test_deployer_is_deployed():
    from conformance.deployer import Deployer

    d = Deployer()
    assert not d.is_deployed("foo")
    d._deployed.add("foo")
    assert d.is_deployed("foo")
    d._deployed.discard("foo")
    assert not d.is_deployed("foo")


def test_cluster_gpu_count_total(monkeypatch):
    """cluster_gpu_count sums allocatable nvidia.com/gpu across nodes, ignoring blank (CPU) nodes."""
    from conformance.deployer import Deployer

    d = Deployer()
    monkeypatch.setattr(d, "kubectl", lambda *a, **k: "1\n\n2\n")
    assert d.cluster_gpu_count() == 3


def test_list_workload_pods_parses_and_filters_blanks(monkeypatch):
    """list_workload_pods returns pod names from jsonpath output, dropping blank lines."""
    from conformance.deployer import Deployer

    d = Deployer()
    monkeypatch.setattr(d, "kubectl", lambda *a, **k: "pod-a\n\npod-b\n")
    assert d.list_workload_pods("my-isvc") == ["pod-a", "pod-b"]


def _write_manifest(tmp_path, body: str) -> "tuple":
    """Write a manifest into a temp manifest_dir; return (Deployer, tc) wired to it."""
    from conformance.deployer import Deployer

    if body.lstrip().startswith("spec:"):
        body = "apiVersion: serving.kserve.io/v1alpha2\nkind: LLMInferenceService\n" + body
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "case.yaml").write_text(body)
    d = Deployer(manifest_dir=str(manifest_dir))
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.manifest_path = "case.yaml"
    return d, tc


def test_manifest_replicas_sums_decode_and_prefill(tmp_path):
    """manifest_replicas = spec.replicas + spec.prefill.replicas (P/D)."""
    d, tc = _write_manifest(
        tmp_path,
        "spec:\n  replicas: 2\n  prefill:\n    replicas: 3\n",
    )
    assert d.manifest_replicas(tc) == 5


def test_manifest_replicas_defaults_to_one(tmp_path):
    """Absent spec.replicas defaults to 1 (matches KServe)."""
    d, tc = _write_manifest(tmp_path, "spec:\n  model:\n    name: x\n")
    assert d.manifest_replicas(tc) == 1


def test_manifest_gpu_needed_sums_decode_and_prefill(tmp_path):
    """manifest_gpu_needed = sum over decode+prefill of replicas x peak container gpu limit."""
    body = (
        "spec:\n"
        "  replicas: 2\n"
        "  template:\n"
        "    containers:\n"
        "      - name: main\n"
        "        resources:\n"
        "          limits:\n"
        "            nvidia.com/gpu: 1\n"
        "  prefill:\n"
        "    replicas: 3\n"
        "    template:\n"
        "      containers:\n"
        "        - name: main\n"
        "          resources:\n"
        "            limits:\n"
        "              nvidia.com/gpu: 2\n"
    )
    d, tc = _write_manifest(tmp_path, body)
    assert d.manifest_gpu_needed(tc) == 2 * 1 + 3 * 2  # 8


def test_manifest_gpu_needed_zero_when_no_gpu(tmp_path):
    """A CPU-only manifest requests zero GPUs (so _require_gpu won't skip on GPU count)."""
    body = "spec:\n  replicas: 1\n  template:\n    containers:\n      - name: main\n"
    d, tc = _write_manifest(tmp_path, body)
    assert d.manifest_gpu_needed(tc) == 0


def test_manifest_counts_sum_across_multi_document_manifest(tmp_path):
    """Multi-document manifests (e.g. multi-pool) are summed across LLMISVC docs, not crashed on."""
    doc = (
        "apiVersion: serving.kserve.io/v1alpha2\n"
        "kind: LLMInferenceService\n"
        "spec:\n"
        "  replicas: 1\n"
        "  template:\n"
        "    containers:\n"
        "      - name: main\n"
        "        resources:\n"
        "          limits:\n"
        "            nvidia.com/gpu: 1\n"
    )
    d, tc = _write_manifest(tmp_path, doc + "---\n" + doc)
    assert d.manifest_replicas(tc) == 2  # 1 + 1
    assert d.manifest_gpu_needed(tc) == 2  # 1 + 1


def test_manifest_counts_ignore_non_llmisvc_documents(tmp_path):
    """Non-LLMInferenceService documents in a manifest are skipped, not counted."""
    body = (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n  name: noise\n"
        "data:\n  replicas: '99'\n"
        "---\n"
        "apiVersion: serving.kserve.io/v1alpha2\n"
        "kind: LLMInferenceService\n"
        "spec:\n  replicas: 2\n"
    )
    d, tc = _write_manifest(tmp_path, body)
    assert d.manifest_replicas(tc) == 2  # ConfigMap ignored


def test_require_gpu_skips_in_mock_mode():
    """A requiresGpu test case skips in mock mode."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.requires_gpu = True
    with pytest.raises(pytest.skip.Exception, match="has requiresGpu set"):
        tc_mod._require_gpu(deployer=None, tc=tc, mock_mode=True, test_mode="deploy")


def test_require_gpu_runs_when_cluster_has_enough_gpu():
    """With enough GPUs available (manifest need <= cluster), the case does not skip."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    class FakeDeployer:
        def manifest_gpu_needed(self, tc):
            return 2

        def cluster_gpu_count(self):
            return 2

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.requires_gpu = True
    tc_mod._require_gpu(deployer=FakeDeployer(), tc=tc, mock_mode=False, test_mode="deploy")


def test_require_gpu_skips_when_not_enough_gpus():
    """Skips when the cluster has fewer GPUs than the manifest requests."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    class FakeDeployer:
        def manifest_gpu_needed(self, tc):
            return 3

        def cluster_gpu_count(self):
            return 1

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    with pytest.raises(pytest.skip.Exception, match="needs 3 GPU"):
        tc_mod._require_gpu(deployer=FakeDeployer(), tc=tc, mock_mode=False, test_mode="deploy")


def test_apply_with_webhook_retry_succeeds_after_transient_error(monkeypatch):
    """Regression: deploy() used to fail immediately if the LLMInferenceService
    admission webhook wasn't serving yet (e.g. right after install/upgrade)."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise RuntimeError(
                "kubectl apply failed: Internal error occurred: failed calling webhook "
                '"llminferenceservice.kserve-webhook-server.v1alpha2.defaulter": '
                'failed to call webhook: Post "https://...": no endpoints available for service "llmisvc-webhook-server-service"'
            )
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 3


def test_apply_with_webhook_retry_raises_immediately_on_other_errors(monkeypatch):
    """Non-webhook errors (e.g. a real manifest problem) must not be retried."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("kubectl apply failed: error validating data: unknown field")

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="unknown field"):
        d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 1


def test_apply_with_webhook_retry_times_out(monkeypatch):
    """If the webhook never comes up within the grace period, fail with a clear error."""
    from conformance.deployer import Deployer

    d = Deployer()

    def fake_kubectl(*args, **kwargs):
        raise RuntimeError(
            'kubectl apply failed: failed calling webhook "llminferenceservice...": no endpoints available for service'
        )

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="waiting for webhook"):
        d._apply_with_webhook_retry("dummy.yaml", timeout=0.05, interval=0.01)


def test_wait_for_ready_fails_fast_on_persistent_error(monkeypatch):
    """wait_for_ready should fail fast when any Ready=False condition with the
    same reason+message persists across 3 consecutive polls, regardless of the
    specific reason string (RBAC, missing CRD, webhook, etc.)."""
    from conformance.deployer import Deployer

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SchedulerReconcileError"
            if "message}" in str(arg):
                return (
                    'roles.rbac.authorization.k8s.io "epp-role" is forbidden: '
                    "user is attempting to grant RBAC permissions not currently held"
                )
            if "CrashLoopBackOff" in str(arg) or "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    logs = []
    with pytest.raises(RuntimeError, match="Persistent controller error.*SchedulerReconcileError.*RBAC permissions"):
        d.wait_for_ready(tc, timeout=600, print_fn=logs.append)

    assert any("RBAC permissions" in line for line in logs), f"Expected RBAC error message in log output, got: {logs}"


def test_wait_for_ready_fails_fast_on_any_repeated_error(monkeypatch):
    """Any controller error reason should trigger fast-fail when it persists,
    not just a hardcoded list."""
    from conformance.deployer import Deployer

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SomeFutureUnknownError"
            if "message}" in str(arg):
                return "CRD llm-d.ai/v1 InferenceObjective not found on the cluster"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Persistent controller error.*SomeFutureUnknownError.*not found"):
        d.wait_for_ready(tc, timeout=600, print_fn=lambda _: None)


def test_wait_for_ready_does_not_fast_fail_on_changing_reasons(monkeypatch):
    """If the error reason/message keeps changing between polls, that
    indicates the controller is making progress -- do NOT fast-fail."""
    from conformance.deployer import Deployer

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    poll = 0

    def fake_kubectl(*args, **kwargs):
        nonlocal poll
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                poll += 1
                return f"TransientError{poll}"
            if "message}" in str(arg):
                return f"Some transient message variant {poll}"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="not ready after"):
        d.wait_for_ready(tc, timeout=0.01, print_fn=lambda _: None)


def test_wait_for_ready_includes_message_in_timeout(monkeypatch):
    """When wait_for_ready times out, the TimeoutError should include the last
    known reason and message for actionable diagnostics."""
    from conformance.deployer import Deployer

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    poll = 0

    def fake_kubectl(*args, **kwargs):
        nonlocal poll
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                poll += 1
                return f"TransientReason{poll}"
            if "message}" in str(arg):
                return f"Controller is retrying something (attempt {poll})"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match=r"not ready after.*TransientReason.*Controller is retrying"):
        d.wait_for_ready(tc, timeout=0.01, print_fn=lambda _: None)


def test_wait_for_ready_logs_message_field(monkeypatch):
    """wait_for_ready should include the condition .message in progress logs."""
    from conformance.deployer import Deployer

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SchedulerReconcileError"
            if "message}" in str(arg):
                return "Missing RBAC permissions for llm-d.ai resources"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    logs = []
    with pytest.raises(RuntimeError):
        d.wait_for_ready(tc, timeout=600, print_fn=logs.append)

    assert any("message=" in line and "Missing RBAC" in line for line in logs), (
        f"Expected condition message in log lines, got: {logs}"
    )


def test_apply_with_webhook_retry_does_not_retry_unrelated_connectivity_errors(monkeypatch):
    """Connectivity substrings alone (no 'webhook' mention) must not trigger retries."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("kubectl apply failed: error validating data: invalid geofence field")

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="geofence"):
        d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 1


def test_apply_with_webhook_retry_always_attempts_once(monkeypatch):
    """Even with timeout=0, at least one apply attempt must happen."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)

    d._apply_with_webhook_retry("dummy.yaml", timeout=0, interval=1)
    assert len(calls) == 1


def test_apply_with_webhook_retry_recovers_from_crd_not_registered(monkeypatch):
    """CRD/API-version-not-registered errors must also be retried."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        if len(calls) < 2:
            raise RuntimeError("Error from server (NotFound): the server could not find the requested resource")
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 2


def test_apply_with_webhook_retry_no_matches_for_kind_is_transient(monkeypatch):
    """'no matches for kind' (stale discovery cache after a CRD apply) is also transient."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        if len(calls) < 2:
            raise RuntimeError('no matches for kind "LLMInferenceService" in version "serving.kserve.io/v1alpha1"')
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 2


def test_wait_for_ready_surfaces_operator_image_pull_errors(monkeypatch):
    """When wait_for_ready hits a persistent error and operator pods have
    ImagePullBackOff, the error message should include the failing image."""
    from conformance.deployer import Deployer

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SchedulerReconcileError"
            if "message}" in str(arg):
                return "failed to reconcile scheduler resources"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)
    monkeypatch.setattr(
        d,
        "_check_operator_image_issues",
        lambda: [
            "redhat-ods-applications/kserve-module-controller-manager-abc: "
            "ImagePullBackOff (quay.io/rhoai/odh-kserve-module-operator-rhel9:latest)"
        ],
    )

    logs = []
    with pytest.raises(RuntimeError, match=r"(?s)Operator image pull failures.*ImagePullBackOff.*kserve-module"):
        d.wait_for_ready(tc, timeout=600, print_fn=logs.append)

    assert any("WARNING" in line and "ImagePullBackOff" in line for line in logs), (
        f"Expected WARNING about ImagePullBackOff in logs, got: {logs}"
    )


def test_wait_for_ready_timeout_includes_image_pull_errors(monkeypatch):
    """When wait_for_ready times out, the TimeoutError should still include
    operator ImagePullBackOff info."""
    from conformance.deployer import Deployer

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    poll = 0

    def fake_kubectl(*args, **kwargs):
        nonlocal poll
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                poll += 1
                return f"TransientReason{poll}"
            if "message}" in str(arg):
                return f"retrying attempt {poll}"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)
    monkeypatch.setattr(
        d,
        "_check_operator_image_issues",
        lambda: ["redhat-ods-applications/bad-operator-pod-xyz: ErrImagePull (quay.io/rhoai/some-private-image:v1)"],
    )

    with pytest.raises(TimeoutError, match=r"(?s)Operator image pull failures.*ErrImagePull.*some-private-image"):
        d.wait_for_ready(tc, timeout=0.01, print_fn=lambda _: None)


def test_check_operator_image_issues_returns_empty_when_healthy(monkeypatch):
    """No false positives when all operator pods are healthy."""
    from conformance.deployer import Deployer

    d = Deployer()

    def fake_kubectl(*args, **kwargs):
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    assert d._check_operator_image_issues() == []


def test_check_operator_image_issues_parses_image_pull_back_off(monkeypatch):
    """_check_operator_image_issues correctly parses ImagePullBackOff from kubectl output."""
    from conformance.deployer import Deployer

    d = Deployer()

    def fake_kubectl(*args, **kwargs):
        joined = " ".join(str(a) for a in args)
        if "redhat-ods-applications" in joined:
            return "kserve-ctrl-abc|ImagePullBackOff=quay.io/rhoai/private-img:v1 |"
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    issues = d._check_operator_image_issues()
    assert len(issues) == 1
    assert "ImagePullBackOff" in issues[0]
    assert "quay.io/rhoai/private-img:v1" in issues[0]
    assert "kserve-ctrl-abc" in issues[0]


def test_env_overrides_applied_to_decode_and_prefill():
    """_patch_manifest should inject env_overrides into main containers of both templates."""
    import tempfile

    import yaml
    from conformance.deployer import Deployer

    manifest = {
        "apiVersion": "serving.kserve.io/v1alpha2",
        "kind": "LLMInferenceService",
        "metadata": {"name": "test"},
        "spec": {
            "model": {"uri": "hf://test/model", "name": "test/model"},
            "template": {
                "containers": [
                    {"name": "main", "env": [{"name": "EXISTING", "value": "keep"}]},
                ]
            },
            "prefill": {
                "replicas": 1,
                "template": {
                    "containers": [
                        {"name": "main", "env": [{"name": "EXISTING", "value": "keep"}]},
                    ]
                },
            },
        },
    }

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.env_overrides = {"FOO": "bar", "EXISTING": "overwritten"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(manifest, f)
        tmp = f.name

    try:
        d = Deployer()
        result = d._patch_manifest(Path(tmp), tc)
        spec = result["spec"]

        for section_name in ("template", "prefill"):
            if section_name == "prefill":
                containers = spec["prefill"]["template"]["containers"]
            else:
                containers = spec["template"]["containers"]
            main = [c for c in containers if c["name"] == "main"][0]
            env_dict = {e["name"]: e["value"] for e in main["env"]}
            assert env_dict["FOO"] == "bar", f"{section_name}: FOO not injected"
            assert env_dict["EXISTING"] == "overwritten", f"{section_name}: EXISTING not overwritten"
    finally:
        Path(tmp).unlink()
