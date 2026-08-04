"""Unit/smoke tests for the llm-d-e2e framework — no cluster required.

Validates config loading, client helpers, deployer retry/fast-fail logic,
manifest setup, and scaffolding scripts without kubectl against a live cluster.
Run with: ``uv run pytest tests/test_smoke.py -v`` (or ``make unittest``).

Coverage areas:
  Config — duration parsing; load testcase/profile/dir; LoRA single/multi YAML
  Metrics — Prometheus text exposition parsing
  Client — bearer token headers; chat() string vs message-list prompts
  Deployer — is_deployed tracking; workload pod listing; webhook/CRD transient
    apply retries; wait_for_ready persistent-error fast-fail and timeout messages;
    operator ImagePullBackOff surfacing; env_overrides on decode+prefill
  Manifests — --setup pruning of stale YAML; _require_manifest skip helpers
  Scaffolding — scripts/new-testcase.sh generates loadable config; rejects dupes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.config import load_testcase, load_profile, load_testcases_from_dir, parse_duration
from conformance.metrics import parse_prometheus
from conformance.client import LLMClient


def test_parse_duration():
    assert parse_duration("15m").total_seconds() == 900
    assert parse_duration("2h").total_seconds() == 7200
    assert parse_duration("300s").total_seconds() == 300
    assert parse_duration("1h30m").total_seconds() == 5400


def test_load_testcase():
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    assert tc.name == "single-gpu-smoke"
    assert tc.model.name == "Qwen/Qwen3-0.6B"
    assert tc.deployment.replicas == 1
    assert tc.deployment.resources.gpus == 1
    assert tc.validation.health_port == 8000
    assert tc.validation.test_prompts


def test_load_profile():
    profile = load_profile("configs/profiles/smoke.yaml")
    assert profile.name == "smoke"
    assert "single-gpu-smoke" in profile.test_cases


def test_load_all_testcases():
    cases = load_testcases_from_dir("configs/testcases")
    assert len(cases) >= 1
    names = [tc.name for tc in cases]
    assert "single-gpu-smoke" in names


def test_dir_loaders_skip_readme(tmp_path):
    """README.md (and README*.yaml) next to configs must not be loaded as cases/profiles."""
    from conformance.config import iter_config_yamls, load_profiles_from_dir, load_testcases_from_dir

    cases_dir = tmp_path / "testcases"
    profiles_dir = tmp_path / "profiles"
    cases_dir.mkdir()
    profiles_dir.mkdir()

    (cases_dir / "README.md").write_text("# docs\n")
    (cases_dir / "ok.yaml").write_text("name: ok\nmodel:\n  name: m\ndeployment:\n  manifestPath: x.yaml\n")
    (cases_dir / "README.yaml").write_text("name: should-skip\n")
    (profiles_dir / "README.md").write_text("# docs\n")
    (profiles_dir / "smoke.yaml").write_text("name: smoke\ntestCases:\n  - ok\n")

    assert [p.name for p in iter_config_yamls(cases_dir)] == ["ok.yaml"]
    assert [p.name for p in iter_config_yamls(profiles_dir)] == ["smoke.yaml"]
    assert [tc.name for tc in load_testcases_from_dir(cases_dir)] == ["ok"]
    assert [p.name for p in load_profiles_from_dir(profiles_dir)] == ["smoke"]


def test_parse_prometheus_text():
    text = """# HELP vllm:request_success_total Total requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="Qwen/Qwen3-0.6B"} 42.0
vllm:gpu_cache_usage_perc 0.15
"""
    metrics = parse_prometheus(text)
    assert "vllm:request_success_total" in metrics
    assert metrics["vllm:request_success_total"][0].value == 42.0
    assert metrics["vllm:request_success_total"][0].labels["model_name"] == "Qwen/Qwen3-0.6B"
    assert metrics["vllm:gpu_cache_usage_perc"][0].value == 0.15


def test_llm_client_init():
    c = LLMClient(base_url="http://localhost:8000", bearer_token="test-token")
    assert c._client.headers.get("authorization") == "Bearer test-token"
    c.close()


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
    # jsonpath output: one line per node, blank for nodes without GPUs.
    monkeypatch.setattr(d, "kubectl", lambda *a, **k: "1\n\n2\n")
    assert d.cluster_gpu_count() == 3


def test_list_workload_pods_parses_and_filters_blanks(monkeypatch):
    """list_workload_pods returns pod names from jsonpath output, dropping blank lines."""
    from conformance.deployer import Deployer

    d = Deployer()
    monkeypatch.setattr(d, "kubectl", lambda *a, **k: "pod-a\n\npod-b\n")
    assert d.list_workload_pods("my-isvc") == ["pod-a", "pod-b"]


def test_require_gpu_skips_when_flag_not_set():
    """A requiresGpu test case skips by default (unless --need-gpu is passed)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.requires_gpu = True
    with pytest.raises(pytest.skip.Exception, match="requires a GPU"):
        tc_mod._require_gpu(deployer=None, tc=tc, mock_mode=False, need_gpu=False, test_mode="deploy")


def test_require_gpu_runs_when_flag_and_gpu_present():
    """With --need-gpu and GPUs available (cluster_gpu_count > 0), the case does not skip."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    class FakeDeployer:
        def cluster_gpu_count(self):
            return 2

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.requires_gpu = True
    tc_mod._require_gpu(deployer=FakeDeployer(), tc=tc, mock_mode=False, need_gpu=True, test_mode="deploy")


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
    from conformance.config import load_testcase

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

    # Should have logged the message field
    assert any("RBAC permissions" in line for line in logs), f"Expected RBAC error message in log output, got: {logs}"


def test_wait_for_ready_fails_fast_on_any_repeated_error(monkeypatch):
    """Any controller error reason should trigger fast-fail when it persists,
    not just a hardcoded list."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

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
    from conformance.config import load_testcase

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
                # Return a different reason each poll
                return f"TransientError{poll}"
            if "message}" in str(arg):
                return f"Some transient message variant {poll}"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    # Should time out normally, NOT raise RuntimeError
    with pytest.raises(TimeoutError, match="not ready after"):
        d.wait_for_ready(tc, timeout=0.01, print_fn=lambda _: None)


def test_wait_for_ready_includes_message_in_timeout(monkeypatch):
    """When wait_for_ready times out, the TimeoutError should include the last
    known reason and message for actionable diagnostics."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

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
    from conformance.config import load_testcase

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

    # Verify message field appears in log output
    assert any("message=" in line and "Missing RBAC" in line for line in logs), (
        f"Expected condition message in log lines, got: {logs}"
    )


def test_apply_with_webhook_retry_does_not_retry_unrelated_connectivity_errors(monkeypatch):
    """Connectivity substrings alone (no 'webhook' mention) must not trigger retries.

    Regression: an early version matched bare substrings like 'connection refused' or
    'eof' anywhere in the error, which could misfire on unrelated failures (e.g. a
    manifest field containing 'geofence') or mask a real, non-webhook outage behind a
    misleading 'waiting for webhook' message.
    """
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
    """CRD/API-version-not-registered errors (a distinct upgrade race from webhook
    readiness) must also be retried. Seen in CI as:
    'the server could not find the requested resource' right after a CRD is
    installed/updated but the client's API discovery hasn't caught up yet.
    """
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


def test_setup_manifests_removes_stale_files(tmp_path, monkeypatch):
    """Switching manifest branches must remove stale files from the previous branch.

    Regression: _setup_manifests used to copy new files on top of existing ones
    without pruning. Switching main→3.4-stable left flow-control-tokens.yaml
    behind, causing it to appear available when the 3.4 EPP would crash on it.
    """
    import shutil
    from unittest.mock import MagicMock, patch
    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    # Simulate manifests left over from a previous `--setup main` run
    manifest_dir = tmp_path / "deploy" / "manifests"
    manifest_dir.mkdir(parents=True)
    for stale in ["flow-control-tokens.yaml", "flow-control.yaml", "pd-performance.yaml"]:
        (manifest_dir / stale).write_text("stale: true")

    # Pre-create what `git clone` would produce for 3.4-stable
    clone_dir = Path("/tmp/llm-d-manifests")
    clone_dir.mkdir(exist_ok=True)
    for new in ["single-gpu.yaml", "cache-aware.yaml"]:
        (clone_dir / new).write_text("branch: 3.4-stable")

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[0] == "rm":
            shutil.rmtree(str(clone_dir), ignore_errors=True)
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("3.4-stable")

    remaining = {f.name for f in manifest_dir.glob("*.yaml")}
    assert "flow-control-tokens.yaml" not in remaining
    assert "flow-control.yaml" not in remaining
    assert "pd-performance.yaml" not in remaining
    assert "single-gpu.yaml" in remaining
    assert "cache-aware.yaml" in remaining


def test_setup_manifests_uses_custom_repo(tmp_path, monkeypatch):
    """--manifest-repo <URL> must clone from the given repo, not the default.

    The custom URL has to reach ``git clone`` and be recorded in .manifest-ref
    so a later run can tell which fork the manifests came from.
    """
    import shutil
    from unittest.mock import MagicMock, patch
    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    custom_repo = "https://github.com/my-org/my-repo.git"

    clone_dir = Path("/tmp/llm-d-manifests")
    clone_dir.mkdir(exist_ok=True)
    (clone_dir / "single-gpu.yaml").write_text("branch: my-branch")

    clone_cmds = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[:2] == ["git", "clone"]:
            clone_cmds.append(cmd)
        if cmd[0] == "rm":
            shutil.rmtree(str(clone_dir), ignore_errors=True)
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("my-branch", custom_repo)

    # custom repo URL is shown in the git clone cmd.
    assert clone_cmds, "git clone was never invoked"
    assert custom_repo in clone_cmds[0]
    assert cli_mod.MANIFEST_REPO not in clone_cmds[0]

    ref_file = tmp_path / "deploy" / "manifests" / ".manifest-ref"
    assert f"repo: {custom_repo}" in ref_file.read_text()


def test_setup_manifests_defaults_to_upstream_repo(tmp_path, monkeypatch):
    """Without --manifest-repo, _setup_manifests clones the upstream default."""
    import shutil
    from unittest.mock import MagicMock, patch
    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    clone_dir = Path("/tmp/llm-d-manifests")
    clone_dir.mkdir(exist_ok=True)
    (clone_dir / "single-gpu.yaml").write_text("branch: main")

    clone_cmds = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[:2] == ["git", "clone"]:
            clone_cmds.append(cmd)
        if cmd[0] == "rm":
            shutil.rmtree(str(clone_dir), ignore_errors=True)
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("main")

    assert clone_cmds, "git clone was never invoked"
    assert cli_mod.MANIFEST_REPO in clone_cmds[0]


def test_require_manifest_skips_when_missing(tmp_path):
    """test_01_prereq and test_02_deploy skip when the manifest file is absent."""
    from dataclasses import dataclass

    @dataclass
    class FakeDeployConfig:
        manifest_path: str = "nonexistent.yaml"

    @dataclass
    class FakeTestCase:
        deployment: FakeDeployConfig = None

        def __post_init__(self):
            self.deployment = FakeDeployConfig()

    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    original = tc_mod._MANIFEST_DIR
    try:
        tc_mod._MANIFEST_DIR = tmp_path
        with pytest.raises(pytest.skip.Exception, match="nonexistent.yaml"):
            tc_mod._require_manifest(FakeTestCase())
    finally:
        tc_mod._MANIFEST_DIR = original


def test_require_manifest_does_not_skip_when_present(tmp_path):
    """_require_manifest should not skip when the manifest exists."""
    from dataclasses import dataclass

    @dataclass
    class FakeDeployConfig:
        manifest_path: str = "exists.yaml"

    @dataclass
    class FakeTestCase:
        deployment: FakeDeployConfig = None

        def __post_init__(self):
            self.deployment = FakeDeployConfig()

    (tmp_path / "exists.yaml").write_text("kind: LLMInferenceService")

    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    original = tc_mod._MANIFEST_DIR
    try:
        tc_mod._MANIFEST_DIR = tmp_path
        tc_mod._require_manifest(FakeTestCase())
    finally:
        tc_mod._MANIFEST_DIR = original


def test_new_testcase_script_generates_loadable_config(tmp_path, monkeypatch):
    """new-testcase.sh must produce a config YAML that load_testcase() can parse."""
    import subprocess

    import yaml

    script = Path(__file__).parent.parent / "scripts" / "new-testcase.sh"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "testcases").mkdir(parents=True)
    (tmp_path / "deploy" / "manifests").mkdir(parents=True)

    result = subprocess.run([str(script), "my-gen-test"], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    config_path = tmp_path / "configs" / "testcases" / "my-gen-test.yaml"
    assert config_path.exists()

    tc = load_testcase(str(config_path))
    assert tc.name == "my-gen-test"
    assert tc.deployment.manifest_path == "my-gen-test.yaml"
    assert tc.deployment.replicas == 1
    assert tc.deployment.resources.gpus == 1
    assert tc.validation.health_port == 8000
    assert tc.validation.test_prompts == ["What is 2+2?"]
    assert tc.validation.metrics_check.check_vllm is True
    assert tc.validation.metrics_check.check_scheduler is True
    assert tc.model.name == "Qwen/Qwen3-0.6B"

    manifest_path = tmp_path / "deploy" / "manifests" / "my-gen-test.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["kind"] == "LLMInferenceService"
    assert manifest["metadata"]["name"] == "my-gen-test"
    assert manifest["spec"]["replicas"] == 1


def test_wait_for_ready_surfaces_operator_image_pull_errors(monkeypatch):
    """When wait_for_ready hits a persistent error and operator pods have
    ImagePullBackOff, the error message should include the failing image."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

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
    """When wait_for_ready times out (no persistent error detected), the
    TimeoutError should still include operator ImagePullBackOff info."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

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


def test_load_lora_single_testcase():
    """LoRA single-adapter testcase YAML should parse correctly."""
    tc = load_testcase("configs/testcases/lora-single.yaml")
    assert tc.name == "lora-single"
    assert tc.model.lora is not None
    assert len(tc.model.lora.adapters) == 1
    assert tc.model.lora.adapters[0]["name"] == "sql-adapter"
    assert tc.model.lora.adapters[0]["uri"] == "hf://edbeeching/opt-125m-lora"
    assert tc.model.lora.max_adapters == 0
    assert tc.validation.metrics_check.check_lora is True


def test_load_lora_multi_testcase():
    """LoRA multi-adapter testcase YAML should parse with all adapters and settings."""
    tc = load_testcase("configs/testcases/lora-multi.yaml")
    assert tc.name == "lora-multi"
    assert tc.model.lora is not None
    assert len(tc.model.lora.adapters) == 2
    adapter_names = [a["name"] for a in tc.model.lora.adapters]
    assert "sql-adapter" in adapter_names
    assert "code-adapter" in adapter_names
    assert tc.model.lora.max_rank == 64
    assert tc.model.lora.max_adapters == 2


def test_load_testcase_without_lora():
    """Testcase YAML without LoRA should have lora=None."""
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    assert tc.model.lora is None


def test_chat_string_prompt_wraps_as_user_message(monkeypatch):
    """chat() with a plain string should wrap it as [{'role': 'user', 'content': ...}]."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        return FakeResp()

    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt="hello")
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    c.close()


def test_chat_list_prompt_passes_through(monkeypatch):
    """chat() with a list of message dicts should pass them through unmodified."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        return FakeResp()

    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt=msgs)
    assert captured["json"]["messages"] == msgs
    c.close()


def test_env_overrides_applied_to_decode_and_prefill():
    """_patch_manifest should inject env_overrides into main containers of both templates."""
    import yaml
    from conformance.deployer import Deployer
    from conformance.config import load_testcase
    from pathlib import Path
    import tempfile

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


def test_new_testcase_script_rejects_duplicate(tmp_path, monkeypatch):
    """new-testcase.sh must refuse to overwrite an existing config."""
    import subprocess

    script = Path(__file__).parent.parent / "scripts" / "new-testcase.sh"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "testcases").mkdir(parents=True)
    (tmp_path / "deploy" / "manifests").mkdir(parents=True)

    subprocess.run([str(script), "dupe-test"], capture_output=True, text=True)
    result = subprocess.run([str(script), "dupe-test"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "already exists" in result.stdout or "already exists" in result.stderr
