"""LLMInferenceService lifecycle via kubectl (no Python K8s client).

``Deployer`` applies manifests, waits for readiness, port-forwards gateway
and workload pods, and cleans up. Cluster I/O is always ``kubectl``
subprocess calls.

Deploy / patch:
  - Apply with retry for transient webhook / CRD-not-found errors
  - Manifest patching: model URI (hf→pvc), mock simulator image, LoRA
    adapters, pull secrets, ``--disable-auth``, env overrides, network
    attach, P/D node selectors, render sidecar
  - Pull-secret propagation from operator namespaces; gateway
    ``allowedRoutes`` so the test namespace is accepted
  - EPP metrics RBAC (``ClusterRoleBinding``) for authenticated scrape

Wait / diagnose:
  - Service, Gateway, pods, Ready condition; CrashLoopBackOff early fail
  - Persistent Ready reason+message fast-fail; operator ImagePullBackOff
    hints in ``OPERATOR_NAMESPACES``

Endpoints:
  - Gateway PF → inference (``get_endpoint``)
  - Direct pod PF → health / models (``get_pod_endpoint``)
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from conformance.config import TestCase

log = logging.getLogger(__name__)

WORKLOAD_LABEL = "app.kubernetes.io/name={name},app.kubernetes.io/component=llminferenceservice-workload"
PREFILL_LABEL = "app.kubernetes.io/name={name},app.kubernetes.io/component=llminferenceservice-workload-prefill"

OPERATOR_NAMESPACES = ("redhat-ods-applications", "redhat-ods-operator", "rhaii")
_IMAGE_PULL_FAILURE_REASONS = ("ImagePullBackOff", "ErrImagePull")


def _parse_node_selector(value: str) -> dict[str, str]:
    """Parse 'key=value' into a dict, or return empty dict."""
    if not value or "=" not in value:
        return {}
    k, v = value.split("=", 1)
    return {k.strip(): v.strip()}


@dataclass
class DeployResult:
    name: str = ""
    namespace: str = ""
    success: bool = False
    error: str = ""
    duration: float = 0.0
    logs: list[str] = field(default_factory=list)


class Deployer:
    """Manages deploy, wait, and cleanup of LLMInferenceService resources via kubectl."""

    def __init__(
        self,
        kubeconfig: str = "",
        platform: str = "any",
        namespace: str = "llm-conformance-test",
        model_source: str = "hf",
        mock_image: str = "",
        render_image: str = "",
        pull_secret: str = "",
        disable_auth: bool = False,
        manifest_dir: str = "deploy/manifests",
        decode_node_selector: str = "",
        prefill_node_selector: str = "",
    ):
        self.kubeconfig = kubeconfig
        self.platform = platform
        self.namespace = namespace
        self.model_source = model_source
        self.mock_image = mock_image
        self._render_image_override = render_image
        self.pull_secret = pull_secret
        self.disable_auth = disable_auth
        self.manifest_dir = Path(manifest_dir)
        self.decode_node_selector = _parse_node_selector(decode_node_selector)
        self.prefill_node_selector = _parse_node_selector(prefill_node_selector)
        self._port_forward_proc: subprocess.Popen | None = None
        self._port_forward_port: int = 0
        self._pod_pf_proc: subprocess.Popen | None = None
        self._pod_pf_port: int = 0
        self._pod_pf_name: str = ""
        self._deployed: set[str] = set()
        self._render_image_cached: str | None = None
        self._gpu_count: int | None = None

    @property
    def render_image(self) -> str:
        if self._render_image_override:
            return self._render_image_override
        if self._render_image_cached is None:
            self._render_image_cached = self._discover_render_image()
        return self._render_image_cached

    def _discover_render_image(self) -> str:
        """Return the default render image. Requires vLLM >= 0.19 for 'vllm launch render'."""
        default = "vllm/vllm-openai-cpu:v0.19.1"
        log.info("Using render image: %s", default)
        return default

    def kubectl(self, *args: str, check: bool = True) -> str:
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += list(args)
        log.debug("kubectl %s", " ".join(args))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if check and result.returncode != 0:
            raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def cluster_gpu_count(self) -> int:
        """Total allocatable nvidia.com/gpu across all cluster nodes (0 if none).

        Cached after the first query.
        """
        if self._gpu_count is not None:
            return self._gpu_count
        try:
            out = self.kubectl(
                "get",
                "nodes",
                "-o",
                r'jsonpath={range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}',
            )
        except RuntimeError as e:
            log.warning("Could not query node GPU capacity: %s", e)
            return 0
        total = 0
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                total += int(line)
            except ValueError:
                continue
        self._gpu_count = total
        return total

    def ensure_namespace(self):
        try:
            self.kubectl("get", "namespace", self.namespace, check=True)
        except RuntimeError:
            self.kubectl("create", "namespace", self.namespace)
        self.kubectl(
            "label",
            "namespace",
            self.namespace,
            "inference-gateway-access=true",
            "--overwrite",
        )

    def ensure_pull_secret(self, secret_name: str, source_namespaces: list[str] | None = None):
        """Copy a pull secret into the test namespace if it doesn't already exist."""
        try:
            self.kubectl("get", "secret", secret_name, "-n", self.namespace)
            return
        except RuntimeError:
            pass
        for ns in source_namespaces or ["rhaii", "redhat-ods-applications", "default"]:
            try:
                secret_json = self.kubectl("get", "secret", secret_name, "-n", ns, "-o", "json")
                if not secret_json:
                    continue
                secret = json.loads(secret_json)
                secret["metadata"] = {"name": secret_name, "namespace": self.namespace}
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump(secret, f)
                    tmp = f.name
                try:
                    self.kubectl("apply", "-f", tmp)
                    log.info("Copied pull secret %s from %s to %s", secret_name, ns, self.namespace)
                finally:
                    Path(tmp).unlink(missing_ok=True)
                return
            except RuntimeError:
                continue
        log.warning("Pull secret %s not found in any source namespace", secret_name)

    @staticmethod
    def _collect_pull_secrets(manifest: dict) -> list[str]:
        """Extract all imagePullSecret names referenced in a manifest."""
        names = set()
        spec = manifest.get("spec", {})
        for section_key in ("template", "prefill", "router"):
            section = spec.get(section_key, {})
            if isinstance(section, dict):
                for s in section.get("imagePullSecrets", []):
                    if s.get("name"):
                        names.add(s["name"])
                sub = section.get("scheduler", {}).get("template", {})
                for s in sub.get("imagePullSecrets", []):
                    if s.get("name"):
                        names.add(s["name"])
        return sorted(names)

    def ensure_metrics_rbac(self, name: str):
        """Bind the EPP service account to kserve-metrics-reader-cluster-role for metrics scraping."""
        sa_name = f"{name}-epp-sa"
        binding_name = f"{self.namespace}-{name}-metrics-reader"
        try:
            self.kubectl("get", "clusterrolebinding", binding_name, check=False)
            existing = self.kubectl(
                "get",
                "clusterrolebinding",
                binding_name,
                "-o",
                "jsonpath={.metadata.name}",
                check=False,
            )
            if existing:
                return
        except RuntimeError:
            pass
        try:
            self.kubectl(
                "create",
                "clusterrolebinding",
                binding_name,
                "--clusterrole=kserve-metrics-reader-cluster-role",
                f"--serviceaccount={self.namespace}:{sa_name}",
            )
            log.info("Created metrics RBAC binding %s for %s", binding_name, sa_name)
        except RuntimeError as e:
            log.warning("Could not create metrics RBAC binding: %s", e)

    def cleanup_metrics_rbac(self, name: str):
        binding_name = f"{self.namespace}-{name}-metrics-reader"
        self.kubectl("delete", "clusterrolebinding", binding_name, "--ignore-not-found", check=False)

    def is_deployed(self, name: str) -> bool:
        return name in self._deployed

    def check_crd_exists(self, crd_name: str) -> bool:
        try:
            self.kubectl("get", "crd", crd_name)
            return True
        except RuntimeError:
            return False

    def check_resource_exists(self, kind: str, name: str) -> bool:
        try:
            self.kubectl("get", kind, name, "-n", self.namespace)
            return True
        except RuntimeError:
            return False

    def _ensure_clean_slate(self, name: str, timeout: float = 120):
        """If the LLMInferenceService already exists, delete it and wait for all pods to terminate."""
        if not self.check_resource_exists("llminferenceservice", name):
            return
        log.info("LLMInferenceService '%s' already exists, deleting before redeploy", name)
        self.cleanup_metrics_rbac(name)
        self.kubectl(
            "delete",
            "llminferenceservice",
            name,
            "-n",
            self.namespace,
            "--timeout",
            f"{int(timeout)}s",
            "--ignore-not-found",
            check=False,
        )
        label = f"app.kubernetes.io/name={name}"
        deadline = time.time() + timeout
        while time.time() < deadline:
            output = self.kubectl(
                "get",
                "pods",
                "-n",
                self.namespace,
                "-l",
                label,
                "-o",
                "jsonpath={.items[*].metadata.name}",
                check=False,
            )
            if not output.strip():
                log.info("All pods for '%s' terminated", name)
                return
            log.info("Waiting for pods to terminate: %s", output.strip())
            time.sleep(5)
        log.warning("Timed out waiting for pods to terminate for '%s'", name)

    # Connectivity substrings seen in kubectl apply errors when the LLMInferenceService
    # admission webhook is registered but its backing pod isn't serving yet (e.g. right
    # after a fresh install or upgrade). Only checked when the error also mentions
    # "webhook", so unrelated transient errors (API server overload, a manifest field
    # that happens to contain "eof", etc.) aren't mistakenly retried under this path.
    _WEBHOOK_CONNECTIVITY_MARKERS = (
        "no endpoints available",
        "connection refused",
        "context deadline exceeded",
        "eof",
    )

    # Substrings seen when kubectl apply targets a CRD/API version that isn't
    # registered on the server yet (e.g. an upgrade race between installing new
    # CRDs and applying resources that use them, or a stale client-side API
    # discovery cache). Distinct from webhook readiness: these phrases are
    # specific enough on their own and don't require a "webhook" anchor.
    _CRD_NOT_REGISTERED_MARKERS = (
        "the server could not find the requested resource",
        "no matches for kind",
    )

    @classmethod
    def _is_webhook_not_ready_error(cls, error: str) -> bool:
        lowered = error.lower()
        return "webhook" in lowered and any(m in lowered for m in cls._WEBHOOK_CONNECTIVITY_MARKERS)

    @classmethod
    def _is_crd_not_registered_error(cls, error: str) -> bool:
        lowered = error.lower()
        return any(m in lowered for m in cls._CRD_NOT_REGISTERED_MARKERS)

    @classmethod
    def _is_transient_apply_error(cls, error: str) -> bool:
        return cls._is_webhook_not_ready_error(error) or cls._is_crd_not_registered_error(error)

    def _apply_with_webhook_retry(self, tmp_path: str, timeout: float = 120, interval: float = 10) -> None:
        """Apply a manifest, retrying on known transient post-upgrade races:
        the admission webhook not serving yet, or the target CRD/API version
        not registered on the server yet.

        Always attempts at least once, regardless of `timeout`.
        """
        deadline = time.time() + timeout
        while True:
            try:
                self.kubectl("apply", "-n", self.namespace, "-f", tmp_path)
                return
            except RuntimeError as e:
                last_error = str(e)
                if not self._is_transient_apply_error(last_error):
                    raise
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"kubectl apply failed after {timeout}s waiting for webhook/CRD readiness: {last_error}"
                    ) from e
                log.warning("Webhook/CRD not ready yet, retrying apply: %s", last_error)
                time.sleep(interval)

    def deploy(self, tc: TestCase) -> DeployResult:
        start = time.time()
        result = DeployResult(name=tc.name, namespace=self.namespace)

        manifest_path = self.manifest_dir / tc.deployment.manifest_path
        if not manifest_path.exists():
            result.error = f"Manifest not found: {manifest_path}"
            return result

        manifest = self._patch_manifest(manifest_path, tc)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(manifest, f)
            tmp_path = f.name

        try:
            self.ensure_namespace()
            self._ensure_clean_slate(tc.name)
            for secret_name in self._collect_pull_secrets(manifest):
                self.ensure_pull_secret(secret_name)
            self._apply_with_webhook_retry(tmp_path)
            self.ensure_metrics_rbac(tc.name)
            result.success = True
            self._deployed.add(tc.name)
        except RuntimeError as e:
            result.error = str(e)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            result.duration = time.time() - start

        return result

    def wait_for_ready(self, tc: TestCase, timeout: float | None = None, print_fn=None) -> bool:
        if timeout is None:
            timeout = tc.deployment.ready_timeout.total_seconds()
        deadline = time.time() + timeout
        name = tc.name
        start = time.time()
        crashloop_count = 0
        # Track any persistent Ready=False condition with a stable reason+message
        _prev_error_key: str | None = None
        _error_repeat_count = 0
        last_reason = ""
        last_message = ""

        while time.time() < deadline:
            elapsed = int(time.time() - start)
            try:
                status = self.kubectl(
                    "get",
                    "llminferenceservice",
                    name,
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.status.conditions[?(@.type=='Ready')].status}",
                    check=False,
                )
                reason = (
                    self.kubectl(
                        "get",
                        "llminferenceservice",
                        name,
                        "-n",
                        self.namespace,
                        "-o",
                        "jsonpath={.status.conditions[?(@.type=='Ready')].reason}",
                        check=False,
                    )
                    or "waiting"
                )
                message = (
                    self.kubectl(
                        "get",
                        "llminferenceservice",
                        name,
                        "-n",
                        self.namespace,
                        "-o",
                        "jsonpath={.status.conditions[?(@.type=='Ready')].message}",
                        check=False,
                    )
                    or ""
                )
                last_reason = reason
                last_message = message
                if print_fn:
                    log_line = f"[{elapsed}s/{int(timeout)}s] Ready={status or 'Unknown'} reason={reason}"
                    if message:
                        log_line += f" message={message}"
                    print_fn(log_line)
                if status == "True":
                    return True

                # Fail fast on any persistent controller error: if the same
                # non-empty reason+message pair repeats across 3 consecutive
                # polls (~45s), the error is unlikely to self-heal (RBAC,
                # missing CRD, webhook misconfiguration, etc.).
                if status == "False" and reason != "waiting" and message:
                    error_key = f"{reason}:{message}"
                    if error_key == _prev_error_key:
                        _error_repeat_count += 1
                    else:
                        _prev_error_key = error_key
                        _error_repeat_count = 1
                    if _error_repeat_count >= 3:
                        detail = f"reason={reason}: {message}"
                        image_issues = self._check_operator_image_issues()
                        if image_issues:
                            detail += "\nOperator image pull failures:\n  " + "\n  ".join(image_issues)
                        raise RuntimeError(f"Persistent controller error for {name}: {detail}")
                else:
                    _prev_error_key = None
                    _error_repeat_count = 0

                if print_fn and _error_repeat_count == 1:
                    image_issues = self._check_operator_image_issues()
                    for issue in image_issues:
                        print_fn(f"WARNING: {issue}")

                crash_pods = self._check_crashloop(name)
                if crash_pods:
                    crashloop_count += 1
                    if print_fn:
                        print_fn(f"CrashLoopBackOff detected: {', '.join(crash_pods)}")
                    if crashloop_count >= 3:
                        raise RuntimeError(f"Pods in CrashLoopBackOff for {name}: {', '.join(crash_pods)}")
                else:
                    crashloop_count = 0
            except RuntimeError:
                raise
            except Exception:
                if print_fn:
                    print_fn(f"[{elapsed}s/{int(timeout)}s] resource not found yet")
            time.sleep(15)

        detail = f"last reason={last_reason}"
        if last_message:
            detail += f": {last_message}"
        image_issues = self._check_operator_image_issues()
        if image_issues:
            detail += "\nOperator image pull failures:\n  " + "\n  ".join(image_issues)
        raise TimeoutError(f"{name} not ready after {timeout}s ({detail})")

    def _check_crashloop(self, name: str) -> list[str]:
        """Return pod names in CrashLoopBackOff for this LLMInferenceService."""
        output = self.kubectl(
            "get",
            "pods",
            "-n",
            self.namespace,
            "-l",
            f"app.kubernetes.io/name={name}",
            "-o",
            "jsonpath={range .items[*]}{.metadata.name}={.status.containerStatuses[*].state.waiting.reason} {end}",
            check=False,
        )
        crash_pods = []
        for entry in output.strip().split() if output.strip() else []:
            parts = entry.split("=", 1)
            if len(parts) == 2 and "CrashLoopBackOff" in parts[1]:
                crash_pods.append(parts[0])
        return crash_pods

    def _check_operator_image_issues(self) -> list[str]:
        """Scan operator namespaces for pods stuck in ImagePullBackOff/ErrImagePull."""
        issues = []
        for ns in OPERATOR_NAMESPACES:
            output = self.kubectl(
                "get",
                "pods",
                "-n",
                ns,
                "-o",
                "jsonpath={range .items[*]}{.metadata.name}|"
                "{range .status.containerStatuses[*]}{.state.waiting.reason}={.image} {end}|"
                "{range .status.initContainerStatuses[*]}{.state.waiting.reason}={.image} {end}"
                "\\n{end}",
                check=False,
            )
            for line in (output or "").strip().splitlines():
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                pod_name = parts[0].strip()
                for container_info in " ".join(parts[1:]).split():
                    for reason in _IMAGE_PULL_FAILURE_REASONS:
                        if container_info.startswith(f"{reason}="):
                            image = container_info.split("=", 1)[1]
                            issues.append(f"{ns}/{pod_name}: {reason} ({image})")
        return issues

    def wait_for_service(self, name: str, timeout: float = 300) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                output = self.kubectl(
                    "get",
                    "svc",
                    "-n",
                    self.namespace,
                    "-l",
                    f"app.kubernetes.io/name={name}",
                    "-o",
                    "jsonpath={.items[0].metadata.name}",
                    check=False,
                )
                if output:
                    return output
            except RuntimeError:
                pass
            time.sleep(10)
        raise TimeoutError(f"Service for {name} not found after {timeout}s")

    def wait_for_httproute(self, name: str, timeout: float = 300) -> bool:
        deadline = time.time() + timeout
        route_name = f"{name}-kserve-route"
        while time.time() < deadline:
            if self.check_resource_exists("httproute", route_name):
                return True
            time.sleep(10)
        raise TimeoutError(f"HTTPRoute {route_name} not found after {timeout}s")

    def wait_for_gateway(self, timeout: float = 300) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                output = self.kubectl(
                    "get",
                    "gateway",
                    "-A",
                    "-o",
                    "jsonpath={.items[0].status.addresses[0].value}",
                    check=False,
                )
                if output:
                    return output
            except RuntimeError:
                pass
            time.sleep(10)
        raise TimeoutError(f"Gateway not programmed after {timeout}s")

    def wait_for_pods(self, name: str, timeout: float = 600, print_fn=None) -> list[str]:
        label = f"app.kubernetes.io/name={name}"
        deadline = time.time() + timeout
        start = time.time()
        crashloop_count = 0
        while time.time() < deadline:
            elapsed = int(time.time() - start)
            try:
                output = self.kubectl(
                    "get",
                    "pods",
                    "-n",
                    self.namespace,
                    "-l",
                    label,
                    "-o",
                    "jsonpath={range .items[*]}{.metadata.name}={.status.phase} {end}",
                    check=False,
                )
                pod_statuses = output.strip().split() if output.strip() else []
                if print_fn and pod_statuses:
                    print_fn(f"[{elapsed}s/{int(timeout)}s] pods: {', '.join(pod_statuses)}")
                elif print_fn:
                    print_fn(f"[{elapsed}s/{int(timeout)}s] no pods found yet")

                pods = []
                all_running = True
                for ps in pod_statuses:
                    parts = ps.split("=")
                    if len(parts) == 2:
                        pods.append(parts[0])
                        if parts[1] != "Running":
                            all_running = False
                if pods and all_running:
                    return pods

                crash_pods = self._check_crashloop(name)
                if crash_pods:
                    crashloop_count += 1
                    if print_fn:
                        print_fn(f"CrashLoopBackOff detected: {', '.join(crash_pods)}")
                    if crashloop_count >= 3:
                        raise RuntimeError(f"Pods in CrashLoopBackOff for {name}: {', '.join(crash_pods)}")
                else:
                    crashloop_count = 0
            except RuntimeError:
                raise
            except Exception:
                if print_fn:
                    print_fn(f"[{elapsed}s/{int(timeout)}s] waiting for pods...")
            time.sleep(15)
        raise TimeoutError(f"Pods for {name} not running after {timeout}s")

    def wait_for_inference_pool(self, name: str, timeout: float = 300) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                output = self.kubectl(
                    "get",
                    "inferencepool",
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath={.items[*].metadata.name}",
                    check=False,
                )
                if output:
                    return True
            except RuntimeError:
                pass
            time.sleep(10)
        raise TimeoutError(f"InferencePool for {name} not found after {timeout}s")

    def get_endpoint(self, name: str) -> str:
        try:
            url = self.kubectl(
                "get",
                "llminferenceservice",
                name,
                "-n",
                self.namespace,
                "-o",
                "jsonpath={.status.url}",
            )
            if url:
                path = url.split("//", 1)[-1].split("/", 1)
                path_suffix = f"/{path[1]}" if len(path) > 1 else f"/{self.namespace}/{name}"
                local_url = self._ensure_port_forward(path_suffix)
                if local_url:
                    return local_url
                return url
        except RuntimeError:
            pass
        gateway_addr = self.wait_for_gateway()
        return f"http://{gateway_addr}/{self.namespace}/{name}"

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _ensure_port_forward(self, path: str) -> str | None:
        if self._port_forward_proc and self._port_forward_proc.poll() is None:
            return f"http://localhost:{self._port_forward_port}{path}"

        local_port = self._find_free_port()
        gateway_svc = "svc/inference-gateway-istio"
        gateway_ns = "redhat-ods-applications"

        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["port-forward", "-n", gateway_ns, gateway_svc, f"{local_port}:80"]

        log.info("Starting port-forward: localhost:%d → %s:80", local_port, gateway_svc)
        self._port_forward_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._port_forward_port = local_port

        time.sleep(3)
        if self._port_forward_proc.poll() is not None:
            log.warning("Port-forward failed to start")
            self._port_forward_proc = None
            return None

        return f"http://localhost:{local_port}{path}"

    def get_pod_endpoint(self, name: str) -> str:
        """Port-forward directly to a workload pod, bypassing the gateway/EPP."""
        if self._pod_pf_proc and self._pod_pf_proc.poll() is None:
            if self._pod_pf_name == name:
                return f"https://localhost:{self._pod_pf_port}"
            self._stop_pod_port_forward()

        label = WORKLOAD_LABEL.format(name=name)
        output = self.kubectl(
            "get",
            "pods",
            "-n",
            self.namespace,
            "-l",
            label,
            "-o",
            "jsonpath={.items[0].metadata.name}",
        )
        pod_name = output.strip()
        if not pod_name:
            raise RuntimeError(f"No workload pod found for {name}")

        local_port = self._find_free_port()
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["port-forward", "-n", self.namespace, pod_name, f"{local_port}:8000"]

        log.info("Starting pod port-forward: localhost:%d → %s:8000", local_port, pod_name)
        self._pod_pf_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._pod_pf_port = local_port
        self._pod_pf_name = name

        time.sleep(3)
        if self._pod_pf_proc.poll() is not None:
            self._pod_pf_proc = None
            raise RuntimeError(f"Pod port-forward to {pod_name} failed to start")

        return f"https://localhost:{local_port}"

    def _stop_pod_port_forward(self):
        if self._pod_pf_proc:
            self._pod_pf_proc.terminate()
            self._pod_pf_proc.wait(timeout=5)
            self._pod_pf_proc = None
            self._pod_pf_port = 0
            self._pod_pf_name = ""

    def stop_port_forward(self):
        if self._port_forward_proc:
            self._port_forward_proc.terminate()
            self._port_forward_proc.wait(timeout=5)
            self._port_forward_proc = None
        self._stop_pod_port_forward()
        log.info("Port-forwards stopped")

    def cleanup(self, tc: TestCase, timeout: float = 120):
        log.info("Cleaning up %s", tc.name)
        self._deployed.discard(tc.name)
        self.cleanup_metrics_rbac(tc.name)
        self.kubectl(
            "delete",
            "llminferenceservice",
            tc.name,
            "-n",
            self.namespace,
            "--timeout",
            f"{int(timeout)}s",
            "--ignore-not-found",
            check=False,
        )
        deadline = time.time() + timeout
        label = f"app.kubernetes.io/name={tc.name}"
        while time.time() < deadline:
            output = self.kubectl(
                "get",
                "pods",
                "-n",
                self.namespace,
                "-l",
                label,
                "-o",
                "jsonpath={.items[*].metadata.name}",
                check=False,
            )
            if not output.strip():
                return
            time.sleep(5)

    def get_platform_info(self) -> dict:
        info = {"platform": self.platform}
        try:
            info["k8s_version"] = self.kubectl("version", "--short", "--client", check=False)
        except RuntimeError:
            pass
        return info

    def _patch_manifest(self, path: Path, tc: TestCase) -> dict:
        with open(path) as f:
            manifest = yaml.safe_load(f)

        manifest.setdefault("metadata", {})["name"] = tc.name

        spec = manifest.get("spec", {})

        if self.mock_image:
            model = spec.setdefault("model", {})
            model["name"] = tc.model.name
            model["uri"] = tc.model.uri
            self._inject_lora_spec(model, tc.model.lora)
            self._replace_vllm_image(spec, self.mock_image, tc.model.name, lora=tc.model.lora)
        elif tc.model.uri:
            model = spec.setdefault("model", {})
            model["uri"] = tc.model.uri
            model["name"] = tc.model.name
            self._inject_lora_spec(model, tc.model.lora)

        if self.pull_secret:
            self._inject_pull_secret(spec, self.pull_secret)

        if self.disable_auth:
            annotations = manifest.setdefault("metadata", {}).setdefault("annotations", {})
            annotations["serving.kserve.io/disable-auth"] = "true"

        if self.decode_node_selector:
            spec.setdefault("template", {})["nodeSelector"] = self.decode_node_selector

        if self.prefill_node_selector:
            prefill = spec.get("prefill", {})
            if prefill:
                prefill.setdefault("template", {})["nodeSelector"] = self.prefill_node_selector

        if tc.deployment.env_overrides:
            for container in self._main_containers(spec):
                env_list = container.setdefault("env", [])
                existing = {e.get("name"): i for i, e in enumerate(env_list) if e.get("name")}
                for k, v in tc.deployment.env_overrides.items():
                    entry = {"name": k, "value": v}
                    if k in existing:
                        env_list[existing[k]] = entry
                    else:
                        env_list.append(entry)

        return manifest

    @staticmethod
    def _pod_templates(spec: dict) -> list[dict]:
        """Return pod template dicts for both decode (spec.template) and prefill (spec.prefill.template)."""
        templates = [spec.get("template", {})]
        prefill_tmpl = spec.get("prefill", {}).get("template", {})
        if prefill_tmpl:
            templates.append(prefill_tmpl)
        return templates

    @staticmethod
    def _main_containers(spec: dict) -> list[dict]:
        """Return the 'main' container dict from each pod template."""
        return [c for t in Deployer._pod_templates(spec) for c in t.get("containers", []) if c.get("name") == "main"]

    @staticmethod
    def _inject_lora_spec(model: dict, lora) -> None:
        """Inject spec.model.lora block from LoRAConfig into the manifest."""
        if not lora or not lora.adapters:
            return
        lora_spec: dict = {"adapters": lora.adapters}
        if lora.max_rank:
            lora_spec["maxRank"] = lora.max_rank
        if lora.max_adapters:
            lora_spec["maxAdapters"] = lora.max_adapters
        model["lora"] = lora_spec

    def _replace_vllm_image(self, spec: dict, image: str, model_name: str = "", lora=None):
        for container in self._main_containers(spec):
            container["image"] = image
            container["command"] = ["/app/llm-d-inference-sim"]
            sim_model = "sim-model"
            container["args"] = [
                "--model",
                sim_model,
                "--served-model-name",
                model_name or sim_model,
                "--port",
                "8000",
                "--self-signed-certs",
                "--mode",
                "random",
                "--enable-kvcache",
                "true",
            ]
            if lora and lora.adapters:
                container["args"].append("--lora-modules")
                for adapter in lora.adapters:
                    container["args"].append(
                        json.dumps({"name": adapter["name"], "path": f"/fake/lora/{adapter['name']}"})
                    )
                if lora.max_adapters:
                    container["args"].extend(["--max-loras", str(lora.max_adapters)])
            env_list = container.setdefault("env", [])
            if not any(e.get("name") == "POD_IP" for e in env_list):
                env_list.append(
                    {
                        "name": "POD_IP",
                        "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
                    }
                )
            resources = container.get("resources", {})
            for section in ("limits", "requests"):
                resources.get(section, {}).pop("nvidia.com/gpu", None)

    def _inject_pull_secret(self, spec: dict, secret_name: str):
        for template in self._pod_templates(spec):
            if template:
                secrets = template.setdefault("imagePullSecrets", [])
                if not any(s.get("name") == secret_name for s in secrets):
                    secrets.append({"name": secret_name})
