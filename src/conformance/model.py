"""HuggingFace model download into a PVC for pre-cached test runs.

Used when ``--model-source pvc`` or ``--mode cache``. ``ModelDownloader``
creates a PVC (optional ``storageClass`` / size override) and a Job that
runs ``huggingface-cli download`` into ``/models``. If ``cache.keepPVC``
is true and the PVC already exists, the download is skipped.

``pvc_uri()`` returns ``pvc://<name>/models`` so the deployer can rewrite
the LLMInferenceService model URI from ``hf://`` to the cached PVC.
``CacheResult.status`` is one of: ready, downloading, failed, not_found.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from conformance.config import TestCase

log = logging.getLogger(__name__)

JOB_TEMPLATE = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: model-cache-{safe_name}
  namespace: {namespace}
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: download
          image: python:3.11-slim
          command: ["sh", "-c"]
          args:
            - |
              pip install -q huggingface-hub &&
              huggingface-cli download {model_name} --local-dir /models
          volumeMounts:
            - name: model-storage
              mountPath: /models
          resources:
            requests:
              cpu: "1"
              memory: 4Gi
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: {pvc_name}
"""

PVC_TEMPLATE = """\
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {pvc_name}
  namespace: {namespace}
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: {storage_size}
"""


@dataclass
class CacheResult:
    model_name: str
    pvc_name: str
    status: str  # "ready", "downloading", "failed", "not_found"
    duration: float = 0.0
    error: str = ""


class ModelDownloader:
    """Downloads models to PVCs for pre-cached test runs."""

    def __init__(self, kubectl_fn, namespace: str, storage_class: str = "", storage_size: str = ""):
        self._kubectl = kubectl_fn
        self.namespace = namespace
        self.storage_class = storage_class
        self.storage_size_override = storage_size

    def download(self, tc: TestCase) -> CacheResult:
        cache = tc.model.cache
        if not cache.enabled:
            return CacheResult(tc.model.name, "", "not_found")

        pvc_name = cache.pvc_name or f"model-{tc.name}"
        storage_size = self.storage_size_override or cache.storage_size
        safe_name = tc.name.replace("_", "-")

        start = time.time()
        result = CacheResult(tc.model.name, pvc_name, "downloading")

        try:
            existing = self._kubectl(
                "get",
                "pvc",
                pvc_name,
                "-n",
                self.namespace,
                "-o",
                "jsonpath={.metadata.name}",
            )
            if existing and cache.keep_pvc:
                log.info("PVC %s already exists, reusing", pvc_name)
                result.status = "ready"
                result.duration = time.time() - start
                return result
        except RuntimeError:
            pass

        pvc_yaml = PVC_TEMPLATE.format(pvc_name=pvc_name, namespace=self.namespace, storage_size=storage_size)
        self._kubectl("apply", "-f", "-", input_data=pvc_yaml)

        job_yaml = JOB_TEMPLATE.format(
            safe_name=safe_name,
            namespace=self.namespace,
            model_name=tc.model.name,
            pvc_name=pvc_name,
        )
        self._kubectl("apply", "-f", "-", input_data=job_yaml)

        timeout = cache.timeout.total_seconds()
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._kubectl(
                "get",
                "job",
                f"model-cache-{safe_name}",
                "-n",
                self.namespace,
                "-o",
                "jsonpath={.status.conditions[0].type}",
            )
            if status == "Complete":
                result.status = "ready"
                break
            elif status == "Failed":
                result.status = "failed"
                result.error = "Job failed"
                break
            time.sleep(30)
        else:
            result.status = "failed"
            result.error = f"Timed out after {timeout}s"

        result.duration = time.time() - start
        return result

    def pvc_uri(self, tc: TestCase) -> str:
        pvc_name = tc.model.cache.pvc_name or f"model-{tc.name}"
        return f"pvc://{pvc_name}/models"
