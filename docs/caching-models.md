# Caching Models

Pre-download models into a PVC to skip the storage-initializer download on every test case. One shared PVC per model — all test cases using the same model share it.

## Quick Start

```bash
# 1. Cache the default model (Qwen/Qwen3-0.6B)
uv run llm-d-e2e --cache-model

# 2. Run tests using the cached model
uv run llm-d-e2e -p configs/profiles/3.5.yaml --model-source pvc -v
```

## How It Works

### Without caching (default)

Each test case deploys an LLMInferenceService with `hf://Qwen/Qwen3-0.6B`. The storage-initializer init container downloads the model from HuggingFace on every pod start. For a 9-test-case profile, that's 9 separate downloads of the same model (~2-3 min each).

### With caching

1. `--cache-model` creates a PVC named `model-qwen-qwen3-0-6b` and runs a Job to download the model once
2. `--model-source pvc` patches each LLMInferenceService manifest to use `pvc://model-qwen-qwen3-0-6b/` instead of `hf://Qwen/Qwen3-0.6B`
3. The storage-initializer mounts the PVC directly — no download

### PVC naming

PVC names are derived from the model name, not the test case name. All test cases using the same model share one PVC.

| Model | PVC name |
|-------|----------|
| `Qwen/Qwen3-0.6B` | `model-qwen-qwen3-0-6b` |

> **Note:** `--cache-model` caches the default model used by all test case configs (Qwen/Qwen3-0.6B). To test with a different model, update the test case config YAMLs to match.

## Commands

### Cache a model

```bash
# Cache the default model (Qwen/Qwen3-0.6B — matches all test case configs)
uv run llm-d-e2e --cache-model

# With custom storage class (e.g. Azure Files for multi-node)
uv run llm-d-e2e --cache-model --storage-class azurefile-csi

# With custom storage size
uv run llm-d-e2e --cache-model --storage-size 50Gi

# With custom namespace
uv run llm-d-e2e --cache-model -n my-namespace
```

### Run tests with cached model

```bash
# Single test case
uv run llm-d-e2e -t single-gpu --model-source pvc -v

# Full profile
uv run llm-d-e2e -p configs/profiles/3.5.yaml --model-source pvc -v

# With mock (requiresGpu tests auto-skip)
uv run llm-d-e2e -p configs/profiles/3.5.yaml --model-source pvc --mock -v
```

### Check PVC status

```bash
kubectl get pvc -n llm-conformance-test
```

### Delete cached model

```bash
kubectl delete pvc model-qwen-qwen3-0-6b -n llm-conformance-test
```

## Multi-Node Clusters

The default PVC uses `ReadWriteOnce` (RWO) — it binds to one node. On multi-node GPU clusters, pods on other nodes can't mount it.

### Platform-specific storage classes

#### AKS (Azure)

```bash
# Azure Managed Disk (default) — RWO, single node only
uv run llm-d-e2e --cache-model --storage-class managed-csi

# Azure Files — RWX, all nodes can mount (recommended for multi-node)
uv run llm-d-e2e --cache-model --storage-class azurefile-csi

# Azure Files Premium — RWX, faster I/O
uv run llm-d-e2e --cache-model --storage-class azurefile-csi-premium
```

#### EKS (AWS)

```bash
# EBS gp3 (default) — RWO, single node only
uv run llm-d-e2e --cache-model --storage-class gp3

# EFS — RWX, all nodes can mount (requires EFS CSI driver + filesystem)
uv run llm-d-e2e --cache-model --storage-class efs-sc

# Local NVMe (instance store) — fastest, node-local only
uv run llm-d-e2e --cache-model --storage-class local-storage
```

EFS setup requires the [EFS CSI driver](https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html) and a pre-created EFS filesystem. Create the StorageClass:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-0123456789abcdef0
  directoryPerms: "700"
```

#### CoreWeave

```bash
# HDD shared storage — RWX, all nodes
uv run llm-d-e2e --cache-model --storage-class shared-hdd-ord1

# SSD shared storage — RWX, faster
uv run llm-d-e2e --cache-model --storage-class shared-ssd-ord1

# Local NVMe — fastest, node-local only
uv run llm-d-e2e --cache-model --storage-class local-nvme-ord1
```

#### OpenShift (OCP)

```bash
# ODF (OpenShift Data Foundation) CephFS — RWX
uv run llm-d-e2e --cache-model --storage-class ocs-storagecluster-cephfs

# ODF RBD — RWO, block storage
uv run llm-d-e2e --cache-model --storage-class ocs-storagecluster-ceph-rbd

# NFS (if available)
uv run llm-d-e2e --cache-model --storage-class nfs-storage
```

### Choosing the right storage class

| Scenario | Recommended | Access mode |
|----------|-------------|-------------|
| Single GPU node | Default (no flag) | RWO |
| Multi-node, same model | Azure Files / EFS / CephFS | RWX |
| Multi-node, fastest I/O | Local NVMe + cache per node | RWO (per node) |
| CI pipeline | Default or Azure Files | RWO or RWX |
| Production | KServe `LocalModelCache` CRD | Per-node local storage |

For CI and testing, RWX storage (Azure Files, EFS, CephFS) is simplest — one cache, all nodes can mount it. For production, use KServe's `LocalModelCache` which pre-caches models to local NVMe on each node via a DaemonSet agent.

### List available storage classes

```bash
kubectl get storageclass
```

## Troubleshooting

### Pod crashes with "Invalid repository ID or local directory"

The model files aren't at the expected path. Check PVC contents:

```bash
kubectl run pvc-check --rm -it --restart=Never --image=busybox \
  -n llm-conformance-test \
  --overrides='{"spec":{"containers":[{"name":"c","image":"busybox","command":["ls","-la","/data"],"volumeMounts":[{"mountPath":"/data","name":"m"}]}],"volumes":[{"name":"m","persistentVolumeClaim":{"claimName":"model-qwen-qwen3-0-6b"}}]}}'
```

You should see `config.json`, `*.safetensors`, etc. at the root.

### PVC stuck Pending

Check storage class and capacity:

```bash
kubectl describe pvc model-qwen-qwen3-0-6b -n llm-conformance-test
```

### Pod can't mount PVC (multi-node)

The PVC is RWO and bound to a different node. Use `--storage-class` with an RWX-capable storage class, or delete the PVC and re-cache.
