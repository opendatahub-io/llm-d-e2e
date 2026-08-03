#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-}"
if [ -z "$NAME" ]; then
    echo "Usage: $0 <test-case-name>"
    echo "Example: $0 my-new-test"
    exit 1
fi

CONFIG="configs/testcases/${NAME}.yaml"
MANIFEST="deploy/manifests/${NAME}.yaml"

if [ -f "$CONFIG" ]; then
    echo "ERROR: $CONFIG already exists"
    exit 1
fi

cat > "$CONFIG" << EOF
name: ${NAME}
description: "TODO: describe what this test validates"
model:
  name: Qwen/Qwen3-0.6B
  uri: hf://Qwen/Qwen3-0.6B
  displayName: qwen3-0.6b-${NAME}
  category: single-node-gpu
  cache:
    enabled: true
    storageSize: 10Gi
    keepPVC: true
    timeout: 15m
deployment:
  manifestPath: ${NAME}.yaml
  replicas: 1
  readyTimeout: 10m
  resources:
    cpu: "2"
    memory: 8Gi
    gpus: 1
    rdma: false
validation:
  healthEndpoint: /health
  healthPort: 8000
  healthScheme: HTTPS
  inferenceCheck: true
  testPrompts:
    - "What is 2+2?"
  expectedCodes: [200]
  timeout: 2m
  retryAttempts: 3
  retryInterval: 15s
  metricsCheck:
    enabled: true
    checkVLLM: true
    checkScheduler: true
    # Uncomment as needed:
    # checkEPP: true
    # checkPrefixCache: true
    # checkFlowControl: true
    # checkPD: true
    # checkNIXL: true
cleanup: true
EOF

cat > "$MANIFEST" << EOF
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: ${NAME}
spec:
  model:
    uri: hf://Qwen/Qwen3-0.6B
    name: Qwen/Qwen3-0.6B
  replicas: 1
  router:
    scheduler:
      template:
        imagePullSecrets:
        - name: rhai-pull-secret
        containers:
        - name: main
    route: {}
    gateway: {}
  template:
    imagePullSecrets:
    - name: rhai-pull-secret
    containers:
      - name: main
        resources:
          limits:
            cpu: '2'
            memory: 8Gi
            nvidia.com/gpu: "1"
          requests:
            cpu: '1'
            memory: 4Gi
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
            scheme: HTTPS
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 30
          failureThreshold: 5
EOF

echo "Created:"
echo "  $CONFIG"
echo "  $MANIFEST (template — move to conformance-manifests repo when ready)"
echo ""
echo "Next steps:"
echo "  1. Edit $CONFIG — update description, model, resources, metricsCheck flags"
echo "  2. Edit $MANIFEST — add scheduler/EPP config for your topology"
echo "  3. Move $MANIFEST to the conformance-manifests repo (correct branch)"
echo "  4. Run: uv run llm-d-e2e --setup <branch>"
echo "  5. Run: uv run llm-d-e2e -t ${NAME} --mock -v"
