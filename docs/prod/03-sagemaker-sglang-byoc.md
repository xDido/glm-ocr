# 03 — SageMaker BYOC container for SGLang

**Purpose:** how to package `lmsysorg/sglang` as a SageMaker-compatible real-time endpoint container, and how to stage the GLM-OCR weights in S3.

---

## SageMaker contract (what the platform expects)

A SageMaker real-time endpoint launches a container and expects:

| Endpoint | Method | Purpose | Must respond |
|---|---|---|---|
| `/ping` | GET | health check | `200 OK`, empty body, within the health-check timeout (default 2 s) |
| `/invocations` | POST | inference | response in whatever format the client agreed to |

Both on port **8080** by default. The container's environment has `SAGEMAKER_*` variables, and the model artifacts (if any) are mounted at `/opt/ml/model/`.

Our CPU-side sidecar (see `05-sigv4-proxy-sidecar.md`) sends the **original glmocr OCR request body** — an OpenAI `/v1/chat/completions` payload — verbatim as the `InvokeEndpoint` body. So `/invocations` must accept exactly that shape and return exactly what OpenAI-compatible `/v1/chat/completions` returns.

---

## Directory layout: `docker/sglang/`

```
docker/sglang/
├── Dockerfile
├── serve.py
├── requirements.txt
└── entrypoint.sh           ← boots SGLang on :30000, boots serve.py on :8080
```

---

## Dockerfile

```dockerfile
# Pin the SGLang version explicitly — don't track `latest` in prod.
# Bump on purpose, not on surprise. The dev repo used whatever `latest` was
# at 2026-04; pin to that concrete tag on first bake.
ARG SGL_IMAGE_TAG=v0.5.10.post1
FROM lmsysorg/sglang:${SGL_IMAGE_TAG}

# SageMaker expects port 8080 and /ping /invocations on it.
EXPOSE 8080

# serve.py is a tiny aiohttp/FastAPI proxy that:
#   GET  /ping         → returns 200 when SGLang on :30000 responds to /health
#   POST /invocations  → forwards to localhost:30000/v1/chat/completions
COPY serve.py /opt/serve/serve.py
COPY requirements.txt /opt/serve/requirements.txt
RUN pip install --no-cache-dir -r /opt/serve/requirements.txt

COPY entrypoint.sh /opt/serve/entrypoint.sh
RUN chmod +x /opt/serve/entrypoint.sh

ENTRYPOINT ["/opt/serve/entrypoint.sh"]
```

### `requirements.txt`

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
httpx==0.27.*
```

### `entrypoint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- SGLang launch (ported from dev docker/gpu/entrypoint.sh) -------------
: "${SGL_MODEL_PATH:=/opt/ml/model}"        # weights mounted here by SageMaker
: "${SGL_SERVED_MODEL_NAME:=glm-ocr}"
: "${SGL_TP_SIZE:=1}"
: "${SGL_DTYPE:=float16}"

ARGS=(
    --model-path          "${SGL_MODEL_PATH}"
    --served-model-name   "${SGL_SERVED_MODEL_NAME}"
    --host                127.0.0.1
    --port                30000
    --tp-size             "${SGL_TP_SIZE}"
    --dtype               "${SGL_DTYPE}"
    --trust-remote-code
    --enable-metrics
)

# Copy the optional-knob pattern verbatim from the dev entrypoint. Each
# knob is wired in the SagemakerStack's Environment[] of the EndpointConfig.
[[ -n "${SGL_MAX_RUNNING_REQUESTS:-}" ]] && ARGS+=(--max-running-requests  "${SGL_MAX_RUNNING_REQUESTS}")
[[ -n "${SGL_MAX_PREFILL_TOKENS:-}"   ]] && ARGS+=(--max-prefill-tokens    "${SGL_MAX_PREFILL_TOKENS}")
[[ -n "${SGL_MAX_TOTAL_TOKENS:-}"     ]] && ARGS+=(--max-total-tokens      "${SGL_MAX_TOTAL_TOKENS}")
[[ -n "${SGL_MEM_FRACTION_STATIC:-}"  ]] && ARGS+=(--mem-fraction-static   "${SGL_MEM_FRACTION_STATIC}")
[[ -n "${SGL_CONTEXT_LENGTH:-}"       ]] && ARGS+=(--context-length        "${SGL_CONTEXT_LENGTH}")
[[ -n "${SGL_SCHEDULE_POLICY:-}"           ]] && ARGS+=(--schedule-policy           "${SGL_SCHEDULE_POLICY}")
[[ -n "${SGL_CUDA_GRAPH_MAX_BS:-}"         ]] && ARGS+=(--cuda-graph-max-bs         "${SGL_CUDA_GRAPH_MAX_BS}")
[[ -n "${SGL_SCHEDULE_CONSERVATIVENESS:-}" ]] && ARGS+=(--schedule-conservativeness "${SGL_SCHEDULE_CONSERVATIVENESS}")

CHUNKED="${SGL_CHUNKED_PREFILL:-false}"
if [[ "${CHUNKED,,}" == "true" || "${CHUNKED}" == "1" ]]; then
    : "${SGL_CHUNKED_PREFILL_SIZE:=8192}"
    ARGS+=(--chunked-prefill-size "${SGL_CHUNKED_PREFILL_SIZE}")
fi

SPEC="${SGL_SPECULATIVE:-false}"
if [[ "${SPEC,,}" == "true" || "${SPEC}" == "1" ]]; then
    : "${SGL_SPEC_ALGORITHM:=NEXTN}"
    : "${SGL_SPEC_NUM_STEPS:=3}"
    : "${SGL_SPEC_EAGLE_TOPK:=1}"
    : "${SGL_SPEC_NUM_DRAFT_TOKENS:=4}"
    export SGLANG_ENABLE_SPEC_V2=1
    ARGS+=(
        --speculative-algorithm        "${SGL_SPEC_ALGORITHM}"
        --speculative-num-steps        "${SGL_SPEC_NUM_STEPS}"
        --speculative-eagle-topk       "${SGL_SPEC_EAGLE_TOPK}"
        --speculative-num-draft-tokens "${SGL_SPEC_NUM_DRAFT_TOKENS}"
    )
fi

echo "[entrypoint] launching SGLang on 127.0.0.1:30000 with args:"
printf '  %s\n' "${ARGS[@]}"

# Launch SGLang in the background on :30000.
python3 -m sglang.launch_server "${ARGS[@]}" &
SGL_PID=$!

# Launch the SageMaker-contract shim on :8080 in the foreground.
# It proxies /ping → SGLang /health, /invocations → /v1/chat/completions.
exec uvicorn --app-dir /opt/serve serve:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
    --log-level info
```

(Trap forwarding SIGTERM omitted for brevity — add in real implementation.)

### `serve.py` (FastAPI shim, ~40 lines)

```python
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

SGL_URL = os.environ.get("SGL_URL", "http://127.0.0.1:30000")
CONNECT_TIMEOUT = float(os.environ.get("SGL_CONNECT_TIMEOUT", "5"))
REQUEST_TIMEOUT = float(os.environ.get("SGL_REQUEST_TIMEOUT", "60"))

app = FastAPI(title="sagemaker-sglang-shim")
client = httpx.AsyncClient(
    base_url=SGL_URL,
    timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
    limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
)

@app.get("/ping")
async def ping() -> Response:
    try:
        r = await client.get("/health", timeout=1.5)
        return Response(status_code=200 if r.status_code == 200 else 503)
    except httpx.HTTPError:
        return Response(status_code=503)

@app.post("/invocations")
async def invocations(request: Request):
    body = await request.body()
    ct = request.headers.get("content-type", "application/json")
    upstream = await client.post(
        "/v1/chat/completions",
        content=body,
        headers={"content-type": ct},
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
```

**Why this shape.** glmocr's client POSTs an OpenAI `/v1/chat/completions` body. SageMaker's `InvokeEndpoint` forwards body-bytes to `/invocations` unchanged. We forward those same bytes to SGLang's `/v1/chat/completions` without re-parsing — fastest path, no JSON round-trip.

**Streaming.** If later we enable SGLang streaming, `InvokeEndpointWithResponseStream` is required. This shim currently does not support streaming; add a `stream: true` branch that returns a `StreamingResponse` when ready. Not in scope for first ship.

---

## Weights: S3 staging

### Why not download from HuggingFace at endpoint boot

- 1.8 GB pull over HF's CDN is unpredictable (dev saw 2–8 min depending on time of day).
- SageMaker boot timeout is generous (up to 60 min) but each retry is a fresh instance, which is slow feedback.
- Prod needs determinism. S3 in the same region gives consistent ~30–60 s cold-start.

### How to bake

`scripts/bake-weights.sh` (runs on your laptop or a CI job; **not** at SageMaker boot time):

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-zai-org/GLM-OCR}"
BUCKET="${WEIGHTS_BUCKET:?set WEIGHTS_BUCKET}"
KEY="${KEY:-glm-ocr/$(date +%Y%m%d)/model.tar.gz}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

export HF_HOME="${WORK}/hf"
pip install --upgrade huggingface_hub > /dev/null
huggingface-cli download "${MODEL_ID}" --local-dir "${WORK}/model" --token "${HF_TOKEN}"

# SageMaker expects everything relative to /opt/ml/model.
# Tar from *inside* the model dir so the archive's root is the model contents,
# not a nested folder.
tar -C "${WORK}/model" -czvf "${WORK}/model.tar.gz" .

aws s3 cp "${WORK}/model.tar.gz" "s3://${BUCKET}/${KEY}" \
    --sse AES256 --storage-class STANDARD

echo "uploaded: s3://${BUCKET}/${KEY}"
```

The CDK `SagemakerStack` reads the key from SSM (`/glmocr/prod/sagemaker/model_data_key`) so bumping the weights is a two-step: run `bake-weights.sh`, update the SSM param → redeploy the endpoint (blue-green).

### Tar archive shape (SageMaker requirement)

```
model.tar.gz
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model-00001-of-000xx.safetensors
├── ...
├── special_tokens_map.json
└── <any other HF artifact in the repo>
```

Root of the tar must be the files themselves, not a subfolder. SageMaker extracts into `/opt/ml/model/`, and `entrypoint.sh` passes `/opt/ml/model` as `--model-path`.

---

## CDK `SagemakerStack` shape

```go
// docker image (from ECR)
image := awssagemaker.NewCfnModel_ContainerDefinitionProperty(...)
image.Image     = jsii.String(sglangEcrRepo.RepositoryUri() + ":" + cfg.SglangImageTag)
image.ModelDataUrl = jsii.String("s3://" + weightsBucket + "/" + modelDataKey)
image.Environment  = map[string]*string{
    "SGL_SERVED_MODEL_NAME":       jsii.String("glm-ocr"),
    "SGL_MAX_RUNNING_REQUESTS":    jsii.String(sslKnobs["SGL_MAX_RUNNING_REQUESTS"]),
    "SGL_MAX_PREFILL_TOKENS":      jsii.String(sslKnobs["SGL_MAX_PREFILL_TOKENS"]),
    "SGL_MAX_TOTAL_TOKENS":        jsii.String(sslKnobs["SGL_MAX_TOTAL_TOKENS"]),
    "SGL_MEM_FRACTION_STATIC":     jsii.String(sslKnobs["SGL_MEM_FRACTION_STATIC"]),
    "SGL_CONTEXT_LENGTH":          jsii.String(sslKnobs["SGL_CONTEXT_LENGTH"]),
    "SGL_CHUNKED_PREFILL":         jsii.String(sslKnobs["SGL_CHUNKED_PREFILL"]),
    "SGL_CHUNKED_PREFILL_SIZE":    jsii.String(sslKnobs["SGL_CHUNKED_PREFILL_SIZE"]),
    "SGL_SCHEDULE_POLICY":         jsii.String(sslKnobs["SGL_SCHEDULE_POLICY"]),
    "SGL_SPECULATIVE":             jsii.String(sslKnobs["SGL_SPECULATIVE"]),
    "SGL_SPEC_ALGORITHM":          jsii.String(sslKnobs["SGL_SPEC_ALGORITHM"]),
    // ... the other SGL_SPEC_* knobs
    // Optional: flip on for CUDA IPC once smoke-tested on g4dn.2xlarge
    // "SGLANG_USE_CUDA_IPC_TRANSPORT": jsii.String("1"),
}

endpointConfig := awssagemaker.NewCfnEndpointConfig(stack, "glmocr-sglang-cfg", ...)
endpointConfig.ProductionVariants = []ProductionVariant{{
    VariantName:           jsii.String("Primary"),
    ModelName:             model.AttrModelName(),
    InitialInstanceCount:  jsii.Number(1),
    InstanceType:          jsii.String("ml.g4dn.2xlarge"),
    InitialVariantWeight:  jsii.Number(1),
    ModelDataDownloadTimeoutInSeconds: jsii.Number(600),
    ContainerStartupHealthCheckTimeoutInSeconds: jsii.Number(1800), // SGLang + weights = ~3-10 min
}}

endpoint := awssagemaker.NewCfnEndpoint(stack, "glmocr-sglang", ...)
// Deployment config for blue-green; see SagemakerStack tests
```

---

## Tuned knob values (from dev, carry over unchanged)

See `reference/env-tuned.md` for the authoritative list. For SageMaker specifically:

```
SGL_SERVED_MODEL_NAME=glm-ocr
SGL_MAX_RUNNING_REQUESTS=64
SGL_MAX_PREFILL_TOKENS=8192
SGL_MAX_TOTAL_TOKENS=200000
SGL_MEM_FRACTION_STATIC=0.95
SGL_CONTEXT_LENGTH=24576
SGL_CHUNKED_PREFILL=true
SGL_CHUNKED_PREFILL_SIZE=8192
SGL_SCHEDULE_POLICY=lpm
SGL_SPECULATIVE=true
SGL_SPEC_ALGORITHM=NEXTN
SGL_SPEC_NUM_STEPS=3
SGL_SPEC_EAGLE_TOPK=1
SGL_SPEC_NUM_DRAFT_TOKENS=4
```

`SGLANG_USE_CUDA_IPC_TRANSPORT` — leave unset on first ship. Smoke with it enabled once after endpoint is `InService`, watching for CUDA OOM in CloudWatch logs. See the CUDA IPC + MTP memory in `reference/memory-seed.md`.

---

## Testing the image before deploy

```bash
# Local sanity — requires nvidia-docker
docker run --rm -it \
    --gpus all --shm-size 32g --ipc host \
    -v $(pwd)/test-model:/opt/ml/model:ro \
    -e SGL_MAX_RUNNING_REQUESTS=8 \
    -e SGL_CONTEXT_LENGTH=4096 \
    -p 8080:8080 \
    <repo>/glmocr-sglang:latest

# In another terminal
curl localhost:8080/ping
curl -X POST localhost:8080/invocations \
    -H 'content-type: application/json' \
    -d '{"model":"glm-ocr","messages":[{"role":"user","content":"hello"}]}'
```

This verifies the `/ping`, `/invocations` contract without booting a SageMaker endpoint.

---

Next: [`04-fargate-cpu-task.md`](./04-fargate-cpu-task.md).
