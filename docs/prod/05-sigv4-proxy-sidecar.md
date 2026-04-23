# 05 — SigV4-proxy sidecar

**Purpose:** the sidecar that makes `glmocr.ocr_client` Just Work against a SageMaker endpoint. Listens on task-local `127.0.0.1:30000`, SigV4-signs every request, and forwards to `sagemaker-runtime:InvokeEndpoint`.

---

## Why we need this

The dev stack has glmocr POSTing to `http://sglang:30000/v1/chat/completions`. Two things change in prod:

1. The destination is now a SageMaker endpoint, reachable via `sagemaker-runtime.<region>.amazonaws.com/endpoints/<name>/invocations`.
2. That URL requires **SigV4 signing** on every request.

`glmocr.ocr_client` (pip package) owns its HTTP transport. We could:

- **A. Fork glmocr** to inject boto3 signing. Rejected — couples the prod repo to a fork; we'd own the glmocr upgrade path forever.
- **B. Wrap the entire CPU app** in a shim that mutates outbound calls. Rejected — fragile, framework-specific.
- **C. Run a SigV4-proxy sidecar**. The CPU container calls a **local** URL; the proxy handles signing. Zero change in glmocr. Zero change in `runtime_app.py`. **Chosen.**

---

## Option 1 — upstream `aws-sigv4-proxy` (default)

AWS publishes [`aws/aws-sigv4-proxy`](https://github.com/awslabs/aws-sigv4-proxy). Public ECR image: `public.ecr.aws/aws-observability/aws-sigv4-proxy`.

Task-def command:

```
[
  "-v",
  "--name=sagemaker",
  "--region=<region>",
  "--host=runtime.sagemaker.<region>.amazonaws.com",
  "--port=:30000",
  "--unsigned-payload"
]
```

- `--name=sagemaker` — the IAM service name (not the AWS service name) used when signing.
- `--host=runtime.sagemaker.<region>.amazonaws.com` — the outbound destination.
- `--port=:30000` — where glmocr connects locally.
- `--unsigned-payload` — SageMaker runtime accepts unsigned-payload signing; it's faster (we don't hash the body).

### Path rewrite

glmocr POSTs to `/v1/chat/completions`. The SageMaker endpoint expects `/endpoints/<endpoint-name>/invocations`. The proxy does **not** rewrite paths on its own. Two fixes:

**Simplest:** point glmocr at the SageMaker runtime path directly:

```
SGLANG_HOST=127.0.0.1
SGLANG_PORT=30000
SGLANG_SCHEME=http
OCR_API_PATH_OVERRIDE=/endpoints/<endpoint-name>/invocations    # if glmocr supports this
```

This works if glmocr's config exposes the API path override. If it doesn't (check `pip show glmocr` → look at `ocr_client.py` on the new MacBook), use Option 2 below or a thin nginx rewrite in the same sidecar image.

**Nginx rewrite layer** (15 lines in the proxy image):

```nginx
server {
    listen 30001;
    location /v1/chat/completions {
        proxy_pass http://127.0.0.1:30000/endpoints/<endpoint-name>/invocations;
    }
}
```

...then glmocr points at `:30001`, nginx rewrites to `:30000/endpoints/.../invocations`, sigv4-proxy signs and forwards to SageMaker. Ugly, but two containers become three files on disk.

---

## Option 2 — 50-line Go fallback

If the upstream proxy doesn't fit (auth mode quirks, streaming, path rewrite), ship a tiny Go service in place of it. Code shape:

```go
// docker/sigv4-proxy/main.go
package main

import (
    "context"
    "io"
    "log"
    "net/http"
    "os"

    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/sagemakerruntime"
)

var (
    client       *sagemakerruntime.Client
    endpointName = os.Getenv("SM_ENDPOINT_NAME")
)

func main() {
    cfg, err := config.LoadDefaultConfig(context.Background())
    if err != nil { log.Fatalf("aws config: %v", err) }
    client = sagemakerruntime.NewFromConfig(cfg)

    http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })
    http.HandleFunc("/v1/chat/completions", handleInvoke)

    log.Printf("listening on :30000; forwarding to SageMaker endpoint %q", endpointName)
    if err := http.ListenAndServe(":30000", nil); err != nil {
        log.Fatal(err)
    }
}

func handleInvoke(w http.ResponseWriter, r *http.Request) {
    body, err := io.ReadAll(r.Body)
    if err != nil { http.Error(w, err.Error(), http.StatusBadRequest); return }

    out, err := client.InvokeEndpoint(r.Context(), &sagemakerruntime.InvokeEndpointInput{
        EndpointName: &endpointName,
        ContentType:  strPtr("application/json"),
        Accept:       strPtr("application/json"),
        Body:         body,
    })
    if err != nil { http.Error(w, err.Error(), http.StatusBadGateway); return }

    w.Header().Set("content-type", *out.ContentType)
    w.WriteHeader(http.StatusOK)
    _, _ = w.Write(out.Body)
}

func strPtr(s string) *string { return &s }
```

Dockerfile (`docker/sigv4-proxy/Dockerfile`):

```dockerfile
FROM golang:1.23 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
RUN CGO_ENABLED=0 go build -o /out/proxy .

FROM gcr.io/distroless/static:nonroot
COPY --from=build /out/proxy /usr/local/bin/proxy
EXPOSE 30000
ENTRYPOINT ["/usr/local/bin/proxy"]
```

Env: `SM_ENDPOINT_NAME=glmocr-sglang` (from the CDK SagemakerStack's output). IAM: task role has `sagemaker:InvokeEndpoint` on the endpoint — standard AWS SDK signing picks up the task role via IMDS.

**Pick this over Option 1** if:
- You need path rewrite and don't want nginx.
- glmocr sends an `Accept` header the upstream proxy strips.
- You want structured logs per request (it's 10 more lines).

**Stay on Option 1** if the smoke test in `prompts/08-first-smoke.md` returns 200 end-to-end on the first try — don't over-engineer.

---

## Request shape verification

The body glmocr POSTs is a standard OpenAI chat-completions payload:

```json
{
  "model": "glm-ocr",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "max_tokens": 8192,
  "temperature": 0.0
}
```

SageMaker `InvokeEndpoint`'s `Body` is an opaque `[]byte` — it forwards to the container's `/invocations` verbatim. Inside the container, our `serve.py` forwards to SGLang's `/v1/chat/completions` verbatim. **No transformation in either layer.** If glmocr changes its client to send something SGLang doesn't accept, the fault is at glmocr; the proxy chain is transparent.

---

## Streaming (future)

`/v1/chat/completions` with `"stream": true` returns a chunked SSE response. The current shim and the upstream sigv4-proxy don't handle this — SageMaker requires `InvokeEndpointWithResponseStream` for streaming.

Current glmocr config does not use streaming (confirmed: `docker/cpu/config.yaml.template` line 29 pipeline is request-response). If prod later adds streaming:

- Swap `InvokeEndpoint` → `InvokeEndpointWithResponseStream` in the Go fallback.
- Flush each chunk to the client as it arrives.
- Update `serve.py` to detect `stream: true` and return `StreamingResponse`.

Tracked as a known future change in `09-runbook.md` §"Future work."

---

## Health endpoint

The task-def healthcheck on the proxy is a simple TCP probe of port 30000. The upstream `aws-sigv4-proxy` binary returns 200 on any GET — a port-level TCP probe is enough. The Go fallback exposes `/healthz`.

We do **not** have the proxy health-check actually hit SageMaker. That would couple task health to endpoint health, causing flapping on SageMaker deploys. If SageMaker is down, OCR requests return 500s with a clear upstream error; CloudWatch alarms page on that, not on task health.

---

## Cost note

For a 200 req/s workload with ~20 regions/page, the proxy handles ~4,000 signatures/s. SigV4 signing on modern x86_64 is ~100 μs, so ~400 ms of CPU work per wall-second → ~0.4 vCPU sustained. Our 0.5 vCPU reservation covers burst headroom with ~20% buffer.

---

Next: [`06-observability-prod.md`](./06-observability-prod.md).
