# Prompt 03 — EcrStack + image builds

> Prerequisite: prompt 02 deployed. Expect ~45 min.

---

```
Implement EcrStack and build + push both Docker images.

Read first:
  - docs/prod/03-sagemaker-sglang-byoc.md
  - docs/prod/02-cdk-go-structure.md

## Part A — EcrStack (internal/stacks/ecr.go)

1. Create two repos:
   - glmocr-cpu
   - glmocr-sglang

2. Each with:
   - Image scanning on push: enabled
   - Lifecycle policy: retain last 20 untagged, last 20 tagged (keep recent SHAs)
   - Encryption: AES256

3. Export accessors:
   - CpuRepository, SglangRepository (awsecr.IRepository)

4. Tests: one-liner that asserts both repositories exist.

5. Deploy:
     cdk deploy --context stage=prod glmocr-ecr-prod
   Expect ~30 seconds.

Commit as "feat(ecr): two repositories for cpu + sglang images".

## Part B — glmocr-sglang image (docker/sglang/)

1. Create the directory from scratch:
     mkdir -p docker/sglang
   Copy ONLY the ARG pattern from docker/gpu/entrypoint.sh (the dev version).

2. Write Dockerfile, serve.py, requirements.txt, entrypoint.sh following
   docs/prod/03-sagemaker-sglang-byoc.md EXACTLY.

3. Local sanity — NOT required unless you have an NVIDIA GPU on this machine.
   Otherwise proceed to ECR push.

4. Build + push:
     aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR
     docker build -t glmocr-sglang:$(git rev-parse --short HEAD) docker/sglang/
     docker tag glmocr-sglang:... $ECR/glmocr-sglang:$(git rev-parse --short HEAD)
     docker tag glmocr-sglang:... $ECR/glmocr-sglang:latest
     docker push $ECR/glmocr-sglang:$(git rev-parse --short HEAD)
     docker push $ECR/glmocr-sglang:latest

## Part C — glmocr-cpu image (docker/cpu/) — BUILD FROM SCRATCH

1. Read docs/prod/12-cpu-container-spec.md cover-to-cover. It has the exact
   contents for every file we'll create.

2. Create the directory and write:
     docker/cpu/Dockerfile
     docker/cpu/entrypoint.sh
     docker/cpu/wsgi.py
     docker/cpu/gunicorn_conf.py
     docker/cpu/config.yaml.template
     docker/cpu/config.layout-off.template
   Each file is fully specified in 12-cpu-container-spec.md — copy the blocks
   verbatim. chmod +x entrypoint.sh.

3. Local smoke build (optional but recommended):
     docker build -t glmocr-cpu:local docker/cpu/
     docker run --rm -p 5002:5002 \
         -e CPU_WORKERS=2 -e CPU_THREADS=8 -e GLMOCR_PORT=5002 \
         -e OCR_MAX_WORKERS=4 -e OCR_CONN_POOL=128 \
         -e OCR_CONNECT_TIMEOUT=5 -e OCR_REQUEST_TIMEOUT=30 \
         -e OCR_RETRY_MAX=1 -e OCR_RETRY_BACKOFF_BASE=0.5 -e OCR_RETRY_BACKOFF_MAX=4 \
         -e OCR_MODEL_NAME=glm-ocr \
         -e SGLANG_HOST=127.0.0.1 -e SGLANG_PORT=30000 -e SGLANG_SCHEME=http \
         -e LAYOUT_ENABLED=true -e LAYOUT_DEVICE=cpu \
         -e LAYOUT_MODEL_DIR=PaddlePaddle/PP-DocLayoutV3_safetensors \
         -e LAYOUT_USE_POLYGON=false \
         -e GUNICORN_TIMEOUT=180 \
         glmocr-cpu:local
     # In another terminal:
     curl localhost:5002/health   # {"status":"ok"}
     curl localhost:5002/metrics

4. Build + push to ECR (same pattern as Part B):
     docker build -t glmocr-cpu:$(git rev-parse --short HEAD) docker/cpu/
     docker tag glmocr-cpu:... $ECR/glmocr-cpu:$(git rev-parse --short HEAD)
     docker tag glmocr-cpu:... $ECR/glmocr-cpu:latest
     docker push $ECR/glmocr-cpu:$(git rev-parse --short HEAD)
     docker push $ECR/glmocr-cpu:latest

**MVP scope reminder.** This container ships with glmocr's stock pipeline +
ONNX layout backend + tuned HTTP/retry knobs. It does NOT include the dev
repo's custom numpy postproc, layout coalescer, or async sidecar — those
are documented as future-port work in 12-cpu-container-spec.md §"Full-
parity phase." The MVP gets the biggest single win (1.76× ONNX on layout)
and ships functional.

## Part D — Bake weights to S3

1. Create an S3 bucket (either manually or add to NetworkStack — your call;
   I'd add a tiny WeightsBucket construct inside SecretsStack or a new
   ArtifactsStack).

2. Run scripts/bake-weights.sh with the bucket name. Requires HF_TOKEN in env.
   Upload takes ~5-10 min depending on network.

3. Create an SSM parameter:
     aws ssm put-parameter \
       --name /glmocr/prod/sagemaker/model_data_key \
       --value "glm-ocr/<yyyymmdd>/model.tar.gz" \
       --type String \
       --description "S3 key for GLM-OCR weights tarball"

Report at end: both image URIs (with tags), weights S3 key, bucket name.
Commit as "feat(images): sglang byoc + cpu slim pushed to ecr; weights baked".
```
