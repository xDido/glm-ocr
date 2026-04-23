---
name: CUDA IPC Transport blocked by 8 GB VRAM; MTP already on — 2026-04-23
description: Tested two GLM-OCR cookbook optimizations. MTP (Multi-Token Prediction) is already effectively enabled — SGLang aliases our SGL_SPEC_ALGORITHM=NEXTN to EAGLE internally, matching the cookbook's recommended values exactly. CUDA IPC Transport crash-loops on 8 GB 3060 Ti at every mem_fraction tested (0.95, 0.82, 0.70) — needs ≥16 GB VRAM. Retry on Fargate if GPU is L4/A10/A100-class.
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
**Source:** https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-OCR

## MTP: already on (no-op)

SGLang internally aliases `--speculative-algorithm NEXTN` to `'EAGLE'`. Our current `.env` already produces these runtime server_args, matching the cookbook recommendation byte-for-byte:

```
speculative_algorithm='EAGLE'
speculative_num_steps=3
speculative_eagle_topk=1
speculative_num_draft_tokens=4
```

Flipping `SGL_SPEC_ALGORITHM=NEXTN` → `EAGLE` is a cosmetic change with zero behavioral effect on this SGLang version. Don't chase this as an optimization — it's already implemented.

## CUDA IPC Transport: needs ≥16 GB VRAM

`SGLANG_USE_CUDA_IPC_TRANSPORT=1` is supposed to speed up multimodal feature transfer between the vision encoder and the language model by sharing GPU memory instead of serializing. But the implementation allocates a dedicated `MmItemMemoryPool` via `torch.storage._share_cuda_()` (`sglang/srt/utils/cuda_ipc_transport_utils.py:127`), and that pool needs multi-GB of contiguous free VRAM on top of the model + KV cache.

On RTX 3060 Ti 8 GB (dev box), GLM-OCR + EAGLE draft heads already take ~5-6 GB at baseline. Tested three `SGL_MEM_FRACTION_STATIC` values, all crash-looped with `torch.AcceleratorError: CUDA error: out of memory`:

| mem_fraction | Result | OOM events observed |
|---|---|---:|
| 0.95 (baseline) | crash loop, smoke HTTP 500 | 33 |
| 0.82 | crash loop, smoke HTTP 500 | 18 |
| 0.70 | crash loop, smoke HTTP 500 | 20 |

Lowering mem_fraction below 0.70 would mean giving up more than a quarter of the KV pool — a significant throughput penalty on its own — and there's no evidence CUDA IPC would fit even then. The feature is simply sized for data-center GPUs.

**Reverted to baseline** (`SGL_MEM_FRACTION_STATIC=0.95`, `SGLANG_USE_CUDA_IPC_TRANSPORT` unset). Stack healthy after revert.

## Plumbing kept in tree

`docker-compose.yml` now forwards `SGLANG_USE_CUDA_IPC_TRANSPORT` to the sglang service with `${SGLANG_USE_CUDA_IPC_TRANSPORT:-}` — unset by default, zero behavior change. This means retrying on Fargate is a one-line flip in `.env`, not a re-implementation.

## When to retry on Fargate

If the Fargate SGLang tier runs on a GPU with ≥16 GB VRAM (L4 24 GB, A10 24 GB, A100 40/80 GB, H100 80 GB — most modern inference hosts qualify):

1. Set `SGLANG_USE_CUDA_IPC_TRANSPORT=1` in the Fargate task env.
2. Keep `SGL_MEM_FRACTION_STATIC=0.95` initially; only lower if VRAM pressure surfaces.
3. Smoke test first: `curl /glmocr/parse` should return 200 cleanly; `docker logs glmocr-sglang` should have zero `out of memory` lines.
4. Run the matrix, look at `Time-to-first-token (prefill+first decode)` in the SGLang phase tables — baseline was 2,845 ms mean at c=12. Cookbook claims "significantly improves TTFT," so that's the metric to watch.

On GPUs <16 GB (T4 16 GB, RTX 3060 Ti 8 GB, A10G 24 GB (safe but check), …) the story is uncertain — may or may not fit depending on the KV pool split. Always smoke test before a matrix run.
