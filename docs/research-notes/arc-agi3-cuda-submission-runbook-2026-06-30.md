# ARC-AGI-3 CUDA Submission Runbook (2026-06-30)

**Goal:** ship the Kaggle submission on **CUDA** (the real eval hardware), not the dev-box ROCm/iGPU
proxy. Operator-only steps (B–D) are flagged; preparation (A + staging) is done.

## The hardware reality (Kaggle env probe, `results/kaggle_env_probe.json`)
- **Kaggle GPU: Tesla P100-PCIE-16GB**, driver **580.159.04**, **CUDA 12.8** toolkit (`nvcc 12.8`).
- Dev box: **CUDA 13.3** + RTX 3090 (sm_86). A *native* dev-box build links `libcudart.so.13` and the
  3090's arch → would not run on Kaggle's P100 / CUDA-12.8 (silent CPU fallback).
- **RESOLVED (2026-06-30): the binary is now cross-built locally via Docker, no Kaggle build step.**
  `nvcc` emits SASS for whatever `-DCMAKE_CUDA_ARCHITECTURES` lists regardless of the host GPU, so a
  `nvidia/cuda:12.8.1-devel` container on the dev box builds the exact Kaggle-compatible binary (CUDA 12.8
  runtime libs + archs 60=P100/75=T4/86/89=L4). The submission is now **fully self-contained** — the
  binary + its 12-SONAME .so closure (real files, no symlinks, 1.4GB) is staged, smoke-validated on the
  3090 via the exact `main.py` copy-to-/kaggle/working path. See
  `/home/ianblenke/carnot_submission_staging/carnot-llamacpp-mtp-binary/build/BUILD_README.md`.

## The 4 payloads
| Payload | Kaggle dataset / kernel | Status |
|---|---|---|
| Agent code (Python) | `iancblenke/carnot-agent-code` | **STAGED** at `/tmp/carnot_cuda_submission/carnot-agent-code/` (python/carnot + ops, guards passed) |
| CUDA llama-server binary | `iancblenke/carnot-llamacpp-mtp-binary` | **PRE-BUILT (CUDA 12.8, Docker cross-build)** — staged + smoke-validated; just upload |
| Generator GGUF (5.5 GB) | `iancblenke/carnot-qwen35-9b-mtp-gguf` | upload from `~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/.../Qwen3.5-9B-Q4_K_M.gguf` |
| Submission kernel | `iancblenke/carnot-arc-agi3-submission` | **STAGED** at `/tmp/carnot_cuda_submission/kernel/` (main.py + kernel-metadata.json) |

## Step A — (DONE locally) Cross-build the CUDA binary via Docker
Already done + staged. The binary was cross-built on the dev box:
```bash
docker run --rm -v /tmp/llamacpp_cuda128_build:/work nvidia/cuda:12.8.1-devel-ubuntu22.04 bash /work/build.sh
```
`-DCMAKE_CUDA_ARCHITECTURES="60;75;86;89;89-virtual"` (P100/T4/dev/L4 + PTX JIT) against CUDA 12.8;
`-lcuda` forced via the toolkit stub at link time (real driver comes from Kaggle at runtime). Output
(`llama-server` + 12-SONAME .so closure, real files no symlinks) is staged at
`carnot-llamacpp-mtp-binary/`. Smoke-validated on the 3090 via the exact `main.py` copy-to-working path.
The legacy build-ON-Kaggle notebook (`scripts/kaggle/build_verify_llamacpp_mtp.py`) is kept as a fallback.

## Step B — Upload the datasets (operator)
- `carnot-agent-code`: `kaggle datasets version -p /tmp/carnot_cuda_submission/carnot-agent-code -m "cuda submission refresh"` (or `create` if first time).
- `carnot-qwen35-9b-mtp-gguf`: upload the GGUF (one-time; stable).
- `carnot-llamacpp-mtp-binary`: from Step A.

## Step C — Push the kernel + verify (operator; internet OFF for the run)
- `kernel-metadata.json` already binds the 3 datasets, `enable_gpu: true`, `enable_internet: false`,
  competition `arc-prize-2026-arc-agi-3`.
- `kaggle kernels push -p /tmp/carnot_cuda_submission/kernel`, then **Save & Run**.
- **Verify in the log:** it must print `LLM TIER RESOLVED: server=... mtp=False ctx=16384 kv=q8_0`
  (CUDA binary found; MTP auto-OFF on the P100). If it prints
  `LLM TIER DISABLED ... CPU graph-explore ONLY`, the binary dataset was not found/attached — fix Step A/B
  before submitting (the agent would run the weaker CPU tier).

## Step D — Submit (OPERATOR-ONLY)
- Submit the kernel output through the Kaggle UI/API. (Per the Operator-Only External Publication rule,
  no automated step submits.)

## Notes
- **MTP is intentionally OFF on the P100** (`main.py` sets `CARNOT_ARC_MTP=0`): the 2026-06-21 probe found
  MTP self-draft doubles VRAM (11.8 GB vs 5.9 GB) for *no* throughput gain on a P100 (27.5 vs 25.3 tok/s).
  The binary still builds with MTP available; it's just not used on this GPU. (Wheel fallback exists if the
  binary proves fiddly, but the binary is the validated path.)
- `kernel-metadata.json` requests `machine_shape: NvidiaL4`; if Kaggle assigns a P100/T4 instead, the
  binary's `60;75;89` arch list covers it — no change needed. VRAM footprint ~11.5 GB fits the 16 GB P100.
- **Score expectation: ~0.08** (unchanged). The CUDA generator runs the same agent; the generation wall
  (the 9B inducer's proposal distribution) is what bounds the score, not the GPU/runtime. This submission
  is for a clean, valid CUDA entry on the 6/30 milestone — not a score improvement.

## Cross-references
- `scripts/kaggle/build_verify_llamacpp_mtp.py` — the build-on-Kaggle notebook
- `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md` — packaging manifest (PACKAGING MANIFEST + Local CUDA validation)
- `results/kaggle_env_probe.json` — the P100/CUDA-12.8/driver-580 evidence
- `results/experiment_4986_submission_package_harden.json` — the latest local harden (iGPU proxy)

## Staged (2026-06-30): /home/ianblenke/carnot_submission_staging/
All four payloads staged + verified (agent-code guards pass, GGUF hardlinked, kernel parses, **binary
pre-built CUDA 12.8 + smoke-validated on the 3090**). Submission is fully self-contained (no Kaggle build
step). See that dir's MANIFEST.md for the operator upload/submit steps.
