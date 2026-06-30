# ARC-AGI-3 CUDA Submission Runbook (2026-06-30)

**Goal:** ship the Kaggle submission on **CUDA** (the real eval hardware), not the dev-box ROCm/iGPU
proxy. Operator-only steps (B–D) are flagged; preparation (A + staging) is done.

## The hardware reality (Kaggle env probe, `results/kaggle_env_probe.json`)
- **Kaggle GPU: Tesla P100-PCIE-16GB**, driver **580.159.04**, **CUDA 12.8** toolkit (`nvcc 12.8`).
- Dev box: **CUDA 13.3** + RTX 3090 (sm_86). A binary built here links `libcudart.so.13` and the 3090's
  arch → **will not run on Kaggle's P100 / CUDA-12.8** (silent CPU fallback). The dev-box iGPU/HIP build
  used in the harden checks was only a *local proxy*.
- **Conclusion: the CUDA `llama-server` binary MUST be built ON Kaggle** (CUDA-version + GPU-arch match).
  This is the one piece that cannot be prepared locally.

## The 4 payloads
| Payload | Kaggle dataset / kernel | Status |
|---|---|---|
| Agent code (Python) | `iancblenke/carnot-agent-code` | **STAGED** at `/tmp/carnot_cuda_submission/carnot-agent-code/` (python/carnot + ops, guards passed) |
| CUDA llama-server binary | `iancblenke/carnot-llamacpp-mtp-binary` | **BUILD ON KAGGLE** — Step A (the CUDA-vs-iGPU step) |
| Generator GGUF (5.5 GB) | `iancblenke/carnot-qwen35-9b-mtp-gguf` | upload from `~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/.../Qwen3.5-9B-Q4_K_M.gguf` |
| Submission kernel | `iancblenke/carnot-arc-agi3-submission` | **STAGED** at `/tmp/carnot_cuda_submission/kernel/` (main.py + kernel-metadata.json) |

## Step A — Build the CUDA binary ON Kaggle (operator; internet ON)
1. New Kaggle notebook, **GPU on, internet ON**.
2. Run `scripts/kaggle/build_verify_llamacpp_mtp.py` (paste the `# %% CELL` blocks, or run the file).
   - Clones llama.cpp `b9714` (has native MTP), builds `llama-server` with
     `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;75;89;89-virtual"` against **Kaggle's CUDA 12.8**
     (60=P100, 75=T4, 89=L4, 89-virtual=PTX JIT fallback — covers the whole Kaggle pool).
   - Self-verifies MTP present, smoke-tests load+generate on the GGUF, probes real per-GPU VRAM.
3. Save `/kaggle/working/llamacpp-cuda-mtp/` (the binary + its `.so` closure:
   `libllama-server-impl, libllama-common.so.0, libmtmd.so.0, libllama.so.0, libggml.so.0,
   libggml-cpu.so.0, libggml-cuda.so.0, libggml-base.so.0`) as the **`carnot-llamacpp-mtp-binary`**
   dataset.

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
