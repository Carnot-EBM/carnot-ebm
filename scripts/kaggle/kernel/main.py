import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub'], check=False)
"""Kaggle build + verify notebook: produce a CUDA llama-server-with-MTP artifact and settle the two open
ARC-submission unknowns in one run. Paste each `# %% CELL` block into a Kaggle notebook cell (GPU on),
or run the whole file as a script. Internet ON for the BUILD cell (git clone); the SMOKE + PROBE cells
need only the built binary + the GGUF.

Why this exists (see docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md "PACKAGING
MANIFEST"): the live generator is Qwen3.5-9B-MTP, and native MTP (`--spec-type draft-mtp`) lives in
libllama-common -> we must bundle the CUDA `llama-server` BINARY (not the llama-cpp-python wheel, which
cannot do native MTP). This notebook builds that binary for the Kaggle GPU arch, self-verifies MTP is
present, smoke-tests load+MTP+q8-KV+generate on Qwen3.5-9B-MTP, and probes the REAL per-GPU VRAM (T4 16GB
vs L4 24GB) -- the two assumptions the whole GGUF plan rests on.

After a clean run: zip /kaggle/working/llamacpp-cuda-mtp/ as a Kaggle Dataset; at submission set
CARNOT_LLAMA_SERVER=/kaggle/input/<that-dataset>/llama-server and CARNOT_ARC_GGUF_PATH=<the GGUF>."""

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "llamacpp-cuda-mtp"  # <- save this dir as a Kaggle Dataset
SRC = WORK / "llama.cpp"
CUDA_ARCHS = "60;70;75;89"  # P100=60, V100=70, T4=75, L4=89 -- cover ALL Kaggle GPUs (probe got a P100!)
# Pin a known-good upstream commit for reproducibility; bump only after re-verifying the MTP smoke below.
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
GGUF_REPO = "unsloth/Qwen3.5-9B-MTP-GGUF"
GGUF_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
PORT = 8920


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=kw.pop("check", True), **kw)


# %% CELL 1 — BUILD the CUDA llama-server with MTP (internet ON) ------------------------------------
PR = "24423"  # the validated branch our local build (9b4dae81f) is on; HAS --spec-type draft-mtp + diffusion


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        sh(f"git clone {LLAMA_REPO} {SRC}")
        # fetch + checkout the validated PR branch (plain master may lack the draft-mtp CLI value)
        sh(f"cd {SRC} && git fetch origin pull/{PR}/head && git checkout FETCH_HEAD")
    # CUDA toolkit (nvcc) + cmake are present on Kaggle GPU images; install cmake if missing.
    sh("cmake --version || pip install -q cmake", check=False)
    # Kaggle/Colab CUDA quirk: FindCUDAToolkit cannot locate the driver stub libcuda.so, so the
    # CUDA::cuda_driver IMPORTED target is missing and ggml-cuda fails at GENERATE. Point CMAKE_LIBRARY_PATH
    # at the stubs dir (auto-detect the cuda root). EXAMPLES/TESTS OFF: smaller configure + skips the
    # diffusion-example targets we do not need (llama-server lives in tools/, still built).
    stubs = "/usr/local/cuda/lib64/stubs"
    sh(
        f"cmake -S {SRC} -B {SRC}/build -DGGML_CUDA=ON "
        f'-DCMAKE_CUDA_ARCHITECTURES="{CUDA_ARCHS}" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release '
        f"-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON "
        f'-DCMAKE_LIBRARY_PATH="{stubs}" -DCUDA_cuda_driver_LIBRARY="{stubs}/libcuda.so"'
    )
    sh(f"cmake --build {SRC}/build --target llama-server -j $(nproc)")
    # collect the binary + EVERY shared lib it needs (incl. libllama-common, where MTP lives)
    server = next(SRC.glob("build/**/llama-server"))
    shutil.copy2(server, OUT / "llama-server")
    n = 0
    for so in SRC.glob("build/**/*.so*"):
        shutil.copy2(so, OUT / so.name)
        n += 1
    print(f"copied llama-server + {n} shared libs to {OUT}", flush=True)
    # SELF-VERIFY: the bundled libs MUST contain the MTP symbol, else the build is too old for draft-mtp
    libs = " ".join(str(p) for p in OUT.glob("*.so*"))
    r = subprocess.run(
        f"strings {libs} 2>/dev/null | grep -c common_speculative_impl_draft_mtp",
        shell=True,
        capture_output=True,
        text=True,
    )
    cnt = int((r.stdout or "0").strip() or 0)
    assert cnt > 0, (
        "FATAL: built llama.cpp has NO MTP (common_speculative_impl_draft_mtp). "
        "Use a newer upstream commit (draft-mtp must be merged)."
    )
    print(f"MTP symbol present ({cnt}) -- binary is MTP-capable.", flush=True)


# %% CELL 2 — fetch the GGUF (build-notebook only; at submission it is a bundled Dataset) ------------
def fetch_gguf():
    p = os.environ.get("CARNOT_ARC_GGUF_PATH")
    if p and Path(p).exists():
        return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(GGUF_REPO, GGUF_FILE)


# %% CELL 3 — SMOKE: load Qwen3.5-9B-MTP with MTP + q8 KV, confirm it generates --------------------
def _healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def smoke(gguf):
    log = open(WORK / "server.log", "w")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{OUT}:" + os.environ.get("LD_LIBRARY_PATH", ""))
    proc = subprocess.Popen(
        [
            str(OUT / "llama-server"),
            "-m",
            gguf,
            "-ngl",
            "999",
            "-c",
            "4096",
            "--spec-type",
            "draft-mtp",
            "--model-draft",
            gguf,  # MTP self-draft
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",  # 8-bit KV cache
            "--port",
            str(PORT),
            "--host",
            "127.0.0.1",
        ],
        stdout=log,
        stderr=log,
        env=env,
    )
    ok = False
    for _ in range(120):
        if _healthy():
            ok = True
            break
        time.sleep(2)
    res = {"server_started": ok}
    if ok:
        body = json.dumps(
            {
                "prompt": "/no_think\nWrite a python function is_even(n):",
                "n_predict": 48,
                "temperature": 0.2,
            }
        ).encode()
        t0 = time.time()
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion",
                data=body,
                headers={"Content-Type": "application/json"},
            ),
            timeout=300,
        ) as r:
            resp = json.load(r)
        res["generation"] = resp.get("content", "")[:200]
        res["tokens"] = resp.get("tokens_predicted")
        res["wall_s"] = round(time.time() - t0, 1)
        res["mtp_active"] = "common_speculative_impl_draft_mtp" in (WORK / "server.log").read_text()
    proc.terminate()
    return res


# %% CELL 4 — PROBE the real GPU + VRAM, then VERDICT --------------------------------------------
def probe_and_verdict(gguf, smoke_res):
    q = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free "
        "--format=csv,noheader,nounits",
        shell=True,
        capture_output=True,
        text=True,
    )
    gpu = q.stdout.strip().splitlines()[0].split(", ")
    name, total, used, free = gpu[0], int(gpu[1]), int(gpu[2]), int(gpu[3])
    gguf_gb = Path(gguf).stat().st_size / 1e9
    verdict = {
        "gpu": name,
        "vram_total_MB": total,
        "vram_used_with_model_MB": used,
        "vram_free_MB": free,
        "gguf_GB": round(gguf_gb, 1),
        "fits_with_kv_headroom": free > 1500,  # >~1.5GB free after model+MTP+q8KV loaded
        "mtp_works": smoke_res.get("mtp_active"),
        "engine_generates_offline": bool(smoke_res.get("generation")),
        "is_24gb_L4": total > 20000,  # if true, the 16GB framing relaxes (27B Q4 fits)
    }
    print(json.dumps({"smoke": smoke_res, "verdict": verdict}, indent=2))
    (WORK / "kaggle_verify_report.json").write_text(
        json.dumps({"smoke": smoke_res, "verdict": verdict}, indent=2)
    )
    print("\n=== SETTLES ===")
    print(
        f"  Real GPU: {name} ({total} MB) -> {'L4 24GB (16GB framing RELAXES)' if total > 20000 else 'T4 16GB (small-model plan stands)'}"
    )
    print(
        f"  Engine offline: built binary {'GENERATES' if verdict['engine_generates_offline'] else 'FAILED'}; "
        f"MTP {'ACTIVE' if verdict['mtp_works'] else 'NOT active'}"
    )
    print(
        f"  VRAM after load (model+MTP self-draft+q8 KV): used {used}MB, free {free}MB "
        f"-> {'adequate KV headroom' if verdict['fits_with_kv_headroom'] else 'TIGHT -- consider CARNOT_ARC_MTP=0'}"
    )
    return verdict


# %% CELL 5 — RUN ALL --------------------------------------------------------------------------------
if __name__ == "__main__":
    build()
    gguf = fetch_gguf()
    sres = smoke(gguf)
    probe_and_verdict(gguf, sres)
    print(
        f"\nDONE. Save {OUT} as a Kaggle Dataset; at submission set "
        f"CARNOT_LLAMA_SERVER={OUT}/llama-server and CARNOT_ARC_GGUF_PATH=<bundled {GGUF_FILE}>."
    )
