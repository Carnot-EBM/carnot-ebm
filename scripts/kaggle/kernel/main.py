"""Kaggle BUILD-ONLY kernel: compile the CUDA llama-server-with-MTP binary for the ARC submission and
save it (+ its shared libs) as the kernel output, to be turned into a Kaggle Dataset that the
submission attaches and points CARNOT_LLAMA_SERVER at.

Why build-only (the v4 failure fix): the Kaggle GPU image has only ~20 GB writable /kaggle/working.
The earlier kernel downloaded the 5.9 GB GGUF *and* built 4 CUDA archs into that 20 GB -> almost
certainly disk exhaustion. This kernel does NOT touch the GGUF at all (the GGUF + the smoke test are
separate concerns done where we control disk). It ALSO captures df/free + the full build log to
/kaggle/working so that, if it fails again, `kaggle kernels output` retrieves the real error instead of
just the source tree.

Output (becomes a Dataset): /kaggle/working/llamacpp-cuda-mtp/{llama-server, *.so*} + build_report.json
+ build.log. Build env matches Kaggle's CUDA 12.8 / driver exactly (no local-CUDA-version mismatch)."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "llamacpp-cuda-mtp"
SRC = WORK / "llama.cpp"
# P100=60 (probed eval GPU), T4=75, L4/RTX6000-Ada=89, +PTX JIT fallback. Without the 5.9GB GGUF the
# 4-arch build should fit the 20GB disk; if build_report shows disk pressure, cut to "60;60-virtual".
CUDA_ARCHS = "60;75;89;89-virtual"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
RELEASE_TAG = "b9714"  # MTP (--spec-type draft-mtp) is in this upstream release
REPORT = {"stages": [], "ok": False}
LOG = WORK / "build.log"


def _disk():
    r = subprocess.run("df -m /kaggle/working | tail -1", shell=True, capture_output=True, text=True)
    parts = r.stdout.split()
    return {"used_MB": int(parts[2]), "avail_MB": int(parts[3])} if len(parts) >= 4 else {}


def stage(name):
    snap = {"stage": name, "t": round(time.time(), 1), "disk": _disk()}
    REPORT["stages"].append(snap)
    print(f"[stage] {name} disk={snap['disk']}", flush=True)


def sh(cmd):
    # tee EVERYTHING to build.log so a failure is recoverable via kaggle kernels output
    print("+", cmd, flush=True)
    with open(LOG, "a") as lf:
        lf.write(f"\n$ {cmd}\n")
    rc = subprocess.run(f"({cmd}) 2>&1 | tee -a {LOG}", shell=True).returncode
    if rc != 0:
        raise RuntimeError(f"command failed (rc={rc}): {cmd}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")
    try:
        stage("start")
        if not SRC.exists():
            sh(f"git clone --depth 1 --branch {RELEASE_TAG} {LLAMA_REPO} {SRC}")
        stage("cloned")
        sh("cmake --version || pip install -q cmake")
        # ggml-cuda LINKS against the CUDA driver lib (libcuda). v5 failed here: the toolkit stub
        # /usr/local/cuda/lib64/stubs/libcuda.so does NOT exist on the Kaggle image. Locate the REAL
        # driver lib (libcuda.so.1, present because the GPU is on) and symlink it to the unversioned
        # name cmake wants, in a WRITABLE dir; point cmake at that.
        stubs = str(WORK / "cudastub")
        os.makedirs(stubs, exist_ok=True)
        find = subprocess.run(
            "find /usr /lib -name 'libcuda.so*' 2>/dev/null | sort", shell=True, capture_output=True, text=True
        )
        cands = [ln for ln in find.stdout.splitlines() if ln.strip()]
        REPORT["libcuda_candidates"] = cands
        real = next((c for c in cands if c.endswith("libcuda.so")), None) or (cands[0] if cands else None)
        if not real:
            raise RuntimeError("no libcuda.so* found on the image; cannot link ggml-cuda")
        sh(f"ln -sf {real} {stubs}/libcuda.so")
        sh(
            f"cmake -S {SRC} -B {SRC}/build -DGGML_CUDA=ON "
            f'-DCMAKE_CUDA_ARCHITECTURES="{CUDA_ARCHS}" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release '
            f"-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON "
            f'-DCMAKE_LIBRARY_PATH="{stubs}" -DCUDA_cuda_driver_LIBRARY="{stubs}/libcuda.so"'
        )
        stage("configured")
        # -j2 (not nproc): nvcc is memory-heavy; fewer parallel jobs avoids an OOM-kill on the builder.
        sh(f"cmake --build {SRC}/build --target llama-server -j 2")
        stage("built")
        server = next(SRC.glob("build/**/llama-server"))
        shutil.copy2(server, OUT / "llama-server")
        n = 0
        for so in SRC.glob("build/**/*.so*"):
            shutil.copy2(so, OUT / so.name)
            n += 1
        # MTP self-verify: the symbol common_speculative_impl_draft_mtp MUST be present
        libs = " ".join(str(p) for p in OUT.glob("*.so*"))
        r = subprocess.run(
            f"strings {libs} 2>/dev/null | grep -c common_speculative_impl_draft_mtp",
            shell=True, capture_output=True, text=True,
        )
        mtp = int((r.stdout or "0").strip() or 0)
        bin_mb = round((OUT / "llama-server").stat().st_size / 1e6, 1)
        REPORT.update({"ok": mtp > 0, "shared_libs": n, "binary_MB": bin_mb, "mtp_symbol_count": mtp})
        stage("collected")
        # free the source tree so the Dataset output is just the binary + libs (smaller)
        shutil.rmtree(SRC, ignore_errors=True)
        stage("cleaned")
        print(f"BUILD OK: llama-server {bin_mb}MB + {n} libs, MTP symbol x{mtp}", flush=True)
    except Exception as e:  # capture the failure into the report so the output download shows it
        REPORT["error"] = f"{type(e).__name__}: {e}"
        REPORT["log_tail"] = LOG.read_text(errors="replace").splitlines()[-40:] if LOG.exists() else []
        print(f"BUILD FAILED: {REPORT['error']}", flush=True)
    (WORK / "build_report.json").write_text(json.dumps(REPORT, indent=2))
    print("wrote build_report.json", flush=True)


if __name__ == "__main__":
    main()
