"""OFFLINE dry-run #1 (the critical gate): does the v7-built CUDA llama-server load Qwen3.5-9B-MTP on
the REAL Kaggle P100 (16GB) with MTP + q8 KV and GENERATE -- with internet OFF, using only the bundled
Datasets? The binary was built for sm_60 but has never run on a P100; locally it was only validated on
a 3090. This settles the last real submission unknown before the full-agent dry-run.

Attaches: carnot-llamacpp-mtp-binary (the llama-server + libs) + carnot-qwen35-9b-mtp-gguf (the 5.9GB
GGUF). enable_internet=FALSE (the eval sandbox). Writes /kaggle/working/smoke_report.json."""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
BIN_DIR = Path("/kaggle/input/carnot-llamacpp-mtp-binary")
GGUF = Path("/kaggle/input/carnot-qwen35-9b-mtp-gguf/Qwen3.5-9B-Q4_K_M.gguf")
SERVER = BIN_DIR / "llama-server"
PORT = 8920
REPORT = {"ok": False}


def vram():
    q = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True,
    )
    p = (q.stdout.strip().splitlines() or [""])[0].split(", ")
    return {"gpu": p[0], "total_MB": p[1], "used_MB": p[2], "free_MB": p[3]} if len(p) >= 4 else {}


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    global SERVER, GGUF, BIN_DIR
    REPORT["gpu_before"] = vram()
    REPORT["kaggle_input"] = sorted(os.listdir("/kaggle/input")) if Path("/kaggle/input").exists() else []
    # Kaggle's dataset mount path varies (here it nests under /kaggle/input/datasets/...), so SELF-LOCATE
    # the bundled files anywhere under /kaggle/input rather than assume a fixed path.
    servers = list(Path("/kaggle/input").rglob("llama-server"))
    ggufs = list(Path("/kaggle/input").rglob("*.gguf"))
    if servers:
        SERVER = servers[0]; BIN_DIR = SERVER.parent
    if ggufs:
        GGUF = ggufs[0]
    REPORT["found_server"] = str(SERVER) if servers else None
    REPORT["found_gguf"] = str(GGUF) if ggufs else None
    REPORT["binary_exists"] = SERVER.exists()
    REPORT["gguf_exists"] = GGUF.exists()
    if not SERVER.exists() or not GGUF.exists():
        REPORT["error"] = "missing binary or gguf dataset"
        (WORK / "smoke_report.json").write_text(json.dumps(REPORT, indent=2))
        print(json.dumps(REPORT, indent=2)); return
    os.chmod(SERVER, 0o755)  # datasets may drop the exec bit
    log = open(WORK / "server.log", "w")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{BIN_DIR}:" + os.environ.get("LD_LIBRARY_PATH", ""))
    proc = subprocess.Popen(
        [str(SERVER), "-m", str(GGUF), "-ngl", "999", "-c", "4096",
         "--spec-type", "draft-mtp", "--model-draft", str(GGUF),  # MTP self-draft
         "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",       # 8-bit KV
         "--port", str(PORT), "--host", "127.0.0.1"],
        stdout=log, stderr=log, env=env,
    )
    ok = False
    for _ in range(120):  # model load on a P100 can take ~30-60s
        if healthy():
            ok = True; break
        if proc.poll() is not None:  # server died -> capture why
            break
        time.sleep(2)
    REPORT["server_started"] = ok
    REPORT["gpu_after_load"] = vram()
    if ok:
        body = json.dumps({"prompt": "/no_think\nWrite a python function is_even(n):",
                           "n_predict": 48, "temperature": 0.2}).encode()
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion", data=body,
                headers={"Content-Type": "application/json"}), timeout=300) as r:
                resp = json.load(r)
            REPORT["generation"] = (resp.get("content") or "")[:200]
            REPORT["tokens_predicted"] = resp.get("tokens_predicted")
            REPORT["wall_s"] = round(time.time() - t0, 1)
            REPORT["tok_per_s"] = round((resp.get("tokens_predicted") or 0) / max(time.time() - t0, 0.1), 1)
            REPORT["ok"] = bool(REPORT["generation"])
        except Exception as e:
            REPORT["error"] = f"generation failed: {type(e).__name__}: {e}"
    log.flush()
    tail = (WORK / "server.log").read_text(errors="replace").splitlines()[-25:]
    REPORT["mtp_active"] = any("draft" in ln.lower() and "mtp" in ln.lower() for ln in tail)
    REPORT["server_log_tail"] = tail
    proc.terminate()
    (WORK / "smoke_report.json").write_text(json.dumps(REPORT, indent=2))
    print(json.dumps({k: v for k, v in REPORT.items() if k != "server_log_tail"}, indent=2))


if __name__ == "__main__":
    main()
