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
        shell=True,
        capture_output=True,
        text=True,
    )
    p = (q.stdout.strip().splitlines() or [""])[0].split(", ")
    return {"gpu": p[0], "total_MB": p[1], "used_MB": p[2], "free_MB": p[3]} if len(p) >= 4 else {}


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


# The POSITIVE marker llama.cpp prints when draft-mtp is genuinely wired up. Duplicated here rather
# than imported because this file is a STANDALONE Kaggle kernel -- the `carnot` package is not on
# its path -- and it is kept byte-identical to
# `LocalGGUFProposer._MTP_ENGAGED_MARKER` in python/carnot/agentic/arc_executable_world_model.py.
MTP_ENGAGED_MARKER = "common_speculative_impl_draft_mtp: adding speculative implementation"


def resolve_mtp_draft(main_gguf):
    """The file to pass as `--model-draft`, or `(None, reason)` when MTP is not available.

    WHY THIS IS NOT `--model-draft <main_gguf>` (fixed 2026-07-28). That is what this kernel used to
    do, and it is correct for exactly one model: a SELF-DRAFTING MTP build such as the retired
    Qwen3.5-9B-MTP, which carries its `nextn` heads inside the main GGUF. The live pin is
    gemma-4-31B-it, whose MTP head is a SEPARATE 491 MiB file (`mtp-gemma-4-31B-it-Q8_0.gguf`,
    architecture `gemma4-assistant`). Handing gemma its own weights as a draft does NOT fail --
    llama.cpp warns and serves normally with speculation SILENTLY DISABLED -- so this dry-run,
    whose entire job is to report whether MTP works on the target hardware, would have reported a
    passing "mtp" arm while measuring the mtp-off regime. A dry-run that cannot fail is not a gate.

    Head detection is by the `mtp-` FILENAME PREFIX, which is upstream's convention. Note that a
    plain substring test would be wrong in the other direction: `Qwen3.5-9B-MTP-Q4_K_M.gguf`
    CONTAINS "MTP-" and is main weights, not a head.
    """
    name = Path(str(main_gguf)).name
    low = name.lower()
    if low.startswith("mtp-"):
        return None, f"refusing to use the draft head {name!r} as the main model"
    if "-mtp" in low or "_mtp" in low:
        return main_gguf, ""  # self-drafting build: its own file carries the heads
    root = Path("/kaggle/input")
    if root.exists():
        for p in sorted(root.rglob("*.gguf")):
            if p.name.lower().startswith("mtp-"):
                return p, ""
    return None, (
        f"{name!r} is not a self-drafting MTP build and no `mtp-*.gguf` head was found under "
        "/kaggle/input. Launching WITHOUT speculative decoding -- this arm is MTP-OFF and must not "
        "be reported as MTP-on. Attach the iancblenke/carnot-gemma4-31b-mtp-head dataset."
    )


def main():
    global SERVER, GGUF, BIN_DIR
    REPORT["gpu_before"] = vram()
    REPORT["kaggle_input"] = (
        sorted(os.listdir("/kaggle/input")) if Path("/kaggle/input").exists() else []
    )
    # Kaggle's dataset mount path varies (here it nests under /kaggle/input/datasets/...), so SELF-LOCATE
    # the bundled files anywhere under /kaggle/input rather than assume a fixed path.
    servers = list(Path("/kaggle/input").rglob("llama-server"))
    ggufs = list(Path("/kaggle/input").rglob("*.gguf"))
    if servers:
        SERVER = servers[0]
        BIN_DIR = SERVER.parent
    if ggufs:
        GGUF = ggufs[0]
    REPORT["found_server"] = str(SERVER) if servers else None
    REPORT["found_gguf"] = str(GGUF) if ggufs else None
    REPORT["binary_exists"] = SERVER.exists()
    REPORT["gguf_exists"] = GGUF.exists()
    if not SERVER.exists() or not GGUF.exists():
        REPORT["error"] = "missing binary or gguf dataset"
        (WORK / "smoke_report.json").write_text(json.dumps(REPORT, indent=2))
        print(json.dumps(REPORT, indent=2))
        return
    # /kaggle/input is a READ-ONLY filesystem, so chmod-in-place fails. Copy the binary to writable
    # /kaggle/working and chmod the copy; the libs stay read-only in /kaggle/input (readable is enough).
    import shutil

    lib_dir = SERVER.parent
    run_server = WORK / "llama-server"
    shutil.copy2(SERVER, run_server)
    os.chmod(run_server, 0o755)
    SERVER = run_server
    REPORT["lib_dir"] = str(lib_dir)
    env = dict(os.environ, LD_LIBRARY_PATH=f"{lib_dir}:" + os.environ.get("LD_LIBRARY_PATH", ""))

    # 2026-06-21: the SUBMISSION agent runs the generator at n_ctx=16384 (arc_executable_world_model.py),
    # NOT the 4096 the prior smoke used. ctx=16384+MTP on a 16GB P100 is the suspected OOM cliff that
    # silently degrades the eval agent to CPU graph-explore (-> the stuck 0.08). Test the REAL submission
    # config AND the proposed CARNOT_ARC_MTP=0 fallback, each in a fresh server, so the result is directly
    # actionable: which config actually loads on the eval GPU.
    CONFIGS = [
        {
            "name": "submission_ctx16384_mtp_on",
            "n_ctx": 16384,
            "mtp": True,
        },  # the exact frozen config
        {"name": "fallback_ctx16384_mtp_off", "n_ctx": 16384, "mtp": False},  # the MTP=0 fix
        {"name": "baseline_ctx4096_mtp_on", "n_ctx": 4096, "mtp": True},  # known-good control
    ]
    results = []
    for cfg in CONFIGS:
        logp = WORK / f"server_{cfg['name']}.log"
        log = open(logp, "w")
        args = [
            str(SERVER),
            "-m",
            str(GGUF),
            "-ngl",
            "999",
            "-c",
            str(cfg["n_ctx"]),
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--port",
            str(PORT),
            "--host",
            "127.0.0.1",
        ]
        if cfg["mtp"]:
            # THE DRAFT IS THE HEAD, NOT THE MAIN WEIGHTS -- unless the main weights are a
            # self-drafting MTP build. This line used to read `--model-draft str(GGUF)`
            # unconditionally, which is right only for the retired Qwen3.5-9B-MTP (heads embedded
            # in the main GGUF) and wrong for the live gemma-4-31B-it, whose head is a SEPARATE
            # 491 MiB file. Passing the main weights as their own draft under gemma does not fail:
            # llama.cpp warns and serves with speculation SILENTLY DISABLED, so this dry-run would
            # report an "mtp" arm that measures the mtp-off regime.
            _draft, _why_not = resolve_mtp_draft(GGUF)
            if _draft:
                args += ["--spec-type", "draft-mtp", "--model-draft", str(_draft)]
            else:
                cfg["mtp_engaged"] = False
                cfg["mtp_not_engaged_reason"] = _why_not
                print(f"[dryrun] MTP NOT ENGAGED: {_why_not}", flush=True)
        proc = subprocess.Popen(args, stdout=log, stderr=log, env=env)
        ok = False
        for _ in range(180):  # ctx=16384 load can be slower; allow ~360s
            if healthy():
                ok = True
                break
            if proc.poll() is not None:  # server died (OOM) -> stop waiting, capture why
                break
            time.sleep(2)
        # MTP ENGAGEMENT IS READ FROM THE SERVER'S OWN LOG, NEVER INFERRED FROM `ok`. A healthy
        # server is NOT evidence that speculation is on: llama.cpp accepts an unusable draft, warns,
        # and serves normally, so the mtp-on and silently-disabled cases are identical from the
        # client side except by tok/s. Three-valued -- None means "MTP was not requested".
        mtp_engaged = None
        if cfg["mtp"]:
            try:
                mtp_engaged = MTP_ENGAGED_MARKER in logp.read_text(errors="replace")
            except OSError:
                mtp_engaged = None  # log unreadable -> undetermined, NOT success
        r = {
            "config": cfg["name"],
            "n_ctx": cfg["n_ctx"],
            "mtp": cfg["mtp"],
            "mtp_engaged": mtp_engaged,
            "mtp_not_engaged_reason": cfg.get("mtp_not_engaged_reason", ""),
            "server_started": ok,
            "gpu_after_load": vram() if ok else None,
            "exited_early": proc.poll() is not None,
        }
        if ok:
            body = json.dumps(
                {
                    "prompt": "/no_think\nWrite a python function is_even(n):",
                    "n_predict": 32,
                    "temperature": 0.2,
                }
            ).encode()
            t0 = time.time()
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(
                        f"http://127.0.0.1:{PORT}/completion",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=300,
                ) as resp_r:
                    resp = json.load(resp_r)
                r["generated_ok"] = bool(resp.get("content"))
                r["tok_per_s"] = round(
                    (resp.get("tokens_predicted") or 0) / max(time.time() - t0, 0.1), 1
                )
            except Exception as e:
                r["generated_ok"] = False
                r["gen_error"] = f"{type(e).__name__}: {e}"
        log.flush()
        tail = logp.read_text(errors="replace").splitlines()[-12:]
        r["oom_in_log"] = any(
            (
                "out of memory" in ln.lower()
                or "cudamalloc" in ln.lower()
                or "failed to allocate" in ln.lower()
            )
            for ln in tail
        )
        r["log_tail"] = tail
        results.append(r)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        # let VRAM free before the next config
        for _ in range(15):
            if not healthy():
                break
            time.sleep(2)
        time.sleep(3)

    REPORT["configs_tested"] = results
    REPORT["ok"] = any(r["server_started"] for r in results)
    REPORT["VERDICT"] = {
        "submission_config_loads": next(
            (r["server_started"] for r in results if r["config"] == "submission_ctx16384_mtp_on"),
            None,
        ),
        "mtp_off_fallback_loads": next(
            (r["server_started"] for r in results if r["config"] == "fallback_ctx16384_mtp_off"),
            None,
        ),
    }
    (WORK / "smoke_report.json").write_text(json.dumps(REPORT, indent=2))
    print(
        json.dumps(
            {
                "VERDICT": REPORT["VERDICT"],
                "configs": [{k: v for k, v in r.items() if k != "log_tail"} for r in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # always leave a report, even on an unhandled crash
        import traceback

        REPORT["error"] = f"{type(e).__name__}: {e}"
        REPORT["traceback"] = traceback.format_exc().splitlines()[-12:]
        (WORK / "smoke_report.json").write_text(json.dumps(REPORT, indent=2))
        print("CRASH:", REPORT["error"])
