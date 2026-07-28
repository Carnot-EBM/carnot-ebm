"""Measure dense-FFN CPU offload for gemma-4-31B-it on one RTX 3090.

Per the operator directive (2026-07-28) deliverable (3): the dense lever is `-ot` with a
regex matching the FFN tensors (`-cmoe`/`-ncmoe` are MoE-only and inert on a dense model).
We measure, per arm: per-PID resident VRAM (nvidia-smi compute-apps, NOT the env var) and
throughput on a FIXED prompt.

Nothing here submits anything and nothing touches the repo. Teardown is by explicit PID.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

GGUF = (
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
    "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
)
SERVER = os.path.expanduser("~/.cache/llama.cpp-master/build/bin/llama-server")
PORT = int(os.environ.get("MEAS_PORT", "8971"))
GPU = os.environ.get("MEAS_GPU", "1")
N_LAYERS = 60  # gemma4.block_count, read from the GGUF header

# A fixed, realistic prompt: same shape class as induce_prompt() but small enough that
# prefill does not dominate the decode-rate measurement.
PROMPT = (
    "/no_think\nYou are given a grid world. Write a Python function `step(grid, action)` that "
    "returns the next grid. The grid is a 2D list of ints 0-9. Actions are 1..5.\n"
    "Observed transitions:\n"
    + "\n".join(
        f"action={a} before=[[0,0,{a}],[0,{a},0]] after=[[0,{a},0],[{a},0,0]]" for a in range(1, 6)
    )
    + "\n\nWrite the complete function now.\n```python\n"
)


def ffn_regex(n_cpu_layers: int) -> str:
    """`-ot` pattern keeping the FFN weights of the FIRST n layers on the CPU.

    Tensor names come from the GGUF header (833 tensors, arch `gemma4`): every block has
    `blk.<i>.ffn_gate.weight`, `blk.<i>.ffn_up.weight`, `blk.<i>.ffn_down.weight`. There is
    no `ffn_*_exps` tensor -- this model is DENSE, which is exactly why -cmoe does nothing.
    """
    idx = "|".join(str(i) for i in range(n_cpu_layers))
    return rf"blk\.({idx})\.ffn_(gate|up|down)\.weight=CPU"


def pid_vram_mib(pid: int) -> int:
    """Resident VRAM for one PID, read from nvidia-smi's compute-apps table.

    Deliberately per-PID and not `memory.used` / not the env var: the directive requires
    proving WHICH card the process actually landed on from residency, and a whole-card
    reading cannot attribute memory to our server.
    """
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return -1


def pid_card_index(pid: int) -> str:
    """Which physical GPU the PID is resident on, by UUID join -- proof, not the env var."""
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    uuid = None
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            uuid = parts[1]
    if not uuid:
        return "unknown"
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    for line in gpus.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1] == uuid:
            return parts[0]
    return "unknown"


def run_arm(label: str, n_cpu_layers: int, n_ctx: int, n_predict: int = 256) -> dict:
    logpath = f"/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/ffn_{label}.err"
    args = [
        SERVER, "-m", GGUF,
        "-ngl", "999",
        "-c", str(n_ctx),
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    ]
    if n_cpu_layers > 0:
        args += ["-ot", ffn_regex(n_cpu_layers)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    rec: dict = {
        "arm": label,
        "n_cpu_ffn_layers": n_cpu_layers,
        "n_ctx": n_ctx,
        "argv_has_ot": any(a == "-ot" for a in args),
        "ot_regex": ffn_regex(n_cpu_layers) if n_cpu_layers else None,
        "argv": args,
    }
    err = open(logpath, "w")
    proc = subprocess.Popen(args, stdout=err, stderr=err, env=env)
    try:
        ok = False
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                rec["error"] = f"server exited rc={proc.returncode}"
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                    if r.status == 200:
                        ok = True
                        break
            except Exception:
                time.sleep(2)
        rec["load_s"] = round(time.time() - t0, 1)
        if not ok:
            rec.setdefault("error", "health timeout")
            return rec
        time.sleep(3)
        rec["resident_mib"] = pid_vram_mib(proc.pid)
        rec["resident_card_index"] = pid_card_index(proc.pid)
        rec["pid"] = proc.pid

        # two passes: first warms caches, second is the reported number
        for i in range(2):
            body = json.dumps({
                "prompt": PROMPT,
                "n_predict": n_predict,
                "temperature": 0.0,
                "cache_prompt": False,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion", data=body,
                headers={"Content-Type": "application/json"},
            )
            t1 = time.time()
            with urllib.request.urlopen(req, timeout=1200) as r:
                d = json.loads(r.read())
            wall = time.time() - t1
            tim = d.get("timings", {}) or {}
            if i == 1:
                rec["gen_wall_s"] = round(wall, 2)
                rec["predicted_n"] = d.get("tokens_predicted") or tim.get("predicted_n")
                rec["prompt_n"] = d.get("tokens_evaluated") or tim.get("prompt_n")
                rec["decode_tok_s"] = round(float(tim.get("predicted_per_second", 0.0)), 2)
                rec["prefill_tok_s"] = round(float(tim.get("prompt_per_second", 0.0)), 2)
        rec["resident_mib_after"] = pid_vram_mib(proc.pid)
    finally:
        proc.terminate()  # explicit PID, never pkill -f
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        err.close()
    # scrape the server's own load log for the offload accounting -- proof the flag was
    # not merely accepted-and-ignored
    try:
        txt = open(logpath, errors="replace").read()
        for pat, key in (
            (r"load_tensors:.*CUDA0 model buffer size\s*=\s*([\d.]+)", "log_cuda0_model_mib"),
            (r"load_tensors:.*CPU model buffer size\s*=\s*([\d.]+)", "log_cpu_model_mib"),
            (r"llama_context:.*KV self size\s*=\s*([\d.]+)", "log_kv_mib"),
        ):
            m = re.search(pat, txt)
            if m:
                rec[key] = float(m.group(1))
        rec["log_tensor_override_lines"] = len(
            [ln for ln in txt.splitlines() if "override" in ln.lower() and "tensor" in ln.lower()]
        )
    except Exception as exc:
        rec["log_scrape_error"] = repr(exc)
    return rec


if __name__ == "__main__":
    arms = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [[0, 32768]]
    out = []
    for n_cpu, n_ctx in arms:
        label = f"cpu{n_cpu}_ctx{n_ctx}"
        print(f"=== arm {label} ===", flush=True)
        rec = run_arm(label, n_cpu, n_ctx)
        print(json.dumps({k: v for k, v in rec.items() if k != "argv"}), flush=True)
        out.append(rec)
        time.sleep(5)
    # OUTPUT PATH IS AN ARGUMENT, not a constant. It used to be a hardcoded
    # `ffn_offload_results.json`, so the second invocation (the n_ctx 81920 cross-check) silently
    # OVERWROTE the first (the 0/12/24/40 sweep at n_ctx 32768) and the sweep's VRAM column
    # survived only as prose in a docstring. A measurement you cannot re-read is a claim, not
    # evidence.
    default_out = (
        "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
        "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/ffn_offload_results.json"
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_out
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"=== wrote {out_path} ===", flush=True)
