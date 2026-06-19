"""OFFLINE full-agent dry-run TIER 1: does the CARNOT AGENT STACK run offline on Kaggle with the LLM?
Isolates the two carnot-side unknowns (the game harness is the competition's, provided at eval):
  (1) does `make_carnot_agent` import offline? -- its closure pulls JAX + ~83 carnot modules (the GAP
      verifiers drag in the EBM stack); if jax/dep is missing on the Kaggle image, this fails here.
  (2) does the LLM proposer run THROUGH the bundled binary? -- create a LocalGGUFProposer pointed at the
      bundled llama-server + GGUF and generate, confirming the carnot->binary->Qwen path works end-to-end.
Attaches: carnot-agent-code + carnot-llamacpp-mtp-binary + carnot-qwen35-9b-mtp-gguf. internet OFF.
Writes /kaggle/working/agent_report.json (always, even on crash)."""

import json
import os
import shutil
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPORT = {"ok": False}


def main():
    inp = Path("/kaggle/input")
    REPORT["kaggle_input"] = sorted(os.listdir(inp)) if inp.exists() else []
    # self-locate the 3 bundles anywhere under /kaggle/input (mount nests under .../datasets/...)
    carnot = next((p.parent for p in inp.rglob("carnot/agentic/arc_competition_agent.py")), None)
    server = next(iter(inp.rglob("llama-server")), None)
    gguf = next(iter(inp.rglob("*.gguf")), None)
    REPORT.update(carnot_pkg=str(carnot), server=str(server), gguf=str(gguf))
    if not (carnot and server and gguf):
        raise RuntimeError("missing carnot-code / binary / gguf bundle under /kaggle/input")

    # binary -> writable + on LD_LIBRARY_PATH so its libs (read-only in /kaggle/input) resolve
    run_server = WORK / "llama-server"
    shutil.copy2(server, run_server)
    os.chmod(run_server, 0o755)
    os.environ["LD_LIBRARY_PATH"] = f"{server.parent}:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["CARNOT_LLAMA_SERVER"] = str(run_server)
    os.environ["CARNOT_ARC_GGUF_PATH"] = str(gguf)
    sys.path.insert(0, str(carnot))

    # (1) IMPORT TEST -- the jax / dep closure
    import time

    t0 = time.time()
    from carnot.agentic.arc_competition_agent import make_carnot_agent  # noqa: F401
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    REPORT["agent_import_ok"] = True
    REPORT["import_s"] = round(time.time() - t0, 1)
    REPORT["jax_imported"] = "jax" in sys.modules

    # (2) PROPOSER-THROUGH-BINARY TEST -- the carnot->binary->Qwen path
    prop = LocalGGUFProposer(
        model_path=str(gguf), mtp=True, kv_quant="q8_0", no_think_prefix="/no_think\n", max_tokens=256
    )
    t1 = time.time()
    ok, code = prop.generate(
        "Write a python function is_win(grid) that returns grid[0][0] == 1. Return ONLY the function.",
        required=("is_win",),
    )
    REPORT["proposer_ran"] = bool(ok)
    REPORT["proposer_wall_s"] = round(time.time() - t1, 1)
    REPORT["proposer_output"] = (code or "")[:240]
    REPORT["ok"] = bool(REPORT.get("agent_import_ok") and ok)
    q = os.popen("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits").read().strip()
    REPORT["gpu_after"] = q


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        REPORT["error"] = f"{type(e).__name__}: {e}"
        REPORT["traceback"] = traceback.format_exc().splitlines()[-15:]
    (WORK / "agent_report.json").write_text(json.dumps(REPORT, indent=2))
    print(json.dumps({k: v for k, v in REPORT.items() if k != "traceback"}, indent=2))
