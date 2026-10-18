"""OFFLINE import probe — does the CARNOT AGENT IMPORT under the competition RERUN's exact
dependency conditions? Retires the one residual submission risk (jax/numpy preinstalled on the
rerun image for the carnot import) BEFORE a scored submission is spent on it.

Mirrors the rerun environment as closely as a regular kernel can:
  * internet OFF;
  * deps from the COMPETITION wheels via `pip install --no-index --find-links .../arc_agi_3_wheels`
    (arc-agi python-dotenv) — exactly what the canonical control notebook installs;
  * the real Agent base loaded from the COMPETITION-PROVIDED ARC-AGI-3-Agents framework;
  * carnot from our attached dataset.

Reports a clear matrix so we know whether the real submission will import clean, or whether we
must stage missing deps (jax/numpy/...) as a wheels dataset. Writes /kaggle/working/probe_report.json
always (even on crash). No GPU needed — import only.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
COMP = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
REPORT = {"ok": False}


def _try(name, fn):
    try:
        REPORT[name] = fn()
    except Exception as e:
        REPORT[name] = f"FAIL: {type(e).__name__}: {e}"


def main():
    inp = Path("/kaggle/input")
    REPORT["kaggle_input"] = sorted(os.listdir(inp)) if inp.exists() else []
    REPORT["competition_dir_present"] = COMP.exists()
    wheels = COMP / "arc_agi_3_wheels"
    REPORT["comp_wheels_present"] = wheels.exists()
    REPORT["comp_wheels_sample"] = sorted([p.name for p in wheels.glob("*")])[:12] if wheels.exists() else []

    # (a) BASE IMAGE deps — the load-bearing question: are jax + numpy already there?
    _try("base_jax", lambda: __import__("jax").__version__)
    _try("base_numpy", lambda: __import__("numpy").__version__)
    _try("base_pandas", lambda: __import__("pandas").__version__)

    # (b) competition wheels install (offline, --no-index) — exactly the rerun's deps
    if wheels.exists():
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index",
             "--find-links", str(wheels), "arc-agi", "python-dotenv", "--quiet"],
            capture_output=True, text=True,
        )
        REPORT["pip_install_arc_agi_rc"] = cp.returncode
        REPORT["pip_install_tail"] = (cp.stderr or cp.stdout or "")[-400:]
    else:
        REPORT["pip_install_arc_agi_rc"] = "skipped_no_wheels"

    _try("arcengine_import", lambda: __import__("arcengine").__name__)
    _try("arc_agi_import", lambda: __import__("arc_agi").__name__)

    # (c) the real Agent base from the COMPETITION-PROVIDED framework (load agent.py directly to
    #     dodge the stock agents/__init__.py which eagerly imports langgraph/smolagents).
    def _load_agent_base():
        import importlib.util

        agent_py = COMP / "ARC-AGI-3-Agents" / "agents" / "agent.py"
        if not agent_py.exists():
            # fall back to our local clone path shape if competition framework isn't mounted
            raise FileNotFoundError(f"no framework agent.py at {agent_py}")
        # agents.agent imports `from .recorder import Recorder` etc -> need the package on path
        fwroot = str((COMP / "ARC-AGI-3-Agents"))
        if fwroot not in sys.path:
            sys.path.insert(0, fwroot)
        spec = importlib.util.spec_from_file_location("agents.agent", str(agent_py))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Agent.__name__

    _try("framework_agent_base", _load_agent_base)

    # (d) CARNOT import under these conditions — the actual risk
    carnot = next((p.parents[2] for p in inp.rglob("carnot/agentic/arc_competition_agent.py")), None)
    REPORT["carnot_pkg_root"] = str(carnot)
    if carnot:
        sys.path.insert(0, str(carnot))
        import time

        t0 = time.time()
        _try("carnot_import", lambda: __import__(
            "carnot.agentic.arc_competition_agent", fromlist=["make_carnot_agent"]).__name__)
        REPORT["carnot_import_s"] = round(time.time() - t0, 1)
        REPORT["jax_after_carnot"] = "jax" in sys.modules

        # (e) make_carnot_agent builds the class against the real Agent base
        def _build():
            from arcengine import GameAction  # noqa: F401
            from carnot.agentic.arc_competition_agent import make_carnot_agent

            class _B:
                def __init__(self, *a, **k):
                    self.game_id = "probe"

            cls = make_carnot_agent(_B)
            return {"class": cls.__name__, "max_actions": cls.MAX_ACTIONS}

        _try("make_carnot_agent_builds", _build)

    # bundled engine present (the offline generator)?
    REPORT["llama_server_present"] = bool(next(iter(inp.rglob("llama-server")), None))
    REPORT["gguf_present"] = bool(next(iter(inp.rglob("*.gguf")), None))

    REPORT["ok"] = (
        isinstance(REPORT.get("carnot_import"), str)
        and not str(REPORT.get("carnot_import", "")).startswith("FAIL")
        and isinstance(REPORT.get("make_carnot_agent_builds"), dict)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        REPORT["fatal"] = f"{type(e).__name__}: {e}"
        REPORT["traceback"] = traceback.format_exc().splitlines()[-12:]
    (WORK / "probe_report.json").write_text(json.dumps(REPORT, indent=2))
    print(json.dumps(REPORT, indent=2))
