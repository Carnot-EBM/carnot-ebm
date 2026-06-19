"""ARC-AGI-3 Kaggle competition SUBMISSION notebook — CarnotAgent.

Modeled EXACTLY on the canonical arcprize control notebook pattern (Ronan McGovern's
`arc3-random-control`, the reference submission shape). The competition is a CODE
competition with an INTERNAL game GATEWAY — NOT the offline-bundled-games shape our
earlier checklist assumed. The real contract:

  * Two modes, gated on env `KAGGLE_IS_COMPETITION_RERUN`:
      - NON-rerun (when you Save & Run to submit): write the placeholder
        `/kaggle/working/submission.parquet` so the notebook is a valid submission.
      - RERUN (Kaggle's actual scored eval): wait for the internal game gateway at
        `http://gateway:8001/api/games`, copy the COMPETITION-PROVIDED ARC-AGI-3-Agents
        framework, drop in our CarnotAgent as `my_agent.py`, register it, and run
        `main.py --agent=carnotagent` against the gateway.
  * Dependencies come from competition-provided wheels (`--no-index`); internet is OFF.
    The game API is the internal gateway, not the public three.arcprize.org.

Our local Qwen3.5-9B-MTP generator runs OFFLINE via the bundled `llama-server` binary +
GGUF (attached datasets) — internet-off is fine because that path is a pure disk read +
local GPU. If the bundled engine is unavailable the agent degrades gracefully to the
CPU graph-explore cascade (try/except in arc_executable_world_model._induce_and_plan), so
the submission still plays games even if the LLM tier is unavailable.

Attach as datasets: carnot-agent-code, carnot-llamacpp-mtp-binary, carnot-qwen35-9b-mtp-gguf.
Add the competition as a data source. GPU on. Internet OFF (the gateway is internal).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"

# 1) competition-provided wheels, offline (mirrors the canonical control notebook)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index",
     "--find-links", f"{COMP}/arc_agi_3_wheels", "arc-agi", "python-dotenv", "--quiet"],
    check=False,
)

# 2) author my_agent.py — our CarnotAgent wired to the bundled OFFLINE generator.
#    The carnot import (jax + the agentic stack) is validated to import offline on the
#    Kaggle image (agent dry-run: 3.6s). The bundled llama-server is copied to a writable
#    path (/kaggle/input is read-only) and pointed at via CARNOT_LLAMA_SERVER / GGUF env.
AGENT_SRC = r'''
import os, shutil, sys
from pathlib import Path

inp = Path("/kaggle/input")
# self-locate the bundled carnot package (mount nests under .../datasets/<owner>/<slug>/);
# arc_competition_agent.py is at .../python/carnot/agentic/... -> sys.path needs .../python
carnot = next(p.parents[2] for p in inp.rglob("carnot/agentic/arc_competition_agent.py"))
sys.path.insert(0, str(carnot))

server = next(iter(inp.rglob("llama-server")), None)
gguf = next(iter(inp.rglob("*.gguf")), None)
if server and gguf:
    run_server = Path("/kaggle/working/llama-server")
    shutil.copy2(server, run_server)
    os.chmod(run_server, 0o755)
    os.environ["LD_LIBRARY_PATH"] = f"{server.parent}:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["CARNOT_LLAMA_SERVER"] = str(run_server)
    os.environ["CARNOT_ARC_GGUF_PATH"] = str(gguf)

from agents.agent import Agent
from carnot.agentic.arc_competition_agent import make_carnot_agent

# the verifier-routed cascade (graph-explore -> E3 induction via the bundled Qwen).
# registered under "carnotagent" in the rewritten agents/__init__.py below.
CarnotAgent = make_carnot_agent(Agent)
'''
Path("/kaggle/working/my_agent.py").write_text(AGENT_SRC)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    # 3) wait for the internal game gateway to come up (canonical pattern)
    subprocess.run(
        "curl --fail --retry 999 --retry-all-errors --retry-delay 5 "
        "--retry-max-time 600 http://gateway:8001/api/games",
        shell=True, check=False,
    )
    # 4) copy the COMPETITION-PROVIDED framework to a writable location
    fw = "/kaggle/working/ARC-AGI-3-Agents"
    shutil.rmtree(fw, ignore_errors=True)
    shutil.copytree(f"{COMP}/ARC-AGI-3-Agents", fw)
    shutil.copy2("/kaggle/working/my_agent.py", f"{fw}/agents/templates/my_agent.py")
    # 5) minimal agents/__init__.py — the stock one eagerly imports langgraph / smolagents
    #    templates whose deps are NOT installed (would crash). Register only what we use.
    Path(f"{fw}/agents/__init__.py").write_text(
        "from typing import Type, cast\n"
        "from dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\n"
        "from .swarm import Swarm\n"
        "from .templates.random_agent import Random\n"
        "from .templates.my_agent import CarnotAgent\n"
        "load_dotenv()\n"
        'AVAILABLE_AGENTS: dict[str, Type[Agent]] = '
        '{"random": Random, "carnotagent": CarnotAgent}\n'
    )
    # 6) CRITICAL: point main.py at the gateway. main.py builds its game-API URL from
    #    SCHEME/HOST/PORT env (default localhost:8001) and loads .env LAST with override=True.
    #    Without this .env it queries localhost, finds no games, records no scorecard -> ERROR.
    #    (This is the omission that errored submission 53862044; mirrors the canonical control nb.)
    Path(f"{fw}/.env").write_text(
        "SCHEME=http\n"
        "HOST=gateway\n"
        "PORT=8001\n"
        "ARC_API_KEY=test-key-123\n"
        "ARC_BASE_URL=http://gateway:8001/\n"
        "OPERATION_MODE=online\n"
        "ENVIRONMENTS_DIR=\n"
        "RECORDINGS_DIR=/kaggle/working/server_recording\n"
    )
    # 7) play all gateway games (12h Kaggle cap). main.py fetches the game list from the
    #    gateway, runs the swarm, and the gateway records the scorecard that is scored.
    run_env = os.environ.copy()
    run_env["MPLBACKEND"] = "agg"  # headless matplotlib (canonical nb sets this)
    subprocess.run(
        [sys.executable, "main.py", "--agent", "carnotagent"],
        cwd=fw, env=run_env, timeout=43200, check=False,
    )
else:
    # NON-rerun: write the placeholder submission so Save & Run produces a valid entry.
    import pandas as pd

    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet("/kaggle/working/submission.parquet", index=False)

print("CarnotAgent submission notebook complete.")
