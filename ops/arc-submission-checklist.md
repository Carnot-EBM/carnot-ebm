# ARC-AGI-3 submission checklist (operator handoff) — 2026-06-19

**STATUS: package VERIFIED OFFLINE on the real Kaggle P100. Ready for OPERATOR submission.**
Submission itself is OPERATOR-ONLY (External Publication discipline) — the outer loop prepared and
verified the package; it did NOT and will NOT submit. Deadline: **2026-06-30** (milestone #1, prize +
open-source eligible — the project is MIT-0 ✅).

## What was verified (so the operator can trust the package)

| Gate | Result |
|---|---|
| Runtime binary builds for the eval GPU | v7 `llama-server` (CUDA, sm_60/75/89 + PTX), MTP symbol present ✅ |
| Binary loads the model on the **real P100** | Qwen3.5-9B-MTP + MTP + q8 KV, **11.5 GB / 16 GB**, 29 tok/s, generates, internet OFF ✅ |
| Carnot agent imports **offline** on Kaggle | `make_carnot_agent` imports; jax IS preinstalled on Kaggle ✅ |
| LLM proposer runs **through the bundled binary** | generated correct code (`def is_win(grid): return grid[0][0]==1`), 12 GB / 16 GB ✅ |
| License / prize eligibility | whole project **MIT-0** ✅ |

Not yet exercised end-to-end (by design): the full game-play loop uses the **competition-provided**
`arcengine` + `ARC-AGI-3-Agents` framework + the held-out games, which only exist in the eval sandbox.
The no-LLM cascade was validated locally; the LLM proposer path is validated above.

## The 3 Kaggle Datasets (uploaded, private, ready)

| Dataset | Contents | Note |
|---|---|---|
| `iancblenke/carnot-llamacpp-mtp-binary` | `llama-server` + 22 shared libs (incl. `libggml-cuda.so`) | built ON Kaggle to match CUDA 12.8/driver |
| `iancblenke/carnot-qwen35-9b-mtp-gguf` | `Qwen3.5-9B-Q4_K_M.gguf` (5.9 GB) | the live generator (Apache-2.0 weights) |
| `iancblenke/carnot-agent-code` | `python/carnot` (NO `_rust.so`) + `ops/` + `results/` data | the agent + registry/survey/ledger |

## The submission agent (the proven setup)

The submission subclasses the competition's `Agent` and wraps the Carnot policy:

```python
import os, sys, shutil
from pathlib import Path
inp = Path("/kaggle/input")
# self-locate the 3 bundles (mount nests under /kaggle/input/datasets/<owner>/<slug>/)
carnot = next(p.parents[2] for p in inp.rglob("carnot/agentic/arc_competition_agent.py"))  # .../python
server = next(iter(inp.rglob("llama-server")))
gguf   = next(iter(inp.rglob("*.gguf")))
run_server = Path("/kaggle/working/llama-server")   # /kaggle/input is READ-ONLY -> copy + chmod
shutil.copy2(server, run_server); os.chmod(run_server, 0o755)
os.environ["LD_LIBRARY_PATH"] = f"{server.parent}:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CARNOT_LLAMA_SERVER"] = str(run_server)
os.environ["CARNOT_ARC_GGUF_PATH"] = str(gguf)
sys.path.insert(0, str(carnot))

from agents.agent import Agent           # the competition framework's base class (eval-provided)
from carnot.agentic.arc_competition_agent import make_carnot_agent
CarnotAgent = make_carnot_agent(Agent)   # cascade: explorer -> E3 induction via the bundled Qwen
```

The generator binds **GPU 0** by default (`-ngl 999`); the agent uses graph-explore (CPU) + the verifier
(CPU) + the LLM proposer (the binary). No 3090/local assumptions — all offline.

## Hard-won packaging lessons (do NOT regress these)

1. **`/kaggle/input` is read-only** — copy the binary to `/kaggle/working` + `chmod +x`; libs stay read-only.
2. **Datasets mount nested** under `/kaggle/input/datasets/<owner>/<slug>/` — `rglob`-locate, never hardcode.
3. **No machine-specific compiled extensions in the bundle** — the box's `_rust.cpython-*.so` SIGILLs on
   Kaggle's CPU; dropped it (carnot falls back to pure-Python `_rust_compat`).
4. Build the binary ON Kaggle (matches CUDA 12.8/driver; a local CUDA-13.3 build w/o sm_60 won't run).
5. Datasets must be fully **processed/ready** before a kernel attaches them (else a silent attach race).

## Operator submission steps (operator-only)

1. Open the ARC Prize 2026 / ARC-AGI-3 competition submission notebook (the `ARC-AGI-3-Agents` template).
2. Attach the 3 datasets above (Add Data) + enable GPU; **internet OFF** (eval requirement).
3. Drop in the agent setup above; confirm `CarnotAgent` instantiates.
4. Save/submit per the competition's Code Competition flow (≤12 h runtime).
5. Confirm the per-day submission limit on the My Submissions tab (not publicly documented; check there).
6. First submission = pipeline validation on the real leaderboard (expected score modest — generic
   transfer is partial); iterate toward the 2026-06-30 deadline (multiple submissions expected).

Cross-refs: `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md`,
`python/carnot/agentic/arc_competition_agent.py:make_carnot_agent`, `[[project_arc_live_generator]]`.
