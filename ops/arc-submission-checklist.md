# ARC-AGI-3 submission checklist (operator handoff) — 2026-06-19

> **CONTRACT CORRECTION (2026-06-19, late).** A diff against the canonical arcprize control
> notebook (Ronan McGovern's `arc3-random-control`, pulled via the kaggle CLI) showed the real
> submission contract is a **code competition with an internal game GATEWAY**, NOT the
> offline-bundled-`environment_files` shape the steps below originally assumed. The actual entry
> is `scripts/kaggle/submission_kernel/main.py` (gateway pattern). The component gates in the
> table below are still real and reused (binary builds + loads Qwen + agent imports offline +
> proposer generates), and the agent shape is confirmed correct vs the reference — but the
> "Operator submission steps" section is superseded by **§ Real submission flow** at the bottom.
> Deadline on Kaggle is **2026-11-02** (our docs say 06-30 — likely a self-imposed milestone).

**STATUS: components VERIFIED on the real Kaggle P100; submission notebook authored against the
correct gateway contract. NOT yet run end-to-end through the real gateway (only the eval sandbox
has it). Ready for an OPERATOR test-submit.**
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
| **Agent interface diffed vs the live framework** | `make_carnot_agent` checked against the real `agents/agent.py` `Agent` ABC + `arcengine.GameAction`; 2 submission-breaking bugs found & fixed (commit `22cc26e5d`) ✅ |
| License / prize eligibility | whole project **MIT-0** ✅ |

### Live-framework interface diff (the risk retired 2026-06-19)

The offline validators drive `policy.next_move()` directly and never touch the adapter's
`choose_action` — the method the real harness calls. A diff against the cloned
`ARC-AGI-3-Agents` framework (`/home/ianblenke/arc3_agents`) caught two bugs invisible to
the whole offline suite, now fixed + regression-tested (`tests/python/test_arc_competition_agent_adapter.py`):

1. **(critical)** `choose_action` returned `GameAction.set_data(data)`, which yields the inner
   `ComplexAction`, not the enum. The framework reads `action.action_data` off the return, so a
   `ComplexAction` crashed **every click/coordinate action** (`AttributeError`). Fixed: mutate the
   enum in place, return the enum, carry the required `game_id`. Verified against real `arcengine`.
2. The framework's `Agent.MAX_ACTIONS` default is **80** — below even our deepest banked replay
   (lp85 → L5). Raised to 400 so multi-level replays + held-out-game explore have room (the ≤12h
   wall-clock is the real bound).

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

## Operator submission steps (operator-only) — SUPERSEDED, see § Real submission flow

(The original offline-bundle steps assumed we author a standalone notebook driving bundled
`environment_files`. That is wrong — see the correction banner. Kept for history per never-prune.)

1. ~~Open the ARC Prize 2026 / ARC-AGI-3 competition submission notebook (the `ARC-AGI-3-Agents` template).~~
2. ~~Attach the 3 datasets above (Add Data) + enable GPU; internet OFF.~~
3. ~~Drop in the agent setup above; confirm `CarnotAgent` instantiates.~~
4. ~~Save/submit per the competition's Code Competition flow.~~

## § Real submission flow (the corrected, gateway contract)

The entry is `scripts/kaggle/submission_kernel/main.py` (modeled on Ronan McGovern's canonical
`arc3-random-control`). The competition provides the `ARC-AGI-3-Agents` framework + games (via an
internal `gateway:8001`) + dep wheels in the rerun sandbox; we only drop in our agent.

1. Create a new notebook in the **arc-prize-2026-arc-agi-3** competition; paste
   `scripts/kaggle/submission_kernel/main.py`.
2. Add the competition as a data source + attach the 3 datasets
   (`carnot-agent-code`, `carnot-llamacpp-mtp-binary`, `carnot-qwen35-9b-mtp-gguf`). GPU on; internet OFF.
3. **Save & Run All** — in non-rerun mode it just writes the placeholder `submission.parquet`
   (cheap; confirms the notebook is valid + the datasets/competition attach cleanly).
4. **Submit** the notebook version to the competition. Kaggle then RE-RUNS it with
   `KAGGLE_IS_COMPETITION_RERUN=1` against the hidden game gateway (≤12 h) — that scored run is the
   leaderboard entry.
5. First submission = pipeline validation on the real gateway (expected score modest — generic
   transfer is partial). Residual risk to watch in the first run's logs: that the Kaggle rerun
   image has jax/numpy preinstalled for the carnot import (the standard image does; the rerun image
   is unconfirmed) — if not, the agent import fails and we bundle those deps as a wheels dataset.
6. Plenty of runway (Kaggle deadline 2026-11-02); iterate.

Cross-refs: `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md`,
`python/carnot/agentic/arc_competition_agent.py:make_carnot_agent`, `[[project_arc_live_generator]]`.
