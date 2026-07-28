# ARC-AGI-3 submission checklist (operator handoff) — 2026-06-19

> **GENERATOR SWITCH (2026-07-28) — TWO OPERATOR ACTIONS OUTSTANDING.** Per operator directive
> ("We must use Gemma-4-31B and stop using Qwen-3.5-9B and Qwen-3.6-27B... the Kaggle hardware is
> 96G since May, that is not a problem when we submit"), the live/scored ARC generator is now
> **gemma-4-31B-it Q4_K_M (18.3 GB)**, not Qwen3.5-9B-MTP (5.9 GB). Every Qwen reference below is
> **historical** — it accurately records what was measured and shipped at the time, and is kept per
> the never-prune rule. It does **not** describe the current stack.
>
> Evidence for the switch: 13 games x 3 replicates, Q4_K_M both sides, n_ctx 32768 —
> gemma-4-31B induced an importable world model on 38/39 attempts (fail-as-zero 0.3843) vs
> Qwen3.6-27B's 21/39 (0.0627); matched per-game tally 11-0-2, two-sided sign p = 0.00098. The
> dominant driver is **loadability**, not subtle induction quality.
>
> **Code changes are landed. These two are not, and only the operator can do them:**
>
> 1. **Upload the model dataset.** `scripts/kaggle/submission_kernel/kernel-metadata.json` now
>    requests `iancblenke/carnot-gemma4-31b-it-gguf`, which **does not exist yet**. Create it from
>    `~/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/*/gemma-4-31B-it-Q4_K_M.gguf`
>    (18.3 GB). Until then a push fails at dataset resolution — deliberately loud, rather than
>    silently running the retired 9B.
> 2. **Confirm `machine_shape` on the next submission log.** It changed `NvidiaL4` (24 GB — too
>    small for 18.3 GB of weights plus an 81920-cell q8 KV pool) to **`NvidiaRtxPro6000`**. That
>    identifier is **not verified by us**: it comes from the arcprize.org 2026 starter kit's
>    `rtx6000` accelerator entry and from a real scored 3rd-place kernel in this competition
>    (`external/arc-m1-3rd-forge/kernel-metadata.json`, server-assigned `id_no` 124697453, which
>    also declares a `gemma-4-31b-it` model source). The local kagglesdk cannot validate it — it
>    documents only T4/P100/TPU and omits even NvidiaL4, which we use successfully, so its silence
>    is stale docs, not counter-evidence. The kernel prints an `LLM GPU HARDWARE:` nvidia-smi line;
>    read the answer off the next real run. **Do not submit merely to test this.**
>
> **Local (non-Kaggle) note:** on a 24 GB RTX 3090 the 31B at the default n_ctx 81920 resides at
> 23888 MiB, leaving 688 MiB — so the local free-VRAM guard correctly declines the CUDA card and
> falls back to the iGPU build. The two levers are `CARNOT_ARC_INDUCE_N_CTX` (smaller pool) and the
> new opt-in `CARNOT_ARC_FFN_CPU_LAYERS` (dense-FFN weights to system RAM, ~195 MiB freed per
> layer). The offload is **not free**: 12 layers frees 2344 MiB but costs 58% of decode throughput
> and 79% of prefill, and prefill is what the induce path is bound by.

> **CONTRACT CORRECTION (2026-06-19, late).** A diff against the canonical arcprize control
> notebook (Ronan McGovern's `arc3-random-control`, pulled via the kaggle CLI) showed the real
> submission contract is a **code competition with an internal game GATEWAY**, NOT the
> offline-bundled-`environment_files` shape the steps below originally assumed. The actual entry
> is `scripts/kaggle/submission_kernel/main.py` (gateway pattern). The component gates in the
> table below are still real and reused (binary builds + loads Qwen + agent imports offline +
> proposer generates), and the agent shape is confirmed correct vs the reference — but the
> "Operator submission steps" section is superseded by **§ Real submission flow** at the bottom.
> **Deadlines / prizes (arcprize.org/competitions/2026/arc-agi-3, confirmed 2026-06-19):**
> **Milestone #1 = 2026-06-30** ($25K/$10K/$2.5K to top 3; REAL, 11 days out — land a scored
> submission before it); **Milestone #2 = 2026-09-30** (same split); **final submissions due
> 2026-11-02**; results 2026-12-04. Track total $850K, incl. a $700K Grand Prize for the first
> eligible agent at 100% on the fully-private set. **Milestone eligibility REQUIRES the solution
> be open-sourced by the milestone deadline** — our MIT-0 satisfies the CC0/MIT-0 requirement.
>
> **Import probe (kernel `carnot-arc-import-probe`, 2026-06-19) — risk retired + 1 bug caught:**
> jax 0.7.2 / numpy 2.4.6 ARE preinstalled on the Kaggle image; the competition wheels
> (`arc_agi_3_wheels`) + framework ARE mounted in a regular notebook; `arcengine`/`arc_agi`
> install clean offline; carnot imports in 1.1s. It CAUGHT that the `carnot-agent-code` dataset
> was STALE (predated the adapter fix) — **re-versioned 2026-06-19 22:41Z with the fix** (the
> `langgraph.store.sqlite` error was a probe artifact of loading `agent.py` directly; the real
> notebook's minimal `__init__.py` rewrite avoids it).

**STATUS: FIRST SCORED SUBMISSION — public score 0.08 (2026-06-19 23:40Z).** `ref 53862349`
("carnot v1.1", kernel v3) COMPLETE. The full pipeline is validated end-to-end live: gateway
handshake, agent plays games, verifier-routed cascade + local Qwen3.5-9B-MTP generator all run in
the scored rerun. Leaderboard context: leaders ~0.66–1.21; 0.08 is modest-but-real (above random).
- `ref 53862044` (v1) ERRORED — missing gateway `.env` (main.py hit localhost not the gateway);
  fixed by writing the gateway `.env` in the rerun branch. The error did NOT consume the daily slot.
- ITERATION PLAN (1/day, ~10 cycles to 06-30): the conductor banks more games/levels offline each
  milestone; before each daily resubmit, re-version the `carnot-agent-code` dataset with the latest
  `python/carnot` so the live agent picks up the new solvers/operators, then resubmit + watch the climb.

**Submit API note:** the CLI `kaggle competitions submit -k ... -v ...` returned a 400; the working
path is the Python API `api.competition_submit_code(file_name="submission.parquet",
message=..., competition="arc-prize-2026-arc-agi-3", kernel="iancblenke/carnot-arc-agi3-submission",
kernel_version=1)`. The kernel's `/code/` URL 404s publicly because it is `is_private: true` — open
it from Kaggle "Your Work → Notebooks" when logged in as iancblenke.
Submission itself is OPERATOR-ONLY (External Publication discipline) — the outer loop prepared and
verified the package; it did NOT and will NOT submit. Deadline: **2026-06-30** (milestone #1, prize +
open-source eligible — the project is MIT-0 ✅).

## Daily iteration routine (1 submission/day → climb from 0.08 before 06-30)

`scripts/kaggle/prep_daily_submission.py` is the standing routine. Each day:

1. **Prep (automated, never submits):** `.venv/bin/python scripts/kaggle/prep_daily_submission.py`
   — downloads the `carnot-agent-code` dataset, overlays the repo's latest `python/carnot`
   (so the live agent picks up the conductor's newly-banked solvers), runs safety guards
   (adapter fix present + no `_rust.so` leak), re-versions the dataset, re-pushes the PUBLIC
   kernel, validates the save-run, and prints `READY ... kernel vN`.
2. **Approve + submit (OPERATOR-ONLY):** `.venv/bin/python scripts/kaggle/prep_daily_submission.py
   --submit-only --kver N` (N = the version the prep printed). This is the only step that submits.
3. **Watch:** the [Submissions page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/submissions);
   the scored rerun takes up to 12h.

Scheduling: **[DISABLED 2026-06-20 per operator — re-enable with `systemctl --user enable --now carnot-arc-daily-prep.timer` once the submitted config is no longer a regression]** durable systemd user timer `carnot-arc-daily-prep.timer` (daily 09:37 EDT, `Persistent=true`
— survives session end + reboots; validated end-to-end 2026-06-19). It runs the prep `--default` (refresh +
re-push + validate, NEVER submits) and writes `ops/arc-daily-prep-status.json` (`{kernel_version, ready,
submit_command}`). The timer cannot push-notify or submit (agent-only / operator-only), so the in-session
watchdog surfaces the status file and the operator runs the `--submit-only` command to approve. Units live at
`~/.config/systemd/user/carnot-arc-daily-prep.{service,timer}` (copies tracked at `ops/systemd/`). The
session-only cron approach was retired in favor of this.

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
6. **Test-submit SOON** — the first ARC-AGI-3 Milestone deadline is **2026-06-30** (11 days out).
   The first scored run surfaces the jax/numpy-on-rerun-image risk; leave time to fix + resubmit
   before the milestone. The final competition deadline (2026-11-02) is the longer horizon.

**Submission limits (Kaggle API, authoritative, 2026-06-19):** `max_daily_submissions = 1`
(ONE scored submission PER DAY), `is_kernels_submissions_only = True` (notebook, not a file),
`max_team_size = 8`, `new_entrant_deadline`/`merger_deadline = 2026-10-26` (for the FINAL),
`submissions_disabled = False` (open now), `evaluation_metric = "ARC-AGI-3 Metric"`. With 1/day
and each scored rerun taking up to 12h, there are **~11 iterate-measure-fix cycles before
06-30** — FRONT-LOAD: every day of delay is one lost attempt. The milestone prize goes to the
best public-leaderboard score AT 06-30.

Cross-refs: `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md`,
`python/carnot/agentic/arc_competition_agent.py:make_carnot_agent`, `[[project_arc_live_generator]]`.
