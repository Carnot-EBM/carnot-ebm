# Environment hardening + milestone 2026.04.38 stranded-experiment reruns

Status: Draft change proposal. Origin: manual intervention on 2026-04-19 after
two consecutive headline-credibility experiments in milestone 2026.04.38 were
blocked by environment gaps that the conductor had no way to detect at
startup. Target milestone: 2026.04.39 (immediately after the current
milestone's retrospective Exp 512 lands).

## Why this exists

Milestone 2026.04.38 is hitting a new class of failure that is neither the
zombie-VRAM problem (closed by GPUVRAMGateV2 in Exp 487) nor the Gemma4
tokenizer bug (closed by Exp 450) nor the silent deliverable drop (closed
by DeliverableGuard in Exp 462). Two experiments are now stuck:

- **Exp 503** (Live 200q VeriCoT+VPRM v4): `honest_verdict=blocked` with
  CUDA OOM because an unrelated Python subprocess was holding ~8.96 GiB on
  GPU 0 at the moment the quantized Gemma4 tried to load. The gate check
  passed (no zombie at that instant), but a new long-running process
  joined in between the gate and the model load and won the VRAM race.
- **Exp 504** (GSM-Symbolic Adversarial v4): `honest_verdict=blocked` with
  `ImportError: sentencepiece` / `tiktoken` / `llama-cpp-python` — the
  GGUF tokenizer initialisation needs three Python packages that nobody
  had ever installed into the project venv. They were never in
  ``pyproject.toml`` either; the research code was written assuming they
  would be present.

During the manual intervention on 2026-04-19 the author:

1. Killed six stale pytest-related processes that had accumulated over
   the previous 3-4 hours and were collectively pinning **~48 GiB** of
   GPU 0 VRAM (GPUVRAMGate kills zombies *at the start of a GPU
   experiment*; it has no reaper for long-lived pytest workers that
   outlive the experiment that spawned them).
2. Installed the three missing GGUF dependencies into the project venv.
3. Verified both GPUs returned to 24,123 MiB free / 5 MiB used.

This change proposal turns that manual procedure into code.

## Proposed experiments for milestone 2026.04.39

Pick at least the first two. Together they cost roughly one milestone
slot and close a whole class of failure that is currently responsible for
the headline RETRO-033 / RETRO-038 / RETRO-039 miss count.

### Exp N: Conductor startup environment check

- **Deliverable:** ``results/experiment_<N>_conductor_env_check.json`` plus
  a new function ``scripts/research_conductor.py::assert_env_ready()``
  invoked once at ``main()`` entry (ordered after ``assert_startup_invariants``
  from the earlier conductor-regression-prevention proposal).
- **Scope:** Before entering the loop, verify:
  1. **Python imports resolve** for every package listed in a new
     ``scripts/conductor_required_packages.txt`` file. Seed the file with
     ``sentencepiece``, ``tiktoken``, ``llama-cpp-python``, ``torch``,
     ``transformers``, ``numpy``, ``jax`` (optional -- warn only),
     ``pyyaml``. Each import either succeeds or is skipped with a clear
     ``optional`` tag.
  2. **No stale pytest processes.** Run ``pgrep -fa 'pytest tests/python'``,
     subtract the conductor's own children, and if any entry has
     ``etimes > 1800`` (30 min) log a warning with the PIDs and suggest
     ``pkill -9 -f 'pytest tests/python'``. Fail only when
     ``CARNOT_STRICT_ENV=1``; by default warn so the conductor still
     starts on a messy host.
  3. **GPU VRAM sanity.** On each GPU, ``memory.free`` must be at least
     ``CARNOT_MIN_GPU_FREE_MB`` (default 16000). On failure, list the top
     three processes by VRAM so the operator can see who to kill.
  4. **Disk free space** in ``/home`` and ``/tmp`` (> 5 GiB each).
     Several earlier conductor runs died silently when the session log
     filled the FS.
  5. **Git remote reachable** (``git ls-remote origin HEAD``). No point
     running experiments that cannot be pushed.
- **Why this matters:** Exp 503 and Exp 504 would both have been caught
  here and either skipped cleanly (with a pending-env status) or
  unblocked before the subagent was ever invoked. Compare the cost: a
  60-second pre-flight check vs a 60-minute subagent run that produces a
  deliverable tagged ``blocked`` and still costs the conductor budget.
- **Scale:** ~80 lines of code, 6 tests (one per check + one integration
  fixture simulating a dirty environment). CPU-only, no GPU, no Claude
  subagent cost.

### Exp N+1: Zombie pytest reaper + retry of stranded Exp 503 / Exp 504

- **Deliverable:** ``results/experiment_<N+1>_pytest_reaper_and_reruns.json``
  plus ``scripts/research_conductor.py::reap_stale_pytest()`` called once
  per iteration just before the ``run_research_step()`` enters.
- **Scope:**
  1. Walk ``pgrep -fa 'pytest tests/python'``, skip the conductor's own
     children (compare against ``os.getpid()`` subtree), and SIGKILL any
     process whose ``etimes > PYTEST_REAPER_THRESHOLD_S`` (default 1800).
     Log PIDs and CPU/VRAM footprint before killing for auditability.
  2. Use Exp 503 and Exp 504 as the first real verification targets:
     re-run both with the now-fixed env and the reaper armed. Capture
     ``status=success`` vs ``status=blocked_again`` as the experiment's
     honest verdict; if still blocked, the pre-flight check from Exp N
     will report the specific reason.
  3. Add a new SCENARIO-INFRA entry asserting that pytest workers
     spawned by a killed subagent are reaped within 30 s of the
     subagent's own death, using process-group kill semantics that
     already exist in ``_kill_subagent_group``.
- **Scale:** ~60 lines of code + two experiment retries. The retries
  cost real subagent time but are exactly the live numbers milestone
  2026.04.38 was aiming for -- no additional research scope, just
  converting the ``blocked`` deliverables to ``complete``.

### Exp N+2: JEPA Live Retrain v4 rerun with real CoT pairs (hard prerequisite: Exp N+1 success)

- **Deliverable:** ``results/experiment_<N+2>_jepa_live_retrain_v4_rerun.json``
  overwriting the stranded ``experiment_510_jepa_live_retrain_v4.json``
  from milestone 2026.04.38.
- **Why this experiment exists as a distinct slot:** Exp 510 produced
  ``honest_verdict=fr11_synthetic_only`` with ``n_live_pairs=0`` because
  its upstream data source (live CoT pairs generated by Exp 503 /
  Exp 504's pipeline runs) was empty -- 503 and 504 were both blocked.
  The committed Exp 510 script is already correct (the ``training_error``
  recorded in the stranded deliverable was a stale capture from an
  earlier subagent draft; the committed kwargs match the class
  signature). What's missing is the data, not the code. This rerun
  does **nothing new** research-wise: it re-executes the same
  committed Exp 510 script after Exp N+1 has produced non-empty
  ``results/live_cot_pairs_*.jsonl`` inputs.
- **Why this validation matters:** the JEPA AUC 0.967 recorded in
  Exp 492 (milestone 2026.04.37) is currently the largest headline
  number on the whole roadmap, and it is unvalidated on live data.
  The ``docs/roadmap.md`` Breakthrough Results table carries it
  explicitly as "pending Exp 510" and instructs the reader not to cite
  it externally. This rerun is the gate. If AUC on real live pairs
  holds at ~0.9+, the breakthrough is real and the row gets promoted
  to LIVE. If it collapses to 0.5-0.7, the 0.967 was leakage from the
  Exp 442 single-session training capture and the row gets demoted to
  a negative finding.
- **Scheduling constraint:** this experiment must run **after** Exp
  N+1 completes with ``success`` on both 503 and 504 rerun targets.
  The planner should not schedule it before Exp N+1 even if there are
  free slots -- a premature rerun just re-produces the synthetic-only
  result and burns the milestone slot. If N+1 itself partially
  succeeds (only one of 503/504 unblocks), rerun N+2 anyway using
  whichever produced live pairs; a single-source live validation
  beats none.
- **Scale:** zero new code. The committed ``scripts/experiment_510_jepa_live_retrain_v4.py``
  runs as-is. Budget: one subagent call, maybe two if the first one
  hits max-turns on the training loop.
- **Acceptance criterion:** the deliverable's ``n_live_pairs > 0``
  AND ``honest_verdict`` is one of ``{fr11_live_confirmed,
  fr11_live_regressed, fr11_live_insufficient_signal}``. A fourth
  ``fr11_synthetic_only`` would mean N+1 did not actually produce live
  data and is itself only half-complete.

### Exp N+3 (stretch): Dependency lockfile + automated install on
``./scripts/bootstrap.sh``

- **Deliverable:** ``scripts/bootstrap.sh`` plus an updated
  ``pyproject.toml`` listing every package the repo's scripts actually
  import (audited via ``ast`` walk over ``scripts/`` and ``python/``).
- **Why:** The root cause of Exp 504 is that ``sentencepiece``,
  ``tiktoken``, and ``llama-cpp-python`` were imported by
  ``python/carnot/pipeline/gemma4_quantized_loader.py`` without being
  listed as dependencies anywhere. A bootstrap script that reads the
  lockfile and runs ``pip install`` is idempotent and adds nothing to
  runtime; combined with Exp N's startup check, it turns "missing
  package" from a silent failure into a two-line "run bootstrap.sh"
  diagnostic.
- **Scale:** ~100 lines + a pytest that imports every package the
  lockfile lists. Schedule only after N and N+1 ship; the lockfile's
  exact contents are easier to audit after the env check is running.

## Interim state captured (for the 2026.04.39 retrospective)

These items were completed manually on 2026-04-19 before this proposal
was written. The next milestone's retro agent should cite them as the
baseline it is improving on:

- 6 stale pytest-related PIDs (107234, 107270, 128279, 128322, 128328,
  130462, 130497) killed; ~48 GiB GPU 0 VRAM freed; both GPUs back to
  24,123 MiB / 5 MiB used.
- ``sentencepiece==0.2.1``, ``tiktoken==0.12.0``, ``llama-cpp-python==0.3.20``
  installed into the project venv.
- Exp 503 and Exp 504 remain marked ``blocked`` in their deliverable
  JSON. The OK entries in ``ops/conductor-log.md`` mean the conductor
  will not retry them automatically; Exp N+1 above explicitly reruns
  them under the fixed env.

## References

- Session observations 2026-04-19 during milestone 2026.04.38 that
  surfaced the concurrent-process OOM (Exp 503) and missing-package
  (Exp 504) failure modes.
- ``scripts/research_conductor.py`` ``_kill_subagent_group`` and
  ``GPUVRAMGateV2`` -- the existing machinery Exp N+1 extends.
- Adjacent proposals:
  ``openspec/change-proposals/conductor-regression-prevention.md``
  (regression tests for ``pick_next_task``/``run_agent``),
  ``openspec/change-proposals/research-roadmap-vNEXT-dspy-signatures.md``
  (typed experiment contract).
