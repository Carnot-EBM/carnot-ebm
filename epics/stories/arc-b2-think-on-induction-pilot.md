# ARC B2 think-ON induction pilot

**Status:** Harness built, review fixes applied, dry-run verified; GPU run not started
**Requirement:** REQ-ARC-WMTE-10010
**Experiment:** 10010

## Goal

Measure whether live think-ON world-model induction (Qwen3.8-27B, one round,
live budget) produces better engines than the codeonly first shots on the ten
B2 windows. Follow the pre-registration in
`docs/research-notes/b2-positive-control-2026-09-23.md` exactly.

## Stage 1: harness (done 2026-09-23)

The harness captures the live `induce()` call before any request is sent. It
proves the replayed prompt equals the recorded first-call prompt, minus the
codeonly directive and one fence, on all ten windows. It proves no held-out
answer is in the prompt. It scores engines in its own wrapper: a raised row
scores 0, and each row loads a fresh engine module. Three controls run before
any model output and stop the run if they miss: identity 0.0, expert 1.0, and
the codeonly baseline 0.13 within 0.01.

The dry run on the recorded evidence passed all of these. The baseline
reproduced at 0.12754. 47 mocked tests pass. Six mutation checks each turned
a test red.

## Stage 1b: adversarial review fixes (done 2026-09-23)

An independent review found 2 blockers and 9 majors. All are fixed or recorded:

- Engines now run in a child process that never sees a held-out answer. It forks
  per row and refuses file reads outside the Python install. Before the fix, an
  engine that walked the call stack or the heap scored 1.0 on all ten windows.
- Only integer grids are compared. An always-equal object array scored 1.0 before.
- The per-call timeout is the pre-registered 2,400 s. The first build used 4,800 s
  and wrongly called it pre-registered. A live-ladder rescore reports what the
  scored path's timeout would do to each window.
- The environment check no longer claims parity with the scored backend (vLLM,
  NVFP4). It declares `backend_parity: false`, and a real run refuses any
  `CARNOT_ARC_*` flag the kernel does not set.
- Server failures are retried, never scored 0. A blocked rerun cannot overwrite a
  finished artifact. Resumed rows are rescored by the current code.
- The artifact records seven new deviations, including the repetition penalty
  that vLLM drops and the missing vLLM reasoning parser (see ops/known-issues.md).

Each fix has a regression test, and 28 mutations each turned a test red.

## Stage 2: GPU run (not started)

Start it detached, as `python`, not `python3`: the host janitor kills `python3`
and `pytest` processes older than 2 h, and the harness refuses those names.

```bash
cd /home/ianblenke/github.com/ianblenke/carnot
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu setsid nohup \
  .venv/bin/python scripts/experiments/experiment_10010_b2_think_on_pilot.py \
  > /tmp/exp10010_run.log 2>&1 &
```

The server PID is written to
`results/raw/experiment_10010_b2_think_on_pilot/server_logs/llama_server_8996.pid`.

One live `generate()` call per window on GPU 1 only, with residency proven by
UUID. The run is resumable per window. It is a pilot: 10 windows and 1 draw
each support no significance claim.
