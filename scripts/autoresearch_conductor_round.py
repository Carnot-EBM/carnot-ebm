#!/usr/bin/env python3
"""Unattended autoresearch mutation round for the conductor (REQ-AUTO-019/020).

**What this closes.** `python/carnot/autoresearch/` (REQ-AUTO-001..015) already
implements the whole AVO-style loop -- an LLM proposes a hypothesis, the
sandbox runs it, the 3-gate evaluator scores it against the current baseline,
and a circuit breaker halts on repeated failure. `scripts/research_conductor.py`
never called any of it. This script is the missing wiring: run one bounded
round of that loop, and when a hypothesis wins, persist it as its own git
commit carrying its score -- the one piece the orchestrator itself does not
do (it only updates an in-memory baseline and appends to a JSON log).

**Fitness is independently measured, not self-reported (REQ-AUTO-021, fixed
2026-09-12).** Adversarial review found the original wiring trusted a
hypothesis's own `final_energy` claim verbatim (`sandbox.py:run_in_sandbox`
took the `run(benchmark_data)` return value as-is; `evaluator.py` compared
whatever number was in it). A `def run(d): return {'double_well':
{'final_energy': -999999.0}}` was accepted and committed as if it were real.
Fixed properly, not just mitigated: the hypothesis contract now asks for a
`final_state` (the point it converged to), never a bare energy number
(`AUTORESEARCH_SYSTEM_PROMPT` below); `_energy_verification_patch` makes
every sandboxed run go through `_verified_execute_hypothesis`, which
discards any self-reported `final_energy` and recomputes it from
`final_state` via `toy_benchmarks.py`'s real DoubleWell/Rosenbrock potential
functions -- code the hypothesis never touches and cannot influence except
by actually finding a lower-energy state. A benchmark name this module
cannot score (including anything an LLM invents) never gets a `final_energy`
at all, so it can never register as an improvement. See
`test_fabricated_energy_claim_is_never_committed_end_to_end` for the exact
reproduction from the adversarial review, now passing through the real
pipeline with nothing mocked below `codex_generate_hypotheses`.

**How it is invoked.** Exactly like the milestone-close audits already wired
into `research_step()` (`pages_adversarial_audit.py`, `verifier_authenticity_
audit.py`, etc.): a subprocess launched via `_run_audit_with_receipt`, gated
by `CARNOT_AUTORESEARCH_UNATTENDED=1` (default off), writing a receipt file
the caller checks for freshness. If the `codex` CLI is unavailable, or the
hypothesis generator's home-grown constitution forbids the run, this writes
a clean non-fatal receipt and exits 0 -- the milestone-close path is never
blocked by this step, matching the existing audits' own contract.

**Hypothesis generator: codex CLI, model gpt-6-astra (2026-09-12 operator
directive).** The mutation-operator's "propose a hypothesis" role is the same
category of task as the planner/retro/audit tiers this project already runs
via `codex exec` (never a locally-served model) -- an agent deciding what to
try next, not the ARC live agent's own inference-latency-bound generation
that IS local-first by contract. `call_codex()` below mirrors the exact
subprocess pattern the sibling audit scripts already use
(`pages_adversarial_audit.py:call_codex`), reusing `hypothesis_generator.py`'s
`_build_user_prompt` / `_extract_hypotheses` for the context/parsing halves,
but its own `AUTORESEARCH_SYSTEM_PROMPT` (below) for the contract itself --
see REQ-AUTO-021 for why. This is the project's own
internal R&D tooling talking to a closed-weight model, the same as every
other autonomous conductor role; it is not a Carnot CAPABILITY (the thing
`python/carnot/verify`/`pipeline`/`samplers` ship to users), so the
Decentralization-Respecting Design Constraints' local-first mandate (which
targets shipped capabilities) does not bind it, any more than it binds the
planner's own codex calls.

**Fitness target #1.** The synthetic DoubleWell/Rosenbrock energy benchmarks
from `scripts/demo_autoresearch.py:create_initial_baselines()` -- cheap,
deterministic, already-implemented, no live LLM inference needed to SCORE a
candidate (only to PROPOSE one). Deliberately NOT the ARC live agent (see
`docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md` Part 3
for why that was rejected). Both benchmarks were driven to machine-precision
zero on the first production fire (2026-09-13); every fire since has landed
`accepted > 0, committed = 0` -- saturated, not broken (see
`docs/research-notes/rsi-levels-and-autoresearch-fitness-target-2026-09-14.md`).

**Fitness target #2 (2026-09-16, REQ-AUTO-025).** `verifier_auroc` --
correcting the note above, which said no reusable AUROC harness existed:
`python/carnot/autoresearch/verifier_auroc_benchmark.py` now provides one,
against a held-out split of `data/fover_corpus_v4.json` and
`carnot.verify.pcib_probe.PCIBProbe`'s two tunable weights. See that
module's own docstring for the trust boundary (same shape as REQ-AUTO-021's)
and the measured, real headroom (default weights: AUROC 0.3465; a single
sign flip: AUROC 0.7219) that motivated building it.

**Where the lineage lives.** `ops/autoresearch_discoveries/<benchmark>/
<experiment_id>.json` -- a self-contained record (code, description, metrics,
before/after baseline) rather than a live `.py` module. This sidesteps
ruff/mypy strict-mode friction on raw LLM-generated snippets entirely: the
record does not need to satisfy production code-quality gates to be a valid,
readable audit trail, any more than a results/*.json artifact does. Each
accepted hypothesis lands as its own commit: `git add`-ing only that one
file (never `-A`) AND `git commit -- <path>` (a pathspec, not a bare flag),
so this can never sweep unrelated in-flight work into its commit -- the
exact collision class `harness_integrity_lint.py` exists to catch, and the
exact bug an adversarial review reproduced 2026-09-12 against an earlier
version of this function that added the path but committed with a bare
`git commit -m` (which commits the WHOLE index, not what was just added).
Scoping BOTH the add and the commit avoids the class outright rather than
needing that guard.

**Local state, not the research record.** `ops/.autoresearch_baselines.json`
and `ops/.autoresearch_experiment_log.json` are gitignored fast-resume
caches (see `.gitignore`) -- the durable record is the committed lineage
above. `ops/autoresearch_conductor_report.md` IS tracked, matching every
sibling audit's report file; it is picked up by the conductor's own regular
end-of-step commit, not committed by this script.

Spec: REQ-AUTO-019, REQ-AUTO-020, SCENARIO-AUTO-019-*, SCENARIO-AUTO-020-*
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from carnot.autoresearch import calibrated_decision_benchmark as _calibrated_decision_module  # noqa: E402
from carnot.autoresearch import orchestrator as _orchestrator_module  # noqa: E402
from carnot.autoresearch import verifier_auroc_benchmark as _verifier_auroc_module  # noqa: E402
from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics  # noqa: E402
from carnot.autoresearch.calibrated_decision_benchmark import (  # noqa: E402
    CALIBRATED_DECISION_BENCHMARK_NAME,
    HIDDEN_DIM,
    INPUT_DIM,
    measure_default_calibration_energy,
    train_features_for_prompt,
)
from carnot.autoresearch.constitution import ActionCategory, ConstitutionChecker  # noqa: E402
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog  # noqa: E402
from carnot.autoresearch.hypothesis_generator import (  # noqa: E402
    _build_user_prompt,
    _extract_hypotheses,
)
from carnot.autoresearch.orchestrator import AutoresearchConfig, run_loop_with_generator  # noqa: E402
from carnot.autoresearch.sandbox import (  # noqa: E402
    BLOCKED_MODULES,
    SandboxConfig,
    SandboxResult,
    execute_hypothesis,
)
from carnot.autoresearch.toy_benchmarks import (  # noqa: E402
    BENCHMARK_ENERGY_FUNCTIONS,
)
from carnot.autoresearch.verifier_auroc_benchmark import (  # noqa: E402
    VERIFIER_AUROC_BENCHMARK_NAME,
    measure_default_weight_energy,
    train_rows_for_prompt,
)
from carnot.models.gibbs import GibbsConfig, GibbsModel  # noqa: E402
from carnot.training.nce import nce_loss  # noqa: E402
from carnot.verify.pcib_probe import PCIBProbe  # noqa: E402

# REQ-AUTO-025 CRITICAL-2 fix (2026-09-16 adversarial review): every benchmark's
# post-sandbox energy is now recomputed in THIS fresh-interpreter worker
# script, never in-process. `recompute_final_energy` and
# `recompute_verifier_auroc_energy` are no longer imported directly above --
# see `_subprocess_recompute_energy` below for why, and
# `scripts/_autoresearch_energy_recompute_worker.py` for the worker itself.
_RECOMPUTE_WORKER = Path(__file__).resolve().parent / "_autoresearch_energy_recompute_worker.py"
_RECOMPUTE_TIMEOUT_S = 30


def default_benchmark_data() -> dict[str, Any]:
    """Built lazily (not a module-level constant) so importing this module --
    every test file does -- never pays the cost of loading and splitting
    `data/fover_corpus_v4.json` (REQ-AUTO-025) unless a round actually runs.
    """
    calibrated_decision_features = train_features_for_prompt()
    return {
        "dim": 2,
        # REQ-AUTO-025: the verifier_auroc benchmark's TRAINING split only --
        # a hypothesis's sandboxed code may read this to search for good
        # weights, but is never shown the held-out split it is actually
        # scored against (see verifier_auroc_benchmark.py's module docstring
        # for why).
        "verifier_auroc_train_rows": train_rows_for_prompt(),
        # REQ-AUTO-025 CRITICAL-2 (Variant A) fix (2026-09-16 adversarial
        # review): PCIBProbe is handed to the hypothesis directly, as a class
        # object in this trusted dict, rather than via `import
        # carnot.verify.pcib_probe` -- `run_round` now blocks the `carnot`
        # import root for the sandboxed execution entirely (see its
        # `sandbox_config` construction), so a hypothesis cannot import THIS
        # module (or any other carnot.* module) to read the held-out split
        # or corrupt the trusted recompute path from inside its own run().
        "PCIBProbe": PCIBProbe,
        # REQ-AUTO-018: same Variant-A pattern as PCIBProbe above --
        # GibbsModel/GibbsConfig/nce_loss are handed to the hypothesis
        # directly rather than via `from carnot.models.gibbs import ...` /
        # `from carnot.training.nce import ...`, both of which are blocked
        # by the same `carnot` import-root block. Only the TRAINING split's
        # raw PCIB features are exposed; the held-out split stays reachable
        # only from this trusted process.
        "calibrated_decision_train_correct": calibrated_decision_features["correct"],
        "calibrated_decision_train_incorrect": calibrated_decision_features["incorrect"],
        "GibbsModel": GibbsModel,
        "GibbsConfig": GibbsConfig,
        "nce_loss": nce_loss,
    }


DEFAULT_MODEL = os.environ.get("CARNOT_AUTORESEARCH_MODEL", "gpt-6-astra")
DEFAULT_CODEX_TIMEOUT_S = 300
# Measured 2026-09-12: `claude --model fable --effort max` genuinely needs
# more than 100s for a substantive coding response (a trivial "reply OK"
# prompt returns in ~15-30s; the real autoresearch hypothesis prompt timed
# out at 100s). This is slower reasoning at max effort, not a bug -- give it
# real headroom rather than reusing codex's budget. The overall round has an
# 3600s outer timeout (_run_audit_with_receipt, bumped from 1800s alongside
# REQ-AUTO-023's retry budget) and this only fires on the fallback path, so
# there is room.
DEFAULT_FABLE_TIMEOUT_S = 600

# REQ-AUTO-021: unlike hypothesis_generator.DEFAULT_SYSTEM_PROMPT (which asks
# for a self-reported final_energy -- the exact self-report gap adversarial
# review 2026-09-12 finding 1 exploited), this prompt asks for a final_state
# and tells the hypothesis its own final_energy claim, if any, is IGNORED.
# The real energy is recomputed by trusted harness code from final_state,
# never by the sandboxed hypothesis.
#
# CLAUDE.md "Energy-Based Calibrated-Decision Training Floor" (2026-09-18):
# double_well/rosenbrock (REQ-AUTO-021's original fitness target #1) were
# retired from this prompt -- both were driven to exact machine-precision
# zero on the first production fire (2026-09-13) and every round since has
# re-solved an already-solved problem with zero headroom left. Removing them
# frees the slot for calibrated_decision (REQ-AUTO-018) below.
AUTORESEARCH_SYSTEM_PROMPT = """\
You are proposing an optimization procedure for a benchmark in the Carnot \
autoresearch pipeline. Two benchmarks exist:

- verifier_auroc: `benchmark_data["verifier_auroc_train_rows"]` is a list of \
{"step_text": str, "label": "correct" or "incorrect"} training examples. \
`benchmark_data["PCIBProbe"]` is a class -- construct it as \
`Probe = benchmark_data["PCIBProbe"]; probe = Probe(entity_weight=..., \
falsifiability_weight=...)`. Do NOT write `import carnot` or anything under \
it -- all `carnot` imports are BLOCKED in this sandbox and will raise \
ImportError; PCIBProbe is provided to you directly for exactly this reason. \
Find two weights (entity_weight, falsifiability_weight) that best separate \
"incorrect" from "correct" rows by AUROC on THIS training set -- try any \
real search (grid search, random search, anything real) over the probe's \
`.score(step_text, "")` output. Return {"verifier_auroc": {"final_state": \
[entity_weight, falsifiability_weight], ...}}. The harness independently \
rescores your weights against a DIFFERENT held-out set you cannot read (you \
have no working route to it: the training rows above are the only corpus \
data you are given, and `carnot` imports that could reach the held-out set \
directly are blocked) -- your own AUROC on the training rows is never \
trusted or seen by the evaluator. Do not assume the probe's own documented \
default weights (0.5, 0.5) are good; measure and search. A weight pair that \
scores every training row identically (e.g. (0.0, 0.0)) will be rejected as \
degenerate, not accepted.

- calibrated_decision: `benchmark_data["calibrated_decision_train_correct"]` \
and `benchmark_data["calibrated_decision_train_incorrect"]` are each a list \
of [entity_uptake, falsifiability_score] pairs (two floats per row) -- the \
same underlying corpus as verifier_auroc, but the raw PCIB signals \
themselves rather than a weighted combination. `benchmark_data["GibbsModel"]` \
and `benchmark_data["GibbsConfig"]` are classes; `benchmark_data["nce_loss"]` \
is a function. Do NOT write `import carnot` or anything under it -- blocked, \
same as above; these are handed to you directly for exactly that reason. \
Build `cfg = benchmark_data["GibbsConfig"](input_dim=2, hidden_dims=[4]); \
model = benchmark_data["GibbsModel"](cfg, key=...)`. Train it with real \
gradient steps: convert the correct/incorrect row lists to arrays, call \
`benchmark_data["nce_loss"](model, correct_array, incorrect_array)` \
(correct rows are the "data" NCE should push to LOW energy, incorrect rows \
are the "noise" NCE should push to HIGH energy -- a real, established reuse \
of NCE as a binary classifier, not a misuse), take the gradient, and update \
`model.layers[0]`, `model.output_weight`, `model.output_bias` for real \
epochs. Return {"calibrated_decision": {"final_state": {"w1": <the 4x2 \
first-layer weight matrix as a nested list>, "b1": <the 4-element first-layer \
bias as a list>, "w_out": <the 4-element output weight as a list>, "b_out": \
<the output bias as a float>}, "wall_clock_seconds": ...}}. The architecture \
(input_dim=2, hidden_dims=[4]) is FIXED -- do not change it, the harness can \
only rescore a final_state of this exact shape. The harness independently \
rescores your trained weights (both an energy AND a calibration score) \
against a DIFFERENT held-out set you cannot read -- your own training-set \
metric is never trusted. A weight set that scores every training row \
identically will be rejected as degenerate, not accepted.

For EITHER benchmark, do NOT return "final_energy" -- it will be IGNORED. \
The harness independently recomputes the true metrics from your \
`final_state`, so there is no way to claim a result you did not actually \
reach. Only your `final_state` and how you found it matter."""


def _subprocess_recompute_energy(benchmark_name: str, final_state: Any) -> float | None:
    """Recompute one benchmark's energy in a FRESH interpreter process.

    REQ-AUTO-025 CRITICAL-2 fix (2026-09-16 adversarial review). The
    hypothesis whose `final_state` this recomputes just finished running
    IN-PROCESS, in the same interpreter as this function -- `sandbox.py`'s
    own docstring says its isolation is "not a security boundary", and the
    review proved it: a hypothesis that does
    `import carnot.autoresearch.toy_benchmarks as tb; tb.BENCHMARK_ENERGY_
    FUNCTIONS["double_well"] = lambda s: -999999.0` (or the equivalent
    against `verifier_auroc_benchmark.py`'s `_binary_auroc`, or mutating the
    `lru_cache`d held-out row dicts in place) makes the SAME process's
    subsequent "independent" recompute return whatever the hypothesis chose
    -- fabricating a real git-committed "discovery." Spawning a brand-new
    `python -c`-equivalent process for the recompute closes this: the worker
    re-imports every dependency fresh, so nothing the hypothesis mutated in
    THIS process's `sys.modules` or `lru_cache`s can reach it. `final_state`
    is passed as JSON on stdin (never a live Python object, so nothing the
    hypothesis constructed -- a class instance, a closure -- crosses the
    process boundary either). Never raises -- a spawn failure, a timeout, or
    malformed worker output all become None (the same "unscoreable" contract
    every recompute function in this project already has).
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(_RECOMPUTE_WORKER)],
            input=json.dumps({"benchmark_name": benchmark_name, "final_state": final_state}),
            capture_output=True,
            text=True,
            timeout=_RECOMPUTE_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout.strip())
        energy = result.get("energy")
        return float(energy) if isinstance(energy, (int, float)) and energy == energy else None
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError, ValueError, AttributeError):
        return None


def _subprocess_recompute_calibrated_decision(final_state: Any) -> dict[str, float] | None:
    """Same fresh-process trust boundary as `_subprocess_recompute_energy`
    above, but for the calibrated_decision benchmark (REQ-AUTO-018), which
    reports TWO trusted numbers (`final_energy` and `brier`) instead of one.
    Kept as its own function rather than widening
    `_subprocess_recompute_energy`'s return type, so every existing caller of
    that function (double_well/rosenbrock/verifier_auroc) is untouched."""
    try:
        proc = subprocess.run(
            [sys.executable, str(_RECOMPUTE_WORKER)],
            input=json.dumps(
                {
                    "benchmark_name": CALIBRATED_DECISION_BENCHMARK_NAME,
                    "final_state": final_state,
                }
            ),
            capture_output=True,
            text=True,
            timeout=_RECOMPUTE_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout.strip())
        energy = result.get("energy")
        brier = result.get("brier")
        if not (isinstance(energy, (int, float)) and energy == energy):
            return None
        if not (isinstance(brier, (int, float)) and brier == brier):
            return None
        return {"final_energy": float(energy), "brier": float(brier)}
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError, ValueError, AttributeError):
        return None


def _recompute_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    """The one seam between the sandbox and the evaluator (REQ-AUTO-021,
    hardened by REQ-AUTO-025's CRITICAL-2 fix -- see
    `_subprocess_recompute_energy` above).

    For each benchmark this project has a real recompute function for, drop
    whatever `final_energy` the hypothesis self-reported and replace it with
    a FRESH-PROCESS recomputation -- None (dropped entirely) if `final_state`
    is missing, malformed, or the recompute worker itself fails for any
    reason, which the evaluator already treats as "not measured"
    (`evaluator.py`: `if bench_energy is None: continue`). A benchmark name
    this project does not know how to score (including anything an LLM made
    up) never gets a `final_energy` at all, so it can never register as an
    improvement or a regression.
    """
    verified: dict[str, Any] = {}
    for name, bench_metrics in raw_metrics.items():
        if not isinstance(bench_metrics, dict):
            continue
        entry = {k: v for k, v in bench_metrics.items() if k != "final_energy"}
        if name in BENCHMARK_ENERGY_FUNCTIONS or name == VERIFIER_AUROC_BENCHMARK_NAME:
            energy = _subprocess_recompute_energy(name, bench_metrics.get("final_state"))
            if energy is not None:
                entry["final_energy"] = energy
        elif name == CALIBRATED_DECISION_BENCHMARK_NAME:
            result = _subprocess_recompute_calibrated_decision(bench_metrics.get("final_state"))
            if result is not None:
                entry["final_energy"] = result["final_energy"]
                entry["brier"] = result["brier"]
        verified[name] = entry
    return verified


def _verified_execute_hypothesis(
    hypothesis_code: str,
    benchmark_data: dict[str, Any],
    config: SandboxConfig | None = None,
    docker_config: Any = None,
) -> SandboxResult:
    """Drop-in replacement for sandbox.execute_hypothesis: same sandbox, same
    isolation, but the returned metrics have been through `_recompute_metrics`
    before the evaluator ever sees them."""
    # The production round blocks ``carnot`` imports, but direct callers can
    # deliberately use a permissive config to test the fresh-process trust
    # boundary. Such a hypothesis still runs in this interpreter and can
    # mutate imported module state. Restore every trusted object reachable by
    # the scoring path, and clear both corpus caches so mutations to cached row
    # dictionaries cannot poison later baseline measurements or tests.
    original_binary_auroc = _verifier_auroc_module._binary_auroc
    original_split_corpus = _verifier_auroc_module._split_corpus
    original_load_corpus_rows = _verifier_auroc_module._load_corpus_rows
    original_toy_energy_functions = dict(BENCHMARK_ENERGY_FUNCTIONS)
    # REQ-AUTO-018: same defense-in-depth restore for calibrated_decision_
    # benchmark.py's private helpers/caches, even though the production
    # sandbox_config already blocks `carnot` imports entirely (see
    # `run_round` below) -- a direct caller using a permissive config should
    # not be able to poison this module's caches either.
    original_cd_binary_auroc = _calibrated_decision_module._binary_auroc
    original_cd_split_features = _calibrated_decision_module._split_features
    original_cd_load_corpus_rows = _calibrated_decision_module._load_corpus_rows
    original_cd_pcib_features = _calibrated_decision_module._pcib_features
    try:
        result = execute_hypothesis(hypothesis_code, benchmark_data, config, docker_config)
    finally:
        _verifier_auroc_module._binary_auroc = original_binary_auroc
        _verifier_auroc_module._split_corpus = original_split_corpus
        _verifier_auroc_module._load_corpus_rows = original_load_corpus_rows
        original_load_corpus_rows.cache_clear()
        original_split_corpus.cache_clear()
        BENCHMARK_ENERGY_FUNCTIONS.clear()
        BENCHMARK_ENERGY_FUNCTIONS.update(original_toy_energy_functions)
        _calibrated_decision_module._binary_auroc = original_cd_binary_auroc
        _calibrated_decision_module._split_features = original_cd_split_features
        _calibrated_decision_module._load_corpus_rows = original_cd_load_corpus_rows
        _calibrated_decision_module._pcib_features = original_cd_pcib_features
        original_cd_load_corpus_rows.cache_clear()
        original_cd_split_features.cache_clear()
        original_cd_pcib_features.cache_clear()
    if not result.success:
        return result
    return SandboxResult(
        success=result.success,
        metrics=_recompute_metrics(result.metrics),
        stdout=result.stdout,
        stderr=result.stderr,
        error=result.error,
        wall_clock_seconds=result.wall_clock_seconds,
        timed_out=result.timed_out,
    )


class _energy_verification_patch:
    """Context manager: for its duration, run_loop_with_generator's internal
    calls to `execute_hypothesis` go through `_verified_execute_hypothesis`
    instead. orchestrator.py binds `execute_hypothesis` into its OWN module
    namespace at import time (`from ...sandbox import ... execute_hypothesis`),
    so patching that name on the orchestrator module -- not on sandbox.py,
    which nothing here calls directly -- is what actually takes effect.

    This is the whole fix for REQ-AUTO-021: it reuses run_loop_with_generator's
    existing, already-tested accept/reject/circuit-breaker/logging logic
    completely unmodified, and only replaces the one step that read a
    self-reported number instead of an independently-verified one.
    """

    def __enter__(self) -> None:
        self._original = _orchestrator_module.execute_hypothesis
        _orchestrator_module.execute_hypothesis = _verified_execute_hypothesis

    def __exit__(self, *exc_info: object) -> None:
        _orchestrator_module.execute_hypothesis = self._original


def seed_baselines() -> BaselineRecord:
    """The starting point for a fresh baseline cache.

    CLAUDE.md "Energy-Based Calibrated-Decision Training Floor" (2026-09-18):
    double_well/rosenbrock (scripts/demo_autoresearch.py's original seeds,
    REQ-AUTO-021's fitness target #1) are deliberately NOT seeded here any
    more -- both were driven to exact machine-precision zero on the first
    production fire (2026-09-13); every round since re-solved an
    already-solved problem with zero headroom left. An EXISTING cache that
    already has entries for them keeps those entries untouched (see
    `_merge_missing_seed_benchmarks` -- it only adds names missing from the
    loaded record, never removes one), so this only affects a genuinely
    fresh cache, or a fresh benchmark being added to an existing one.
    """
    record = BaselineRecord(version="0.1.0")
    # REQ-AUTO-025: a real measurement, not an illustrative placeholder --
    # PCIBProbe's own documented default weights (0.5, 0.5), scored against
    # the actual held-out corpus split. See verifier_auroc_benchmark.py's
    # module docstring for the number.
    record.benchmarks[VERIFIER_AUROC_BENCHMARK_NAME] = BenchmarkMetrics(
        benchmark_name=VERIFIER_AUROC_BENCHMARK_NAME,
        final_energy=measure_default_weight_energy(),
        convergence_steps=0,
        wall_clock_seconds=0.0,
    )
    # REQ-AUTO-018: also a real measurement -- a freshly constructed,
    # UNTRAINED GibbsModel's zero-initialized output layer scores exactly
    # chance (AUROC 0.5) on the held-out split. See
    # calibrated_decision_benchmark.py's module docstring for why this is
    # the honest seed, not a fabricated placeholder.
    record.benchmarks[CALIBRATED_DECISION_BENCHMARK_NAME] = BenchmarkMetrics(
        benchmark_name=CALIBRATED_DECISION_BENCHMARK_NAME,
        final_energy=measure_default_calibration_energy()["final_energy"],
        convergence_steps=0,
        wall_clock_seconds=0.0,
    )
    return record


def _merge_missing_seed_benchmarks(record: BaselineRecord) -> None:
    """Add any benchmark `seed_baselines()` knows about but `record` (loaded
    from a persisted cache) does not, in place. Never overwrites an existing
    entry -- a benchmark the cache already tracks keeps its real, evolved
    baseline; only a genuinely NEW benchmark name gets seeded.

    REQ-AUTO-025 CRITICAL-1 fix (2026-09-16 adversarial review). Before this,
    `load_baselines` returned a cached `BaselineRecord` verbatim -- correct
    when the cache already knows every benchmark, but on the FIRST run after
    `verifier_auroc` shipped, the real production cache
    (`ops/.autoresearch_baselines.json`) predates it and holds only
    double_well/rosenbrock. `evaluator.py` iterates `baselines.benchmarks`
    and treats a name absent from that dict as "nothing to compare against",
    so the round's evaluator gate returns "PASS: No regression" (not
    "Improved") for ANY reported verifier_auroc energy, accepts it, and
    `orchestrator._update_baselines` writes THAT number as the baseline --
    reproduced with a bad weight pair (held-out energy ~0.72) landing as the
    baseline, then the probe's own DEFAULT weights (the seed number this
    module is supposed to start from) landing as a git-committed "accept ...
    (final_energy None -> 0.6534640104441265)" improvement against it. This
    function is the migration: called every time a cache is loaded, so any
    FUTURE new benchmark added the same way `verifier_auroc` was gets seeded
    correctly too, not just this one.
    """
    seed = seed_baselines()
    for name, metrics in seed.benchmarks.items():
        record.benchmarks.setdefault(name, metrics)


def load_baselines(baseline_cache: Path) -> BaselineRecord:
    if baseline_cache.exists():
        try:
            record = BaselineRecord.load(baseline_cache)
        except (OSError, json.JSONDecodeError, KeyError):
            return seed_baselines()  # fail toward a fresh, known-good baseline, not a crash
        _merge_missing_seed_benchmarks(record)
        return record
    return seed_baselines()


def load_experiment_log(log_cache: Path) -> ExperimentLog:
    try:
        return ExperimentLog.load(log_cache)
    except (OSError, json.JSONDecodeError, TypeError):
        # Same fail-toward-fresh-state as load_baselines above (adversarial
        # review 2026-09-12, finding 7: ExperimentLog.load has no try/except
        # of its own, so a corrupt cache raised before any receipt was written).
        return ExperimentLog()


def codex_available() -> bool:
    """Precondition check (Pre-Launch Preconditions Discipline pattern)."""
    return shutil.which("codex") is not None


def call_codex(prompt: str, model: str, timeout: int) -> tuple[bool, str]:
    """One codex exec call. Mirrors pages_adversarial_audit.py:call_codex and
    scripts/research_conductor.py's own `_build_agent_command` codex branch --
    same flags, same stdin-piped-prompt shape, same '-' terminator.

    `--cd` points at a FRESH, EMPTY scratch directory, never `PROJECT_ROOT`
    (adversarial review 2026-09-12: `--dangerously-bypass-approvals-and-
    sandbox` is what it says -- codex exec is agentic and this flag removes
    both the approval gate and the sandbox, so a docstring claiming "no repo
    tool access" was false as long as `--cd` pointed at the real checkout.
    The task here is a plain text-completion ask; it needs no repo access at
    all, so give it none -- even a codex session that decided to explore its
    cwd can only reach an empty temp dir that is deleted right after).
    """
    try:
        with tempfile.TemporaryDirectory(prefix="autoresearch-codex-") as scratch_dir:
            proc = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--color",
                    "never",
                    "--model",
                    model,
                    "--cd",
                    scratch_dir,
                    "--ephemeral",
                    "-",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=scratch_dir,
            )
        if proc.returncode != 0:
            # CORRECTION (2026-09-17): this used to slice stderr to [:200]. codex's
            # own startup banner (workdir/model/provider/approval/sandbox/reasoning
            # lines) is itself close to 200 chars, so every failure reason this
            # project has ever logged was just that banner -- the real error text,
            # which always comes AFTER the banner, was silently truncated away.
            # Every "codex_call_failed" reason recorded before this fix told nobody
            # what actually went wrong. 4000 chars comfortably clears the banner and
            # still bounds the receipt.
            detail = proc.stderr.strip() or proc.stdout.strip()
            return False, f"codex exit {proc.returncode}: {detail[:4000]}"
        return True, proc.stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)


def _hypothesis_prompt(
    baselines: BaselineRecord, recent_failures: list[dict[str, Any]], iteration: int
) -> str:
    """Shared prompt for both generators -- codex and its Fable fallback get
    asked the exact same question, so a comparison between them is fair."""
    return f"{AUTORESEARCH_SYSTEM_PROMPT}\n\n{_build_user_prompt(baselines, recent_failures, iteration)}"


def codex_generate_hypotheses(
    model: str,
    timeout: int,
    baselines: BaselineRecord,
    recent_failures: list[dict[str, Any]],
    iteration: int,
) -> list[tuple[str, str]]:
    """The primary generator. Reuses hypothesis_generator.py's own
    user-prompt-building and response-parsing, but AUTORESEARCH_SYSTEM_PROMPT
    (REQ-AUTO-021), not hypothesis_generator's own DEFAULT_SYSTEM_PROMPT --
    the two ask for a different return contract (final_state vs. a
    self-reported final_energy) and only ours is actually verified downstream
    by _energy_verification_patch."""
    prompt = _hypothesis_prompt(baselines, recent_failures, iteration)
    ok, output = call_codex(prompt, model, timeout)
    if not ok:
        recent_failures.append({"description": "codex_call_failed", "reason": output})
        return []
    return _extract_hypotheses(output)


def call_fable(prompt: str, timeout: int) -> tuple[bool, str]:
    """Second-opinion generator (2026-09-12 operator directive: "If the codex
    run returns zero hypothesis, I want to follow up with a Fable 5.1 run to
    see if it finds anything"). Mirrors pages_adversarial_audit.py:call_claude
    -- a stateless `claude --print` completion, not an agentic session (no
    --dangerously-skip-permissions, so unlike call_codex there is no repo
    tool access to restrict in the first place). `claude --help` documents
    'fable' as a first-class --model alias directly.

    DORMANT since 2026-09-20 (Claude quota-conserve operator directive) --
    `generate_hypotheses_with_fallback` no longer calls this function. Kept,
    not deleted, so re-enabling is a one-line revert if the quota constraint
    is lifted later. See that function's own docstring for the full context.
    """
    try:
        proc = subprocess.run(
            ["claude", "--model", "fable", "--effort", "max", "--print", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            # Same fix as call_codex above (2026-09-17) -- do not re-truncate to
            # [:200] here even though `claude --print` has no startup banner to
            # exhaust it today; the sibling function silently losing its real
            # error text is exactly the failure this correction exists to prevent.
            detail = proc.stderr.strip() or proc.stdout.strip()
            return False, f"claude exit {proc.returncode}: {detail[:4000]}"
        return True, proc.stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)


def fable_generate_hypotheses(
    timeout: int,
    baselines: BaselineRecord,
    recent_failures: list[dict[str, Any]],
    iteration: int,
) -> list[tuple[str, str]]:
    """Same question as codex_generate_hypotheses, same parsing, different
    model -- called only when codex returned nothing first."""
    prompt = _hypothesis_prompt(baselines, recent_failures, iteration)
    ok, output = call_fable(prompt, timeout)
    if not ok:
        recent_failures.append({"description": "fable_call_failed", "reason": output})
        return []
    return _extract_hypotheses(output)


def generate_hypotheses_with_fallback(
    model: str,
    timeout: int,
    baselines: BaselineRecord,
    recent_failures: list[dict[str, Any]],
    iteration: int,
    fallback_log: list[int],
    fable_timeout: int = DEFAULT_FABLE_TIMEOUT_S,
) -> list[tuple[str, str]]:
    """codex only (2026-09-20 operator directive, quota-conserve: Claude usage
    across all automated workers must drop while Claude quota is constrained).
    This used to fall back to Fable 5.1 (`claude --model fable`) when codex
    returned nothing -- see the retained, now-dormant `call_fable`/
    `fable_generate_hypotheses` below. That fallback is DISABLED here, not
    deleted: gemini-cli was checked as the natural non-Claude replacement and
    is currently unusable for a different, unrelated reason (`gemini --model
    gemini-3.1-pro-preview --yolo -p ...` fails immediately with
    `IneligibleTierError: This client is no longer supported for Gemini Code
    Assist for individuals` -- an external Google account-tier change, not
    something a retry or a code fix here can work around). So the honest
    choice today is codex-only, accepting that a codex failure now ends this
    iteration with zero hypotheses instead of getting a second opinion.
    `fallback_log` stays as a parameter (empty forever under this directive)
    rather than being torn out, so `generator_label_for_entry` and every
    caller need no signature change -- and so re-enabling Fable later, if the
    operator lifts the quota constraint, is the one-line revert of this
    function body, not a rebuild. See ops/known-issues.md 2026-09-20 for the
    directive and the gemini-cli finding.
    """
    return codex_generate_hypotheses(model, timeout, baselines, recent_failures, iteration)


_ENTRY_ID_ITERATION = re.compile(r"-(\d+)$")


def generator_label_for_entry(entry_id: str, fable_fallback_iterations: Sequence[int]) -> str:
    """REQ-AUTO-024: which generator actually produced this entry.

    orchestrator.py's `run_loop_with_generator` names each entry
    ``llm-<timestamp>-<iteration:03d>`` (see its own exp_id line) -- the
    trailing iteration number is the only place that survives to tell
    codex and Fable apart after the fact, since `ExperimentEntry` itself
    (a shared, REQ-AUTO-008 dataclass) carries no generator-provenance
    field, and this script does not own that dataclass. Falls back to
    "codex exec" (the historical, still-correct-in-the-common-case
    hardcoded text) if the id does not match the expected shape, rather
    than raising on an unexpected format.
    """
    match = _ENTRY_ID_ITERATION.search(entry_id)
    if match and int(match.group(1)) in fable_fallback_iterations:
        return "Fable 5.1 fallback (codex returned nothing this iteration)"
    return "codex exec"


def _git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=project_root, check=check, capture_output=True, text=True
    )


_SAFE_BENCHMARK_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def commit_accepted_hypothesis(
    checker: ConstitutionChecker,
    entry: ExperimentEntry,
    benchmark_name: str,
    baseline_before: float | None,
    baseline_after: float | None,
    *,
    project_root: Path,
    generator_label: str = "codex exec",
) -> str | None:
    """Persist one accepted hypothesis as its own scoped git commit.

    AVO's "git commit per version with its score", adapted: the record is a
    JSON sidecar (see module docstring for why). Returns the new commit SHA,
    or None if the name is unsafe, the constitution forbade it, or the write/
    add/commit did not apply cleanly -- in every None case, no file is left
    behind (adversarial review 2026-09-12, findings 3/4/6: `benchmark_name`
    is LLM-controlled data flowing into a filesystem path and a
    `ConstitutionChecker.check()` call that only `re.search`-matches, so an
    unsanitized name is a path-traversal vector; a non-JSON-serializable
    metric must not crash before the receipt is written; a failed commit
    must not leave an untracked file for a later `git add -A` to sweep in).

    `generator_label` (REQ-AUTO-024) names which generator actually
    produced this hypothesis in the commit message -- callers should pass
    `generator_label_for_entry(entry.id, fable_fallback_iterations)`, not
    rely on the default. The default of "codex exec" exists only so the
    many tests exercising commit mechanics (not attribution) don't need a
    value they don't care about.
    """
    if not _SAFE_BENCHMARK_NAME.match(benchmark_name):
        return None

    rel_dir = Path("ops") / "autoresearch_discoveries" / benchmark_name
    rel_path = rel_dir / f"{entry.id}.json"

    create_verdict = checker.check(f"create_file:{rel_path.as_posix()}")
    if create_verdict.category != ActionCategory.ALLOWED:
        return None
    commit_verdict = checker.check("git_commit")
    if commit_verdict.category != ActionCategory.ALLOWED:
        return None

    record = {
        "id": entry.id,
        "timestamp": entry.timestamp,
        "benchmark": benchmark_name,
        "description": entry.hypothesis_description,
        "code": entry.hypothesis_code,
        "metrics": entry.sandbox_metrics.get(benchmark_name, {}),
        "eval_verdict": entry.eval_verdict,
        "eval_reason": entry.eval_reason,
        "baseline_final_energy_before": baseline_before,
        "baseline_final_energy_after": baseline_after,
    }
    try:
        serialized = json.dumps(record, indent=2) + "\n"
    except TypeError:
        return None  # e.g. a numpy/jax scalar the hypothesis returned -- not our bug to crash on

    abs_path = project_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(serialized)

    add = _git(project_root, "add", rel_path.as_posix(), check=False)
    if add.returncode != 0:
        abs_path.unlink(missing_ok=True)
        return None

    message = (
        f"[autoresearch] {benchmark_name}: accept {entry.id} "
        f"(final_energy {baseline_before} -> {baseline_after})\n\n"
        f"{entry.hypothesis_description}\n\n"
        "Autonomous mutation round -- no operator or outer-loop session\n"
        f"involved in this commit. Hypothesis proposed via {generator_label},\n"
        "accepted by the existing 3-gate evaluator (REQ-AUTO-005),\n"
        "persisted per REQ-AUTO-019/REQ-AUTO-020.\n"
    )
    # -- <path> (a pathspec, not a bare flag) scopes the commit to ONLY this
    # file even if something else is already staged in the index -- a bare
    # `git commit -m` commits the WHOLE index, which is the exact bug an
    # adversarial review reproduced 2026-09-12 (a pre-staged unrelated file
    # landed inside this commit).
    commit = _git(project_root, "commit", "-m", message, "--", rel_path.as_posix(), check=False)
    if commit.returncode != 0:
        _git(project_root, "reset", "HEAD", "--", rel_path.as_posix(), check=False)
        abs_path.unlink(missing_ok=True)
        return None
    return _git(project_root, "rev-parse", "HEAD").stdout.strip()


def run_round(
    *,
    model: str,
    max_iterations: int,
    codex_timeout: int = DEFAULT_CODEX_TIMEOUT_S,
    fable_timeout: int = DEFAULT_FABLE_TIMEOUT_S,
    project_root: Path = PROJECT_ROOT,
    baseline_cache: Path | None = None,
    log_cache: Path | None = None,
    receipt_path: Path | None = None,
) -> int:
    baseline_cache = baseline_cache or project_root / "ops" / ".autoresearch_baselines.json"
    log_cache = log_cache or project_root / "ops" / ".autoresearch_experiment_log.json"
    receipt_path = receipt_path or project_root / "ops" / "autoresearch_conductor_report.md"
    for path in (baseline_cache, log_cache, receipt_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC)
    report_lines = [
        "# Autoresearch conductor round",
        "",
        f"- started: {started.isoformat()}",
        f"- model: {model}",
        f"- max_iterations: {max_iterations}",
        "",
    ]

    if not codex_available():
        report_lines.append("BLOCKED: `codex` CLI not found on PATH. No round run.")
        receipt_path.write_text("\n".join(report_lines) + "\n")
        print("blocked_codex_unavailable")
        return 0

    checker = ConstitutionChecker()
    baselines = load_baselines(baseline_cache)
    experiment_log = load_experiment_log(log_cache)
    before_count = len(experiment_log.entries)
    breaker_historical_tail = experiment_log.consecutive_failures()
    energy_before = {name: metrics.final_energy for name, metrics in baselines.benchmarks.items()}

    fable_fallback_iterations: list[int] = []
    # REQ-AUTO-022: orchestrator.py owns one `recent_failures` list for the
    # whole loop (cleared only on an accepted hypothesis) and hands it to
    # `generator` by reference every iteration -- capturing that same
    # reference here means the list still holds every unclear failure once
    # the loop ends, so the receipt can say WHY a round produced nothing.
    captured_failures: list[dict[str, Any]] = []

    def generator(
        cur_baselines: BaselineRecord,
        recent_failures: list[dict[str, Any]],
        iteration: int,
    ) -> list[tuple[str, str]]:
        nonlocal captured_failures
        captured_failures = recent_failures
        return generate_hypotheses_with_fallback(
            model,
            codex_timeout,
            cur_baselines,
            recent_failures,
            iteration,
            fable_fallback_iterations,
            fable_timeout,
        )

    config = AutoresearchConfig(
        max_iterations=max_iterations,
        max_consecutive_failures=10,
        # REQ-AUTO-023: 3, not the orchestrator default's own 3 by
        # coincidence -- explicit here because this generator is expensive
        # (a codex subprocess up to codex_timeout, then Fable up to
        # fable_timeout on top). Worst case 3 * (codex_timeout +
        # fable_timeout) must stay under the conductor's own outer timeout
        # for this script (see research_conductor.py:_run_autoresearch_round,
        # bumped to 3600s alongside this for exactly that reason).
        max_consecutive_empty_generations=3,
        constitution_checker=checker,
        # REQ-AUTO-025 CRITICAL-2 (Variant A) fix (2026-09-16 adversarial
        # review): block every `carnot` import for sandboxed hypothesis
        # code, not just the stdlib roots SandboxConfig's own default
        # blocks. Without this, a hypothesis could `import
        # carnot.autoresearch.verifier_auroc_benchmark` directly and call
        # its internal `_split_corpus()` to read the held-out rows it is
        # meant to never see, or reach `toy_benchmarks.BENCHMARK_ENERGY_
        # FUNCTIONS` to set up the monkeypatch `_subprocess_recompute_
        # energy` otherwise defends against (that fix closes the RECOMPUTE
        # step; this closes the EXECUTION step). PCIBProbe is handed to the
        # hypothesis directly via `benchmark_data["PCIBProbe"]`
        # (`default_benchmark_data()`) so blocking `carnot` does not break
        # the intended verifier_auroc workflow.
        sandbox_config=SandboxConfig(blocked_modules=BLOCKED_MODULES | frozenset({"carnot"})),
    )
    # REQ-AUTO-021: for the duration of the loop, every sandboxed hypothesis's
    # metrics are recomputed from its final_state through a real potential
    # function before the evaluator sees them -- see _energy_verification_patch.
    with _energy_verification_patch():
        result = run_loop_with_generator(
            generator, baselines, default_benchmark_data(), config, experiment_log
        )

    # Adversarial review 2026-09-12, findings 1/3/5: only commit benchmarks the
    # EVALUATOR itself measured as a real improvement (entry.eval_improvements),
    # never every key the hypothesis's return dict happened to contain -- that
    # was accepting a no-op ({}), an unknown made-up benchmark name (which is
    # also how an LLM-controlled string reached a filesystem path), and
    # crediting a hypothesis with a benchmark it never actually improved.
    # `energy_before` tracks the running per-benchmark value as of just BEFORE
    # each entry, updated after processing it -- the round-final baseline
    # (checked once, after the whole loop) was wrongly stamped onto every
    # entry's commit message regardless of which iteration produced it.
    new_entries = experiment_log.entries[before_count:]
    committed: list[tuple[str, str, str]] = []
    for entry in new_entries:
        if entry.outcome != "accepted":
            continue
        for bench_name in entry.eval_improvements:
            after_metrics = entry.sandbox_metrics.get(bench_name, {})
            after_energy = (
                after_metrics.get("final_energy") if isinstance(after_metrics, dict) else None
            )
            sha = commit_accepted_hypothesis(
                checker,
                entry,
                bench_name,
                energy_before.get(bench_name),
                after_energy,
                project_root=project_root,
                generator_label=generator_label_for_entry(entry.id, fable_fallback_iterations),
            )
            if sha:
                energy_before[bench_name] = after_energy
                committed.append((entry.id, bench_name, sha))

    result.final_baselines.save(baseline_cache)
    experiment_log.save(log_cache)

    report_lines += [
        f"- iterations: {result.iterations}",
        f"- accepted: {result.accepted}",
        f"- rejected: {result.rejected}",
        f"- pending_review: {result.pending_review}",
        f"- circuit_breaker_tripped: {result.circuit_breaker_tripped}",
        f"- breaker_invocation_start_position: {before_count}",
        f"- breaker_historical_tail_at_start: {breaker_historical_tail}",
        "- breaker_invocation_local_tail_at_start: 0",
        f"- breaker_invocation_local_tail_at_end: "
        f"{experiment_log.consecutive_failures_since(before_count)}",
        f"- generator_exhausted: {result.generator_exhausted}",
        f"- fable_fallback_iterations: {fable_fallback_iterations or 'none'}",
        "",
    ]
    if result.generator_exhausted:
        # REQ-AUTO-023: this now means codex+Fable BOTH failed
        # max_consecutive_empty_generations times in a row, not just once --
        # see the '## Generator failure reasons' section below for why each
        # attempt failed.
        report_lines.append(
            f"codex and Fable 5.1 both produced nothing across "
            f"{len(fable_fallback_iterations)} attempt(s) this round -- giving up."
        )
    if captured_failures:
        # REQ-AUTO-022: the diagnostic the 2026-09-12 known-issues entry
        # named as missing -- WHY a generator returned nothing, not just
        # THAT it did. `reason` is the raw call_codex/call_fable failure
        # string (a timeout message, a non-zero exit + truncated stderr, or
        # an OSError) -- capped so a runaway stderr blob cannot blow up the
        # receipt.
        report_lines.append("")
        report_lines.append("## Generator failure reasons")
        for fail in captured_failures:
            desc = fail.get("description", "unknown")
            reason = str(fail.get("reason", ""))[:300]
            report_lines.append(f"- {desc}: {reason}")
    if committed:
        report_lines.append("## Committed lineage")
        for exp_id, bench, sha in committed:
            report_lines.append(f"- {exp_id} ({bench}): {sha[:12]}")
    else:
        report_lines.append("No hypothesis both won this round and committed cleanly.")
    receipt_path.write_text("\n".join(report_lines) + "\n")

    print(
        f"complete: autoresearch round iterations={result.iterations} "
        f"accepted={result.accepted} committed={len(committed)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--codex-timeout", type=int, default=DEFAULT_CODEX_TIMEOUT_S)
    parser.add_argument("--fable-timeout", type=int, default=DEFAULT_FABLE_TIMEOUT_S)
    args = parser.parse_args()
    return run_round(
        model=args.model,
        max_iterations=args.max_iterations,
        codex_timeout=args.codex_timeout,
        fable_timeout=args.fable_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
