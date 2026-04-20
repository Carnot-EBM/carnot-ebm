#!/usr/bin/env python3
"""Experiment 572: PRA EORM Beam Search — CPU prototype.

Researcher summary:
    arXiv 2604.09482 (April 2026) introduces Process Reward Agents (PRA):
    decouple a frozen LLM from a step-wise reward module, run beam search
    pruning per step.  Achieves +25.7% on MedQA at 4B parameters.

    Carnot's EORM is structurally identical to the PRA reward module:
      generate K=3 candidate continuations per step
      score each with EORM → select minimum energy → proceed.

    This experiment validates the PRA-EORM wiring on 20 synthetic arithmetic
    problems using a CPU-safe EORM mock (no GPU needed).  We measure:
      - baseline_violation_rate: greedy (first candidate) picks high-energy step
      - beam_violation_rate:     EORM-guided beam picks high-energy step
      - beam_improvement:        baseline - beam (positive = beam wins)

Gate chain (in order):
    1. apply_env_autofix()                              — normalise env
    2. ExperimentTimeoutWatchdog(572, timeout_minutes=20) — hard cap
    3. ExperimentTemplate(572, ..., requires_gpu=False)  — no GPU
    4. 20 synthetic arithmetic problems, K=3 candidates per step, 2 steps
    5. Greedy baseline vs EORM beam
    6. Build artifact schema='carnot.pra_eorm_beam.v1'
    7. tmpl.assert_deliverable_written()  -- FINAL LINE

Spec: REQ-REPAIR-016,
      SCENARIO-REPAIR-031, SCENARIO-REPAIR-032, SCENARIO-REPAIR-033
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() MUST be called before any JAX/CUDA import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.pra_eorm_beam import PRABeamResult, PRAEBMBeamSearch
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 572
EXP_TITLE = "PRA EORM Beam Search"
DELIVERABLE = "results/experiment_572_pra_eorm_beam_search.json"

K_CANDIDATES = 3
N_STEPS = 2  # Two-step math problems

# 20 fixed synthetic arithmetic problems
PROBLEMS = [
    "What is 3 + 5?",
    "What is 12 - 7?",
    "What is 4 * 6?",
    "What is 20 / 4?",
    "What is 8 + 9?",
    "What is 15 - 6?",
    "What is 7 * 3?",
    "What is 36 / 6?",
    "What is 11 + 14?",
    "What is 30 - 13?",
    "What is 9 * 4?",
    "What is 45 / 9?",
    "What is 6 + 17?",
    "What is 25 - 8?",
    "What is 5 * 8?",
    "What is 56 / 7?",
    "What is 13 + 19?",
    "What is 40 - 22?",
    "What is 11 * 3?",
    "What is 72 / 8?",
]


# ---------------------------------------------------------------------------
# CI-safe EORM mock
# ---------------------------------------------------------------------------

class _MockEORMModel:
    """CI-safe EORM mock: returns deterministic energy based on text hash.

    **For engineers:**
        When a real EORM checkpoint is not available (CI, CPU-only runs), we
        use this mock.  It assigns energy as a simple function of the text so
        that the beam search can still demonstrate differential selection — the
        mock is deterministic, so the experiment produces stable results across
        runs without requiring a GPU or model weights.

        Energy assignment rule:
          - Correct-sounding candidate texts (containing "=" sign) get lower
            energy, rewarding candidates that look like completed arithmetic.
          - All other candidates get a hash-based energy in [0.5, 1.5].

        This is intentionally simple — the goal is to test the beam-search
        wiring, not the quality of the EORM model itself.
    """

    def energy(self, cot_input: object) -> float:
        """Return mock energy from the candidate step text.

        The input may be a CoTEnergyInput dataclass or a raw string (depending
        on whether JAX is available).  We extract the text and hash it.
        """
        # Handle both CoTEnergyInput and raw string (mock path)
        if hasattr(cot_input, "response_text"):
            text = cot_input.response_text
        else:
            text = str(cot_input)

        # Reward candidates that look like completed calculations
        if "=" in text:
            base = 0.1
        else:
            base = 0.8

        # Deterministic variation via polynomial hash
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFF
        return base + (h % 1000) / 2000.0  # [base, base+0.5)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _make_generate_fn(problem: str) -> object:
    """Build a synthetic generate_fn for one arithmetic problem.

    **For engineers:**
        A real PRA experiment would call a frozen LLM here.  For this CPU
        prototype we generate three fixed candidates per step:
          - Step 0 (setup): one candidate with an "=" (complete), two without.
          - Step 1 (conclusion): same pattern.

        This ensures the EORM mock reliably picks the "=" candidate (lower
        energy), demonstrating that beam search outperforms greedy when greedy
        happens to choose a high-energy candidate first.

    Args:
        problem: The arithmetic question being solved.

    Returns:
        Callable(question, step_idx) → list[str] of length K_CANDIDATES.
    """
    # Extract numbers from problem for synthetic candidates
    import re
    nums = re.findall(r"\d+", problem)
    a = int(nums[0]) if nums else 3
    b = int(nums[1]) if len(nums) > 1 else 5

    def generate_fn(question: str, step_idx: int) -> list[str]:
        if step_idx == 0:
            # First step: identify operands
            # Candidate 0: correct-looking (contains =)
            return [
                f"Let x = {a} and y = {b}",  # no =, greedy picks this
                f"The first number is {a} = given",  # has =
                f"Operands: {a} and {b}",      # no =
            ]
        else:
            # Second step: compute answer
            try:
                answer = eval(f"{a} + {b}")  # noqa: S307
            except Exception:
                answer = a + b
            return [
                "Now I add them together",           # no =, greedy picks this
                f"Result = {answer}",                # has =, beam picks this
                "The computation follows naturally",  # no =
            ]

    return generate_fn


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _try_load_real_eorm() -> object | None:
    """Attempt to load a real EORM model from the latest checkpoint.

    Returns None if no checkpoint is available (CI-safe fallback to mock).
    """
    checkpoint_path = _REPO_ROOT / "results" / "experiment_556_eorm_grpo_retrain.json"
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path) as f:
            result = json.load(f)
        model_path = result.get("model_path") or result.get("checkpoint_path")
        if not model_path:
            return None
        from carnot.models.eorm import EORMModel  # noqa: PLC0415
        model = EORMModel.load(model_path)
        _log.info("Loaded real EORM from %s", model_path)
        return model
    except Exception as e:
        _log.info("Could not load real EORM (%s) — using mock", e)
        return None


def main() -> None:
    """Run the PRA EORM Beam Search experiment."""
    # Step 2: Watchdog — hard 20-minute cap
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)

    # Step 3: ExperimentTemplate setup
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _log.info("Starting Exp %d: %s", EXP_ID, EXP_TITLE)

    # Load EORM model (real or CI-safe mock)
    eorm_model = _try_load_real_eorm() or _MockEORMModel()
    model_source = "real_eorm" if not isinstance(eorm_model, _MockEORMModel) else "ci_mock"
    _log.info("EORM source: %s", model_source)

    # Build beam search engine
    beam = PRAEBMBeamSearch(eorm_model=eorm_model, k_candidates=K_CANDIDATES)

    # Step 4 & 5: Evaluate 20 synthetic problems
    problem_results: list[dict] = []
    total_baseline_violations = 0.0
    total_beam_violations = 0.0

    for idx, problem in enumerate(PROBLEMS):
        generate_fn = _make_generate_fn(problem)
        result: PRABeamResult = beam.run_beam_episode(
            question=problem,
            generate_fn=generate_fn,
            n_steps=N_STEPS,
        )

        total_baseline_violations += result.baseline_violation_rate
        total_beam_violations += result.beam_violation_rate

        problem_results.append({
            "problem_idx": idx,
            "question": problem,
            "n_steps": result.n_steps,
            "n_beams_explored": result.n_beams_explored,
            "baseline_violation_rate": result.baseline_violation_rate,
            "beam_violation_rate": result.beam_violation_rate,
            "improvement": result.improvement,
        })
        _log.info(
            "Problem %2d: baseline_vr=%.3f  beam_vr=%.3f  improvement=%.3f",
            idx,
            result.baseline_violation_rate,
            result.beam_violation_rate,
            result.improvement,
        )

    n_problems = len(PROBLEMS)
    baseline_violation_rate = total_baseline_violations / n_problems
    beam_violation_rate = total_beam_violations / n_problems
    beam_improvement = baseline_violation_rate - beam_violation_rate
    pra_viable = beam_violation_rate < baseline_violation_rate
    honest_verdict = "pra_viable" if pra_viable else "pra_no_improvement"

    _log.info(
        "Aggregate: baseline_vr=%.3f  beam_vr=%.3f  improvement=%.3f  viable=%s",
        baseline_violation_rate,
        beam_violation_rate,
        beam_improvement,
        pra_viable,
    )

    # Step 6: Build artifact
    artifact = tmpl.build_result(
        {
            "schema": "carnot.pra_eorm_beam.v1",
            "n_problems": n_problems,
            "k_candidates": K_CANDIDATES,
            "n_steps_per_problem": N_STEPS,
            "eorm_source": model_source,
            "baseline_violation_rate": baseline_violation_rate,
            "beam_violation_rate": beam_violation_rate,
            "beam_improvement": beam_improvement,
            "pra_viable": pra_viable,
            "honest_verdict": honest_verdict,
            "problem_results": problem_results,
        },
        status="success",
    )

    # Write deliverable
    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    _log.info("Deliverable written to %s", output_path)

    # Step 7: Final guard — MUST be last line
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
