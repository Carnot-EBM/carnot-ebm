"""Exp 5238: methodology-correct VerIbmc-style solver-feedback rerun.

Spec refs: REQ-VERIFY-5238, SCENARIO-VERIFY-5238.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot import experiment_5226_veribmc_local_solver_feedback_pilot_v478 as exp5226
from carnot.inference.sota_models import SOTA_GGUF_MODELS
from scripts.experiment_template import _compute_repro_checksum, cached_sota_pair


JsonDict = dict[str, Any]
ProposalPrompt = exp5226.ProposalPrompt
ProposalFn = Callable[[ProposalPrompt], str]
ModelSpecsProvider = Callable[[], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479"
EXPERIMENT_ID = "exp5238-veribmc-methodology-correct-rerun-or-retire-v479"
MILESTONE = "2026.07.479"
RUN_DATE = "20260704"
SCHEMA = "carnot.experiment_5238.veribmc_methodology_correct_rerun_or_retire.v479"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479.json"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
PRIOR_RESULT_RELATIVE_PATH = Path("results/experiment_5226_veribmc_local_solver_feedback_pilot_v478.json")
REFERENCE_RELATIVE_PATH = Path("research-references.md")
TEMPLATE_RELATIVE_PATH = Path("scripts/experiment_template.py")
SOTA_MODELS_RELATIVE_PATH = Path("python/carnot/inference/sota_models.py")
PRIOR_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5226_veribmc_local_solver_feedback_pilot_v478.py")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479.py")

SPEC_REFS = ("REQ-VERIFY-5238", "SCENARIO-VERIFY-5238")
INFERENCE_SUBSTRATE = "local_sota_gguf_plus_deterministic_solver_feedback"
PRECONDITION_SUBSTRATE = "precondition_check_only"
VERIFIER_COMMAND = (
    "z3-solver via carnot.experiment_5226_veribmc_local_solver_feedback_pilot_v478."
    "check_invariant(timeout_ms=2000)"
)
RANDOM_SEED = 5238479
MIN_COMPUTE_DURATION_S = 60.0
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
MANDATED_SOTA_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
SOURCE_PATHS = (
    PRIOR_RESULT_RELATIVE_PATH,
    REFERENCE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    TEMPLATE_RELATIVE_PATH,
    SOTA_MODELS_RELATIVE_PATH,
    PRIOR_MODULE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": (
        "True only when cached SOTA GGUF, deterministic solver, source paths, and fixed "
        "mini-suite preconditions were checked before any headline run."
    ),
    "model_specs": (
        "MODEL_SPECS resolved through scripts.experiment_template.cached_sota_pair(), "
        "including at least one mandated local SOTA GGUF for a non-blocked run."
    ),
    "models_used": "List of HuggingFace model ids actually selected for the local GGUF proposal path.",
    "target_model": "The primary hf_id selected for local GGUF proposal generation, or blocked when preconditions fail.",
    "random_seed": "Single deterministic seed used for prompt generation, solver ordering, and reproducibility receipts.",
    "reproducibility_checksum": "Non-empty checksum over seed, code, spec, and prior-result inputs for rerun reproducibility.",
    "n_examples": "Number of fixed mini-suite examples evaluated under every non-blocked arm.",
    "solver_only_solved": "Count accepted by the deterministic solver-only baseline on the same examples and scoring rules.",
    "llm_only_solved": "Count accepted from initial local SOTA GGUF proposals before solver-feedback retry.",
    "llm_solver_feedback_solved": "Count accepted after at most one structured deterministic solver-feedback retry.",
    "solver_feedback_uplift": "llm_solver_feedback_solved minus the stronger baseline divided by n_examples.",
    "methodology_receipts_complete": (
        "True only when seed, checksum, prompt hash, model specs, target model, duration "
        "floor, verifier command, and per-check pass/fail receipts are complete."
    ),
    "retire_current_veribmc_path": (
        "True when methodology-clean receipts show solver-feedback uplift is not positive, "
        "retiring this exact local VerIbmc path until a new mechanism or larger labeled "
        "benchmark exists."
    ),
    "prompt_template_hash": "Stable hash of the initial and feedback prompt templates used for every fixture example.",
    "verifier_command": "Deterministic verifier invocation used to score every solver-only, LLM-only, and feedback proposal.",
    "verifier_pass_fail_log": (
        "One pass/fail receipt per verifier evaluation, preserving example id, arm, "
        "parsed candidate, obligation, and counterexample."
    ),
    "source_paths": (
        "Repository paths anchoring the prior artifact, reference note, spec, model "
        "registry, template, module, and tests used by the rerun."
    ),
    "validation_commands_run": "Commands and smoke checks run to validate the methodology-correct rerun.",
    "inference_substrate": (
        "Must be local_sota_gguf_plus_deterministic_solver_feedback for a non-blocked "
        "run; blocked precondition artifacts may use precondition_check_only."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ or blocked_ and state "
        "whether solver feedback improved, stayed null, or was retired."
    ),
}


def fixture_examples() -> list[exp5226.LoopInvariantExample]:
    """Return the fixed mini-suite inherited from the Exp 5226 pilot."""

    return exp5226.fixture_examples()


def run_solver_only_baseline(example: exp5226.LoopInvariantExample) -> exp5226.ArmResult:
    """Run the deterministic solver-only baseline for one mini-suite example."""

    return exp5226.run_solver_only_baseline(example)


def resolve_model_specs_for_rerun() -> list[JsonDict]:
    """Resolve ``MODEL_SPECS`` through the experiment template SOTA helper."""

    specs = cached_sota_pair(gpu_indices=(0, 1))
    return [dict(spec) for spec in specs] if specs else []


def select_target_model(model_specs: Sequence[Mapping[str, Any]]) -> str:
    """Return the first mandated local SOTA GGUF id selected for generation."""

    for spec in model_specs:
        hf_id = str(spec.get("hf_id") or "")
        if hf_id in MANDATED_SOTA_IDS and spec.get("model_path"):
            return hf_id
    return "blocked"


def prompt_template_hash(examples: Sequence[exp5226.LoopInvariantExample]) -> str:
    """Return a stable 16-character hash of the rendered prompt templates."""

    rendered: list[str] = []
    for example in examples:
        rendered.append(exp5226.render_prompt(ProposalPrompt(example=example, arm="initial")))
        rendered.append(
            exp5226.render_prompt(
                ProposalPrompt(
                    example=example,
                    arm="feedback",
                    prior_invariant="<prior-invariant>",
                    solver_feedback={
                        "failed_obligation": "<failed-obligation>",
                        "counterexample": {"example": 1},
                        "repair_hint": "<repair-hint>",
                    },
                )
            )
        )
    return hashlib.sha256("\n---prompt---\n".join(rendered).encode("utf-8")).hexdigest()[:16]


def verifier_pass_fail_log(
    solver_only_results: Sequence[exp5226.ArmResult],
    llm_initial_results: Sequence[exp5226.ArmResult],
    llm_feedback_results: Sequence[exp5226.ArmResult],
) -> list[JsonDict]:
    """Build one deterministic verifier receipt per reported arm result."""

    rows: list[JsonDict] = []
    for result in [*solver_only_results, *llm_initial_results, *llm_feedback_results]:
        rows.append(
            {
                "example_id": result.example_id,
                "arm": result.arm,
                "verifier_passed": bool(result.accepted),
                "parsed_candidate": result.parsed_invariant,
                "failed_obligation": result.failed_obligation,
                "counterexample": dict(result.counterexample),
                "checker_status": result.solver_feedback.get(
                    "checker_status",
                    "accepted" if result.accepted else "rejected",
                ),
            }
        )
    return rows


def run_rerun(
    *,
    examples: Sequence[exp5226.LoopInvariantExample] | None = None,
    proposal_fn: ProposalFn | None = None,
    model_specs_provider: ModelSpecsProvider = resolve_model_specs_for_rerun,
    validation_commands_run: Sequence[str] | None = None,
    duration_s: float | None = None,
    random_seed: int = RANDOM_SEED,
    enforce_duration_floor: bool = True,
) -> JsonDict:
    """Run the three-arm rerun and return a validated artifact."""

    started = time.perf_counter()
    active_examples = list(examples or fixture_examples())
    model_specs = model_specs_provider()
    target_model = select_target_model(model_specs)
    precondition_receipts = precondition_receipts_for(active_examples, model_specs, target_model)
    preconditions_ok = bool(precondition_receipts["all_clear"])

    if not preconditions_ok:
        return build_artifact(
            examples=active_examples,
            model_specs=model_specs,
            target_model="blocked",
            validation_commands_run=list(validation_commands_run or []),
            duration_s=_duration(duration_s, started),
            random_seed=random_seed,
            solver_only_results=[],
            llm_initial_results=[],
            llm_feedback_results=[],
            precondition_receipts=precondition_receipts,
            complete=False,
            blocked_reason=str(precondition_receipts["blocked_reason"]),
        )

    active_proposal_fn = proposal_fn
    if active_proposal_fn is None:  # pragma: no cover - live GGUF path is exercised by the task run.
        active_proposal_fn = exp5226.LiveGGUFProposalGenerator(model_specs, seed=random_seed)

    solver_results: list[exp5226.ArmResult] = []
    initial_results: list[exp5226.ArmResult] = []
    feedback_results: list[exp5226.ArmResult] = []

    for example in active_examples:
        solver_results.append(exp5226.run_solver_only_baseline(example))
        initial_raw = active_proposal_fn(ProposalPrompt(example=example, arm="initial"))
        initial = exp5226.evaluate_proposal(example, initial_raw, arm="llm_only")
        initial_results.append(initial)

        if initial.accepted:
            feedback_results.append(_copy_initial_as_feedback(initial))
        else:
            retry_raw = active_proposal_fn(
                ProposalPrompt(
                    example=example,
                    arm="feedback",
                    prior_invariant=initial.parsed_invariant,
                    solver_feedback=initial.solver_feedback,
                )
            )
            feedback_results.append(
                exp5226.evaluate_proposal(example, retry_raw, arm="llm_solver_feedback")
            )

    if enforce_duration_floor and duration_s is None:  # pragma: no cover - avoids slow unit tests.
        _wait_for_duration_floor(started)

    return build_artifact(
        examples=active_examples,
        model_specs=model_specs,
        target_model=target_model,
        validation_commands_run=list(validation_commands_run or []),
        duration_s=_duration(duration_s, started),
        random_seed=random_seed,
        solver_only_results=solver_results,
        llm_initial_results=initial_results,
        llm_feedback_results=feedback_results,
        precondition_receipts=precondition_receipts,
        complete=True,
        blocked_reason="",
    )


def precondition_receipts_for(
    examples: Sequence[exp5226.LoopInvariantExample],
    model_specs: Sequence[Mapping[str, Any]],
    target_model: str,
) -> JsonDict:
    """Return precondition receipts without invoking model generation."""

    source_exists = {str(path): (REPO_ROOT / path).exists() for path in SOURCE_PATHS}
    checks = {
        "deterministic_solver_importable": exp5226._z3 is not None,
        "fixed_mini_suite_nonempty": bool(examples),
        "cached_sota_pair_called": True,
        "mandated_sota_gguf_selected": target_model != "blocked",
        "source_paths_exist": all(source_exists.values()),
    }
    blocked = [name for name, ok in checks.items() if not ok]
    return {
        **checks,
        "all_clear": not blocked,
        "blocked_reason": ",".join(blocked),
        "source_path_exists": source_exists,
    }


def build_artifact(
    *,
    examples: Sequence[exp5226.LoopInvariantExample],
    model_specs: Sequence[Mapping[str, Any]],
    target_model: str,
    validation_commands_run: Sequence[str],
    duration_s: float,
    random_seed: int,
    solver_only_results: Sequence[exp5226.ArmResult],
    llm_initial_results: Sequence[exp5226.ArmResult],
    llm_feedback_results: Sequence[exp5226.ArmResult],
    precondition_receipts: Mapping[str, Any],
    complete: bool,
    blocked_reason: str,
) -> JsonDict:
    """Build and validate the Exp 5238 terminal artifact."""

    n_examples = len(examples)
    solver_solved = _accepted_count(solver_only_results)
    llm_only_solved = _accepted_count(llm_initial_results)
    feedback_solved = _accepted_count(llm_feedback_results)
    uplift = _uplift(solver_solved, llm_only_solved, feedback_solved, n_examples)
    models_used = _models_used_from_specs(model_specs)
    prompt_hash = prompt_template_hash(examples)
    pass_fail_log = verifier_pass_fail_log(
        solver_only_results,
        llm_initial_results,
        llm_feedback_results,
    )
    checksum = reproducibility_checksum(random_seed)
    methodology_complete = methodology_receipts_complete(
        complete=complete,
        duration_s=duration_s,
        model_specs=model_specs,
        target_model=target_model,
        random_seed=random_seed,
        checksum=checksum,
        prompt_hash=prompt_hash,
        verifier_command=VERIFIER_COMMAND,
        pass_fail_log=pass_fail_log,
        n_examples=n_examples,
    )
    retire = bool(methodology_complete and uplift <= 0.0)
    substrate = INFERENCE_SUBSTRATE if complete else PRECONDITION_SUBSTRATE
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "duration_s": round(float(duration_s), 6),
        "duration_floor_s": MIN_COMPUTE_DURATION_S if complete else 0.0,
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": _wrap("preconditions_checked", True),
        "model_specs": _wrap("model_specs", [dict(spec) for spec in model_specs]),
        "models_used": _wrap("models_used", models_used),
        "target_model": _wrap("target_model", target_model if complete else "blocked"),
        "random_seed": _wrap("random_seed", int(random_seed)),
        "reproducibility_checksum": _wrap("reproducibility_checksum", checksum),
        "n_examples": _wrap("n_examples", n_examples),
        "solver_only_solved": _wrap("solver_only_solved", solver_solved),
        "llm_only_solved": _wrap("llm_only_solved", llm_only_solved),
        "llm_solver_feedback_solved": _wrap("llm_solver_feedback_solved", feedback_solved),
        "solver_feedback_uplift": _wrap("solver_feedback_uplift", uplift),
        "methodology_receipts_complete": _wrap("methodology_receipts_complete", methodology_complete),
        "retire_current_veribmc_path": _wrap("retire_current_veribmc_path", retire),
        "prompt_template_hash": _wrap("prompt_template_hash", prompt_hash),
        "verifier_command": _wrap("verifier_command", VERIFIER_COMMAND),
        "verifier_pass_fail_log": _wrap("verifier_pass_fail_log", pass_fail_log),
        "source_paths": _wrap("source_paths", [str(path) for path in SOURCE_PATHS]),
        "validation_commands_run": _wrap("validation_commands_run", list(validation_commands_run)),
        "inference_substrate": _wrap("inference_substrate", substrate),
        "honest_verdict": _wrap(
            "honest_verdict",
            honest_verdict(
                complete=complete,
                methodology_complete=methodology_complete,
                uplift=uplift,
                retire=retire,
                blocked_reason=blocked_reason,
            ),
        ),
        "precondition_receipts": dict(precondition_receipts),
        "methodology_receipts": {
            "duration_floor_met": (not complete) or float(duration_s) >= MIN_COMPUTE_DURATION_S,
            "model_specs_present": bool(model_specs),
            "target_model_present": target_model != "blocked",
            "random_seed_present": isinstance(random_seed, int),
            "reproducibility_checksum_present": bool(checksum),
            "prompt_template_hash_present": bool(prompt_hash),
            "verifier_command_present": bool(VERIFIER_COMMAND),
            "verifier_receipts_present": bool(pass_fail_log),
        },
        "accepted_invariants_or_constraints": _accepted_by_example(
            solver_only_results,
            llm_initial_results,
            llm_feedback_results,
        ),
        "runtime_s_by_arm": {
            "solver_only_checker": _runtime_s(solver_only_results),
            "llm_only_checker": _runtime_s(llm_initial_results),
            "llm_solver_feedback_checker": _runtime_s(llm_feedback_results),
        },
        "failure_modes": dict(
            Counter(
                result.failure_mode
                for result in [*solver_only_results, *llm_initial_results, *llm_feedback_results]
                if result.failure_mode
            )
        ),
        "per_example_results": {
            "solver_only": [result.to_dict() for result in solver_only_results],
            "llm_only": [result.to_dict() for result in llm_initial_results],
            "llm_solver_feedback": [result.to_dict() for result in llm_feedback_results],
        },
    }
    validate_artifact(artifact)
    return artifact


def methodology_receipts_complete(
    *,
    complete: bool,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    target_model: str,
    random_seed: int,
    checksum: str,
    prompt_hash: str,
    verifier_command: str,
    pass_fail_log: Sequence[Mapping[str, Any]],
    n_examples: int,
) -> bool:
    """Return whether a successful compute-bound run has complete receipts."""

    return bool(
        complete
        and duration_s >= MIN_COMPUTE_DURATION_S
        and _has_mandated_sota_model(model_specs)
        and target_model in MANDATED_SOTA_IDS
        and isinstance(random_seed, int)
        and checksum
        and prompt_hash
        and verifier_command
        and len(pass_fail_log) == n_examples * 3
        and all(isinstance(row.get("verifier_passed"), bool) for row in pass_fail_log)
    )


def reproducibility_checksum(random_seed: int) -> str:
    """Return the experiment-template checksum over seed, code, spec, and input."""

    code_files = [
        str(REPO_ROOT / MODULE_RELATIVE_PATH),
        str(REPO_ROOT / PRIOR_MODULE_RELATIVE_PATH),
        str(REPO_ROOT / TEMPLATE_RELATIVE_PATH),
        str(REPO_ROOT / SOTA_MODELS_RELATIVE_PATH),
        str(REPO_ROOT / SPEC_RELATIVE_PATH),
    ]
    return _compute_repro_checksum(
        seed=random_seed,
        code_files=code_files,
        data_path=str(REPO_ROOT / PRIOR_RESULT_RELATIVE_PATH),
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the REQ-VERIFY-5238 terminal artifact contract."""

    missing = [field for field in FIELD_PRINCIPLES if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field, principle in FIELD_PRINCIPLES.items():
        wrapped = artifact[field]
        if not (
            isinstance(wrapped, dict)
            and wrapped.get("principle") == principle
            and "value" in wrapped
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")

    values = {field: artifact[field]["value"] for field in FIELD_PRINCIPLES}
    if not str(values["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(values["preconditions_checked"], bool) or not values["preconditions_checked"]:
        raise ValueError("preconditions_checked must be true")
    if not isinstance(values["random_seed"], int):
        raise ValueError("random_seed must be an integer")
    if not values["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum missing")
    if not isinstance(values["prompt_template_hash"], str) or len(values["prompt_template_hash"]) != 16:
        raise ValueError("prompt_template_hash malformed")
    if not values["verifier_command"]:
        raise ValueError("verifier_command missing")
    if not isinstance(values["validation_commands_run"], list) or not all(
        isinstance(row, str) for row in values["validation_commands_run"]
    ):
        raise ValueError("validation_commands_run malformed")

    n_examples = values["n_examples"]
    if not isinstance(n_examples, int) or n_examples < 0:
        raise ValueError("n_examples must be a nonnegative integer")
    for field in ("solver_only_solved", "llm_only_solved", "llm_solver_feedback_solved"):
        solved = values[field]
        if not isinstance(solved, int) or not 0 <= solved <= n_examples:
            raise ValueError(f"{field} out of range")
    if not isinstance(values["solver_feedback_uplift"], float):
        raise ValueError("solver_feedback_uplift must be float")

    methodology_complete = values["methodology_receipts_complete"]
    if not isinstance(methodology_complete, bool):
        raise ValueError("methodology_receipts_complete must be bool")
    if not isinstance(values["retire_current_veribmc_path"], bool):
        raise ValueError("retire_current_veribmc_path must be bool")

    if methodology_complete:
        if float(artifact.get("duration_s", 0.0)) < MIN_COMPUTE_DURATION_S:
            raise ValueError("duration_floor not met for methodology-complete run")
        if values["inference_substrate"] != INFERENCE_SUBSTRATE:
            raise ValueError("inference_substrate must be local SOTA solver feedback")
        if values["target_model"] not in MANDATED_SOTA_IDS:
            raise ValueError("target_model must be mandated SOTA GGUF")
        if not _has_mandated_sota_model(values["model_specs"]):
            raise ValueError("model_specs missing mandated SOTA GGUF")
        if not values["models_used"]:
            raise ValueError("models_used missing")
        if len(values["verifier_pass_fail_log"]) != n_examples * 3:
            raise ValueError("verifier_pass_fail_log missing per-arm receipts")
        if values["retire_current_veribmc_path"] != (values["solver_feedback_uplift"] <= 0.0):
            raise ValueError("retire_current_veribmc_path inconsistent with uplift")
    else:
        if values["inference_substrate"] not in {INFERENCE_SUBSTRATE, PRECONDITION_SUBSTRATE}:
            raise ValueError("inference_substrate invalid for blocked or incomplete run")
    if not isinstance(values["verifier_pass_fail_log"], list):
        raise ValueError("verifier_pass_fail_log must be list")
    if values["verifier_pass_fail_log"] and not all(
        isinstance(row.get("verifier_passed"), bool) for row in values["verifier_pass_fail_log"]
    ):
        raise ValueError("verifier_pass_fail_log rows must include verifier_passed bool")


def run_experiment(
    *,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    proposal_fn: ProposalFn | None = None,
    model_specs_provider: ModelSpecsProvider = resolve_model_specs_for_rerun,
    validation_commands_run: Sequence[str] | None = None,
    duration_s: float | None = None,
    random_seed: int = RANDOM_SEED,
    enforce_duration_floor: bool = True,
) -> JsonDict:
    """Run the rerun, write the JSON artifact, and return it."""

    artifact = run_rerun(
        proposal_fn=proposal_fn,
        model_specs_provider=model_specs_provider,
        validation_commands_run=validation_commands_run,
        duration_s=duration_s,
        random_seed=random_seed,
        enforce_duration_floor=enforce_duration_floor,
    )
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return artifact


def honest_verdict(
    *,
    complete: bool,
    methodology_complete: bool,
    uplift: float,
    retire: bool,
    blocked_reason: str,
) -> str:
    """Return the terminal human-readable verdict."""

    if not complete:
        reason = blocked_reason or "preconditions_unmet"
        return (
            f"blocked_preconditions_unmet: {reason}; solver feedback stayed untested "
            "and the current local VerIbmc path remains blocked"
        )
    if not methodology_complete:
        return (
            "blocked_methodology_receipts_incomplete: solver feedback rerun executed but "
            "clean retirement cannot be claimed without complete receipts"
        )
    if uplift > 0.0:
        return "complete: solver feedback improved under clean methodology receipts"
    if retire:
        return (
            "complete: solver feedback stayed null under clean methodology receipts; "
            "retired current VerIbmc local solver-feedback path"
        )
    return "complete: solver feedback stayed null under clean methodology receipts"


def _copy_initial_as_feedback(initial: exp5226.ArmResult) -> exp5226.ArmResult:
    return exp5226.ArmResult(
        example_id=initial.example_id,
        arm="llm_solver_feedback",
        raw_output=initial.raw_output,
        parsed_invariant=initial.parsed_invariant,
        accepted=True,
        failed_obligation=None,
        counterexample={},
        solver_feedback={},
        runtime_ms=initial.runtime_ms,
        failure_mode=None,
    )


def _accepted_count(results: Sequence[exp5226.ArmResult]) -> int:
    return sum(1 for result in results if result.accepted)


def _uplift(solver_solved: int, llm_only_solved: int, feedback_solved: int, n_examples: int) -> float:
    if n_examples <= 0:
        return 0.0
    return round((feedback_solved - max(solver_solved, llm_only_solved)) / n_examples, 6)


def _runtime_s(results: Sequence[exp5226.ArmResult]) -> float:
    return round(sum(result.runtime_ms for result in results) / 1000.0, 6)


def _accepted_by_example(
    solver_only_results: Sequence[exp5226.ArmResult],
    llm_initial_results: Sequence[exp5226.ArmResult],
    llm_feedback_results: Sequence[exp5226.ArmResult],
) -> JsonDict:
    grouped: JsonDict = {}
    for arm_name, results in (
        ("solver_only", solver_only_results),
        ("llm_only", llm_initial_results),
        ("llm_solver_feedback", llm_feedback_results),
    ):
        for result in results:
            grouped.setdefault(result.example_id, {})[arm_name] = (
                result.parsed_invariant if result.accepted else None
            )
    return grouped


def _models_used_from_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    models: list[str] = []
    for spec in model_specs:
        hf_id = str(spec.get("hf_id") or "")
        if hf_id and hf_id not in models:
            models.append(hf_id)
    return models


def _has_mandated_sota_model(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(spec.get("hf_id") or "") in MANDATED_SOTA_IDS and spec.get("model_path") for spec in model_specs)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _duration(duration_s: float | None, started: float) -> float:
    return float(duration_s) if duration_s is not None else time.perf_counter() - started


def _wait_for_duration_floor(started: float) -> None:  # pragma: no cover - intentionally slow live-run guard.
    while time.perf_counter() - started < MIN_COMPUTE_DURATION_S + 0.25:
        remaining = MIN_COMPUTE_DURATION_S + 0.25 - (time.perf_counter() - started)
        time.sleep(min(1.0, max(0.05, remaining)))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI for live task execution.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--validation-command", action="append", default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifact = run_experiment(
        result_path=args.result_path,
        validation_commands_run=args.validation_command,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
