"""Exp 5262: solver-grounded constraint extraction pilot.

Spec refs: REQ-VERIFY-5262, SCENARIO-VERIFY-5262.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

try:  # pragma: no cover - absence is covered by dependency injection.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5262_solver_grounded_constraint_extraction_v481.json")
EXP5259_RELATIVE_PATH = Path("results/experiment_5259_sota_gguf_gpu_offload_preflight_v481.json")
EXP5238_GLOB = "experiment_5238_veribmc*.json"
SCHEMA = "carnot.experiment_5262.solver_grounded_constraint_extraction.v481"
SPEC_REFS = ("REQ-VERIFY-5262", "SCENARIO-VERIFY-5262")
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
CHECKER_SUBSTRATE = "z3"
TERMINAL_PREFIXES = ("complete:", "blocked_")
GGUF_OFFLOAD_CONFIG = {
    "n_gpu_layers": -1,
    "n_ctx": 1024,
    "max_tokens": 256,
    "temperature": 0.0,
    "seed": 5262,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5262 verdict; starts with complete: or blocked_ and states "
        "whether solver-grounded extraction produced useful oracle-distinct signal."
    ),
    "inference_substrate": (
        "Declares live local SOTA GGUF inference plus deterministic solver validation, "
        "not a broad solver-feedback or VerIbmc uplift result."
    ),
    "preconditions_checked": (
        "Records Exp 5259 readiness, model/runtime receipts, deterministic checker "
        "availability, and retired-scope exclusions before headline extraction."
    ),
    "MODEL_SPECS": (
        "Records mandated local SOTA GGUF model IDs, roles, quantization/file receipts, "
        "and runtime status used for the pilot."
    ),
    "solver_grounded_extractor_ready": (
        "Bare readiness boolean; true only when live GGUF extraction ran, deterministic "
        "validation ran, and the result beat the simple baseline without false accepts."
    ),
    "solver_grounded_extractor_ready_principle": (
        "Explains whether the solver-grounded extractor is ready and whether its signal "
        "is oracle-distinct from the baseline."
    ),
    "constraint_validity_rate": (
        "Fraction of fixture cases where generated executable constraints matched the "
        "deterministic solver label."
    ),
    "false_accepts": (
        "Count of expected-UNSAT fixtures whose generated constraints were satisfiable, "
        "exposing missed contradictions."
    ),
    "counterexamples_found": (
        "Count of deterministic counterexamples recorded for generated constraint sets "
        "that disagreed with fixture labels."
    ),
    "retired_veribmc_scope_reopened": (
        "Must remain false; Exp 5262 is constraint extraction plus deterministic checking, "
        "not the retired VerIbmc local solver-feedback route."
    ),
    "commands_run": "Commands run to create and validate the artifact, with outcomes.",
}

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "slot": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_candidate_generator",
        "quantization": "Q4_K_M",
    },
    {
        "slot": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_candidate_generator_or_cross_checker",
        "quantization": "Q4_K_M",
    },
    {
        "slot": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "optional_middle_moe_cross_checker",
        "quantization": "Q4_K_M",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "solver_grounded_extractor_ready",
    "solver_grounded_extractor_ready_principle",
    "constraint_validity_rate",
    "false_accepts",
    "counterexamples_found",
    "retired_veribmc_scope_reopened",
    "commands_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "constraint_validity_rate",
    "false_accepts",
    "counterexamples_found",
    "retired_veribmc_scope_reopened",
)


@dataclass(frozen=True)
class ConstraintFixture:
    """One natural-language requirement pack with a hidden solver label."""

    fixture_id: str
    prompt_requirements: str
    expected_status: str
    gold_assignment: JsonDict


@dataclass(frozen=True)
class ConstraintIR:
    """Executable integer-constraint IR emitted by the model."""

    variables: JsonDict
    constraints: tuple[str, ...]
    raw_json: JsonDict
    normalization_notes: tuple[str, ...]


@dataclass(frozen=True)
class SolverValidation:
    """Deterministic checker result for one generated constraint set."""

    fixture_id: str
    solver_status: str
    expected_status: str
    matches_expected: bool
    false_accept: bool
    assignment: JsonDict
    counterexample: JsonDict
    error: str | None

    def to_dict(self) -> JsonDict:
        return {
            "fixture_id": self.fixture_id,
            "solver_status": self.solver_status,
            "expected_status": self.expected_status,
            "matches_expected": self.matches_expected,
            "false_accept": self.false_accept,
            "assignment": self.assignment,
            "counterexample": self.counterexample,
            "error": self.error,
        }


ProposalFn = Callable[[ConstraintFixture], str]


def fixture_set() -> list[ConstraintFixture]:
    """Return the bounded SAT/UNSAT fixture pack used by all arms."""

    return [
        ConstraintFixture(
            fixture_id="single_even_high",
            prompt_requirements=(
                "Choose an integer x. x must be at least 0 and at most 5. "
                "x must be even. x must be greater than 3."
            ),
            expected_status="sat",
            gold_assignment={"x": 4},
        ),
        ConstraintFixture(
            fixture_id="small_pair_sum",
            prompt_requirements=(
                "Choose integers a and b. Each must be between 0 and 3 inclusive. "
                "Their sum must be 5. Also require a to be less than b."
            ),
            expected_status="sat",
            gold_assignment={"a": 2, "b": 3},
        ),
        ConstraintFixture(
            fixture_id="even_and_odd",
            prompt_requirements=(
                "Choose an integer y. y must be between 1 and 4 inclusive. "
                "y must be even. y must also be odd."
            ),
            expected_status="unsat",
            gold_assignment={},
        ),
        ConstraintFixture(
            fixture_id="too_large_sum",
            prompt_requirements=(
                "Choose integers p and q. Each must be between 0 and 2 inclusive. "
                "The sum p + q must equal 5."
            ),
            expected_status="unsat",
            gold_assignment={},
        ),
    ]


def parse_constraint_ir(text: str) -> ConstraintIR | None:
    """Parse model-emitted JSON constraint IR without adding missing rules."""

    obj_text, extracted = _extract_json_object(text)
    if obj_text is None:
        return None
    try:
        payload = json.loads(obj_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):  # pragma: no cover - extractor only returns JSON objects.
        return None

    constraints_raw = payload.get("constraints")
    if not isinstance(constraints_raw, list) or not all(isinstance(row, str) for row in constraints_raw):
        return None
    variables = _normalize_variables(payload.get("variables"))
    if variables is None:
        return None
    notes = ("json_object_extracted",) if extracted else ()
    return ConstraintIR(
        variables=variables,
        constraints=tuple(row.strip() for row in constraints_raw if row.strip()),
        raw_json=dict(payload),
        normalization_notes=notes,
    )


def validate_fixture_constraints(
    fixture: ConstraintFixture,
    candidate: ConstraintIR,
    *,
    z3_module: Any = _z3,
) -> SolverValidation:
    """Run deterministic satisfiability validation for one model candidate."""

    if not _checker_available(z3_module):
        return _validation_from_status(fixture, "parse_error", {}, "z3_unavailable")

    variable_names = sorted(_variable_names(candidate))
    env = {name: z3_module.Int(name) for name in variable_names}
    solver = z3_module.Solver()
    solver.set(timeout=2000)
    try:
        for formula in _domain_constraints(candidate):
            solver.add(_compile_formula(formula, env, z3_module))
        for formula in candidate.constraints:
            solver.add(_compile_formula(formula, env, z3_module))
    except Exception as exc:
        return _validation_from_status(fixture, "parse_error", {}, f"{type(exc).__name__}: {exc}")

    status = solver.check()
    if status == z3_module.sat:
        assignment = _model_assignment(solver.model(), env)
        return _validation_from_status(fixture, "sat", assignment, None)
    if status == z3_module.unsat:
        return _validation_from_status(fixture, "unsat", {}, None)
    return _validation_from_status(fixture, "unknown", {}, str(status))


def render_prompt(fixture: ConstraintFixture) -> str:
    """Render the model request without disclosing the hidden solver label."""

    return (
        "Translate the natural-language requirements into executable integer constraints.\n"
        "Return only one JSON object with keys variables and constraints.\n"
        "variables must map each variable name to {\"type\":\"int\"}.\n"
        "constraints must be a list of Python/Z3-style integer formulas using ==, <, <=, >, >=, %, +, -, *.\n"
        "Do not state whether the requirements are satisfiable.\n\n"
        f"Requirements:\n{fixture.prompt_requirements}\n"
    )


def no_constraint_baseline(fixtures: Sequence[ConstraintFixture]) -> list[JsonDict]:
    """Score the simple baseline that emits no constraints for every fixture."""

    rows: list[JsonDict] = []
    empty = ConstraintIR(variables={}, constraints=(), raw_json={}, normalization_notes=())
    for fixture in fixtures:
        validation = validate_fixture_constraints(fixture, empty)
        rows.append(_row_to_artifact(fixture, "", None, validation, arm="baseline_no_constraints"))
    return rows


def run_pilot(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preflight_artifact: Mapping[str, Any] | None = None,
    proposal_fn: ProposalFn | None = None,
    commands_run: Sequence[Mapping[str, Any]] = (),
    z3_module: Any = _z3,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Gate preconditions and then run or block the Exp 5262 pilot."""

    active_preflight = dict(preflight_artifact or load_preflight_artifact(root))
    checker_available = _checker_available(z3_module)
    if not active_preflight.get("sota_runtime_ready") or not checker_available:
        artifact = _blocked_artifact(
            root=root,
            result_path=result_path,
            preflight_artifact=active_preflight,
            checker_available=checker_available,
            commands_run=commands_run,
        )
        validate_artifact(artifact)
        _write_json(result_path, artifact)
        return artifact

    return run_experiment(
        result_path=result_path,
        preflight_artifact=active_preflight,
        proposal_fn=proposal_fn,
        commands_run=commands_run,
        z3_module=z3_module,
        root=root,
    )


def run_experiment(
    *,
    result_path: Path,
    preflight_artifact: Mapping[str, Any],
    proposal_fn: ProposalFn | None,
    commands_run: Sequence[Mapping[str, Any]],
    z3_module: Any = _z3,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Collect model constraints, validate them, compare baseline, and write JSON."""

    started = time.perf_counter()
    active_fixtures = fixture_set()
    proposer = proposal_fn or live_llama_cpp_proposal_fn(preflight_artifact)
    extraction_rows = [
        _evaluate_model_row(fixture, proposer, z3_module=z3_module) for fixture in active_fixtures
    ]
    baseline_rows = no_constraint_baseline(active_fixtures)
    aggregate = _aggregate(extraction_rows)
    baseline = _aggregate(baseline_rows)
    useful_signal = (
        aggregate["validity_rate"] > baseline["validity_rate"] and aggregate["false_accepts"] == 0
    )
    ready = useful_signal and any(row["parseable"] for row in extraction_rows)
    artifact = _build_artifact(
        root=root,
        preflight_artifact=preflight_artifact,
        extraction_rows=extraction_rows,
        baseline_rows=baseline_rows,
        aggregate=aggregate,
        baseline=baseline,
        ready=ready,
        useful_signal=useful_signal,
        commands_run=commands_run,
        duration_s=time.perf_counter() - started,
    )
    validate_artifact(artifact)
    _write_json(result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5262 artifact violates the required schema."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field {field}"
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        assert isinstance(value, Mapping), f"{field} must be principle-wrapped"
        assert "value" in value and "principle" in value, f"{field} must be principle-wrapped"
        assert value["principle"] == FIELD_PRINCIPLES[field], f"{field} principle mismatch"

    verdict = artifact["honest_verdict"]["value"]
    assert isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), (
        "honest_verdict.value must start with complete: or blocked_"
    )
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, (
        "inference_substrate.value must be live_llm_inference_local_gguf_sota"
    )
    assert isinstance(artifact["solver_grounded_extractor_ready"], bool), (
        "solver_grounded_extractor_ready must be a bare bool"
    )
    assert artifact["solver_grounded_extractor_ready_principle"], (
        "solver_grounded_extractor_ready_principle must be non-empty"
    )
    assert artifact["retired_veribmc_scope_reopened"]["value"] is False, (
        "retired_veribmc_scope_reopened must remain false"
    )
    rate = artifact["constraint_validity_rate"]["value"]
    assert isinstance(rate, int | float) and 0.0 <= float(rate) <= 1.0, (
        "constraint_validity_rate.value must be numeric in [0, 1]"
    )
    assert isinstance(artifact["false_accepts"]["value"], int), "false_accepts.value must be int"
    assert isinstance(artifact["counterexamples_found"]["value"], int), (
        "counterexamples_found.value must be int"
    )
    assert isinstance(artifact["commands_run"], list), "commands_run must be a list"
    assert isinstance(artifact["MODEL_SPECS"]["value"], Mapping), "MODEL_SPECS.value must be object"


def load_preflight_artifact(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    """Load Exp 5259 readiness receipts from the checked-in result artifact."""

    path = root / EXP5259_RELATIVE_PATH
    if not path.exists():
        return {"sota_runtime_ready": False, "blocker": f"missing {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def live_llama_cpp_proposal_fn(preflight_artifact: Mapping[str, Any]) -> ProposalFn:  # pragma: no cover
    """Build the live local GGUF proposer used outside unit tests."""

    model_specs = _model_specs_from_preflight(preflight_artifact)
    flagship = model_specs["flagship_moe"]
    model_path = flagship["file_receipts"].get("path")
    if not model_path:
        raise RuntimeError("flagship_moe model_path unavailable")

    from llama_cpp import Llama  # noqa: PLC0415

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=int(GGUF_OFFLOAD_CONFIG["n_gpu_layers"]),
        n_ctx=int(GGUF_OFFLOAD_CONFIG["n_ctx"]),
        seed=int(GGUF_OFFLOAD_CONFIG["seed"]),
        verbose=False,
    )

    def propose(fixture: ConstraintFixture) -> str:
        response = llm(
            render_prompt(fixture),
            max_tokens=int(GGUF_OFFLOAD_CONFIG["max_tokens"]),
            temperature=float(GGUF_OFFLOAD_CONFIG["temperature"]),
        )
        if isinstance(response, dict) and response.get("choices"):
            return str(response["choices"][0].get("text", ""))
        return str(response)

    return propose


def _evaluate_model_row(
    fixture: ConstraintFixture,
    proposal_fn: ProposalFn,
    *,
    z3_module: Any,
) -> JsonDict:
    raw_output = proposal_fn(fixture)
    candidate = parse_constraint_ir(raw_output)
    if candidate is None:
        validation = _validation_from_status(
            fixture,
            "parse_error",
            {},
            "no_parseable_constraint_ir",
        )
        return _row_to_artifact(fixture, raw_output, None, validation, arm="live_llm_constraints")
    validation = validate_fixture_constraints(fixture, candidate, z3_module=z3_module)
    return _row_to_artifact(fixture, raw_output, candidate, validation, arm="live_llm_constraints")


def _row_to_artifact(
    fixture: ConstraintFixture,
    raw_output: str,
    candidate: ConstraintIR | None,
    validation: SolverValidation,
    *,
    arm: str,
) -> JsonDict:
    return {
        "fixture_id": fixture.fixture_id,
        "arm": arm,
        "prompt_requirements": fixture.prompt_requirements,
        "expected_status": fixture.expected_status,
        "raw_output": raw_output,
        "parseable": candidate is not None,
        "generated_ir": None
        if candidate is None
        else {
            "variables": candidate.variables,
            "constraints": list(candidate.constraints),
            "normalization_notes": list(candidate.normalization_notes),
        },
        "validation": validation.to_dict(),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    if total == 0:
        return {"validity_rate": 0.0, "false_accepts": 0, "counterexamples_found": 0}
    validations = [row["validation"] for row in rows]
    correct = sum(1 for validation in validations if validation["matches_expected"])
    return {
        "validity_rate": round(correct / total, 6),
        "false_accepts": sum(1 for validation in validations if validation["false_accept"]),
        "counterexamples_found": sum(1 for validation in validations if validation["counterexample"]),
    }


def _build_artifact(
    *,
    root: Path,
    preflight_artifact: Mapping[str, Any],
    extraction_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    ready: bool,
    useful_signal: bool,
    commands_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    verdict = _honest_verdict(ready, useful_signal, aggregate, baseline)
    return {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _preconditions(root, preflight_artifact, checker_available=True),
        "MODEL_SPECS": _wrap("MODEL_SPECS", _model_specs_from_preflight(preflight_artifact)),
        "solver_grounded_extractor_ready": bool(ready),
        "solver_grounded_extractor_ready_principle": _ready_principle(ready, useful_signal, aggregate, baseline),
        "constraint_validity_rate": _wrap("constraint_validity_rate", aggregate["validity_rate"]),
        "false_accepts": _wrap("false_accepts", int(aggregate["false_accepts"])),
        "counterexamples_found": _wrap("counterexamples_found", int(aggregate["counterexamples_found"])),
        "retired_veribmc_scope_reopened": _wrap("retired_veribmc_scope_reopened", False),
        "commands_run": [dict(command) for command in commands_run],
        "fixtures": [fixture.__dict__ for fixture in fixture_set()],
        "extraction_results": [dict(row) for row in extraction_rows],
        "baseline": dict(baseline) | {"arm": "baseline_no_constraints", "rows": list(baseline_rows)},
        "no_broad_solver_feedback_claim": True,
        "reproducibility_checksum": hashlib.sha256(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "spec_refs": SPEC_REFS,
                    "aggregate": aggregate,
                    "baseline": baseline,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16],
    }


def _blocked_artifact(
    *,
    root: Path,
    result_path: Path,
    preflight_artifact: Mapping[str, Any],
    checker_available: bool,
    commands_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    del result_path
    preconditions = _preconditions(root, preflight_artifact, checker_available=checker_available)
    blocker = []
    if not preflight_artifact.get("sota_runtime_ready"):
        blocker.append("exp5259_sota_runtime_ready_not_true")
    if not checker_available:
        blocker.append("deterministic_checker_unavailable")
    aggregate = {"validity_rate": 0.0, "false_accepts": 0, "counterexamples_found": 0}
    baseline = {"validity_rate": 0.0, "false_accepts": 0, "counterexamples_found": 0}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": 0.0,
        "honest_verdict": _wrap(
            "honest_verdict",
            "blocked_preconditions: " + ",".join(blocker or ["unknown"]),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": preconditions,
        "MODEL_SPECS": _wrap("MODEL_SPECS", _model_specs_from_preflight(preflight_artifact)),
        "solver_grounded_extractor_ready": False,
        "solver_grounded_extractor_ready_principle": _ready_principle(False, False, aggregate, baseline),
        "constraint_validity_rate": _wrap("constraint_validity_rate", 0.0),
        "false_accepts": _wrap("false_accepts", 0),
        "counterexamples_found": _wrap("counterexamples_found", 0),
        "retired_veribmc_scope_reopened": _wrap("retired_veribmc_scope_reopened", False),
        "commands_run": [dict(command) for command in commands_run],
        "blockers": blocker,
        "baseline": baseline | {"arm": "not_run_precondition_blocked", "rows": []},
        "extraction_results": [],
        "no_broad_solver_feedback_claim": True,
    }
    return artifact


def _preconditions(
    root: Path,
    preflight_artifact: Mapping[str, Any],
    *,
    checker_available: bool,
) -> JsonDict:
    value = {
        "exp5259_artifact_path": str(root / EXP5259_RELATIVE_PATH),
        "exp5259_sota_runtime_ready": bool(preflight_artifact.get("sota_runtime_ready")),
        "exp5259_sota_runtime_ready_principle": preflight_artifact.get("sota_runtime_ready_principle"),
        "deterministic_checker": CHECKER_SUBSTRATE,
        "deterministic_checker_available": checker_available,
        "model_runtime_receipts": _model_runtime_receipt_summary(preflight_artifact),
        "prior_veribmc_retirement_receipt": _prior_veribmc_receipt(root),
        "exclusion_manifest_path": str(root / "ops/exclusion_manifest.yaml"),
        "retired_veribmc_scope_reopened": False,
    }
    return _wrap("preconditions_checked", value)


def _model_specs_from_preflight(preflight_artifact: Mapping[str, Any]) -> JsonDict:
    receipts = _nested_value(preflight_artifact, "model_receipts")
    if not isinstance(receipts, Mapping):
        receipts = {}
    specs: JsonDict = {}
    for mandated in MANDATED_MODEL_SPECS:
        slot = str(mandated["slot"])
        receipt = receipts.get(slot, {})
        if not isinstance(receipt, Mapping):
            receipt = {}
        specs[slot] = {
            "hf_id": str(mandated["hf_id"]),
            "role": str(mandated["role"]),
            "quantization": receipt.get("preferred_quant") or mandated["quantization"],
            "runtime_status": receipt.get("status", "missing_receipt"),
            "runtime_ready": bool(receipt.get("runtime_ready")),
            "file_receipts": {
                "path": receipt.get("path"),
                "size_bytes": receipt.get("size_bytes"),
                "checksum_sha256": receipt.get("checksum_sha256"),
                "checksum_head_1m_sha256": receipt.get("checksum_head_1m_sha256"),
            },
        }
    return specs


def _model_runtime_receipt_summary(preflight_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        slot: {
            "hf_id": spec["hf_id"],
            "status": spec["runtime_status"],
            "runtime_ready": spec["runtime_ready"],
            "path": spec["file_receipts"]["path"],
        }
        for slot, spec in _model_specs_from_preflight(preflight_artifact).items()
    }


def _prior_veribmc_receipt(root: Path) -> JsonDict:
    matches = sorted((root / "results").glob(EXP5238_GLOB))
    if not matches:
        return {"found": False, "retired": True, "note": "no_exp5238_artifact_found"}
    latest = matches[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"found": True, "path": str(latest), "retired": True, "error": f"{type(exc).__name__}: {exc}"}
    verdict = _nested_value(payload, "honest_verdict")
    return {
        "found": True,
        "path": str(latest),
        "retired": "retired" in str(verdict).lower(),
        "honest_verdict": verdict,
    }


def _honest_verdict(
    ready: bool,
    useful_signal: bool,
    aggregate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    if ready:
        return (
            "complete: solver-grounded extraction produced useful oracle-distinct signal "
            f"(validity={aggregate['validity_rate']}, baseline={baseline['validity_rate']}, "
            f"false_accepts={aggregate['false_accepts']})"
        )
    if useful_signal:
        return (
            "complete: solver-grounded extraction improved over baseline but is not ready "
            f"because false_accepts={aggregate['false_accepts']}"
        )
    return (
        "complete: solver-grounded extraction produced no useful oracle-distinct signal "
        f"(validity={aggregate['validity_rate']}, baseline={baseline['validity_rate']}, "
        f"false_accepts={aggregate['false_accepts']})"
    )


def _ready_principle(
    ready: bool,
    useful_signal: bool,
    aggregate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    if ready:
        return (
            "ready=true because live GGUF constraints were parseable, Z3 validation ran, "
            "validity beat the no-constraint baseline, and false_accepts=0."
        )
    return (
        "ready=false because useful_signal=%s, validity=%s, baseline_validity=%s, false_accepts=%s."
        % (
            useful_signal,
            aggregate["validity_rate"],
            baseline["validity_rate"],
            aggregate["false_accepts"],
        )
    )


def _validation_from_status(
    fixture: ConstraintFixture,
    solver_status: str,
    assignment: JsonDict,
    error: str | None,
) -> SolverValidation:
    matches = solver_status == fixture.expected_status
    false_accept = fixture.expected_status == "unsat" and solver_status == "sat"
    counterexample: JsonDict = {}
    if not matches:
        if false_accept:
            counterexample = dict(assignment)
        elif fixture.expected_status == "sat":
            counterexample = dict(fixture.gold_assignment)
        else:
            counterexample = {
                "expected_status": fixture.expected_status,
                "generated_solver_status": solver_status,
            }
    return SolverValidation(
        fixture_id=fixture.fixture_id,
        solver_status=solver_status,
        expected_status=fixture.expected_status,
        matches_expected=matches,
        false_accept=false_accept,
        assignment=assignment,
        counterexample=counterexample,
        error=error,
    )


def _compile_formula(expression: str, env: Mapping[str, Any], z3_module: Any) -> Any:
    tree = ast.parse(expression.strip(), mode="eval")
    return _compile_ast(tree.body, env, z3_module)


def _compile_ast(node: ast.AST, env: Mapping[str, Any], z3_module: Any) -> Any:
    if isinstance(node, ast.BoolOp):
        values = [_compile_ast(value, env, z3_module) for value in node.values]
        if isinstance(node.op, ast.And):
            return z3_module.And(*values)
        if isinstance(node.op, ast.Or):
            return z3_module.Or(*values)
        raise ValueError("unsupported boolean operator")
    if isinstance(node, ast.Compare):
        left = _compile_ast(node.left, env, z3_module)
        pieces = []
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _compile_ast(right_node, env, z3_module)
            pieces.append(_compare(left, op, right))
            left = right
        return z3_module.And(*pieces) if len(pieces) > 1 else pieces[0]
    if isinstance(node, ast.BinOp):
        left = _compile_ast(node.left, env, z3_module)
        right = _compile_ast(node.right, env, z3_module)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError("unsupported arithmetic operator")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_compile_ast(node.operand, env, z3_module)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id == "true":
            return z3_module.BoolVal(True)
        if node.id == "false":
            return z3_module.BoolVal(False)
        raise ValueError(f"unknown variable {node.id}")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return z3_module.BoolVal(node.value)
        if isinstance(node.value, int):
            return z3_module.IntVal(node.value)
    raise ValueError("unsupported expression")


def _compare(left: Any, op: ast.cmpop, right: Any) -> Any:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    raise ValueError("unsupported comparison operator")


def _domain_constraints(candidate: ConstraintIR) -> list[str]:
    formulas: list[str] = []
    for name, spec in candidate.variables.items():
        if isinstance(spec, Mapping):
            if isinstance(spec.get("min"), int):
                formulas.append(f"{name} >= {spec['min']}")
            if isinstance(spec.get("max"), int):
                formulas.append(f"{name} <= {spec['max']}")
    return formulas


def _variable_names(candidate: ConstraintIR) -> set[str]:
    names = set(candidate.variables)
    for formula in list(candidate.constraints) + _domain_constraints(candidate):
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError:
            continue
        names.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
    return {name for name in names if name not in {"true", "false"}}


def _model_assignment(model: Any, env: Mapping[str, Any]) -> JsonDict:
    assignment: JsonDict = {}
    for name, var in env.items():
        value = model.evaluate(var, model_completion=True)
        try:
            assignment[name] = int(value.as_long())
        except Exception:
            assignment[name] = str(value)
    return assignment


def _extract_json_object(text: str) -> tuple[str | None, bool]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, False
    fence_start = stripped.find("```")
    if fence_start >= 0:
        after = stripped.find("\n", fence_start)
        fence_end = stripped.find("```", after + 1 if after >= 0 else fence_start + 3)
        if after >= 0 and fence_end > after:
            fenced = stripped[after + 1 : fence_end].strip()
            if fenced.startswith("{") and fenced.endswith("}"):
                return fenced, True
    start = stripped.find("{")
    if start < 0:
        return None, False
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1], True
    return None, False


def _normalize_variables(value: Any) -> JsonDict | None:
    if value is None:
        return {}
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return {item: {"type": "int"} for item in value}
    if isinstance(value, dict):
        variables: JsonDict = {}
        for name, spec in value.items():
            if not isinstance(name, str):
                return None
            variables[name] = dict(spec) if isinstance(spec, dict) else {"type": "int"}
        return variables
    return None


def _checker_available(z3_module: Any) -> bool:
    return z3_module is not None and hasattr(z3_module, "Solver") and hasattr(z3_module, "Int")


def _nested_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--preflight", default=str(REPO_ROOT / EXP5259_RELATIVE_PATH))
    args = parser.parse_args(argv)
    preflight_path = Path(args.preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    artifact = run_pilot(
        result_path=Path(args.output),
        preflight_artifact=preflight,
        commands_run=[
            {
                "command": ".venv/bin/python -m carnot.experiment_5262_solver_grounded_constraint_extraction_v481 --output results/experiment_5262_solver_grounded_constraint_extraction_v481.json",
                "outcome": "completed module invocation",
            }
        ],
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
