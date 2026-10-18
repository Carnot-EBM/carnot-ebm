"""Build the Exp 3289 repair-gate decision v9 artifact.

Spec refs: REQ-VERIFY-3289, SCENARIO-VERIFY-3289.

This module only aggregates upstream evidence. It does not run repair or
reinterpret KAN as a detector. The point is to make the downstream exp3290
decision mechanical: open only when Garak evidence exists, the clean verifier is
non-degenerate with zero false accepts, and KAN is explicitly bounded away from
repair authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.repair_gate_decision.v9"
EXPERIMENT_ID = "exp3289"
TASK_ID = "exp3289-repair-gate-decision-v9-after-garak-abstention"
ARTIFACT = "experiment_3289_repair_gate_decision_v9_after_garak_abstention"
MILESTONE = "2026.05.304"
RUN_DATE = "20260528"
RANDOM_SEED = 3289
INFERENCE_SUBSTRATE = "artifact_aggregation_only"

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json")
EXP3285_REL_PATH = Path("results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json")
EXP3287_REL_PATH = Path("results/experiment_3287_abstention_calibrated_clean_verifier_v15.json")
EXP3288_REL_PATH = Path("results/experiment_3288_kan_sidecar_failure_autopsy_boundary_v1.json")
EXP3276_REL_PATH = Path("results/experiment_3276_repair_gate_decision_v8_after_v4_garak_clean_verifier.json")

SUCCESS_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
BOUNDED_KAN_USES = {
    "offline_failure_autopsy",
    "negative_control_regression_fixture",
    "future_kan_work_prerequisite_evidence_only",
}
FORBIDDEN_KAN_USES = {
    "repair_gate_authority",
    "prompt_injection_headline_detector",
    "standalone_garak_success_evidence",
    "production_triage_without_new_calibrated_false_positive_gate",
}
REQUIRED_FIELDS = {
    "repair_gate_decision_v9_ready",
    "repair_gate_open",
    "garak_redteam_eval_ready",
    "clean_verifier_rerun_ready",
    "repair_gate_input_clean_enough",
    "kan_boundary_decision_ready",
    "gate_inputs",
    "blocked_reasons",
    "permitted_repair_scope",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3289_repair_gate_decision_v9.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/repair_gate_decision_v9.py -m pytest -o addopts='' tests/python/test_experiment_3289_repair_gate_decision_v9.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/repair_gate_decision_v9.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class JsonLoad:
    """Parsed JSON source plus enough read metadata to explain closed gates."""

    payload: JsonDict
    present: bool
    readable: bool
    error: str | None
    path: Path
    sha256: str | None


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3289: decide whether exp3290 may run a bounded repair panel."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    inputs = gate_inputs(sources)
    blocked = blocked_reasons(sources, inputs)
    gate_open = not blocked
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3289", "SCENARIO-VERIFY-3289"],
        "repair_gate_decision_v9_ready": True,
        "repair_gate_open": gate_open,
        "garak_redteam_eval_ready": inputs["exp3285_garak"]["garak_redteam_eval_ready"],
        "clean_verifier_rerun_ready": inputs["exp3287_clean_verifier"]["clean_verifier_rerun_ready"],
        "repair_gate_input_clean_enough": inputs["exp3287_clean_verifier"]["repair_gate_input_clean_enough"],
        "kan_boundary_decision_ready": inputs["exp3288_kan_boundary"]["kan_boundary_decision_ready"],
        "gate_inputs": inputs,
        "blocked_reasons": blocked,
        "permitted_repair_scope": permitted_repair_scope(gate_open, inputs),
        "source_artifacts": source_artifacts(sources),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "random_seed": int(random_seed),
        "duration_s": duration(started, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3289 decision artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        random_seed=random_seed,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_sources(root: Path) -> dict[str, JsonLoad]:
    """Load all upstream gate inputs, including the prior gate as optional context."""

    return {
        "exp3285": read_json_object(root / EXP3285_REL_PATH),
        "exp3287": read_json_object(root / EXP3287_REL_PATH),
        "exp3288": read_json_object(root / EXP3288_REL_PATH),
        "exp3276": read_json_object(root / EXP3276_REL_PATH),
    }


def read_json_object(path: Path) -> JsonLoad:
    """Read a JSON object and preserve missing or malformed evidence as data."""

    if not path.is_file():
        return JsonLoad({}, False, False, "missing", path, None)
    digest = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return JsonLoad({}, True, False, str(exc), path, digest)
    if not isinstance(payload, Mapping):
        return JsonLoad({}, True, False, "json root is not an object", path, digest)
    return JsonLoad(dict(payload), True, True, None, path, digest)


def gate_inputs(sources: Mapping[str, JsonLoad]) -> JsonDict:
    """Extract the exact upstream fields that drive the v9 gate."""

    exp3285 = sources["exp3285"]
    exp3287 = sources["exp3287"]
    exp3288 = sources["exp3288"]
    exp3276 = sources["exp3276"]
    garak = exp3285.payload
    clean = exp3287.payload
    kan = exp3288.payload
    prior = exp3276.payload
    return {
        "exp3285_garak": source_prefix(exp3285)
        | {
            "garak_redteam_eval_ready": bool_field(garak, "garak_redteam_eval_ready"),
            "garak_dataflip_redteam_eval_v2_ready": bool_field(
                garak,
                "garak_dataflip_redteam_eval_v2_ready",
            ),
            "garak_gate_passed": bool_field(garak, "garak_gate_passed"),
            "dataflip_gate_passed": bool_field(garak, "dataflip_gate_passed"),
            "garak_probe_count": int(garak.get("garak_probe_count") or 0),
            "attack_success_rate": rate_field(garak, "attack_success_rate", default=0.0),
            "blocked_reasons": list_field(garak, "blocked_reasons"),
            "model_ids_used": model_ids(garak.get("models_used")),
        },
        "exp3287_clean_verifier": source_prefix(exp3287)
        | {
            "clean_verifier_rerun_ready": bool_field(clean, "clean_verifier_rerun_ready"),
            "repair_gate_input_clean_enough": bool_field(
                clean,
                "repair_gate_input_clean_enough",
            ),
            "false_accept_rate": rate_field(clean, "false_accept_rate", default=1.0),
            "false_reject_rate": rate_field(clean, "false_reject_rate", default=1.0),
            "abstention_rate": rate_field(clean, "abstention_rate", default=1.0),
            "coverage_rate": rate_field(clean, "coverage_rate", default=0.0),
            "n_eval": int(clean.get("n_eval") or 0),
            "exact_checkable_row_count": int(clean.get("exact_checkable_row_count") or 0),
            "selected_model_ids": model_ids(clean.get("models_used")),
            "missing_model_ids": model_ids(clean.get("missing_model_specs")),
            "gate_reasons": list_field(clean, "gate_reasons"),
        },
        "exp3288_kan_boundary": source_prefix(exp3288)
        | {
            "kan_boundary_decision_ready": bool_field(kan, "kan_boundary_decision_ready"),
            "kan_failure_autopsy_ready": bool_field(kan, "kan_failure_autopsy_ready"),
            "kan_boundary_decision": str(kan.get("kan_boundary_decision") or ""),
            "permitted_downstream_use": list_field(kan, "permitted_downstream_use"),
            "prohibited_downstream_use": list_field(kan, "prohibited_downstream_use"),
            "prior_full_corpus_auroc": kan.get("prior_full_corpus_auroc"),
            "prior_delong_noninferiority_passed": bool_field(
                kan,
                "prior_delong_noninferiority_passed",
            ),
            "kan_downstream_use_bounded": kan_downstream_use_bounded(kan),
        },
        "exp3276_prior_gate": source_prefix(exp3276)
        | {
            "status": str(prior.get("status") or ""),
            "schema": str(prior.get("schema") or prior.get("schema_version") or ""),
            "gate_check_summary": str(prior.get("gate_check_summary") or ""),
            "blocked_at_layer": str(prior.get("blocked_at_layer") or ""),
        },
    }


def source_prefix(load: JsonLoad) -> JsonDict:
    """Return common source metadata for a gate input row."""

    return {
        "path": repo_relative_path(load.path),
        "present": load.present,
        "readable": load.readable,
        "error": load.error,
        "sha256": load.sha256,
        "schema_version": load.payload.get("schema_version") or load.payload.get("schema"),
        "experiment_id": load.payload.get("experiment_id") or load.payload.get("experiment"),
        "honest_verdict": load.payload.get("honest_verdict"),
    }


def blocked_reasons(sources: Mapping[str, JsonLoad], inputs: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Collect precise reasons that keep repair generation closed."""

    reasons: list[JsonDict] = []
    reasons.extend(source_blockers(sources))
    garak = inputs["exp3285_garak"]
    clean = inputs["exp3287_clean_verifier"]
    kan = inputs["exp3288_kan_boundary"]
    if garak["garak_redteam_eval_ready"] is not True:
        reasons.append(blocker("garak_redteam_eval_not_ready", EXP3285_REL_PATH, "garak_redteam_eval_ready", True, garak["garak_redteam_eval_ready"]))
    if clean["clean_verifier_rerun_ready"] is not True:
        reasons.append(blocker("clean_verifier_rerun_not_ready", EXP3287_REL_PATH, "clean_verifier_rerun_ready", True, clean["clean_verifier_rerun_ready"]))
    if clean["repair_gate_input_clean_enough"] is not True:
        reasons.append(blocker("repair_gate_input_not_clean_enough", EXP3287_REL_PATH, "repair_gate_input_clean_enough", True, clean["repair_gate_input_clean_enough"]))
    if clean["false_accept_rate"] > 0.0:
        reasons.append(blocker("clean_verifier_false_accept_relaxation", EXP3287_REL_PATH, "false_accept_rate", 0.0, clean["false_accept_rate"]))
    if clean["abstention_rate"] >= 1.0:
        reasons.append(blocker("clean_verifier_abstain_all", EXP3287_REL_PATH, "abstention_rate", "< 1.0", clean["abstention_rate"]))
    if clean["coverage_rate"] <= 0.0:
        reasons.append(blocker("clean_verifier_no_coverage", EXP3287_REL_PATH, "coverage_rate", "> 0.0", clean["coverage_rate"]))
    if kan["kan_boundary_decision_ready"] is not True:
        reasons.append(blocker("kan_boundary_decision_not_ready", EXP3288_REL_PATH, "kan_boundary_decision_ready", True, kan["kan_boundary_decision_ready"]))
    if kan["kan_downstream_use_bounded"] is not True:
        reasons.append(blocker("kan_downstream_use_unbounded", EXP3288_REL_PATH, "permitted_downstream_use", sorted(BOUNDED_KAN_USES), kan["permitted_downstream_use"]))
    return reasons


def source_blockers(sources: Mapping[str, JsonLoad]) -> list[JsonDict]:
    """Mandatory inputs must exist and parse before a repair gate can open."""

    reasons: list[JsonDict] = []
    for key, path in (
        ("exp3285", EXP3285_REL_PATH),
        ("exp3287", EXP3287_REL_PATH),
        ("exp3288", EXP3288_REL_PATH),
    ):
        load = sources[key]
        if not load.present:
            reasons.append(blocker("missing_artifact", path, "present", True, False, detail=load.error))
        elif not load.readable:
            reasons.append(blocker("malformed_artifact", path, "readable_json_object", True, False, detail=load.error))
    return reasons


def blocker(
    code: str,
    source_artifact: Path,
    field: str,
    expected: Any,
    actual: Any,
    *,
    detail: Any | None = None,
) -> JsonDict:
    """Create a stable blocked-reason row for downstream conductor checks."""

    row: JsonDict = {
        "code": code,
        "source_artifact": source_artifact.as_posix(),
        "field": field,
        "expected": expected,
        "actual": actual,
    }
    if detail is not None:
        row["detail"] = detail
    return row


def permitted_repair_scope(gate_open: bool, inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Bound the exp3290 repair panel when the gate opens."""

    clean = inputs["exp3287_clean_verifier"]
    kan = inputs["exp3288_kan_boundary"]
    if not gate_open:
        return {
            "repair_task_id": "exp3290-gated-sota-repair-micro-panel-v10",
            "repair_generation_allowed": False,
            "reason": "repair_gate_closed",
            "sample_size": {"min_cases": 0, "max_cases": 0},
        }
    selected = clean["selected_model_ids"] or [MANDATED_MODEL_IDS[-1]]
    return {
        "repair_task_id": "exp3290-gated-sota-repair-micro-panel-v10",
        "repair_generation_allowed": True,
        "scope_label": "bounded_exact_fixture_code_repair_micro_panel",
        "sample_size": {"min_cases": 4, "max_cases": 8},
        "max_panel_cases": 8,
        "permitted_case_families": [
            "exact_context_fixture_counterexamples",
            "localized_code_or_json_fragment_failures",
            "deterministic_integer_constraint_failures",
        ],
        "model_specs": {
            "mandated_model_ids": list(MANDATED_MODEL_IDS),
            "selected_model_ids": selected,
            "missing_model_ids": clean["missing_model_ids"],
            "runtime": "llama_cpp_local_gguf_only",
        },
        "selected_model_ids": selected,
        "exact_verification_requirements": {
            "authority": [
                "calibrated_clean_verifier_v15",
                "exact_context_checker",
                "python_ast_or_json_parser_when_applicable",
                "deterministic_integer_constraint_evaluator_when_applicable",
            ],
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "coverage_rate_floor": clean["coverage_rate"],
            "abstentions_recorded_separately": True,
            "accepted_repairs_require_exact_pass": True,
        },
        "kan_boundary": {
            "kan_boundary_decision": kan["kan_boundary_decision"],
            "kan_as_repair_gate_authority": False,
            "permitted_downstream_use": kan["permitted_downstream_use"],
        },
        "claim_boundary": {
            "headline_claim_allowed": False,
            "panel_claim": "diagnostic_micro_panel_only",
            "no_generalization_beyond_panel": True,
        },
    }


def kan_downstream_use_bounded(payload: Mapping[str, Any]) -> bool:
    """Return true only when KAN use is explicitly bounded away from promotion."""

    permitted = set(str(item) for item in list_field(payload, "permitted_downstream_use"))
    prohibited = set(str(item) for item in list_field(payload, "prohibited_downstream_use"))
    decision = str(payload.get("kan_boundary_decision") or "")
    return (
        bool(permitted)
        and permitted <= BOUNDED_KAN_USES
        and permitted.isdisjoint(FORBIDDEN_KAN_USES)
        and "repair_gate_authority" in prohibited
        and decision in {
            "retain_sidecar_only",
            "retire_from_prompt_injection_headline",
            "prerequisite_required",
        }
    )


def source_artifacts(sources: Mapping[str, JsonLoad]) -> list[JsonDict]:
    """List input artifact paths and checksums for reproducible aggregation."""

    return [
        {"id": key, **source_prefix(load)}
        for key, load in sources.items()
    ]


def bool_field(payload: Mapping[str, Any], field: str) -> bool:
    """Read a gate boolean without treating truthy non-bools as passing."""

    return payload.get(field) is True


def rate_field(payload: Mapping[str, Any], field: str, *, default: float) -> float:
    """Read a finite [0, 1] rate, defaulting conservatively when malformed."""

    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return float(default)
    rate = float(value)
    return rate if math.isfinite(rate) and 0.0 <= rate <= 1.0 else float(default)


def list_field(payload: Mapping[str, Any], field: str) -> list[Any]:
    """Read a JSON list field without accepting strings as sequences."""

    value = payload.get(field)
    return list(value) if isinstance(value, list) else []


def model_ids(value: Any) -> list[str]:
    """Extract model identifiers from upstream model rows."""

    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    ids: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            model_id = item.get("model_id") or item.get("hf_id")
            if isinstance(model_id, str) and model_id:
                ids.append(model_id)
    return ids


def sha256_file(path: Path) -> str | None:
    """Hash an input file when it exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative_path(path: Path) -> str:
    """Render source paths relative to the repository when possible."""

    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def duration(started: float, finished: float) -> float:
    """Return a non-negative rounded wall-clock duration."""

    return round(max(0.0, float(finished) - float(started)), 6)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Summarize the terminal gate decision without implying repair ran."""

    return (
        f"complete: repair_gate_open={str(artifact['repair_gate_open']).lower()}; "
        f"blocked_reason_count={len(artifact['blocked_reasons'])}; "
        f"exp3290_scope_defined={str(artifact['permitted_repair_scope'].get('repair_generation_allowed') is True).lower()}"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable decision inputs while excluding timing and test-run noise."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum", "tests_run"}
    }
    return stable_hash(stable)


def stable_hash(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hash for JSON-compatible mappings."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the v9 artifact and fail-closed invariants."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("repair_gate_decision_v9_ready") is not True:
        raise ValueError("repair_gate_decision_v9_ready must be true")
    for field in (
        "repair_gate_open",
        "garak_redteam_eval_ready",
        "clean_verifier_rerun_ready",
        "repair_gate_input_clean_enough",
        "kan_boundary_decision_ready",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"gate bool {field} must be a bool")
    if not isinstance(artifact.get("gate_inputs"), Mapping):
        raise ValueError("gate_inputs must be a dict")
    blocked = artifact.get("blocked_reasons")
    if not isinstance(blocked, list):
        raise ValueError("blocked_reasons must be a list")
    scope = artifact.get("permitted_repair_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("permitted_repair_scope must be a dict")
    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, int | float) or isinstance(duration_s, bool) or duration_s < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    checksum = str(artifact.get("reproducibility_checksum") or "")
    if len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a 64-character sha256")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")
    if artifact["repair_gate_open"]:
        if blocked:
            raise ValueError("open gate must not include blocked_reasons")
        if scope.get("repair_generation_allowed") is not True:
            raise ValueError("open gate must define an executable permitted_repair_scope")
    else:
        if not blocked:
            raise ValueError("closed gate must include blocked_reasons")
        if scope.get("repair_generation_allowed") is True:
            raise ValueError("closed gate must not allow repair generation")
