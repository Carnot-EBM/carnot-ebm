"""Reusable terminal-evidence preflight for result artifacts.

Spec refs: REQ-INFRA-6298, SCENARIO-INFRA-6298-1,
SCENARIO-INFRA-6298-2, SCENARIO-INFRA-6298-3,
SCENARIO-INFRA-6298-4, SCENARIO-INFRA-6298-5.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

from carnot.terminal_artifacts import (
    classify_artifact_payload,
    gate_field_eligibility,
    path_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - import path setup.
    sys.path.insert(0, str(SCRIPTS_ROOT))

ARTIFACT_QA_LINT_TESTS_SUBSTRATE = "artifact_qa_lint_tests"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py "
    "-q --no-cov -n 0"
)

DEFAULT_REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "preconditions_checked",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
)
DEFAULT_GATE_FIELDS = (
    {"field": "terminal_evidence_preflight_ready_score", "expected_type": "number"},
)
TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "complete_ready:",
    "complete_null:",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked:",
    "blocked_",
    "skipped:",
    "skipped_",
    "flagged:",
    "flagged_",
)
PROTECTED_DETERMINATION_KEYS = frozenset(
    {
        "flagged_adversarial",
        "corrigendum_pending",
        "corrigendum_note",
        "inference_substrate_correction_note",
        "inference_substrate_original_invalid_value",
        "solve_provenance",
        "solve_provenance_note",
        "verifier_is_oracle",
        "preconditions_checked",
    }
)

FAILURE_TAXONOMY: dict[str, str] = {
    "malformed_payload": "The artifact JSON did not load to a top-level object.",
    "missing_required_field": "A required top-level artifact field is absent.",
    "nonterminal_artifact": "The shared terminal classifier did not find a terminal state.",
    "missing_terminal_prefix": "honest_verdict does not start with an accepted terminal prefix.",
    "field_principles_not_mapping": "field_principles is absent or not an object.",
    "field_provenance_not_mapping": "field_provenance is absent or not an object.",
    "missing_field_principle": "A required field has no field_principles entry.",
    "missing_field_provenance": "A required field has no field_provenance entry.",
    "missing_inference_substrate": "inference_substrate is absent or empty.",
    "unknown_compute_bound_substrate": "A compute-bound artifact used an unknown substrate name.",
    "duration_missing": "duration_s is absent or not a finite number.",
    "duration_floor_violation": "duration_s is below the selected substrate duration floor.",
    "methodology_missing": "Compute-bound evidence is missing model, seed, or checksum receipts.",
    "reproducibility_missing": "A seed, checksum, or precondition receipt is absent.",
    "test_commands_missing": "test_commands is absent, empty, or not a list of strings.",
    "test_exit_codes_not_mapping": "test_exit_codes is absent or not an object.",
    "test_exit_code_missing": "A declared test command has no recorded exit code.",
    "test_exit_code_extra": "An exit-code entry names a command not declared in test_commands.",
    "test_exit_code_not_int": "A recorded exit code is not a bare integer.",
    "test_exit_code_nonzero": "A recorded test command exited nonzero.",
    "gate_field_nonterminal_artifact": "A staged gate points at a nonterminal artifact.",
    "gate_field_missing": "A staged gate field is absent at the top level.",
    "gate_field_not_bare": "A staged gate field is principle-wrapped rather than bare.",
    "gate_field_type_mismatch": "A staged gate field has the wrong bare value type.",
    "determination_dropped": "A protected determination or review marker was dropped.",
}

V542_FAILURE_FIXTURES = (
    {
        "fixture_id": "exp6288",
        "path": "results/experiment_6288_partial_atom_evidence_adapter.json",
        "expected_accept": False,
        "expected_failure_classes": [
            "duration_floor_violation",
            "methodology_missing",
        ],
    },
    {
        "fixture_id": "exp6289",
        "path": "results/experiment_6289_flagship_exact_state_refinement_benchmark.json",
        "expected_accept": False,
        "expected_failure_classes": ["test_exit_code_missing"],
    },
    {
        "fixture_id": "exp6290",
        "path": "results/experiment_6290_revocable_atomic_repair_memory.json",
        "expected_accept": False,
        "expected_failure_classes": ["test_exit_code_nonzero"],
    },
)


def _adversarial_verify() -> Any:
    import adversarial_verify as av  # noqa: PLC0415

    return av


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _has_seed(payload: JsonMap) -> bool:
    return any(
        payload.get(key) not in (None, [], {}, "")
        for key in ("random_seed", "seed", "random_seeds", "random_seeds_used", "seeds")
    )


def _has_model_spec(payload: JsonMap) -> bool:
    return any(
        bool(payload.get(key))
        for key in ("model_specs", "MODEL_SPECS", "target_model", "models_tested")
    )


def _is_substantive(value: Any) -> bool:
    if isinstance(value, Mapping) and "value" in value:
        value = value["value"]
    if value is None or value is False:
        return False
    if isinstance(value, (str, list, tuple, dict)):
        return len(value) > 0
    return True


def _protected_determination_key(key: str, value: Any) -> bool:
    if key in PROTECTED_DETERMINATION_KEYS:
        if key == "flagged_adversarial":
            return value is True
        return _is_substantive(value)
    lowered = key.lower()
    return _is_substantive(value) and any(
        marker in lowered
        for marker in (
            "corrigend",
            "correction",
            "provenance",
            "acknowledg",
            "retract",
            "errat",
            "disclos",
            "caveat",
            "false_negative_risk",
            "forbidden_claims",
        )
    )


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "bool":
        return type(value) is bool
    if expected_type == "int":
        return type(value) is int
    if expected_type == "float":
        return type(value) is float
    if expected_type == "number":
        return _is_finite_number(value)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "list":
        return isinstance(value, list)
    if expected_type == "dict":
        return isinstance(value, Mapping)
    return False


def _required_field_check(payload: JsonMap, required_fields: Sequence[str]) -> JsonDict:
    missing = [field for field in required_fields if field not in payload]
    return {
        "ok": not missing,
        "required_fields": list(required_fields),
        "missing": missing,
        "failure_classes": ["missing_required_field"] if missing else [],
    }


def _terminal_prefix_check(payload: JsonMap) -> JsonDict:
    classification = classify_artifact_payload(payload).to_dict()
    verdict = payload.get("honest_verdict")
    verdict_text = verdict if isinstance(verdict, str) else ""
    has_prefix = verdict_text.lower().startswith(TERMINAL_VERDICT_PREFIXES)
    failures: list[str] = []
    if not classification["terminal"]:
        failures.append("nonterminal_artifact")
    if not has_prefix:
        failures.append("missing_terminal_prefix")
    return {
        "ok": not failures,
        "classification": classification,
        "honest_verdict": verdict,
        "accepted_prefixes": list(TERMINAL_VERDICT_PREFIXES),
        "failure_classes": failures,
    }


def _field_principle_coverage_check(payload: JsonMap, required_fields: Sequence[str]) -> JsonDict:
    principles_raw = payload.get("field_principles")
    provenance_raw = payload.get("field_provenance")
    principles = principles_raw if isinstance(principles_raw, Mapping) else {}
    provenance = provenance_raw if isinstance(provenance_raw, Mapping) else {}
    missing_principles = [field for field in required_fields if field not in principles]
    missing_provenance = [field for field in required_fields if field not in provenance]
    failures: list[str] = []
    if not isinstance(principles_raw, Mapping):
        failures.append("field_principles_not_mapping")
    if not isinstance(provenance_raw, Mapping):
        failures.append("field_provenance_not_mapping")
    if missing_principles:
        failures.append("missing_field_principle")
    if missing_provenance:
        failures.append("missing_field_provenance")
    return {
        "ok": not failures,
        "principle_count": len(principles),
        "provenance_count": len(provenance),
        "missing_field_principles": missing_principles,
        "missing_field_provenance": missing_provenance,
        "failure_classes": failures,
    }


def _substrate_duration_and_methodology_check(payload: JsonMap) -> JsonDict:
    av = _adversarial_verify()
    artifact = dict(payload)
    substrate = payload.get("inference_substrate")
    classification = av._classify_inference_substrate(artifact)
    compute_marker = av._has_compute_bound_marker(artifact)
    duration_floor = av.duration_floor_for_artifact(artifact)
    duration = payload.get("duration_s")
    missing_methodology: list[str] = []
    failures: list[str] = []

    if not isinstance(substrate, str) or not substrate.strip():
        failures.append("missing_inference_substrate")
    if classification["kind"] == av.SUBSTRATE_KIND_UNKNOWN and compute_marker:
        failures.append("unknown_compute_bound_substrate")
    if not _is_finite_number(duration):
        failures.append("duration_missing")
    elif duration_floor is not None and float(duration) < float(duration_floor["min_duration_s"]):
        failures.append("duration_floor_violation")

    floor_reason = str(duration_floor.get("reason")) if isinstance(duration_floor, Mapping) else ""
    live_or_marked = (
        compute_marker
        or classification["kind"] == av.SUBSTRATE_KIND_LIVE_MODEL
        or floor_reason in {"live_model", "local_sota_gguf_small_n", "native_gguf_backend_bisect"}
    )
    if live_or_marked and not _has_model_spec(payload):
        missing_methodology.append("model_specs/target_model")
    if live_or_marked and not _has_seed(payload):
        missing_methodology.append("random_seed")
    if live_or_marked and not payload.get("reproducibility_checksum"):
        missing_methodology.append("reproducibility_checksum")
    if live_or_marked and not _is_finite_number(duration):
        missing_methodology.append("duration_s")
    if missing_methodology:
        failures.append("methodology_missing")

    return {
        "ok": not failures,
        "declared_substrate": substrate,
        "classification": classification,
        "compute_bound_marker_present": bool(compute_marker),
        "duration_s": duration,
        "duration_floor": duration_floor,
        "missing_methodology": missing_methodology,
        "failure_classes": _unique(failures),
    }


def _reproducibility_check(payload: JsonMap) -> JsonDict:
    missing = []
    if not _has_seed(payload):
        missing.append("random_seed")
    if not payload.get("reproducibility_checksum"):
        missing.append("reproducibility_checksum")
    if not _is_substantive(payload.get("preconditions_checked")):
        missing.append("preconditions_checked")
    return {
        "ok": not missing,
        "missing": missing,
        "failure_classes": ["reproducibility_missing"] if missing else [],
    }


def _test_command_and_exit_code_check(payload: JsonMap) -> JsonDict:
    commands_raw = payload.get("test_commands")
    exit_codes_raw = payload.get("test_exit_codes")
    commands = commands_raw if isinstance(commands_raw, list) else []
    command_strings = [command for command in commands if isinstance(command, str) and command]
    exit_codes = exit_codes_raw if isinstance(exit_codes_raw, Mapping) else {}

    missing_exit_codes = [command for command in command_strings if command not in exit_codes]
    extra_exit_codes = [
        str(command) for command in exit_codes if command not in set(command_strings)
    ]
    non_integer = [str(command) for command, code in exit_codes.items() if type(code) is not int]
    nonzero = [
        {"command": str(command), "exit_code": code}
        for command, code in exit_codes.items()
        if type(code) is int and code != 0
    ]
    failures: list[str] = []
    if not commands or len(command_strings) != len(commands):
        failures.append("test_commands_missing")
    if not isinstance(exit_codes_raw, Mapping):
        failures.append("test_exit_codes_not_mapping")
    if missing_exit_codes:
        failures.append("test_exit_code_missing")
    if extra_exit_codes:
        failures.append("test_exit_code_extra")
    if non_integer:
        failures.append("test_exit_code_not_int")
    if nonzero:
        failures.append("test_exit_code_nonzero")
    return {
        "ok": not failures,
        "command_count": len(command_strings),
        "exit_code_count": len(exit_codes),
        "missing_exit_codes": missing_exit_codes,
        "extra_exit_codes": extra_exit_codes,
        "non_integer_exit_codes": non_integer,
        "nonzero_exit_codes": nonzero,
        "executed_commands": [],
        "failure_classes": failures,
    }


def _gate_field_type_check(payload: JsonMap, gate_fields: Sequence[JsonMap]) -> JsonDict:
    rows: list[JsonDict] = []
    failures: list[str] = []
    for spec in gate_fields:
        field = str(spec.get("field") or "")
        expected_type = str(spec.get("expected_type") or "")
        eligibility = gate_field_eligibility(payload, field).to_dict()
        row_failures: list[str] = []
        if not eligibility["classification"]["terminal"]:
            row_failures.append("gate_field_nonterminal_artifact")
        if not eligibility["field_present"]:
            row_failures.append("gate_field_missing")
        elif not eligibility["field_is_bare"]:
            row_failures.append("gate_field_not_bare")
        elif not _type_matches(eligibility.get("value"), expected_type):
            row_failures.append("gate_field_type_mismatch")
        failures.extend(row_failures)
        rows.append(
            {
                "field": field,
                "expected_type": expected_type,
                "eligibility": eligibility,
                "failure_classes": _unique(row_failures),
                "ok": not row_failures,
            }
        )
    return {"ok": not failures, "gate_fields": rows, "failure_classes": _unique(failures)}


def _determination_preservation_check(
    payload: JsonMap,
    baseline_payload: JsonMap | None,
) -> JsonDict:
    dropped: list[str] = []
    if baseline_payload is not None:
        for key, old_value in baseline_payload.items():
            if not _protected_determination_key(str(key), old_value):
                continue
            if key not in payload or not _is_substantive(payload.get(key)):
                dropped.append(str(key))
    return {
        "ok": not dropped,
        "baseline_present": baseline_payload is not None,
        "dropped_determination_fields": sorted(dropped),
        "failure_classes": ["determination_dropped"] if dropped else [],
    }


def preflight_payload(
    payload: JsonMap | Any,
    *,
    required_fields: Sequence[str] = DEFAULT_REQUIRED_ARTIFACT_FIELDS,
    gate_fields: Sequence[JsonMap] = (),
    baseline_payload: JsonMap | None = None,
    fixture_id: str | None = None,
    path: str | None = None,
) -> JsonDict:
    """Validate one already-loaded artifact payload without running its commands."""

    if not isinstance(payload, Mapping):
        return {
            "fixture_id": fixture_id,
            "path": path,
            "accepted": False,
            "failure_classes": ["malformed_payload"],
            "required_field_check": {},
            "terminal_prefix_check": {},
            "field_principle_coverage_check": {},
            "substrate_duration_and_methodology_check": {},
            "reproducibility_check": {},
            "test_command_and_exit_code_check": {},
            "gate_field_type_check": {},
            "determination_preservation_check": {},
        }

    required = _required_field_check(payload, required_fields)
    terminal = _terminal_prefix_check(payload)
    field_coverage = _field_principle_coverage_check(payload, required_fields)
    substrate = _substrate_duration_and_methodology_check(payload)
    reproducibility = _reproducibility_check(payload)
    commands = _test_command_and_exit_code_check(payload)
    gates = _gate_field_type_check(payload, gate_fields)
    determination = _determination_preservation_check(payload, baseline_payload)
    failures = _unique(
        [
            *required["failure_classes"],
            *terminal["failure_classes"],
            *field_coverage["failure_classes"],
            *substrate["failure_classes"],
            *reproducibility["failure_classes"],
            *commands["failure_classes"],
            *gates["failure_classes"],
            *determination["failure_classes"],
        ]
    )
    return {
        "fixture_id": fixture_id,
        "path": path,
        "accepted": not failures,
        "failure_classes": failures,
        "required_field_check": required,
        "terminal_prefix_check": terminal,
        "field_principle_coverage_check": field_coverage,
        "substrate_duration_and_methodology_check": substrate,
        "reproducibility_check": reproducibility,
        "test_command_and_exit_code_check": commands,
        "gate_field_type_check": gates,
        "determination_preservation_check": determination,
    }


def preflight_artifact_path(
    path: Path,
    *,
    required_fields: Sequence[str] = DEFAULT_REQUIRED_ARTIFACT_FIELDS,
    gate_fields: Sequence[JsonMap] = (),
    baseline_payload: JsonMap | None = None,
    fixture_id: str | None = None,
) -> JsonDict:
    """Load and validate one artifact path."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    result = preflight_payload(
        payload,
        required_fields=required_fields,
        gate_fields=gate_fields,
        baseline_payload=baseline_payload,
        fixture_id=fixture_id,
        path=path.as_posix(),
    )
    result["path_sha256"] = path_sha256(path)
    return result


def _principle_map(fields: Sequence[str]) -> dict[str, str]:
    return {field: f"{field} is required so terminal evidence can be audited." for field in fields}


def _provenance_map(fields: Sequence[str]) -> dict[str, JsonDict]:
    return {
        field: {
            "sources": ["REQ-INFRA-6298", "synthetic_fixture"],
            "principle": f"{field} is required so terminal evidence can be audited.",
        }
        for field in fields
    }


def clean_fixture_payload(
    *,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Return the passing synthetic artifact used by the fixture matrix."""

    commands = list(test_commands or [FOCUSED_TEST_COMMAND])
    exits = dict(test_exit_codes or {command: 0 for command in commands})
    fields = (*DEFAULT_REQUIRED_ARTIFACT_FIELDS, "terminal_evidence_preflight_ready_score")
    return {
        "status": "complete",
        "honest_verdict": "complete: synthetic terminal evidence fixture passed",
        "inference_substrate": ARTIFACT_QA_LINT_TESTS_SUBSTRATE,
        "duration_s": 0.01,
        "preconditions_checked": {"synthetic_fixture": True},
        "verifier_is_oracle": False,
        "random_seed": 6298,
        "field_principles": _principle_map(fields),
        "field_provenance": _provenance_map(fields),
        "test_commands": commands,
        "test_exit_codes": exits,
        "reproducibility_checksum": "sha256:synthetic-fixture",
        "terminal_evidence_preflight_ready_score": 1.0,
    }


def build_synthetic_fixture_manifest() -> JsonDict:
    """Build clean and adversarial synthetic fixtures without touching disk."""

    clean = clean_fixture_payload()
    missing_field = copy.deepcopy(clean)
    del missing_field["duration_s"]
    bad_prefix = copy.deepcopy(clean)
    bad_prefix["honest_verdict"] = "synthetic fixture passed"
    bad_gate_type = copy.deepcopy(clean)
    bad_gate_type["terminal_evidence_preflight_ready_score"] = "1.0"
    determination_baseline = copy.deepcopy(clean)
    determination_baseline["flagged_adversarial"] = True
    determination_baseline["corrigendum_pending"] = [
        {"kind": "FIXTURE_FLAG", "detail": "baseline protected record"}
    ]
    determination_drop = copy.deepcopy(clean)

    return {
        "schema": "carnot.terminal_evidence_preflight.synthetic_fixtures.v1",
        "spec_refs": [
            "REQ-INFRA-6298",
            "SCENARIO-INFRA-6298-2",
            "SCENARIO-INFRA-6298-5",
        ],
        "gate_fields": list(DEFAULT_GATE_FIELDS),
        "fixtures": [
            {
                "fixture_id": "clean",
                "expected_accept": True,
                "expected_failure_classes": [],
                "payload": clean,
            },
            {
                "fixture_id": "missing_field",
                "expected_accept": False,
                "expected_failure_classes": ["missing_required_field", "duration_missing"],
                "payload": missing_field,
            },
            {
                "fixture_id": "bad_prefix",
                "expected_accept": False,
                "expected_failure_classes": ["missing_terminal_prefix"],
                "payload": bad_prefix,
            },
            {
                "fixture_id": "bad_gate_type",
                "expected_accept": False,
                "expected_failure_classes": ["gate_field_type_mismatch"],
                "payload": bad_gate_type,
            },
            {
                "fixture_id": "determination_drop",
                "expected_accept": False,
                "expected_failure_classes": ["determination_dropped"],
                "payload": determination_drop,
                "baseline_payload": determination_baseline,
            },
        ],
    }


def evaluate_fixture_manifest(manifest: JsonMap) -> JsonDict:
    """Evaluate the synthetic fixture manifest and compute false counts."""

    fixtures_raw = manifest.get("fixtures")
    fixtures = fixtures_raw if isinstance(fixtures_raw, list) else []
    default_gates = manifest.get("gate_fields")
    gate_fields = default_gates if isinstance(default_gates, list) else list(DEFAULT_GATE_FIELDS)
    results: list[JsonDict] = []
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            continue
        fixture_gate_fields = fixture.get("gate_fields")
        gates = fixture_gate_fields if isinstance(fixture_gate_fields, list) else gate_fields
        result = preflight_payload(
            fixture.get("payload"),
            gate_fields=gates,
            baseline_payload=fixture.get("baseline_payload")
            if isinstance(fixture.get("baseline_payload"), Mapping)
            else None,
            fixture_id=str(fixture.get("fixture_id") or ""),
        )
        result["expected_accept"] = fixture.get("expected_accept") is True
        result["expected_failure_classes"] = list(fixture.get("expected_failure_classes") or [])
        results.append(result)

    false_accept_count = sum(
        1 for row in results if row["accepted"] is True and row["expected_accept"] is False
    )
    false_reject_count = sum(
        1 for row in results if row["accepted"] is False and row["expected_accept"] is True
    )
    return {
        "fixture_results": results,
        "clean_fixture_accept_count": sum(
            1 for row in results if row["expected_accept"] is True and row["accepted"] is True
        ),
        "bad_fixture_reject_count": sum(
            1 for row in results if row["expected_accept"] is False and row["accepted"] is False
        ),
        "false_accept_count": int(false_accept_count),
        "false_reject_count": int(false_reject_count),
        "failure_class_counts": dict(
            sorted(Counter(cls for row in results for cls in row["failure_classes"]).items())
        ),
    }


def replay_v542_failure_fixtures(root: Path = REPO_ROOT) -> list[JsonDict]:
    """Replay the three V542 failure artifacts as immutable fixtures."""

    rows: list[JsonDict] = []
    for fixture in V542_FAILURE_FIXTURES:
        path = root / str(fixture["path"])
        result = preflight_artifact_path(path, fixture_id=str(fixture["fixture_id"]))
        result["path"] = str(fixture["path"])
        result["expected_accept"] = fixture["expected_accept"]
        result["expected_failure_classes"] = list(fixture["expected_failure_classes"])
        rows.append(result)
    return rows
