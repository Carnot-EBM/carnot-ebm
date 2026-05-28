"""Build the Exp 3227 repair-gate decision v7 artifact.

Spec refs: REQ-VERIFY-3227, SCENARIO-VERIFY-3227.

This module aggregates evidence that is already on disk. The repair ladder is
expensive and can change downstream state, so this gate fails closed unless the
local SOTA receipt, clean verifier rerun, and structured proposal preflight are
all strong enough to justify execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.repair_gate_decision.v7"
EXPERIMENT_ID = "exp3227"
MILESTONE = "2026.05.298"
INFERENCE_SUBSTRATE = "artifact_gate_aggregation"

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3227_repair_gate_decision_v7.json")
EXP3222_REL_PATH = Path("results/experiment_3222_full_local_sota_receipt_v6.json")
EXP3225_REL_PATH = Path("results/experiment_3225_clean_live_sota_verifier_rerun_v13.json")
EXP3226_REL_PATH = Path("results/experiment_3226_structured_repair_proposal_preflight_v2.json")
EXP3213_REL_PATH = Path("results/experiment_3213_repair_gate_decision_v6.json")

REQUIRED_FIELD_NAMES = (
    "schema_version",
    "experiment_id",
    "milestone",
    "input_artifacts",
    "receipt_ok",
    "clean_verifier_ok",
    "structured_preflight_ok",
    "blocker_list",
    "blocker_count",
    "repair_gate_state",
    "repair_ladder_allowed",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)
ALLOWED_REPAIR_GATE_STATES = {"unblocked", "blocked", "diagnostic_only"}
MANDATORY_ROLES = {
    "exp3222_full_local_sota_receipt_v6",
    "exp3225_clean_live_sota_verifier_rerun_v13",
    "exp3226_structured_repair_proposal_preflight_v2",
}
CPU_FALLBACK_BOOL_FIELDS = (
    "cpu_fallback",
    "cpu_fallback_used",
    "cpu_fallback_only",
    "cpu_fallback_detected",
    "using_cpu_fallback",
)
CPU_FALLBACK_TEXT_FIELDS = (
    "inference_substrate",
    "substrate_classification",
    "substrate",
    "backend",
    "execution_path",
    "honest_verdict",
)
EXACT_VERIFIER_BOOL_FIELDS = (
    "exact_labels_authoritative",
    "exact_verifier_ready",
    "exact_verifier_handoff_ready",
    "exact_verifier_authority_preserved",
    "exact_verifier_invoked",
    "exact_verifier_scoring_complete",
)
EXACT_VERIFIER_COUNT_FIELDS = (
    "exact_verifier_invocation_count",
    "exact_verifier_row_count",
)
EXACT_VERIFIER_LIST_FIELDS = (
    "exact_verifier_types",
    "exact_verifier_results",
    "exact_fixture_artifacts",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3227_repair_gate_decision_v7.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3227_repair_gate_decision_v7.py -q",
    ".venv/bin/coverage report --include='*/repair_gate_decision_v7.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class SourceSpec:
    """One source file whose bytes support the gate decision."""

    role: str
    path: Path
    mandatory_for_gate: bool
    source_type: str


@dataclass(frozen=True)
class JsonLoad:
    """Parsed JSON source plus the read error, if any."""

    payload: JsonDict
    readable: bool
    error: str | None


SOURCE_SPECS = (
    SourceSpec("agents_repo_instructions", Path("AGENTS.md"), False, "text"),
    SourceSpec("codex_repo_workflow", Path("CODEX.md"), False, "text"),
    SourceSpec("claude_authenticity_rules", Path("CLAUDE.md"), False, "text"),
    SourceSpec("verification_openspec", SPEC_REL_PATH, False, "text"),
    SourceSpec("conductor_gate_helper", Path("scripts/conductor_gates.py"), False, "text"),
    SourceSpec("exp3213_repair_gate_decision_v6", EXP3213_REL_PATH, False, "json"),
    SourceSpec("exp3222_full_local_sota_receipt_v6", EXP3222_REL_PATH, True, "json"),
    SourceSpec("exp3225_clean_live_sota_verifier_rerun_v13", EXP3225_REL_PATH, True, "json"),
    SourceSpec("exp3226_structured_repair_proposal_preflight_v2", EXP3226_REL_PATH, True, "json"),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3227: decide whether the v7 repair ladder gate is unblocked."""

    root_path = Path(root)
    json_sources = load_json_sources(root_path)
    payloads = {
        "exp3222": json_sources["exp3222_full_local_sota_receipt_v6"].payload,
        "exp3225": json_sources["exp3225_clean_live_sota_verifier_rerun_v13"].payload,
        "exp3226": json_sources["exp3226_structured_repair_proposal_preflight_v2"].payload,
    }
    inputs = input_artifacts(root_path, json_sources)
    receipt_ok = receipt_gate_ok(payloads["exp3222"])
    clean_ok = clean_verifier_gate_ok(payloads["exp3225"])
    preflight_ok = structured_preflight_gate_ok(payloads["exp3226"])
    blockers = all_blockers(inputs, payloads)
    any_mandatory_readable = any(
        row["role"] in MANDATORY_ROLES and row["readable_json_object"] is True
        for row in inputs
    )
    if not any_mandatory_readable:
        blockers.append(
            blocker(
                "no_mandatory_gate_artifacts_readable",
                "mandatory_gate_artifacts",
                "readable_json_object",
                "at least one",
                0,
                "no mandatory Exp 3222/3225/3226 gate artifact could be read as a JSON object",
            )
        )
    state = repair_gate_state(
        receipt_ok=receipt_ok,
        clean_verifier_ok=clean_ok,
        structured_preflight_ok=preflight_ok,
        blockers=blockers,
        any_mandatory_readable=any_mandatory_readable,
    )
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": ["REQ-VERIFY-3227", "SCENARIO-VERIFY-3227"],
        "input_artifacts": inputs,
        "receipt_ok": receipt_ok,
        "clean_verifier_ok": clean_ok,
        "structured_preflight_ok": preflight_ok,
        "blocker_list": blockers,
        "blocker_count": len(blockers),
        "repair_gate_state": state,
        "repair_ladder_allowed": state == "unblocked",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the v7 repair-gate decision JSON."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_json_sources(root: Path) -> dict[str, JsonLoad]:
    """Load every JSON source named in the v7 gate contract."""

    return {
        spec.role: read_json_object(root / spec.path)
        for spec in SOURCE_SPECS
        if spec.source_type == "json"
    }


def read_json_object(path: Path) -> JsonLoad:
    """Read a JSON object from disk and make missing or malformed evidence explicit."""

    if not path.is_file():
        return JsonLoad({}, False, "missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return JsonLoad({}, False, str(exc))
    return (
        JsonLoad(dict(payload), True, None)
        if isinstance(payload, Mapping)
        else JsonLoad({}, False, "json root is not an object")
    )


def input_artifacts(root: Path, json_sources: Mapping[str, JsonLoad]) -> list[JsonDict]:
    """Summarize every instruction, provenance, and mandatory gate input path."""

    rows: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        loaded = json_sources.get(spec.role, JsonLoad({}, False, None))
        payload = loaded.payload
        rows.append(
            {
                "role": spec.role,
                "path": spec.path.as_posix(),
                "source_type": spec.source_type,
                "mandatory_for_gate": spec.mandatory_for_gate,
                "present": path.is_file(),
                "readable_json_object": loaded.readable if spec.source_type == "json" else None,
                "error": loaded.error if spec.source_type == "json" and not loaded.readable else None,
                "sha256": sha256_file(path),
                "summary": artifact_summary(payload) if payload else {},
            }
        )
    return rows


def sha256_file(path: Path) -> str | None:
    """Hash source evidence when it exists so the decision can be reproduced."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_summary(payload: Mapping[str, Any]) -> JsonDict:
    """Keep compact machine-readable gate fields beside source provenance."""

    return {
        "experiment_id": payload.get("experiment_id") or payload.get("experiment"),
        "schema_version": payload.get("schema_version") or payload.get("schema"),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "clean_rerun_allowed": payload.get("clean_rerun_allowed"),
        "clean_verifier_ready": payload.get("clean_verifier_ready"),
        "ready_for_repair_gate": payload.get("ready_for_repair_gate"),
        "exact_verifier_handoff_ready": payload.get("exact_verifier_handoff_ready"),
        "repair_correctness_claimed": payload.get("repair_correctness_claimed"),
        "blocked_at_layer": payload.get("blocked_at_layer"),
        "gate_check_summary": payload.get("gate_check_summary"),
    }


def receipt_gate_ok(payload: Mapping[str, Any]) -> bool:
    """Exp 3222 passes only with clean rerun permission and no fallback/skip."""

    return (
        payload.get("clean_rerun_allowed") is True
        and not gate_skipped(payload)
        and not cpu_fallback_detected(payload)
    )


def clean_verifier_gate_ok(payload: Mapping[str, Any]) -> bool:
    """Exp 3225 passes only when local SOTA exact-verifier evidence is ready."""

    return (
        payload.get("clean_verifier_ready") is True
        and not gate_skipped(payload)
        and not cpu_fallback_detected(payload)
        and exact_verifier_available(payload)
    )


def structured_preflight_gate_ok(payload: Mapping[str, Any]) -> bool:
    """Exp 3226 passes only when proposals are ready for exact-verifier handoff."""

    return (
        payload.get("ready_for_repair_gate") is True
        and payload.get("exact_verifier_handoff_ready") is True
        and payload.get("repair_correctness_claimed") is False
        and not gate_skipped(payload)
        and not schema_only_repair_limitation(payload)
    )


def gate_skipped(payload: Mapping[str, Any]) -> bool:
    """Detect conductor pre-gate or explicit gated-skip artifacts."""

    schema = str(payload.get("schema") or payload.get("schema_version") or "")
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    return (
        payload.get("gated_skip") is True
        or payload.get("gate_skipped") is True
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or schema == "blocked_gate_check_v1"
        or status in {"gated_skipped", "gate_skipped"}
        or "blocked_gate_check_failed" in verdict
    )


def cpu_fallback_detected(payload: Mapping[str, Any]) -> bool:
    """Detect CPU fallback evidence that cannot unlock local SOTA repair."""

    if any(payload.get(field) is True for field in CPU_FALLBACK_BOOL_FIELDS):
        return True
    return any(
        "cpu_fallback" in str(payload.get(field) or "").lower()
        or "cpu fallback" in str(payload.get(field) or "").lower()
        for field in CPU_FALLBACK_TEXT_FIELDS
    )


def exact_verifier_available(payload: Mapping[str, Any]) -> bool:
    """Detect exact labels, exact verifier metadata, or exact-verifier counts."""

    if any(payload.get(field) is True for field in EXACT_VERIFIER_BOOL_FIELDS):
        return True
    if any(_positive_int(payload.get(field)) for field in EXACT_VERIFIER_COUNT_FIELDS):
        return True
    if any(_nonempty_sequence(payload.get(field)) for field in EXACT_VERIFIER_LIST_FIELDS):
        return True
    available = payload.get("exact_verifier_available")
    return bool(available.get("ok")) if isinstance(available, Mapping) else bool(available)


def _positive_int(value: Any) -> bool:
    """Return true for integer-like counts greater than zero."""

    return isinstance(value, int) and value > 0


def _nonempty_sequence(value: Any) -> bool:
    """Return true for non-string sequences that carry at least one item."""

    return isinstance(value, Sequence) and not isinstance(value, str) and bool(value)


def schema_only_repair_limitation(payload: Mapping[str, Any]) -> bool:
    """Detect repair preflights that only prove schema shape, not verifier handoff."""

    if payload.get("schema_only_repair_limitation") is True:
        return True
    if payload.get("exact_verifier_handoff_ready") is True:
        return False
    return any(
        "schema_only" in str(payload.get(field) or "").lower()
        or "schema-only" in str(payload.get(field) or "").lower()
        for field in ("structured_decoding_backend", "repair_mode", "honest_verdict")
    )


def all_blockers(
    inputs: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Collect every blocker that explains why the terminal gate failed closed."""

    blockers: list[JsonDict] = []
    blockers.extend(source_blockers(inputs, payloads))
    blockers.extend(receipt_blockers(payloads["exp3222"]))
    blockers.extend(clean_verifier_blockers(payloads["exp3225"]))
    blockers.extend(structured_preflight_blockers(payloads["exp3226"]))
    return blockers


def source_blockers(
    inputs: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Mandatory artifacts must exist, parse, avoid gate skips, and avoid CPU fallback."""

    payload_by_role = {
        "exp3222_full_local_sota_receipt_v6": payloads["exp3222"],
        "exp3225_clean_live_sota_verifier_rerun_v13": payloads["exp3225"],
        "exp3226_structured_repair_proposal_preflight_v2": payloads["exp3226"],
    }
    blockers: list[JsonDict] = []
    for row in inputs:
        role = str(row["role"])
        if role not in MANDATORY_ROLES:
            continue
        if row.get("present") is not True:
            blockers.append(
                blocker(
                    "missing_artifact",
                    str(row["path"]),
                    "present",
                    True,
                    False,
                    "mandatory repair-gate input artifact is absent",
                )
            )
            continue
        if row.get("readable_json_object") is not True:
            blockers.append(
                blocker(
                    "malformed_artifact",
                    str(row["path"]),
                    "readable_json_object",
                    True,
                    False,
                    row.get("error") or "mandatory artifact is not a readable JSON object",
                )
            )
            continue
        payload = payload_by_role[role]
        if gate_skipped(payload):
            blockers.append(
                blocker(
                    "gate_skipped_artifact",
                    str(row["path"]),
                    "blocked_at_layer/status",
                    "not gate skipped",
                    {
                        "blocked_at_layer": payload.get("blocked_at_layer"),
                        "status": payload.get("status"),
                        "schema": payload.get("schema") or payload.get("schema_version"),
                    },
                    payload.get("gate_check_summary") or "artifact records a gate skip",
                )
            )
        if cpu_fallback_detected(payload):
            blockers.append(
                blocker(
                    "cpu_fallback_detected",
                    str(row["path"]),
                    "cpu_fallback",
                    False,
                    {
                        "cpu_fallback_detected": payload.get("cpu_fallback_detected"),
                        "substrate_classification": payload.get("substrate_classification"),
                        "inference_substrate": payload.get("inference_substrate"),
                    },
                    "CPU fallback evidence cannot unlock the repair ladder",
                )
            )
    return blockers


def receipt_blockers(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return Exp 3222 receipt-specific blockers."""

    if payload.get("clean_rerun_allowed") is True:
        return []
    return [
        blocker(
            "clean_rerun_not_allowed",
            EXP3222_REL_PATH.as_posix(),
            "clean_rerun_allowed",
            True,
            payload.get("clean_rerun_allowed"),
            payload.get("gate_check_summary")
            or "full local SOTA receipt did not explicitly allow clean reruns",
        )
    ]


def clean_verifier_blockers(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return Exp 3225 clean-verifier blockers."""

    blockers: list[JsonDict] = []
    if payload.get("clean_verifier_ready") is not True:
        blockers.append(
            blocker(
                "clean_verifier_not_ready",
                EXP3225_REL_PATH.as_posix(),
                "clean_verifier_ready",
                True,
                payload.get("clean_verifier_ready"),
                "clean verifier artifact did not report clean_verifier_ready=true",
            )
        )
    if not exact_verifier_available(payload):
        blockers.append(
            blocker(
                "absent_exact_verifier",
                EXP3225_REL_PATH.as_posix(),
                "exact_verifier",
                "exact labels or exact verifier metadata",
                {
                    "exact_labels_authoritative": payload.get("exact_labels_authoritative"),
                    "exact_verifier_types": payload.get("exact_verifier_types"),
                    "exact_verifier_invocation_count": payload.get(
                        "exact_verifier_invocation_count"
                    ),
                },
                "clean verifier evidence lacks exact verifier authority metadata",
            )
        )
    return blockers


def structured_preflight_blockers(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return Exp 3226 structured repair proposal preflight blockers."""

    blockers: list[JsonDict] = []
    if payload.get("ready_for_repair_gate") is not True:
        blockers.append(
            blocker(
                "structured_preflight_not_ready",
                EXP3226_REL_PATH.as_posix(),
                "ready_for_repair_gate",
                True,
                payload.get("ready_for_repair_gate"),
                "structured proposal preflight is not ready for the repair gate",
            )
        )
    if payload.get("exact_verifier_handoff_ready") is not True:
        blockers.append(
            blocker(
                "exact_verifier_handoff_not_ready",
                EXP3226_REL_PATH.as_posix(),
                "exact_verifier_handoff_ready",
                True,
                payload.get("exact_verifier_handoff_ready"),
                "structured proposal preflight lacks exact-verifier handoff readiness",
            )
        )
    if payload.get("repair_correctness_claimed") is not False:
        blockers.append(
            blocker(
                "repair_correctness_claimed",
                EXP3226_REL_PATH.as_posix(),
                "repair_correctness_claimed",
                False,
                payload.get("repair_correctness_claimed"),
                "proposal preflight must not claim repair correctness",
            )
        )
    if schema_only_repair_limitation(payload):
        blockers.append(
            blocker(
                "schema_only_repair_limitation",
                EXP3226_REL_PATH.as_posix(),
                "schema_only_repair_limitation",
                False,
                True,
                "schema-only proposal validity is not enough to execute repair",
            )
        )
    return blockers


def repair_gate_state(
    *,
    receipt_ok: bool,
    clean_verifier_ok: bool,
    structured_preflight_ok: bool,
    blockers: Sequence[Mapping[str, Any]],
    any_mandatory_readable: bool,
) -> str:
    """Classify the terminal gate state from mandatory booleans and blockers."""

    if receipt_ok and clean_verifier_ok and structured_preflight_ok and not blockers:
        return "unblocked"
    if not any_mandatory_readable:
        return "diagnostic_only"
    return "blocked"


def blocker(
    code: str,
    source_artifact: str,
    field: str,
    expected: Any,
    actual: Any,
    detail: Any,
) -> JsonDict:
    """Create a stable blocker row for downstream exp3228 gate decisions."""

    return {
        "code": code,
        "source_artifact": str(source_artifact),
        "field": field,
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Summarize the gate result without implying any repair has run."""

    return (
        f"complete: repair_gate_state={artifact['repair_gate_state']}; "
        f"repair_ladder_allowed={str(artifact['repair_ladder_allowed']).lower()}; "
        f"blocker_count={artifact['blocker_count']}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the v7 terminal artifact shape and fail-closed invariants."""

    missing = set(REQUIRED_FIELD_NAMES) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    state = artifact.get("repair_gate_state")
    if state not in ALLOWED_REPAIR_GATE_STATES:
        raise ValueError(f"repair_gate_state must be one of {sorted(ALLOWED_REPAIR_GATE_STATES)}")
    gate_fields = ("receipt_ok", "clean_verifier_ok", "structured_preflight_ok")
    if any(not isinstance(artifact.get(field), bool) for field in gate_fields):
        raise ValueError("gate booleans must be bool values")
    if not isinstance(artifact.get("input_artifacts"), list):
        raise ValueError("input_artifacts must be a list")
    blockers = artifact.get("blocker_list")
    if not isinstance(blockers, list):
        raise ValueError("blocker_list must be a list")
    if artifact.get("blocker_count") != len(blockers):
        raise ValueError("blocker_count must match blocker_list length")
    if state == "unblocked":
        if artifact.get("repair_ladder_allowed") is not True:
            raise ValueError("unblocked state must allow the repair ladder")
        if blockers:
            raise ValueError("unblocked state must not include blockers")
        if not all(artifact.get(field) is True for field in gate_fields):
            raise ValueError("unblocked state requires all mandatory gates to pass")
    else:
        if artifact.get("repair_ladder_allowed") is not False:
            raise ValueError("blocked or diagnostic_only state must not allow repair ladder")
        if not blockers:
            raise ValueError("blocked or diagnostic_only state must include blockers")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE!r}")
    if artifact.get("conductor_file_modified") is not False:
        raise ValueError("conductor_file_modified must remain false")
    if artifact.get("active_roadmap_modified") is not False:
        raise ValueError("active_roadmap_modified must remain false")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
