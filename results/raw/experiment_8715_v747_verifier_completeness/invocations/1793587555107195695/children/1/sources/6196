"""Build the Exp 3213 repair-gate decision v6 artifact.

Spec refs: REQ-VERIFY-3213, SCENARIO-VERIFY-3213.

This module is deliberately only an evidence aggregator. It reads the receipt,
clean-verifier, structured-proposal, fixture, prior-gate, and conductor-log
artifacts already on disk and decides whether the next repair ladder is
allowed to run. It does not call an LLM or perform repair work because the
gate's job is to prevent premature repair attempts when upstream evidence is
missing, blocked, or adversarially unhandled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.repair_gate_decision.v6"
EXPERIMENT_ID = "exp3213"
MILESTONE = "2026.05.297"

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3213_repair_gate_decision_v6.json")
EXP3208_REL_PATH = Path("results/experiment_3208_full_local_sota_receipt_v5.json")
EXP3209_REL_PATH = Path("results/experiment_3209_clean_live_sota_verifier_rerun_v12.json")
EXP3210_REL_PATH = Path(
    "results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json"
)
EXP3211_REL_PATH = Path("results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json")
EXP3212_REL_PATH = Path("results/experiment_3212_structured_repair_proposal_preflight_v1.json")
EXP3198_REL_PATH = Path("results/experiment_3198_repair_gate_decision_v5.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")

REQUIRED_FIELD_NAMES = (
    "schema_version",
    "experiment_id",
    "milestone",
    "required_artifacts",
    "missing_artifacts",
    "receipt_gate_passed",
    "clean_verifier_gate_passed",
    "structured_proposal_gate_passed",
    "auxiliary_fixture_artifacts",
    "repair_gate_state",
    "repair_ladder_allowed",
    "blockers",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)
ALLOWED_REPAIR_GATE_STATES = {"unblocked", "blocked", "diagnostic_only"}
MANDATORY_ROLES = {
    "exp3208_full_local_sota_receipt_v5",
    "exp3209_clean_live_sota_verifier_rerun_v12",
    "exp3212_structured_repair_proposal_preflight_v1",
}
AUXILIARY_ROLES = {
    "exp3210_context_cot_clbench_parametric_shortcut_fixtures_v1",
    "exp3211_constraintbench_feasibility_objective_pilot_v1",
}
SOURCE_SPECS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), False, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), False, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False, "text"),
    ("verification_openspec", SPEC_REL_PATH, False, "text"),
    ("exp3208_full_local_sota_receipt_v5", EXP3208_REL_PATH, True, "json"),
    ("exp3209_clean_live_sota_verifier_rerun_v12", EXP3209_REL_PATH, True, "json"),
    ("exp3210_context_cot_clbench_parametric_shortcut_fixtures_v1", EXP3210_REL_PATH, False, "json"),
    ("exp3211_constraintbench_feasibility_objective_pilot_v1", EXP3211_REL_PATH, False, "json"),
    ("exp3212_structured_repair_proposal_preflight_v1", EXP3212_REL_PATH, True, "json"),
    ("exp3198_repair_gate_decision_v5", EXP3198_REL_PATH, False, "json"),
    ("conductor_log", CONDUCTOR_LOG_REL_PATH, False, "text"),
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3213_repair_gate_decision_v6.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3213_repair_gate_decision_v6.py -q",
    ".venv/bin/coverage report --include='*/repair_gate_decision_v6.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3213: decide the v6 repair gate from artifacts only."""

    root_path = Path(root)
    payloads = load_payloads(root_path)
    sources = required_artifacts(root_path)
    receipt_passed = receipt_gate_passed(payloads["exp3208"])
    clean_passed = clean_verifier_gate_passed(payloads["exp3209"])
    proposal_passed = structured_proposal_gate_passed(payloads["exp3212"])
    blockers = all_blockers(sources, payloads)
    any_mandatory_readable = any(
        row["role"] in MANDATORY_ROLES and row["readable_json_object"] is True
        for row in sources
    )
    if not any_mandatory_readable:
        blockers.append(
            blocker(
                "no_mandatory_gate_artifacts_readable",
                "mandatory_gate_artifacts",
                "readable_json_object",
                "at least one",
                0,
                "no mandatory repair-gate input artifact could be read as a JSON object",
            )
        )
    state = repair_gate_state(
        receipt_gate=receipt_passed,
        clean_gate=clean_passed,
        proposal_gate=proposal_passed,
        blockers=blockers,
        any_mandatory_readable=any_mandatory_readable,
    )
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": ["REQ-VERIFY-3213", "SCENARIO-VERIFY-3213"],
        "required_artifacts": sources,
        "missing_artifacts": missing_artifacts(sources),
        "receipt_gate_passed": receipt_passed,
        "clean_verifier_gate_passed": clean_passed,
        "structured_proposal_gate_passed": proposal_passed,
        "auxiliary_fixture_artifacts": auxiliary_fixture_artifacts(sources, payloads),
        "repair_gate_state": state,
        "repair_ladder_allowed": state == "unblocked",
        "blockers": blockers,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "inference_substrate": inference_substrate(),
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
    """Build, validate, and write the v6 repair-gate decision JSON."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_payloads(root: Path) -> dict[str, JsonDict]:
    """Load every JSON artifact the v6 gate knows how to interpret."""

    return {
        "exp3208": read_json_object(root / EXP3208_REL_PATH),
        "exp3209": read_json_object(root / EXP3209_REL_PATH),
        "exp3210": read_json_object(root / EXP3210_REL_PATH),
        "exp3211": read_json_object(root / EXP3211_REL_PATH),
        "exp3212": read_json_object(root / EXP3212_REL_PATH),
        "exp3198": read_json_object(root / EXP3198_REL_PATH),
    }


def read_json_object(path: Path) -> JsonDict:
    """Return a JSON object from disk; missing, malformed, or non-object input fails closed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def required_artifacts(root: Path) -> list[JsonDict]:
    """Summarize all listed instruction, provenance, and upstream artifact paths."""

    rows: list[JsonDict] = []
    for role, rel_path, mandatory, source_type in SOURCE_SPECS:
        path = root / rel_path
        payload = read_json_object(path) if source_type == "json" else {}
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "source_type": source_type,
                "mandatory_for_gate": mandatory,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if source_type == "json" else None,
                "sha256": sha256_file(path),
                "summary": artifact_summary(payload) if payload else {},
            }
        )
    return rows


def missing_artifacts(sources: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return every listed path that is absent, including nonblocking auxiliary paths."""

    return [str(row["path"]) for row in sources if row.get("present") is not True]


def sha256_file(path: Path) -> str | None:
    """Hash source evidence when it exists so the gate decision is auditable."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def artifact_summary(payload: Mapping[str, Any]) -> JsonDict:
    """Keep compact fields beside source provenance without copying full artifacts."""

    return {
        "experiment_id": payload.get("experiment_id") or payload.get("experiment"),
        "schema_version": payload.get("schema_version") or payload.get("schema"),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "clean_rerun_allowed": payload.get("clean_rerun_allowed"),
        "clean_verifier_state": payload.get("clean_verifier_state"),
        "ready_for_repair_gate": payload.get("ready_for_repair_gate"),
        "repair_correctness_claimed": payload.get("repair_correctness_claimed"),
        "repair_gate_state": payload.get("repair_gate_state"),
    }


def receipt_gate_passed(payload: Mapping[str, Any]) -> bool:
    """Exp 3208 passes only when it explicitly allows the clean verifier rerun."""

    return payload.get("clean_rerun_allowed") is True


def clean_verifier_gate_passed(payload: Mapping[str, Any]) -> bool:
    """Exp 3209 passes only with clean state and no unhandled adversarial flag."""

    return payload.get("clean_verifier_state") == "clean" and not clean_verifier_has_unhandled_flag(
        payload
    )


def clean_verifier_has_unhandled_flag(payload: Mapping[str, Any]) -> bool:
    """Detect the adversarial-methodology signals that keep clean verifier evidence blocked."""

    if payload.get("flagged_adversarial") is True:
        return True
    if payload.get("unhandled_adversarial_methodology_flag") is True:
        return True
    if payload.get("adversarial_methodology_flag_unhandled") is True:
        return True
    unhandled_flags = payload.get("unhandled_adversarial_methodology_flags")
    if isinstance(unhandled_flags, list) and unhandled_flags:
        return True
    methodology_status = str(payload.get("adversarial_methodology_status") or "")
    if methodology_status in {"unhandled", "flagged", "blocked"}:
        return True
    flags = payload.get("adversarial_methodology_flags")
    if isinstance(flags, list):
        return any(isinstance(flag, Mapping) and flag.get("handled") is False for flag in flags)
    return False


def structured_proposal_gate_passed(payload: Mapping[str, Any]) -> bool:
    """Exp 3212 passes only when ready while making no repair-correctness claim."""

    return (
        payload.get("ready_for_repair_gate") is True
        and payload.get("repair_correctness_claimed") is False
    )


def auxiliary_invalidity(payload: Mapping[str, Any]) -> str | None:
    """Return an invalidity reason for present auxiliary fixture artifacts, if any."""

    if payload.get("artifact_invalid") is True:
        return "artifact_invalid=true"
    if payload.get("flagged_invalid") is True:
        return "flagged_invalid=true"
    if payload.get("valid") is False:
        return "valid=false"
    invalidity = payload.get("invalidity")
    return str(invalidity) if invalidity else None


def auxiliary_fixture_artifacts(
    sources: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Summarize auxiliary exact fixture coverage without making absence mandatory."""

    source_by_role = {str(row["role"]): row for row in sources}
    rows: list[JsonDict] = []
    for role, key in (
        ("exp3210_context_cot_clbench_parametric_shortcut_fixtures_v1", "exp3210"),
        ("exp3211_constraintbench_feasibility_objective_pilot_v1", "exp3211"),
    ):
        source = source_by_role[role]
        payload = payloads[key]
        rows.append(
            {
                "role": role,
                "path": source["path"],
                "present": source["present"],
                "readable_json_object": source["readable_json_object"],
                "ready_for_clean_verifier": payload.get("ready_for_clean_verifier"),
                "invalidity": auxiliary_invalidity(payload) if payload else None,
                "summary": artifact_summary(payload) if payload else {},
            }
        )
    return rows


def all_blockers(
    sources: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Collect all machine-readable blockers for the terminal decision."""

    blockers: list[JsonDict] = []
    blockers.extend(source_blockers(sources))
    blockers.extend(receipt_blockers(payloads["exp3208"]))
    blockers.extend(clean_verifier_blockers(payloads["exp3209"]))
    blockers.extend(structured_proposal_blockers(payloads["exp3212"]))
    blockers.extend(auxiliary_fixture_blockers(payloads))
    return blockers


def source_blockers(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Mandatory artifact rows must exist and parse as JSON before they can unlock repair."""

    blockers: list[JsonDict] = []
    for row in sources:
        if row.get("mandatory_for_gate") is not True:
            continue
        malformed = row.get("present") is True and row.get("readable_json_object") is not True
        if row.get("present") is not True or malformed:
            blockers.append(
                blocker(
                    "missing_mandatory_artifact",
                    str(row["path"]),
                    "present/readable_json_object",
                    True,
                    {
                        "present": row.get("present"),
                        "readable_json_object": row.get("readable_json_object"),
                    },
                    "mandatory repair-gate artifact is absent or malformed",
                )
            )
    return blockers


def receipt_blockers(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return Exp 3208 receipt blockers."""

    if receipt_gate_passed(payload):
        return []
    return [
        blocker(
            "exp3208_clean_rerun_not_allowed",
            EXP3208_REL_PATH.as_posix(),
            "clean_rerun_allowed",
            True,
            payload.get("clean_rerun_allowed"),
            payload.get("gate_check_summary")
            or "Exp 3208 did not explicitly allow the clean verifier rerun",
        )
    ]


def clean_verifier_blockers(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return Exp 3209 clean-verifier blockers."""

    blockers: list[JsonDict] = []
    if payload.get("clean_verifier_state") != "clean":
        blockers.append(
            blocker(
                "exp3209_clean_verifier_state_not_clean",
                EXP3209_REL_PATH.as_posix(),
                "clean_verifier_state",
                "clean",
                payload.get("clean_verifier_state"),
                "Exp 3209 did not report a clean verifier state",
            )
        )
    if clean_verifier_has_unhandled_flag(payload):
        blockers.append(
            blocker(
                "exp3209_unhandled_adversarial_methodology_flag",
                EXP3209_REL_PATH.as_posix(),
                "adversarial_methodology_flags",
                "none unhandled",
                {
                    "flagged_adversarial": payload.get("flagged_adversarial"),
                    "unhandled_adversarial_methodology_flags": payload.get(
                        "unhandled_adversarial_methodology_flags"
                    ),
                    "adversarial_methodology_status": payload.get("adversarial_methodology_status"),
                },
                "clean verifier evidence has an unhandled adversarial methodology flag",
            )
        )
    return blockers


def structured_proposal_blockers(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Return Exp 3212 structured proposal preflight blockers."""

    blockers: list[JsonDict] = []
    if payload.get("ready_for_repair_gate") is not True:
        blockers.append(
            blocker(
                "exp3212_ready_for_repair_gate_not_true",
                EXP3212_REL_PATH.as_posix(),
                "ready_for_repair_gate",
                True,
                payload.get("ready_for_repair_gate"),
                "structured proposal preflight is not ready for the repair gate",
            )
        )
    repair_claim = payload.get("repair_correctness_claimed")
    if repair_claim is not False:
        blockers.append(
            blocker(
                "exp3212_repair_correctness_claimed"
                if repair_claim is True
                else "exp3212_repair_correctness_claim_not_false",
                EXP3212_REL_PATH.as_posix(),
                "repair_correctness_claimed",
                False,
                repair_claim,
                "proposal preflight must not claim repair correctness",
            )
        )
    return blockers


def auxiliary_fixture_blockers(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Present auxiliary exact fixture artifacts block only when they flag invalidity."""

    blockers: list[JsonDict] = []
    for key, rel_path in (("exp3210", EXP3210_REL_PATH), ("exp3211", EXP3211_REL_PATH)):
        payload = payloads[key]
        invalidity = auxiliary_invalidity(payload) if payload else None
        if invalidity:
            blockers.append(
                blocker(
                    "auxiliary_fixture_invalid",
                    rel_path.as_posix(),
                    "invalidity",
                    None,
                    invalidity,
                    "auxiliary exact fixture artifact is present but flagged invalid",
                )
            )
    return blockers


def repair_gate_state(
    *,
    receipt_gate: bool,
    clean_gate: bool,
    proposal_gate: bool,
    blockers: Sequence[Mapping[str, Any]],
    any_mandatory_readable: bool,
) -> str:
    """Classify the terminal gate state from mandatory booleans and blockers."""

    if receipt_gate and clean_gate and proposal_gate and not blockers:
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
    """Create a stable blocker row that downstream gates can inspect."""

    return {
        "code": code,
        "source_artifact": str(source_artifact),
        "field": field,
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def inference_substrate() -> JsonDict:
    """Record that the decision used no live generation, verification, or repair."""

    return {
        "kind": "artifact_aggregation_only",
        "no_llm_calls": True,
        "live_model_calls": 0,
        "new_live_model_calls": 0,
        "executes_verifiers": False,
        "executes_repairs": False,
        "downloads_models": False,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Summarize the terminal decision without overstating repair correctness."""

    return (
        f"complete: repair_gate_state={artifact['repair_gate_state']}; "
        f"repair_ladder_allowed={str(artifact['repair_ladder_allowed']).lower()}; "
        f"blocker_count={len(artifact['blockers'])}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the v6 terminal artifact shape and gate invariants."""

    missing = set(REQUIRED_FIELD_NAMES) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    state = artifact.get("repair_gate_state")
    if state not in ALLOWED_REPAIR_GATE_STATES:
        raise ValueError(f"repair_gate_state must be one of {sorted(ALLOWED_REPAIR_GATE_STATES)}")
    gate_fields = (
        "receipt_gate_passed",
        "clean_verifier_gate_passed",
        "structured_proposal_gate_passed",
    )
    if any(not isinstance(artifact.get(field), bool) for field in gate_fields):
        raise ValueError("gate booleans must be bool values")
    blockers = artifact.get("blockers")
    if not isinstance(blockers, list):
        raise ValueError("blockers must be a list")
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
    if artifact.get("conductor_file_modified") is not False:
        raise ValueError("conductor_file_modified must remain false")
    if artifact.get("active_roadmap_modified") is not False:
        raise ValueError("active_roadmap_modified must remain false")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
