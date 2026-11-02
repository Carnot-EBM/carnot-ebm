"""Build the Exp 3150 adversarial verifier-evidence corrigendum.

Spec refs: REQ-VERIFY-3150, SCENARIO-VERIFY-3150.

This module is a gate audit, not a verifier rerun. It reads the checked-in
.292 evidence chain and separates deterministic exact-replay recovery from
live-inference claims whose authenticity evidence is incomplete. That
distinction matters because a repair gate should be allowed to consume exact
false-accept regression rows while refusing to unlock repair from live metrics
that lack transcripts, seeds, checksums, or coherent model-load evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3150_adversarial_verifier_evidence_corrigendum_v1"
SCHEMA = "carnot.adversarial_verifier_evidence_corrigendum.v1"
OUTPUT_REL_PATH = Path("results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3150_adversarial_verifier_evidence_corrigendum_v1.py"

EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path(
    "results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json"
)
EXP3139_REL_PATH = Path("results/experiment_3139_live_sota_verifier_rerun_v7.json")
EXP3140_REL_PATH = Path("results/experiment_3140_repair_gate_unlock_decision_v1.json")
EXP3147_REL_PATH = Path("results/experiment_3147_cross_corpus_matrix_v26.json")
EXP3148_REL_PATH = Path("results/experiment_3148_capstone_v292.json")

SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
ALLOWED_REPAIR_IMPLICATIONS = {
    "blocked_pending_clean_rerun",
    "exact_recovery_only_no_repair_unlock",
    "clean_rerun_required_before_unlock",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3150_adversarial_verifier_evidence_corrigendum_v1.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3150_adversarial_verifier_evidence_corrigendum_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/adversarial_verifier_evidence_corrigendum_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    role: str
    required: bool
    source_type: str = "json"


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("agents_repo_instructions", Path("AGENTS.md"), "repo_instructions", False, "md"),
    SourceSpec("codex_repo_workflow", Path("CODEX.md"), "codex_workflow", False, "md"),
    SourceSpec("claude_authenticity_rules", Path("CLAUDE.md"), "authenticity_rules", False, "md"),
    SourceSpec("research_references", Path("research-references.md"), "research_context", False, "md"),
    SourceSpec(
        "verifier_authenticity_lint",
        Path("scripts/verifier_authenticity_lint.py"),
        "authenticity_lint",
        False,
        "py",
    ),
    SourceSpec(
        "harness_fit_lint",
        Path("scripts/harness_fit_lint.py"),
        "harness_fit_lint",
        False,
        "py",
    ),
    SourceSpec(
        "verification_openspec",
        Path("openspec/capabilities/verification/spec.md"),
        "verification_spec",
        True,
        "md",
    ),
    SourceSpec("exp3136", EXP3136_REL_PATH, "false_accept_autopsy", True),
    SourceSpec("exp3137", EXP3137_REL_PATH, "exact_safe_contract", True),
    SourceSpec("exp3138", EXP3138_REL_PATH, "canonical_grounding", True),
    SourceSpec("exp3139", EXP3139_REL_PATH, "live_verifier_rerun", True),
    SourceSpec("exp3140", EXP3140_REL_PATH, "repair_gate_decision", True),
    SourceSpec("exp3147", EXP3147_REL_PATH, "matrix_v26", True),
    SourceSpec("exp3148", EXP3148_REL_PATH, "capstone_v292", True),
    SourceSpec(
        "exp3150_module",
        Path("python/carnot/verify/adversarial_verifier_evidence_corrigendum_v1.py"),
        "corrigendum_module",
        False,
        "py",
    ),
    SourceSpec(
        "exp3150_script",
        Path("scripts/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.py"),
        "corrigendum_script",
        False,
        "py",
    ),
    SourceSpec(
        "exp3150_tests",
        Path("tests/python/test_experiment_3150_adversarial_verifier_evidence_corrigendum_v1.py"),
        "corrigendum_tests",
        False,
        "py",
    ),
)

AUDITED_EXPERIMENTS = {"exp3136", "exp3137", "exp3138", "exp3139", "exp3140", "exp3147", "exp3148"}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, treating malformed files as absent evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a reproducible checksum for source traceability."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration(started_s: float, now_s: float) -> float:
    """Clamp elapsed time so a clock anomaly cannot create negative evidence."""

    return round(max(0.0, float(now_s) - float(started_s)), 6)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3150: build the corrigendum from checked-in artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = source_artifacts(root_path)
    payloads = {
        source.experiment_id: read_json_object(root_path / source.path)
        for source in SOURCE_SPECS
        if source.source_type == "json"
    }
    errors = source_errors(sources)
    audited = audited_artifacts(payloads, sources)
    counts = adversarial_flag_counts(audited)
    flagged_count = sum(1 for row in audited if row["flagged"] is True)
    exact_recovery = known_false_accept_recovery_preserved(payloads)
    live_trusted = live_verifier_evidence_trusted(payloads.get("exp3139", {}), audited)
    repair_implication = repair_gate_implication(payloads.get("exp3140", {}), exact_recovery, live_trusted)
    safe_fields = safe_downstream_fields(payloads, exact_recovery, live_trusted, repair_implication)
    blocked_fields = blocked_downstream_fields(payloads, live_trusted, repair_implication)
    sanity_rows = sanity_check_table(payloads, audited, exact_recovery, live_trusted)
    ready = (
        not errors
        and flagged_count > 0
        and exact_recovery
        and repair_implication in ALLOWED_REPAIR_IMPLICATIONS
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "adversarial_corrigendum_v1_ready": ready,
        "audited_artifacts": audited,
        "flagged_artifact_count": flagged_count,
        "adversarial_flag_counts": counts,
        "sanity_check_table": sanity_rows,
        "safe_downstream_fields": safe_fields,
        "blocked_downstream_fields": blocked_fields,
        "known_false_accept_recovery_preserved": exact_recovery,
        "live_verifier_evidence_trusted": live_trusted,
        "repair_gate_implication": repair_implication,
        "methodology_requirements_for_rerun": methodology_requirements_for_rerun(),
        "source_artifacts": sources,
        "source_errors": errors,
        "source_checksums": {row["path"]: row.get("sha256") for row in sources if row.get("sha256")},
        "inference_substrate": inference_substrate(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, time.perf_counter() if now_s is None else float(now_s)),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3150 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return all files the corrigendum reads or cites."""

    rows: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        payload = read_json_object(path) if spec.source_type == "json" else {}
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "role": spec.role,
                "required": spec.required,
                "source_type": spec.source_type,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if spec.source_type == "json" else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def source_errors(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """List missing required files before any downstream gate can be trusted."""

    errors: list[JsonDict] = []
    for row in sources:
        if row.get("required") is not True:
            continue
        if row.get("present") is not True:
            errors.append(
                {
                    "experiment_id": str(row.get("experiment_id") or ""),
                    "path": str(row.get("path") or ""),
                    "reason": "missing_required_source",
                }
            )
        elif row.get("source_type") == "json" and row.get("readable_json_object") is not True:
            errors.append(
                {
                    "experiment_id": str(row.get("experiment_id") or ""),
                    "path": str(row.get("path") or ""),
                    "reason": "malformed_required_json",
                }
            )
    return errors


def audited_artifacts(
    payloads: Mapping[str, Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Classify direct and inherited evidence flags for audited experiments."""

    source_by_id = {str(row.get("experiment_id")): row for row in sources}
    rows: list[JsonDict] = []
    for experiment_id in sorted(AUDITED_EXPERIMENTS):
        payload = payloads.get(experiment_id, {})
        source = source_by_id.get(experiment_id, {})
        direct_counts = direct_flag_counts(payload)
        derived_counts = derived_flag_counts(experiment_id, payload)
        inherited = inherited_flagged_source(experiment_id, payload)
        kinds = sorted({*direct_counts, *derived_counts, *(["aggregate_inherited_flag"] if inherited else [])})
        direct_flagged = payload.get("flagged_adversarial") is True or bool(_list(payload.get("corrigendum_pending")))
        flagged = bool(kinds) or direct_flagged
        flag_source = "clean"
        if direct_flagged and inherited:
            flag_source = "direct_and_inherited"
        elif direct_flagged:
            flag_source = "direct"
        elif inherited:
            flag_source = "inherited"
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": str(source.get("path") or ""),
                "role": str(source.get("role") or ""),
                "present": source.get("present") is True,
                "readable_json_object": source.get("readable_json_object") is True,
                "flagged": flagged,
                "flag_source": flag_source,
                "flag_kinds": kinds,
                "direct_flag_counts": direct_counts,
                "derived_flag_counts": derived_counts,
                "inherited_flag": inherited,
                "safe_gate_role": safe_gate_role(experiment_id, payload, flagged, inherited),
                "honest_verdict": str(payload.get("honest_verdict") or ""),
            }
        )
    return rows


def direct_flag_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    """Count adversarial flags already recorded on an artifact."""

    counts: dict[str, int] = {}
    for flag in _list(payload.get("corrigendum_pending")):
        if not isinstance(flag, Mapping):
            continue
        normalized = normalize_flag_kind(str(flag.get("kind") or flag.get("reason") or ""))
        if normalized:
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def derived_flag_counts(experiment_id: str, payload: Mapping[str, Any]) -> dict[str, int]:
    """Add flags that the prior linter could not express row-locally."""

    counts: dict[str, int] = {}
    if experiment_id == "exp3139":
        if _live_payload_missing_transcript(payload):
            counts["missing_transcript_evidence"] = 1
        if _live_payload_missing_seed_or_checksum(payload):
            counts["missing_seed_or_checksum"] = 1
        if _live_payload_has_inconsistent_model_load(payload):
            counts["inconsistent_model_load_evidence"] = 1
    return counts


def inherited_flagged_source(experiment_id: str, payload: Mapping[str, Any]) -> bool:
    """Return whether an aggregate artifact inherits flagged source evidence."""

    if experiment_id == "exp3140":
        return any("flagged" in text or "corrigendum" in text for text in _combined_text(payload))
    if experiment_id == "exp3147":
        recovery = _mapping(payload.get("false_accept_recovery_summary"))
        return (
            _int(recovery.get("flagged_adversarial_artifact_count")) > 0
            or _int(recovery.get("corrigendum_pending_count")) > 0
            or "adversarial" in str(recovery.get("recovery_claim_status") or "")
        )
    if experiment_id == "exp3148":
        return any("flagged" in text or "adversarial" in text for text in _combined_text(payload))
    return False


def adversarial_flag_counts(audited: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Aggregate failure classes without merging distinct causes."""

    counts: dict[str, int] = {}
    for row in audited:
        for class_counts in (row.get("direct_flag_counts"), row.get("derived_flag_counts")):
            if not isinstance(class_counts, Mapping):
                continue
            for key, value in class_counts.items():
                counts[str(key)] = counts.get(str(key), 0) + _int(value)
        if row.get("inherited_flag") is True:
            counts["aggregate_inherited_flag"] = counts.get("aggregate_inherited_flag", 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def normalize_flag_kind(kind: str) -> str:
    """Map historical verifier flag labels to corrigendum classes."""

    upper = kind.upper()
    if "TAUTOLOGY" in upper:
        return "tautology"
    if "DURATION" in upper and "TOO" in upper and "SHORT" in upper:
        return "duration_too_short"
    if "METHODOLOGY" in upper and "MISSING" in upper:
        return "methodology_missing"
    if "TRANSCRIPT" in upper:
        return "missing_transcript_evidence"
    return ""


def known_false_accept_recovery_preserved(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return whether exact replay blocked all known false-accept rows."""

    contract = payloads.get("exp3137", {})
    grounding = payloads.get("exp3138", {})
    regression_rows = _list(contract.get("regression_row_set"))
    return (
        contract.get("acceptance_contract_v1_ready") is True
        and contract.get("known_false_accept_rows_blocked") is True
        and _float(contract.get("replay_false_accept_rate")) == 0.0
        and grounding.get("canonical_grounding_pilot_v1_ready") is True
        and _int(grounding.get("false_accept_rows_blocked")) >= len(regression_rows) > 0
        and _list(grounding.get("residual_false_accept_rows")) == []
    )


def live_verifier_evidence_trusted(payload: Mapping[str, Any], audited: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether live fields have enough authenticity evidence for gates."""

    row = next((item for item in audited if item.get("experiment_id") == "exp3139"), {})
    if row.get("flagged") is True:
        return False
    if payload.get("live_verifier_rerun_v7_ready") is not True:
        return False
    if _int(payload.get("live_call_count")) <= 0:
        return False
    return (
        not _live_payload_missing_transcript(payload)
        and not _live_payload_missing_seed_or_checksum(payload)
        and not _live_payload_has_inconsistent_model_load(payload)
    )


def repair_gate_implication(
    repair_gate: Mapping[str, Any],
    exact_recovery: bool,
    live_trusted: bool,
) -> str:
    """Return the machine-readable effect on downstream repair unlocks."""

    if not exact_recovery or not live_trusted:
        return "blocked_pending_clean_rerun"
    if str(repair_gate.get("repair_gate_state") or "") != "unblocked":
        return "exact_recovery_only_no_repair_unlock"
    return "clean_rerun_required_before_unlock"


def safe_downstream_fields(
    payloads: Mapping[str, Mapping[str, Any]],
    exact_recovery: bool,
    live_trusted: bool,
    repair_implication: str,
) -> list[str]:
    """List fields that downstream gates may consume without overstating evidence."""

    fields: list[str] = []
    if exact_recovery:
        fields.extend(
            [
                "exp3136.false_accept_row_ids",
                "exp3136.regression_row_set",
                "exp3137.known_false_accept_rows_blocked",
                "exp3137.replay_false_accept_rate",
                "exp3137.regression_row_set",
                "exp3138.false_accept_rows_blocked",
                "exp3138.residual_false_accept_rows",
            ]
        )
    if live_trusted:
        fields.extend(
            [
                "exp3139.false_accept_rate",
                "exp3139.false_reject_rate",
                "exp3139.regression_rows_included",
            ]
        )
    if repair_implication != "clean_rerun_required_before_unlock" and _mapping(
        payloads.get("exp3140", {})
    ).get("repair_gate_state") != "unblocked":
        fields.append("exp3140.repair_gate_state_blocked_only")
    return sorted(dict.fromkeys(fields))


def blocked_downstream_fields(
    payloads: Mapping[str, Mapping[str, Any]],
    live_trusted: bool,
    repair_implication: str,
) -> list[str]:
    """List fields that must not unlock repair or headline claims."""

    blocked = [
        "exp3139.abstention_rate",
        "exp3139.headline_claim_allowed",
        "exp3139.live_verifier_rerun_v7_ready",
        "exp3139.verifier_gain_delta",
        "exp3147.false_accept_recovery_summary.rerun_verifier_gain_delta",
        "exp3148.verifier_claim_status",
        "exp3140.live_model_ready",
        "exp3140.repair_gate_state_for_unlock",
    ]
    if not live_trusted:
        blocked.extend(
            [
                "exp3139.false_accept_rate",
                "exp3139.false_accept_gate_passed",
                "exp3139.false_reject_rate",
                "exp3140.false_accept_gate_passed",
                "exp3140.false_accept_rate",
            ]
        )
    if repair_implication == "clean_rerun_required_before_unlock":
        repair_gate = _mapping(payloads.get("exp3140", {}))
        if repair_gate.get("repair_gate_state") == "unblocked":
            blocked.remove("exp3140.repair_gate_state_for_unlock")
    return sorted(dict.fromkeys(blocked))


def sanity_check_table(
    payloads: Mapping[str, Mapping[str, Any]],
    audited: Sequence[Mapping[str, Any]],
    exact_recovery: bool,
    live_trusted: bool,
) -> list[JsonDict]:
    """Build sanity checks adapted from hallucination-detector audits."""

    live = _mapping(payloads.get("exp3139", {}))
    direct_tautology = adversarial_flag_counts(audited).get("tautology", 0)
    aggregate_inherited = adversarial_flag_counts(audited).get("aggregate_inherited_flag", 0)
    return [
        {
            "check": "field_provenance",
            "principle": "claims must name their source fields",
            "status": "passed" if all(row.get("path") for row in audited) else "blocked",
            "evidence": "audited artifacts record local paths and source roles",
            "safe_for_downstream": True,
        },
        {
            "check": "non_tautological_recomputation",
            "principle": "a recomputation must not grade itself by equality alone",
            "status": "blocked" if direct_tautology else "passed",
            "evidence": f"tautology_flags={direct_tautology}",
            "safe_for_downstream": direct_tautology == 0,
        },
        {
            "check": "adversarial_regression_rows",
            "principle": "known false accepts must stay in the denominator",
            "status": "passed" if exact_recovery else "blocked",
            "evidence": "Exp 3137/3138 replay blocks the known regression rows",
            "safe_for_downstream": exact_recovery,
        },
        {
            "check": "methodology_completeness",
            "principle": "live claims need seeds, checksums, transcripts, and model-load evidence",
            "status": "passed" if live_trusted else "blocked",
            "evidence": methodology_gap_summary(live),
            "safe_for_downstream": live_trusted,
        },
        {
            "check": "aggregate_source_trust",
            "principle": "aggregate trust requires source trust",
            "status": "blocked" if aggregate_inherited else "passed",
            "evidence": f"aggregate_inherited_flagged_artifacts={aggregate_inherited}",
            "safe_for_downstream": aggregate_inherited == 0,
        },
    ]


def methodology_gap_summary(live: Mapping[str, Any]) -> str:
    """Summarize why live verifier evidence is or is not complete."""

    gaps: list[str] = []
    if _live_payload_missing_seed_or_checksum(live):
        gaps.append("missing seed/checksum")
    if _live_payload_missing_transcript(live):
        gaps.append("missing transcript evidence")
    if _live_payload_has_inconsistent_model_load(live):
        gaps.append("inconsistent model-load evidence")
    if not gaps:
        return "complete live methodology evidence"
    return ", ".join(gaps)


def methodology_requirements_for_rerun() -> list[str]:
    """State the concrete requirements a clean live rerun must satisfy."""

    return [
        "record random_seed or random_seeds_used for every live row",
        "record reproducibility_checksum over prompts, raw outputs, model specs, and runner revision",
        "record transcript path and sha256 for every live call",
        "record model-load evidence that agrees with live_call_count and selected_model_ids",
        "record non-tautological metric recomputation from row-level denominators",
        "include all known false-accept regression row IDs with exact labels",
        "report real wall-clock duration for the selected inference substrate",
    ]


def safe_gate_role(
    experiment_id: str,
    payload: Mapping[str, Any],
    flagged: bool,
    inherited: bool,
) -> str:
    """Describe how each artifact may be consumed downstream."""

    if experiment_id in {"exp3137", "exp3138"} and not flagged:
        return "safe_exact_replay_gate"
    if experiment_id == "exp3140" and inherited:
        return "safe_as_block_only"
    if flagged or inherited:
        return "blocked_for_unlock"
    if payload:
        return "supporting_context"
    return "missing"


def inference_substrate() -> JsonDict:
    """Declare that Exp 3150 performs aggregation and lint only."""

    return {
        "kind": "aggregation_from_checked_in_verifier_evidence",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
        "declares_no_new_live_model_inference": True,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict with the required success/block prefix."""

    if artifact.get("adversarial_corrigendum_v1_ready") is True:
        return (
            "complete: adversarial_corrigendum_v1_ready=true; "
            f"repair_gate_implication={artifact.get('repair_gate_implication')}; "
            f"live_verifier_evidence_trusted={artifact.get('live_verifier_evidence_trusted')}"
        )
    errors = _list(artifact.get("source_errors"))
    if errors:
        return f"blocked_missing_sources: {len(errors)} required source artifact(s) unavailable"
    return "blocked_pending_clean_rerun: adversarial corrigendum preconditions not met"


def _live_payload_missing_transcript(payload: Mapping[str, Any]) -> bool:
    if _int(payload.get("live_call_count")) <= 0:
        return False
    transcript_paths = _list(payload.get("live_transcript_paths")) or _list(
        payload.get("transcript_paths")
    )
    transcript_hashes = payload.get("transcript_sha256s") or payload.get("transcript_sha256")
    return not transcript_paths or transcript_hashes in (None, "", [], {})


def _live_payload_missing_seed_or_checksum(payload: Mapping[str, Any]) -> bool:
    if not _live_claims_model_execution(payload):
        return False
    has_seed = payload.get("random_seed") is not None or bool(_list(payload.get("random_seeds_used")))
    has_checksum = bool(payload.get("reproducibility_checksum"))
    return not (has_seed and has_checksum)


def _live_payload_has_inconsistent_model_load(payload: Mapping[str, Any]) -> bool:
    if not _live_claims_model_execution(payload):
        return False
    substrate = _mapping(payload.get("inference_substrate"))
    gpu_preflight = _mapping(substrate.get("gpu_preflight"))
    return gpu_preflight.get("no_model_loaded") is True or gpu_preflight.get("no_inference_run") is True


def _live_claims_model_execution(payload: Mapping[str, Any]) -> bool:
    substrate = _mapping(payload.get("inference_substrate"))
    return (
        substrate.get("executes_models") is True
        or _int(payload.get("live_call_count")) > 0
        or _int(substrate.get("live_model_calls")) > 0
    )


def _combined_text(payload: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in (
        "repair_blockers",
        "headline_disqualifiers",
        "what_stayed_blocked",
        "what_292_proved",
        "false_accept_recovery_status",
        "live_verifier_status",
        "verifier_claim_status",
        "repair_gate_status",
        "repair_claim_status",
    ):
        value = payload.get(key)
        if isinstance(value, str):
            texts.append(value.lower())
        else:
            texts.extend(str(item).lower() for item in _list(value))
    return texts


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
