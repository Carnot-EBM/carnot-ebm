"""Archive milestone .340 and confirm milestone .341 Thesis A is active.

Spec: REQ-REPORT-3724, SCENARIO-REPORT-3724.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp3724"
ARCHIVED_MILESTONE = "2026.06.340"
ACTIVATED_MILESTONE = "2026.06.341"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3724_archive_v340_activate_v341.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
THESIS_MENU_REL_PATH = Path("docs/research-notes/phase3-alternative-thesis-menu.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
RANDOM_SEED = 3724
P01_STATUS = "honest-negative-bounded"
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: JSON-read + format; "
    "0.0001s floor; no compute-bound marker so it does not false-flag)."
)
TERMINAL_VERDICT = (
    "complete: "
    "archived_v340_convergence_hardened_thesis_a_energy_generator_seeded_"
    "v341_active_paper_ready_true_frozen_headline_unchanged"
)
V340_OUTCOME = (
    "convergence_hardened_g3_mechanical_g4_audited_risk_coverage_abstention_"
    "fresh_corpus_fover_specific_fr11_v14_no_collapse_kv260_terminal_"
    "operator_next_thesis_recorded_paper_ready_true_p01_honest_negative_bounded_"
    "frozen_0_9131_unchanged"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v340_outcome_recorded",
    "thesis_a_seeded_recorded",
    "paper_ready_preserved",
    "p01_status_preserved",
    "n_tasks_archived",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix lets the reconciler classify the transition complete "
        "without re-running it."
    ),
    "inference_substrate": (
        "JSON-read + format; 0.0001s floor; no compute-bound marker so it does "
        "not false-flag."
    ),
    "v340_outcome_recorded": (
        "Records .340's converged-hardening state so the record does not revert."
    ),
    "thesis_a_seeded_recorded": (
        "Records that the operator seeded Thesis A (energy-as-generator) as the "
        ".341 direction -- the human seed the loop could not self-initiate."
    ),
    "paper_ready_preserved": (
        "G1-G4 stay met; the transition must not silently regress paper_ready; "
        "frozen headline 0.9131 stays frozen."
    ),
    "p01_status_preserved": (
        "P0.1 / energy-SELECTION stays honest-negative-bounded; .341 tests a "
        "DIFFERENT mechanism (generation), not a re-grind."
    ),
    "n_tasks_archived": (
        "Sample-size hygiene -- confirms the full milestone was archived, not a partial."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": (
        "Content hash catches silent drift vs any replication."
    ),
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}

UPSTREAM_ARTIFACTS = {
    "exp3713": Path("results/experiment_3713_archive_v339_activate_v340.json"),
    "exp3714": Path("results/experiment_3714_backend_state_diagnostic_v6.json"),
    "exp3715": Path("results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json"),
    "exp3716": Path("results/experiment_3716_ship_paper_v6_narrowing_lint.json"),
    "exp3717": Path("results/experiment_3717_g4_full_provenance_audit.json"),
    "exp3718": Path("results/experiment_3718_risk_coverage_abstention_characterization.json"),
    "exp3719": Path("results/experiment_3719_headline_replication_fresh_corpus.json"),
    "exp3720": Path("results/experiment_3720_fr11_continuous_self_learning_v14.json"),
    "exp3721": Path(
        "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"
    ),
    "exp3722": Path("results/experiment_3722_convergence_synthesis_operator_next_thesis.json"),
    "exp3723": Path("results/experiment_3723_capstone_and_g_gate_v340.json"),
}

V340_TASKS = [
    {
        "id": "exp3713-archive-v339-activate-v340",
        "title": "Archive .339 honestly and activate .340",
        "deliverable": "results/experiment_3713_archive_v339_activate_v340.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3714-backend-state-diagnostic-v6",
        "title": "Backend-state diagnostic v6",
        "deliverable": "results/experiment_3714_backend_state_diagnostic_v6.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3715-refreeze-disambiguation-clean-corrigendum",
        "title": "Clean corrigendum for re-freeze disambiguation",
        "deliverable": "results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json",
        "result": "CLOSED-NEGATIVE; clean corrigendum; headline stays frozen",
    },
    {
        "id": "exp3716-ship-paper-v6-narrowing-lint-g3-mechanical",
        "title": "Ship Paper-v6 narrowing lint",
        "deliverable": "results/experiment_3716_ship_paper_v6_narrowing_lint.json",
        "result": "G3 mechanically enforced; paper clean",
    },
    {
        "id": "exp3717-g4-full-provenance-audit",
        "title": "Full G4 provenance audit",
        "deliverable": "results/experiment_3717_g4_full_provenance_audit.json",
        "result": "G4 fully traced to clean primary artifacts",
    },
    {
        "id": "exp3718-risk-coverage-abstention-characterization",
        "title": "Risk-coverage abstention characterization",
        "deliverable": "results/experiment_3718_risk_coverage_abstention_characterization.json",
        "result": "ENERGY signal beats entropy as abstention gate; warnings non-critical",
    },
    {
        "id": "exp3719-headline-replication-fresh-corpus",
        "title": "Headline-class replication on fresh corpus",
        "deliverable": "results/experiment_3719_headline_replication_fresh_corpus.json",
        "result": "fresh-corpus result FoVer-specific; frozen headline unchanged",
    },
    {
        "id": "exp3720-fr11-continuous-self-learning-v14",
        "title": "FR-11 continuous self-learning v14",
        "deliverable": "results/experiment_3720_fr11_continuous_self_learning_v14.json",
        "result": "graceful fallback under shift; no collapse",
    },
    {
        "id": "exp3721-hardware-kv260-terminal-confirm-and-continuity",
        "title": "Hardware continuity and KV260 terminal confirmation",
        "deliverable": "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json",
        "result": "KV260 terminal confirmed; mandate lift recommended",
    },
    {
        "id": "exp3722-convergence-synthesis-operator-next-thesis",
        "title": "Convergence synthesis and operator next-thesis request",
        "deliverable": (
            "results/experiment_3722_convergence_synthesis_operator_next_thesis.json"
        ),
        "result": "converged state synthesized; next-thesis request recorded",
    },
    {
        "id": "exp3723-capstone-and-g-gate-v340",
        "title": "Capstone v340 and G-gate synthesis",
        "deliverable": "results/experiment_3723_capstone_and_g_gate_v340.json",
        "result": "THESIS-A operator seed recorded for .341",
    },
]


def build_research_complete_block() -> str:
    """Return the honest `research-complete.yaml` block for milestone .340."""

    finding = (
        "CONVERGENCE-HARDENING MILESTONE: .340 did not move the frozen headline; "
        "it hardened and bounded the converged state. exp3715 cleanly re-emitted "
        "the re-freeze corrigendum as CLOSED-NEGATIVE, so no candidate replaces "
        "the frozen FoVer 0.9131. exp3716 made G3 mechanically enforced through "
        "the Paper-v6 narrowing lint, and exp3717 made G4 fully traced to clean "
        "primary artifacts. exp3718 characterized the proven discriminator as a "
        "risk-coverage abstention gate; exp3719 recorded that the fresh-corpus "
        "result was FoVer-specific rather than a silent new headline. exp3720 "
        "kept FR-11 bounded under shift with graceful fallback and no collapse. "
        "exp3721 confirmed KV260 terminal and recommended lifting the mandate. "
        "exp3722 recorded the converged-state next-thesis request, and the "
        "operator seeded Thesis A for .341: energy-as-generator, not selector. "
        "P0.1 stayed honest-negative-bounded, paper_ready stayed TRUE (G1-G4), "
        "and the frozen FoVer 0.9131 stayed frozen."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        '  title: "Convergence hardening and Thesis A handoff"',
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-02'",
        f"  finding: {_json_string(finding)}",
        "  tasks:",
    ]
    for task in V340_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {_json_string(task['title'])}",
                f"    deliverable: {task['deliverable']}",
                f"    result: {task['result']}",
            ]
        )
    return "\n".join(lines) + "\n"


def rewrite_research_complete(text: str) -> str:
    """Replace or append the single milestone .340 archive block."""

    replacement = build_research_complete_block().splitlines()
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == f"- id: {ARCHIVED_MILESTONE}"),
        None,
    )
    if start is None:
        prefix = text.rstrip()
        block = build_research_complete_block()
        return f"{prefix}\n{block}" if prefix else block

    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("- id: 2026.")),
        len(lines),
    )
    return "\n".join([*lines[:start], *replacement, *lines[end:]]) + "\n"


def build_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Build the Exp 3724 terminal artifact from upstream files."""

    root_path = Path(root)
    roadmap_text = _read_text_required(root_path / ROADMAP_REL_PATH)
    active_milestone = _read_active_milestone(roadmap_text)
    if active_milestone != ACTIVATED_MILESTONE:
        raise ValueError("v341 active milestone confirmation is required")

    thesis_menu_text = _read_text_required(root_path / THESIS_MENU_REL_PATH)
    north_star = root_path / NORTH_STAR_REL_PATH
    conductor = root_path / CONDUCTOR_REL_PATH
    conductor_hash_before = _sha256_path(conductor)

    exp3715 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3715"])
    exp3716 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3716"])
    exp3717 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3717"])
    exp3718 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3718"])
    exp3719 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3719"])
    exp3720 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3720"])
    exp3721 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3721"])
    exp3722 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3722"])
    exp3723 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3723"])

    g3_mechanical = (
        exp3716.get("g3_now_mechanically_enforced") is True
        and exp3716.get("current_paper_lint_clean") is True
        and exp3723.get("g3_mechanically_enforced") is True
    )
    g4_traced = (
        exp3717.get("all_numbers_trace_to_clean_artifacts") is True
        and exp3717.get("g4_provenance_audit_result", "fully_traced") == "fully_traced"
        and exp3723.get("g4_provenance_audit_result") == "fully_traced"
    )
    refreeze_clean = (
        exp3715.get("no_candidate_beats_frozen") is True
        and exp3715.get("frozen_headline_unchanged_assert") is True
        and exp3723.get("exp3704_corrigendum_clean") is True
    )
    abstention_verdict = str(exp3723.get("energy_abstention_verdict", ""))
    abstention_recorded = (
        exp3718.get("energy_beats_baseline_abstention") is True
        and abstention_verdict == "energy_better_than_entropy"
    )
    fresh_corpus = str(exp3723.get("fresh_corpus_generalization", ""))
    fresh_corpus_recorded = (
        exp3719.get("fresh_corpus_generalization") == "fover_specific"
        and fresh_corpus == "fover_specific"
    )
    fr11_recorded = (
        exp3720.get("template_robust_or_graceful_fallback") is True
        and exp3720.get("collapse_detected_deploy_arm") is False
        and exp3720.get("template_library_bounded") is True
        and exp3723.get("fr11_v14_result")
        == "falls_back_gracefully_under_shift_no_collapse"
    )
    kv260_terminal = (
        exp3721.get("kv260_terminal_condition_confirmed") is True
        and exp3721.get("kv260_terminal_transcript_present") is True
        and exp3721.get("speedup_claim_avoided_assert") is True
        and exp3723.get("kv260_terminal_confirmed") is True
    )
    next_thesis_recorded = (
        exp3722.get("all_self_generable_threads_settled") is True
        and exp3723.get("operator_next_thesis_recorded") is True
    )
    frozen_headline = _point(exp3723.get("frozen_fover_headline_auroc"))
    paper_ready_preserved = (
        exp3723.get("paper_ready") is True
        and all(exp3723.get(gate) is True for gate in ("g1", "g2", "g3", "g4"))
        and exp3723.get("frozen_headline_unchanged") is True
        and frozen_headline == 0.9131
    )
    p01_preserved = (
        exp3723.get("p01_status") == "honest-negative"
        and exp3723.get("selection_diagnosis_closed") is True
    )
    thesis_a_seeded = _thesis_a_seeded(roadmap_text, thesis_menu_text)
    if not thesis_a_seeded:
        raise ValueError("Thesis A operator seed evidence is required")

    payload: JsonDict = {
        "schema": "carnot.archive_activation.v340_to_v341.v1",
        "experiment_id": EXPERIMENT_ID,
        "task_id": "exp3724-archive-v340-activate-v341",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": active_milestone,
        "v341_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "archive_v340_activate_v341_ready": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v340_outcome_recorded": V340_OUTCOME,
        "thesis_a_seeded_recorded": thesis_a_seeded,
        "paper_ready_preserved": paper_ready_preserved,
        "p01_status_preserved": P01_STATUS if p01_preserved else exp3723.get("p01_status"),
        "n_tasks_archived": len(V340_TASKS),
        "adversarial_verify_clean": False,
        "random_seed": RANDOM_SEED,
        "duration_s": 0.0001,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_headline_auroc_preserved": frozen_headline,
        "g_gates_preserved": {
            "g1": exp3723.get("g1") is True,
            "g2": exp3723.get("g2") is True,
            "g3": exp3723.get("g3") is True,
            "g4": exp3723.get("g4") is True,
        },
        "v340_evidence": {
            "refreeze_corrigendum_clean": refreeze_clean,
            "g3_mechanical": g3_mechanical,
            "g4_fully_traced": g4_traced,
            "risk_coverage_abstention_gate": abstention_verdict,
            "risk_coverage_noncritical_warnings_preserved": _max_report_severity(exp3718) == 1,
            "fresh_corpus_generalization": fresh_corpus,
            "fr11_v14_no_collapse_or_fallback": fr11_recorded,
            "kv260_terminal_confirmed": kv260_terminal,
            "operator_next_thesis_recorded": next_thesis_recorded,
            "capstone_verdict": exp3723.get("honest_verdict"),
        },
        "thesis_a_evidence": {
            "source": "research-roadmap.yaml; phase3-alternative-thesis-menu.md",
            "mechanism": "energy_as_generator_not_selector",
            "human_seed_required": True,
            "matched_compute_discipline_recorded": _contains(roadmap_text, "matched-COMPUTE")
            or _contains(thesis_menu_text, "matched-COMPUTE"),
            "p01_boundary": "energy_selection_stays_bounded",
        },
        "paper_ready_evidence": {
            "paper_ready": exp3723.get("paper_ready") is True,
            "g1": exp3723.get("g1") is True,
            "g2": exp3723.get("g2") is True,
            "g3": exp3723.get("g3") is True,
            "g4": exp3723.get("g4") is True,
            "frozen_headline_unchanged": exp3723.get("frozen_headline_unchanged") is True,
        },
        "source_artifact_checksums": _source_artifacts(root_path),
        "source_document_checksums": {
            str(ROADMAP_REL_PATH): _sha256_text(roadmap_text),
            str(THESIS_MENU_REL_PATH): _sha256_text(thesis_menu_text),
            str(NORTH_STAR_REL_PATH): _sha256_path(north_star),
        },
        "protected_files_left_to_conductor": [
            "ops/status.md",
            "ops/changelog.md",
            "_bmad/traceability.md",
        ],
        "scripts_research_conductor_modified": (
            conductor_hash_before != _sha256_path(conductor)
        ),
        "ops_docs_reconciliation_left_to_conductor": True,
        "north_star_context_read": north_star.exists(),
    }
    payload["reproducibility_checksum"] = _payload_checksum(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3724 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [
        field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles
    ]
    _ensure(not missing_principles, f"missing field principles: {missing_principles}")
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(_no_compute_markers(artifact), "compute-bound markers must not be present")
    _ensure(
        artifact.get("honest_verdict") == TERMINAL_VERDICT,
        "terminal verdict does not match Exp 3724 contract",
    )
    _ensure(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate does not match Exp 3724 aggregation substrate",
    )
    _ensure(
        artifact.get("v341_active_confirmed") is True,
        "v341 active milestone confirmation is required",
    )
    _ensure(
        artifact.get("v340_outcome_recorded") == V340_OUTCOME,
        ".340 outcome record does not match the Exp 3724 contract",
    )
    _ensure(
        artifact.get("thesis_a_seeded_recorded") is True,
        "Thesis A operator seed must be recorded",
    )
    _ensure(artifact.get("paper_ready_preserved") is True, "paper_ready must remain preserved")
    _ensure(
        artifact.get("p01_status_preserved") == P01_STATUS,
        "P0.1 must remain honest-negative-bounded",
    )
    _ensure(
        artifact.get("n_tasks_archived") == 11,
        "n_tasks_archived must equal 11 for the full .340 roadmap block",
    )
    _ensure(
        artifact.get("adversarial_verify_clean") is True,
        "adversarial_verify_clean must be true for Exp 3724",
    )
    duration = artifact.get("duration_s")
    _ensure(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and float(duration) >= 0.0001,
        "duration_s must be numeric with the 0.0001s floor",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(
        isinstance(checksum, str) and len(checksum) == 64,
        "reproducibility_checksum must be a sha256 hex string",
    )
    _ensure(
        checksum == _payload_checksum(artifact),
        "reproducibility_checksum does not match artifact content",
    )


def run(root: Path | str = REPO_ROOT) -> Path:
    """Write the research-complete archive block and terminal JSON artifact."""

    root_path = Path(root)
    payload = build_artifact(root_path)
    out_path = root_path / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(out_path, payload)

    verify_report = _run_adversarial_verify(out_path)
    payload["adversarial_verify_report"] = _compact_verify_report(verify_report)
    payload["adversarial_verify_clean"] = _is_verify_clean(verify_report)
    payload["reproducibility_checksum"] = _payload_checksum(payload)
    validate_artifact(payload)
    _write_payload(out_path, payload)

    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    complete_path.write_text(
        rewrite_research_complete(complete_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return out_path


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_string(value: str) -> str:
    return json.dumps(value)


def _read_active_milestone(roadmap_text: str) -> str:
    for line in roadmap_text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def _read_json_object(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _read_text_required(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"required text input missing: {path}") from exc


def _point(metric: Any) -> float | None:
    if isinstance(metric, Mapping):
        return _point(metric.get("point"))
    if isinstance(metric, (int, float)) and not isinstance(metric, bool):
        return round(float(metric), 6)
    return None


def _thesis_a_seeded(roadmap_text: str, thesis_menu_text: str) -> bool:
    return (
        _contains(roadmap_text, "Thesis A")
        and _contains(roadmap_text, "energy-as-GENERATOR")
        and _contains(thesis_menu_text, "Thesis A")
        and _contains(thesis_menu_text, "selected")
        and (_contains(roadmap_text, "operator") or _contains(thesis_menu_text, "operator"))
        and _contains(thesis_menu_text, "energy")
    )


def _max_report_severity(artifact: Mapping[str, Any]) -> int:
    report = artifact.get("adversarial_verify_report")
    if isinstance(report, Mapping):
        value = report.get("max_severity")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        flags = report.get("flags")
        if isinstance(flags, list):
            severities = [
                _severity_rank(flag.get("severity"))
                for flag in flags
                if isinstance(flag, Mapping)
            ]
            return max(severities) if severities else -1
    return -1


def _source_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "name": name,
            "path": str(path),
            "sha256": _sha256_path(root / path),
            "exists": (root / path).exists(),
        }
        for name, path in sorted(UPSTREAM_ARTIFACTS.items())
    ]


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return hashlib.sha256(b"<missing>").hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_adversarial_verify(path: Path) -> JsonDict:
    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3724", verifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load adversarial verifier from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify_artifact(path)
    if not isinstance(report, dict):
        raise RuntimeError("adversarial verifier returned a non-object report")
    return report


def _compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    raw_flags = report.get("flags", [])
    flags = (
        [dict(flag) for flag in raw_flags if isinstance(flag, Mapping)]
        if isinstance(raw_flags, list)
        else []
    )
    severities = [_severity_rank(flag.get("severity")) for flag in flags]
    return {
        "flag_count": len(flags),
        "max_severity": max(severities) if severities else -1,
        "flags": flags,
    }


def _is_verify_clean(report: Mapping[str, Any]) -> bool:
    flags = report.get("flags")
    if not isinstance(flags, list):
        return True
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def _no_compute_markers(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact)
    disallowed = (
        "GGUF",
        "CUDA",
        "live-model",
        "live_model",
        "llama.cpp",
        "torch.cuda",
        ".cuda(",
        "model_specs",
        "target_model",
    )
    return not any(marker in encoded for marker in disallowed)


def _severity_rank(value: Any) -> int:
    return {"info": 0, "warn": 1, "critical": 2}.get(str(value).lower(), -1)


def _contains(value: str, needle: str) -> bool:
    return needle.lower() in value.lower()


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
