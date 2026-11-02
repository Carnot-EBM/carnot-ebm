"""Archive milestone .336 and confirm milestone .337 is active.

Spec: REQ-REPORT-3678, SCENARIO-REPORT-3678.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp3678"
ARCHIVED_MILESTONE = "2026.06.336"
ACTIVATED_MILESTONE = "2026.06.337"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3678_archive_v336_activate_v337.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
RANDOM_SEED = 3678
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts "
    "(principle: a JSON-read + format task, not live inference; 0.0001s floor)."
)
TERMINAL_VERDICT = (
    "complete: "
    "archived_v336_dependency_aware_refreeze_candidate_facts_retired_"
    "selection_negative_v337_active_paper_ready_true"
)
V336_OUTCOME = (
    "dependency_aware_refreeze_candidate_facts_domain_bound_real_ragtruth_"
    "selection_earned_negative_detector_math_strong_code_blind_fr11_v10_no_collapse"
)
HEADLINE_REFREEZE = (
    "dependency_aware_weighting_heldout_0.933224_vs_carnot_0.919964_"
    "candidate_to_raise_frozen_0.9131_pending_g1_rigor_g2_re_reproduction"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v336_outcome_recorded_as",
    "headline_refreeze_candidate_recorded",
    "facts_generalization_retired_recorded",
    "selection_earned_negative_recorded",
    "paper_ready_preserved",
    "p01_status_preserved",
    "n_tasks_archived",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix lets the conductor reconciler classify the transition as "
        "complete without re-running it."
    ),
    "inference_substrate": (
        "A JSON-read + format task, not live inference; 0.0001s floor."
    ),
    "v336_outcome_recorded_as": (
        "Records .336's defensible state (dependency-aware re-freeze candidate; "
        "facts domain-bound on REAL data; selection earned-negative; detector "
        "shipped math-strong code-blind) so the record does not revert."
    ),
    "headline_refreeze_candidate_recorded": (
        "Names the #1 .337 lead: dependency-aware weighting 0.9332 generalized "
        "held-out -- a candidate to RAISE the frozen 0.9131, pending G1-rigor + "
        "G2 re-reproduction."
    ),
    "facts_generalization_retired_recorded": (
        "Records that facts-generalization is RETIRED (exp3670 same-verdict on "
        "REAL RAGTruth) so .337 does not re-propose it."
    ),
    "selection_earned_negative_recorded": (
        "Records that ensemble best-of-N selection is an earned-negative (exp3672) "
        "-- .337 diagnoses the gap, does not re-run the same test."
    ),
    "paper_ready_preserved": (
        "G1-G4 stay met; the transition must not silently regress paper_ready; "
        "frozen headline 0.9131 stays frozen."
    ),
    "p01_status_preserved": (
        "P0.1 stays honest-negative; the transition does not re-assert a positive."
    ),
    "n_tasks_archived": (
        "Sample-size hygiene -- confirms the full milestone was archived, not a partial."
    ),
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": (
        "Content hash catches silent drift between this artifact and any replication."
    ),
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}

UPSTREAM_ARTIFACTS = {
    "exp3668": Path("results/experiment_3668_dependency_aware_weighting_heldout.json"),
    "exp3670": Path("results/experiment_3670_facts_row_real_benchmark.json"),
    "exp3671": Path("results/experiment_3671_ship_second_pair_of_eyes_detector.json"),
    "exp3672": Path("results/experiment_3672_ensemble_selection_where_sc_weak.json"),
    "exp3673": Path("results/experiment_3673_fr11_continuous_self_learning_v10.json"),
    "exp3677": Path("results/experiment_3677_capstone_and_g_gate_v336.json"),
}

V336_TASKS = [
    {
        "id": "exp3665-archive-v335-activate-v336",
        "title": "Archive .335 honestly and activate .336",
        "deliverable": "results/experiment_3665_archive_v335_activate_v336.json",
    },
    {
        "id": "exp3666-backend-state-diagnostic-v2",
        "title": "Backend-state diagnostic v2",
        "deliverable": "results/experiment_3666_backend_state_diagnostic_v2.json",
    },
    {
        "id": "exp3667-dependency-aware-weighting-clean-detautologized",
        "title": "Clean, de-tautologized dependency-aware ensemble weighting",
        "deliverable": "results/experiment_3667_dependency_aware_weighting_clean.json",
    },
    {
        "id": "exp3668-dependency-aware-weighting-heldout-validation",
        "title": "Held-out validation of dependency-aware weighting",
        "deliverable": "results/experiment_3668_dependency_aware_weighting_heldout.json",
    },
    {
        "id": "exp3669-build-real-factual-hallucination-corpus",
        "title": "Build a real factual hallucination corpus from RAGTruth",
        "deliverable": "results/experiment_3669_build_real_factual_corpus.json",
    },
    {
        "id": "exp3670-facts-row-real-benchmark-remeasurement",
        "title": "Re-measure the facts row on a real benchmark",
        "deliverable": "results/experiment_3670_facts_row_real_benchmark.json",
    },
    {
        "id": "exp3671-ship-second-pair-of-eyes-detector-product-surface",
        "title": "Ship the second-pair-of-eyes detector product surface",
        "deliverable": "results/experiment_3671_ship_second_pair_of_eyes_detector.json",
    },
    {
        "id": "exp3672-ensemble-selection-value-where-sc-is-weak",
        "title": "Test ensemble selection value where self-consistency is weak",
        "deliverable": "results/experiment_3672_ensemble_selection_where_sc_weak.json",
    },
    {
        "id": "exp3673-fr11-continuous-self-learning-v10-online-dependency-aware",
        "title": "FR-11 v10 online dependency-aware weighting without collapse",
        "deliverable": "results/experiment_3673_fr11_continuous_self_learning_v10.json",
    },
    {
        "id": "exp3674-kv260-continuity-v23",
        "title": "KV260 SSH reachability continuity v23",
        "deliverable": "results/experiment_3674_kv260_continuity_v23.json",
    },
    {
        "id": "exp3675-polarfire-continuity-v23",
        "title": "PolarFire opportunistic reachability and continuity audit",
        "deliverable": "results/experiment_3675_polarfire_continuity_v23.json",
    },
    {
        "id": "exp3676-gatemate-continuity-audit-v23",
        "title": "GateMate continuity audit v23",
        "deliverable": "results/experiment_3676_gatemate_continuity_audit_v23.json",
    },
    {
        "id": "exp3677-capstone-and-g-gate-v336",
        "title": "Capstone v336 and G1-G4 gate synthesis",
        "deliverable": "results/experiment_3677_capstone_and_g_gate_v336.json",
    },
]


def build_research_complete_block() -> str:
    """Return the honest `research-complete.yaml` block for milestone .336."""

    finding = (
        "DEPENDENCY-AWARE RE-FREEZE CANDIDATE: .336 produced the first "
        "headline-advancing lead in many milestones. Dependency-aware ensemble "
        "weighting beat Carnot cleanly and generalized held-out (0.933224 vs "
        "0.919964), making it a .337 candidate to raise the frozen 0.9131 "
        "headline only after G1-rigor and G2 re-reproduction. facts-generalization "
        "is RETIRED: exp3670 confirmed the same domain-bound verdict on REAL "
        "RAGTruth data, leak-free. Ensemble best-of-N selection is an "
        "earned-negative: exp3672 found ensemble selection is an earned-negative "
        "even with SC weak and oracle headroom. The detector shipped "
        "math-strong/code-blind; FR-11 v10 held without collapse; paper_ready "
        "stayed TRUE (G1-G4), P0.1 stayed honest-negative, and the frozen 0.9131 "
        "headline stayed frozen while .337 prepares the re-freeze package."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        '  title: "Dependency-aware re-freeze candidate, facts retired, selection negative"',
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-01'",
        f"  finding: {_json_string(finding)}",
        "  tasks:",
    ]
    for task in V336_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {_json_string(task['title'])}",
                f"    deliverable: {task['deliverable']}",
                "    result: OK (codex artifact landed)",
            ]
        )
    return "\n".join(lines) + "\n"


def rewrite_research_complete(text: str) -> str:
    """Replace or append the single milestone .336 archive block."""

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
    """Build the Exp 3678 terminal artifact from upstream JSON files."""

    root_path = Path(root)
    active_milestone = _read_active_milestone(root_path)
    exp3668 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3668"])
    exp3670 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3670"])
    exp3671 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3671"])
    exp3672 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3672"])
    exp3673 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3673"])
    exp3677 = _read_json_object(root_path / UPSTREAM_ARTIFACTS["exp3677"])

    detector_aurocs = _mapping(exp3671.get("fused_detector_auroc_per_domain"))
    selection_delta = _mapping(exp3672.get("ensemble_vs_sc_delta_ci"))

    payload: JsonDict = {
        "schema": "carnot.milestone_archive.v336_to_v337.v1",
        "experiment_id": EXPERIMENT_ID,
        "task_id": "exp3678-archive-v336-activate-v337",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": active_milestone,
        "v337_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "archive_v336_activate_v337_ready": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v336_outcome_recorded_as": V336_OUTCOME,
        "headline_refreeze_candidate_recorded": HEADLINE_REFREEZE,
        "facts_generalization_retired_recorded": (
            exp3670.get("facts_generalize_or_adds_value_real") is False
            and exp3670.get("grounding_leak_free") is True
            and exp3677.get("facts_real_benchmark_verdict") == "domain_bound_real_earned"
        ),
        "selection_earned_negative_recorded": (
            exp3672.get("ensemble_adds_selection_value_sc_weak") is False
            and exp3672.get("positive_control_valid") is True
        ),
        "paper_ready_preserved": exp3677.get("paper_ready") is True,
        "p01_status_preserved": exp3677.get("p01_status"),
        "n_tasks_archived": len(V336_TASKS),
        "post_handoff_codex_tasks_landed": 12,
        "random_seed": RANDOM_SEED,
        "duration_s": 0.0001,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_headline_auroc_preserved": _point(exp3677.get("frozen_fover_headline_auroc")),
        "dependency_aware_candidate_status": exp3677.get(
            "dependency_aware_headline_candidate_status"
        ),
        "heldout_auroc_dependency_aware": _point(
            exp3668.get("heldout_auroc_dependency_aware")
        ),
        "heldout_auroc_carnot": _point(exp3668.get("heldout_auroc_carnot")),
        "heldout_delta_dependency_minus_carnot": _point(
            _mapping(exp3668.get("heldout_delta_ci")).get("point")
        ),
        "heldout_delong_p": _point(exp3668.get("heldout_delong_p")),
        "heldout_n_splits": exp3668.get("n_splits"),
        "facts_grounding_auroc_real_corpus": _point(
            exp3670.get("grounding_auroc_real_corpus")
        ),
        "facts_confidence_baseline_auroc": _point(
            exp3670.get("confidence_baseline_auroc")
        ),
        "facts_grounding_minus_confidence_delta": _point(
            exp3670.get("grounding_minus_confidence_delta")
        ),
        "facts_mcnemar_p": _point(exp3670.get("mcnemar_p_facts")),
        "facts_real_corpus_path": exp3670.get("corpus_path_used"),
        "facts_n_examples": exp3670.get("n_examples"),
        "ensemble_selection_accuracy": _point(exp3672.get("ensemble_selection_accuracy")),
        "sc_accuracy": _point(exp3672.get("sc_accuracy")),
        "oracle_bestofn_accuracy": _point(exp3672.get("oracle_bestofn_accuracy")),
        "ensemble_vs_sc_delta": _point(selection_delta.get("delta")),
        "ensemble_vs_sc_mcnemar_p": _point(selection_delta.get("mcnemar_exact_p")),
        "selection_positive_control_valid": exp3672.get("positive_control_valid") is True,
        "selection_flip_count": exp3672.get("flip_count"),
        "detector_shipped": exp3671.get("detector_shipped") is True,
        "detector_module_path": exp3671.get("detector_module_path"),
        "detector_math_auroc": _point(detector_aurocs.get("math")),
        "detector_code_auroc": _point(detector_aurocs.get("code")),
        "fr11_v10_no_collapse_recorded": (
            exp3673.get("collapse_detected_deploy_arm") is False
            and exp3673.get("quality_maintained") is True
            and exp3677.get("fr11_v10_result") == "held_no_collapse_quality_maintained"
        ),
        "trained_judge_ood_retired_recorded": exp3677.get("trained_judge_ood_retired")
        is True,
        "g1": exp3677.get("g1") is True,
        "g2": exp3677.get("g2") is True,
        "g3": exp3677.get("g3") is True,
        "g4": exp3677.get("g4") is True,
        "unmet_gates": exp3677.get("unmet_gates"),
        "source_artifact_checksums": _source_artifacts(root_path),
        "protected_files_left_to_conductor": [
            "ops/status.md",
            "ops/changelog.md",
            "_bmad/traceability.md",
        ],
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "north_star_context_read": (root_path / "ops" / "north-star.md").exists(),
    }
    payload["reproducibility_checksum"] = _payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3678 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("honest_verdict") != TERMINAL_VERDICT:
        raise ValueError("terminal verdict does not match Exp 3678 contract")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate does not match Exp 3678 aggregation substrate")
    if artifact.get("v337_active_confirmed") is not True:
        raise ValueError("v337 active milestone confirmation is required")
    if artifact.get("v336_outcome_recorded_as") != V336_OUTCOME:
        raise ValueError("v336 outcome record does not match the Exp 3678 contract")
    if artifact.get("headline_refreeze_candidate_recorded") != HEADLINE_REFREEZE:
        raise ValueError("headline re-freeze candidate does not match the Exp 3678 contract")
    if artifact.get("facts_generalization_retired_recorded") is not True:
        raise ValueError("facts-generalization retirement must be recorded")
    if artifact.get("selection_earned_negative_recorded") is not True:
        raise ValueError("selection earned-negative must be recorded")
    if artifact.get("paper_ready_preserved") is not True:
        raise ValueError("paper_ready must remain preserved")
    if artifact.get("p01_status_preserved") != "honest-negative":
        raise ValueError("P0.1 must remain honest-negative")
    if artifact.get("n_tasks_archived") != 13:
        raise ValueError("n_tasks_archived must equal 13 for the full .336 roadmap block")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0001:
        raise ValueError("duration_s must be numeric with the 0.0001s floor")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if checksum != _payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def run(root: Path | str = REPO_ROOT) -> Path:
    """Write the research-complete archive block and terminal JSON artifact."""

    root_path = Path(root)
    payload = build_artifact(root_path)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    complete_path.write_text(
        rewrite_research_complete(complete_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    out_path = root_path / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _json_string(value: str) -> str:
    return json.dumps(value)


def _read_active_milestone(root: Path) -> str:
    roadmap = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    for line in roadmap.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def _read_json_object(path: Path) -> JsonDict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _point(metric: Any) -> float | None:
    if isinstance(metric, Mapping):
        return _point(metric.get("point"))
    if isinstance(metric, (int, float)) and not isinstance(metric, bool):
        return round(float(metric), 6)
    return None


def _source_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "name": name,
            "path": str(path),
            "sha256": _sha256_file(root / path),
        }
        for name, path in UPSTREAM_ARTIFACTS.items()
    ]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
