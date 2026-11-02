"""Archive milestone .344's SKIP cascade and confirm milestone .345 is active.

Spec refs: REQ-REPORT-3765, SCENARIO-REPORT-3765.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml

from scripts import adversarial_verify


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.344"
ACTIVATED_MILESTONE = "2026.06.345"
RANDOM_SEED = 3765
OUTPUT_REL_PATH = Path("results/experiment_3765_archive_v344_activate_v345.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
ROADMAP_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RETRO_REL_PATH = Path("results/operational_retro_2026_06_344.json")
CAPSTONE_REL_PATH = Path("results/experiment_3764_capstone_v344.json")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: "
    "archived_v344_zero_experiments_skip_cascade_recorded_v345_recovery_active_"
    "paper_ready_true_frozen_headline_unchanged"
)
V344_OUTCOME_RECORDED = (
    "zero_completed_experiments_whole_milestone_skip_cascade_agenda_carried_to_v345"
)
V344_SKIP_CAUSE_RECORDED = (
    "unquoted_embedded_colon_yaml_scannererror_test_public_docs_failed_skip_cascade"
)
V345_FOCUS_RECORDED = (
    "reexecute_unlanded_v344_agenda_reconcile_mechanize_gates_bank_verifier_"
    "certified_abstention_self_learning_prm_positioning"
)
UNLANDED_V344_AGENDA = [f"exp{experiment_id}" for experiment_id in range(3754, 3762)]
PARTIAL_ARTIFACTS = ["exp3762", "exp3763", "exp3764"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v344_outcome_recorded",
    "v344_skip_cause_recorded",
    "v345_focus_recorded",
    "research_complete_yaml_parses",
    "paper_ready_preserved",
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
    "v344_outcome_recorded": (
        "Records .344's REAL state: 0 completed experiments, whole-milestone "
        "SKIP cascade from a malformed-YAML poison; the agenda carried into .345."
    ),
    "v344_skip_cause_recorded": (
        "Records the ROOT CAUSE (unquoted embedded colon -> yaml.ScannerError -> "
        "test_public_docs_* failed -> SKIP cascade) so a future planner does not "
        "mis-read .344 as a research negative."
    ),
    "v345_focus_recorded": (
        "Records the .345 pivot: re-execute the un-landed .344 agenda "
        "(reconcile + mechanize-gates + bank-verifier + certified-abstention + "
        "self-learning + PRM-positioning)."
    ),
    "research_complete_yaml_parses": (
        "BARE bool, MUST be true -- confirms research-complete.yaml safe_loads "
        "after the write."
    ),
    "paper_ready_preserved": (
        "G1-G4 stay met; the transition must not silently regress paper_ready; "
        "frozen headline 0.9131 stays frozen."
    ),
    "n_tasks_archived": (
        "Sample-size hygiene -- confirms the full milestone was archived, not a partial."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": "Content hash catches silent drift vs any replication.",
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}

V344_TASKS = [
    {
        "id": "exp3754-archive-v343-activate-v344",
        "title": "Archive .343 and activate .344",
        "deliverable": "results/experiment_3754_archive_v343_activate_v344.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: YAML poison stopped the conductor before this "
            "archive task ran; carried into .345 as exp3765"
        ),
    },
    {
        "id": "exp3755-thesis-a-definitive-reconcile",
        "title": "Thesis-A definitive reconcile",
        "deliverable": "results/experiment_3755_thesis_a_definitive_reconcile.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: substantive Thesis-A reconcile never ran; "
            "carried into .345 as exp3766"
        ),
    },
    {
        "id": "exp3756-g2-mechanical-reproducer",
        "title": "G2 mechanical reproducer",
        "deliverable": "results/experiment_3756_g2_mechanical_reproducer.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: local FoVer reproducer never ran; carried "
            "into .345 as exp3767"
        ),
    },
    {
        "id": "exp3757-g3-narrowing-lint",
        "title": "G3 narrowing lint",
        "deliverable": "results/experiment_3757_g3_narrowing_lint.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: narrowing lint extension never ran; carried "
            "into .345 as exp3768"
        ),
    },
    {
        "id": "exp3758-package-cli-mcp-e2e-smoke",
        "title": "Package CLI MCP E2E smoke",
        "deliverable": "results/experiment_3758_package_cli_mcp_e2e_smoke.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: package CLI MCP smoke never ran; carried "
            "into .345 as exp3769"
        ),
    },
    {
        "id": "exp3759-distribution-mirror-publish-checklist",
        "title": "Distribution mirror and operator publish checklist",
        "deliverable": "results/experiment_3759_distribution_mirror_publish_checklist.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: distribution mirror checklist never ran; "
            "carried into .345 as exp3770"
        ),
    },
    {
        "id": "exp3760-certified-abstention-operating-point",
        "title": "Certified abstention operating point",
        "deliverable": "results/experiment_3760_certified_abstention_operating_point.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: certified abstention point never ran; carried "
            "into .345 as exp3771"
        ),
    },
    {
        "id": "exp3761-fr11-self-learning-v17-verifier-precision-tracker",
        "title": "FR-11 v17 verifier precision tracker",
        "deliverable": "results/experiment_3761_fr11_self_learning_v17_verifier_precision_tracker.json",
        "result": (
            "SKIPPED_BY_PRETEST_GATE: FR-11 verifier precision tracker never ran; "
            "carried into .345 as exp3772"
        ),
    },
    {
        "id": "exp3762-kv260-opportunistic-continuity-audit",
        "title": "KV260 opportunistic continuity audit",
        "deliverable": "results/experiment_3762_kv260_opportunistic_continuity_audit.json",
        "result": (
            "PARTIAL_ARTIFACT: KV260 audit artifact exists, but the .344 retro still "
            "records zero completed experiments under the cascade"
        ),
    },
    {
        "id": "exp3763-next-phase3-thesis-decision-menu",
        "title": "Next Phase-3 thesis decision menu",
        "deliverable": "results/experiment_3763_next_phase3_thesis_decision_menu.json",
        "result": (
            "PARTIAL_ARTIFACT: next-thesis menu artifact exists as an operator "
            "decision surface, not a self-seeded route"
        ),
    },
    {
        "id": "exp3764-capstone-v344",
        "title": "Capstone .344",
        "deliverable": "results/experiment_3764_capstone_v344.json",
        "result": "PARTIAL_ARTIFACT: exp3764 honestly reports missing upstreams",
    },
]


def build_research_complete_block() -> str:
    """Return the single honest `research-complete.yaml` block for .344."""

    finding = (
        "RECORD-HONEST SKIP cascade MILESTONE: .344 produced ZERO completed "
        "experiments. A malformed research-complete.yaml value with an unquoted "
        "embedded colon raised yaml.ScannerError, failed test_public_docs_* in "
        "the conductor pre-test gate, and SKIP-cascaded the whole milestone. "
        "Only exp3762, exp3763, and exp3764 left partial artifacts; exp3764 "
        "honestly reports both_energy_routes_bounded=false because exp3755 was "
        "missing. The substantive exp3754-exp3761 agenda never ran and is "
        "carried into .345 as recovery work, not a research negative. "
        "paper_ready stayed TRUE (G1-G4), and the frozen FoVer 0.9131 stayed frozen."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {yaml_string('V344 zero-experiment SKIP cascade archived')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-03'",
        f"  finding: {yaml_string(finding)}",
        "  tasks:",
    ]
    for task in V344_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {yaml_string(task['title'])}",
                f"    deliverable: {task['deliverable']}",
                f"    result: {yaml_string(task['result'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def rewrite_research_complete(text: str) -> str:
    """Replace or append the `.344` archive block without duplicating it."""

    block = build_research_complete_block()
    replacement = block.splitlines()
    if not text.strip():
        return "milestones:\n" + block

    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == f"- id: {ARCHIVED_MILESTONE}"),
        None,
    )
    if start is None:
        prefix = text.rstrip()
        if any(line.strip() == "milestones:" for line in lines):
            return f"{prefix}\n{block}"
        return f"{prefix}\nmilestones:\n{block}"

    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("- id: 2026.")),
        len(lines),
    )
    return "\n".join([*lines[:start], *replacement, *lines[end:]]) + "\n"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    research_complete_yaml_parses: bool,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the Exp 3765 terminal artifact from checked-in evidence."""

    root_path = Path(root)
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    _ensure(active_milestone == ACTIVATED_MILESTONE, ".345 active milestone confirmation is required")

    roadmap_text = (root_path / active_roadmap_path).read_text(encoding="utf-8")
    design_text = (root_path / ROADMAP_DESIGN_REL_PATH).read_text(encoding="utf-8")
    _ensure(
        all(
            token in f"{roadmap_text}\n{design_text}"
            for token in ("yaml.ScannerError", "SKIP", ".345", "reconcile")
        ),
        ".345 roadmap evidence must record the .344 skip cascade and recovery agenda",
    )

    retro = read_json_object(root_path / RETRO_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_REL_PATH)
    _ensure(retro.get("experiments_completed") == 0, ".344 retro must record zero completed experiments")
    missing_ids = sorted(
        item.get("experiment_id")
        for item in capstone.get("missing_upstream_artifacts", [])
        if isinstance(item, Mapping)
    )
    _ensure(
        missing_ids == list(range(3754, 3762))
        and capstone.get("both_energy_routes_bounded") is False,
        "partial capstone must record missing exp3754-exp3761 upstreams",
    )
    paper_ready_evidence = extract_paper_ready_evidence(capstone)
    paper_ready_preserved = (
        paper_ready_evidence["paper_ready"] is True
        and paper_ready_evidence["frozen_headline_unchanged"] is True
        and paper_ready_evidence["frozen_headline_auroc"] == 0.9131
        and all(paper_ready_evidence[gate] is True for gate in ("g1", "g2", "g3", "g4"))
    )
    _ensure(paper_ready_preserved, "partial capstone must preserve paper_ready and the frozen headline")

    report = compact_verify_report(adversarial_report or {"flags": [], "flag_count": 0, "max_severity": -1})
    duration_s = duration_from(started_s, now_s)
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v344_to_v345_3765.v1",
        "experiment_id": "exp3765",
        "task_id": "exp3765-archive-v344-activate-v345",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "v345_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v344_outcome_recorded": V344_OUTCOME_RECORDED,
        "v344_skip_cause_recorded": V344_SKIP_CAUSE_RECORDED,
        "v345_focus_recorded": V345_FOCUS_RECORDED,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "paper_ready_preserved": paper_ready_preserved,
        "n_tasks_archived": len(V344_TASKS),
        "adversarial_verify_clean": report_is_clean(report),
        "adversarial_verify_report": report,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "experiments_completed": retro.get("experiments_completed"),
        "partial_artifacts_recorded": list(PARTIAL_ARTIFACTS),
        "unlanded_v344_agenda_carried_to_v345": list(UNLANDED_V344_AGENDA),
        "v344_archive_evidence": {
            "operational_retro_path": str(RETRO_REL_PATH),
            "partial_capstone_path": str(CAPSTONE_REL_PATH),
            "missing_upstream_experiment_ids": missing_ids,
            "capstone_both_energy_routes_bounded": capstone.get("both_energy_routes_bounded"),
            "skip_cascade_pattern": "incident_agent_shipped_test_cascade",
        },
        "paper_ready_evidence": paper_ready_evidence,
        "source_artifact_checksums": [
            {"path": str(RETRO_REL_PATH), "sha256": sha256_path(root_path / RETRO_REL_PATH)},
            {"path": str(CAPSTONE_REL_PATH), "sha256": sha256_path(root_path / CAPSTONE_REL_PATH)},
        ],
        "source_document_checksums": [
            {"path": active_roadmap_path, "sha256": sha256_path(root_path / active_roadmap_path)},
            {
                "path": str(ROADMAP_DESIGN_REL_PATH),
                "sha256": sha256_path(root_path / ROADMAP_DESIGN_REL_PATH),
            },
            {"path": str(NORTH_STAR_REL_PATH), "sha256": sha256_path(root_path / NORTH_STAR_REL_PATH)},
        ],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def run(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the honest `.344` archive and terminal Exp 3765 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    rewritten = rewrite_research_complete(complete_path.read_text(encoding="utf-8"))
    _ensure(yaml_parses(rewritten), "rewritten research-complete.yaml must safe-load")
    complete_path.write_text(rewritten, encoding="utf-8")
    research_complete_parses = yaml_parses(complete_path.read_text(encoding="utf-8"))

    out_path = root_path / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_artifact(
        root_path,
        research_complete_yaml_parses=research_complete_parses,
        started_s=start,
        now_s=now_s,
    )
    write_payload(out_path, payload)

    verify_report = adversarial_verify.verify_artifact(out_path)
    payload["adversarial_verify_report"] = compact_verify_report(verify_report)
    payload["adversarial_verify_clean"] = report_is_clean(verify_report)
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    write_payload(out_path, payload)
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3765 archive/activation contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    _ensure(not missing_principles, f"missing field principles: {missing_principles}")
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(no_forbidden_markers(artifact), "artifact must not contain compute-bound markers")
    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "terminal verdict mismatch")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
    _ensure(artifact.get("v344_outcome_recorded") == V344_OUTCOME_RECORDED, ".344 outcome mismatch")
    _ensure(artifact.get("v344_skip_cause_recorded") == V344_SKIP_CAUSE_RECORDED, ".344 skip cause mismatch")
    _ensure(artifact.get("v345_focus_recorded") == V345_FOCUS_RECORDED, ".345 focus mismatch")
    _ensure(artifact.get("v345_active_confirmed") is True, ".345 active confirmation required")
    _ensure(artifact.get("research_complete_yaml_parses") is True, "safe-load confirmation required")
    _ensure(artifact.get("paper_ready_preserved") is True, "paper_ready must remain preserved")
    _ensure(artifact.get("n_tasks_archived") == 11, "n_tasks_archived must equal 11")
    _ensure(artifact.get("experiments_completed") == 0, "experiments_completed must remain 0")
    _ensure(artifact.get("partial_artifacts_recorded") == PARTIAL_ARTIFACTS, "partial artifact list mismatch")
    _ensure(
        artifact.get("unlanded_v344_agenda_carried_to_v345") == UNLANDED_V344_AGENDA,
        "unlanded .344 agenda mismatch",
    )
    _ensure(artifact.get("adversarial_verify_clean") is True, "adversarial_verify_clean must be true")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed must equal 3765")
    duration_s = artifact.get("duration_s")
    _ensure(
        isinstance(duration_s, int | float) and not isinstance(duration_s, bool) and float(duration_s) >= 0.0001,
        "duration_s must be numeric with the 0.0001s floor",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be a sha256 hex string")
    _ensure(checksum == payload_checksum(artifact), "reproducibility_checksum does not match artifact content")


def extract_paper_ready_evidence(capstone: Mapping[str, Any]) -> JsonDict:
    """Extract the stable G1-G4 and frozen-headline evidence from Exp 3764."""

    gate = capstone.get("publication_gate") if isinstance(capstone.get("publication_gate"), Mapping) else {}
    return {
        "g1": gate.get("g1") is True,
        "g2": gate.get("g2") is True,
        "g3": gate.get("g3") is True,
        "g4": gate.get("g4") is True,
        "paper_ready": capstone.get("paper_ready_preserved") is True or gate.get("paper_ready") is True,
        "frozen_headline_auroc": safe_point(capstone.get("frozen_fover_auroc")),
        "frozen_headline_unchanged": capstone.get("frozen_headline_unchanged") is True,
    }


def read_active_milestone(root: Path) -> tuple[str, str]:
    """Return the active milestone and roadmap path used for confirmation."""

    for rel_path in (ROADMAP_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            milestone = milestone_from_text(path.read_text(encoding="utf-8"))
            if milestone != "unknown":
                return milestone, str(rel_path)
    return "unknown", str(ROADMAP_REL_PATH)


def milestone_from_text(text: str) -> str:
    """Parse the first top-level `milestone:` value from roadmap YAML text."""

    for line in text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    _ensure(isinstance(payload, dict), f"expected JSON object in {path}")
    return payload


def yaml_parses(text: str) -> bool:
    """Return true when PyYAML can safe-load the provided text."""

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def yaml_string(value: str) -> str:
    """Render a YAML-safe quoted scalar using JSON string escaping."""

    return json.dumps(value)


def safe_point(value: Any) -> float | None:
    """Return a rounded float from either a number or a `{point: number}` object."""

    if isinstance(value, Mapping):
        return safe_point(value.get("point"))
    if isinstance(value, int | float) and not isinstance(value, bool):
        return round(float(value), 4)
    return None


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Compute a duration with the aggregation plausibility floor."""

    if started_s is None:
        return 0.0001
    end_s = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, end_s - float(started_s)), 6)


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep deterministic adversarial-verifier fields in the artifact."""

    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    severities = [severity_rank(flag.get("severity")) for flag in flags]
    return {
        "flag_count": len(flags),
        "max_severity": max(severities) if severities else -1,
        "flags": flags,
    }


def report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when the adversarial report has no critical flag."""

    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in report.get("flags", [])
    )


def severity_rank(severity: Any) -> int:
    """Map verifier severities to a stable integer order."""

    return {"info": 0, "warn": 1, "critical": 2}.get(str(severity).lower(), -1)


def sha256_path(path: Path) -> str:
    """Return a file checksum, using the repository's missing-file sentinel."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return hashlib.sha256(b"<missing>").hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the reproducibility checksum over payload content."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return true when the value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def no_forbidden_markers(value: Mapping[str, Any]) -> bool:
    """Return true when aggregation output did not copy live-compute markers."""

    encoded = json.dumps(value, sort_keys=True)
    return all(marker not in encoded for marker in ("GGUF", "CUDA", "live-model"))


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
