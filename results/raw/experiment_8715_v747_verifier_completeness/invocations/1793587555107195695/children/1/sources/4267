"""Archive milestone .345's fully-landed recovery and confirm .346 is active.

Spec refs: REQ-REPORT-3776, SCENARIO-REPORT-3776.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from scripts import adversarial_verify


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.345"
ACTIVATED_MILESTONE = "2026.06.346"
RANDOM_SEED = 3776
OUTPUT_REL_PATH = Path("results/experiment_3776_archive_v345_activate_v346.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
ROADMAP_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CAPSTONE_REL_PATH = Path("results/experiment_3775_capstone_v345.json")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: "
    "archived_v345_fully_landed_v346_convergence_active_paper_ready_true_"
    "both_energy_routes_bounded_frozen_headline_unchanged"
)
V345_OUTCOME_RECORDED = (
    "fully_landed_11_of_11_verifier_product_banked_certified_abstention_shipped_"
    "paper_ready_true_both_energy_routes_bounded"
)
V346_FOCUS_RECORDED = (
    "settle_p1_discrete_search_v3_bank_verifier_product_build_anomaly_escalation_"
    "scaffold_edlm_continue_self_learning_regrind_nothing_bounded"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v345_outcome_recorded",
    "v346_focus_recorded",
    "research_complete_yaml_parses",
    "paper_ready_preserved",
    "both_energy_routes_still_bounded",
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
    "v345_outcome_recorded": (
        "Records .345's REAL state: fully landed 11/11, verifier product "
        "banked, paper_ready TRUE."
    ),
    "v346_focus_recorded": (
        "Records the .346 pivot: settle P1 discrete-search v3, bank the verifier "
        "product, build Anomaly-Escalation, scaffold EDLM, continue self-learning "
        "-- re-grinding nothing bounded."
    ),
    "research_complete_yaml_parses": (
        "BARE bool, MUST be true -- confirms research-complete.yaml safe_loads "
        "after the write (anti-recurrence of the .344 poison)."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met (confirmed via publication_gate.py); the "
        "transition must not silently regress paper_ready; frozen 0.9131 stays frozen."
    ),
    "both_energy_routes_still_bounded": (
        "BARE bool -- records that .346 does not reopen the bounded conclusion "
        "(P1 v3 only sharpens the mechanism)."
    ),
    "n_tasks_archived": (
        "Sample-size hygiene -- confirms the full .345 milestone was archived, "
        "not a partial."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": "Content hash catches silent drift vs any replication.",
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}

V345_TASKS = [
    {
        "id": "exp3765-archive-v344-activate-v345",
        "title": "Archive .344 and activate .345 recovery",
        "deliverable": "results/experiment_3765_archive_v344_activate_v345.json",
        "result": (
            "COMPLETE: archived .344 zero-experiment skip cascade honestly and "
            "activated .345 recovery"
        ),
    },
    {
        "id": "exp3766-thesis-a-definitive-reconcile",
        "title": "Thesis-A definitive reconcile restored",
        "deliverable": "results/experiment_3766_thesis_a_definitive_reconcile.json",
        "result": (
            "COMPLETE: Thesis-A reconciled as PASS/discriminative for part-a and "
            "BOUNDED/not-generative for part-b"
        ),
    },
    {
        "id": "exp3767-g2-mechanical-reproducer",
        "title": "G2 local mechanical reproducer",
        "deliverable": "results/experiment_3767_g2_mechanical_reproducer.json",
        "result": (
            "COMPLETE: local FoVer reproducer landed within CI with AUROC 0.913134; "
            "frozen headline unchanged"
        ),
    },
    {
        "id": "exp3768-g3-narrowing-lint",
        "title": "G3 narrowing lint wired",
        "deliverable": "results/experiment_3768_g3_narrowing_lint.json",
        "result": "COMPLETE: narrowing lint extended, twelfth retraction added, pre-commit wired",
    },
    {
        "id": "exp3769-package-cli-mcp-e2e-smoke",
        "title": "Package CLI MCP E2E smoke",
        "deliverable": "results/experiment_3769_package_cli_mcp_e2e_smoke.json",
        "result": "COMPLETE: package import, pipeline, protocol exchange, and CLI smoke passed",
    },
    {
        "id": "exp3770-distribution-mirror-publish-checklist",
        "title": "Distribution mirror and publish checklist",
        "deliverable": "results/experiment_3770_distribution_mirror_publish_checklist.json",
        "result": (
            "COMPLETE: PyPI workflow, HF mirror, IPFS plan, and operator-only "
            "publish checklist ready; agent published nothing"
        ),
    },
    {
        "id": "exp3771-certified-abstention-operating-point",
        "title": "Certified abstention operating point",
        "deliverable": "results/experiment_3771_certified_abstention_operating_point.json",
        "result": (
            "COMPLETE: deployable certified abstention point shipped at threshold "
            "0.733216 with coverage 0.998218"
        ),
    },
    {
        "id": "exp3772-fr11-self-learning-v17-verifier-precision-tracker",
        "title": "FR-11 v17 verifier precision tracker",
        "deliverable": "results/experiment_3772_fr11_self_learning_v17_verifier_precision_tracker.json",
        "result": (
            "COMPLETE: Tier-1 verifier precision tracker preserved the memory "
            "contribution on the verifier product"
        ),
    },
    {
        "id": "exp3773-verifier-product-prm-positioning",
        "title": "Verifier product PRM positioning",
        "deliverable": "results/experiment_3773_verifier_product_prm_positioning.json",
        "result": (
            "COMPLETE: verifier product positioned honestly versus PRM SOTA; "
            "peer numbers treated as reported"
        ),
    },
    {
        "id": "exp3774-kv260-opportunistic-continuity-audit",
        "title": "KV260 opportunistic continuity audit",
        "deliverable": "results/experiment_3774_kv260_opportunistic_continuity_audit.json",
        "result": "COMPLETE: KV260 terminal state held; reachable and overlay loadable",
    },
    {
        "id": "exp3775-capstone-v345",
        "title": "Capstone .345",
        "deliverable": "results/experiment_3775_capstone_v345.json",
        "result": (
            "COMPLETE: capstone .345 aggregated all 11/11 tasks; paper_ready TRUE, "
            "FoVer 0.9131 frozen, both energy routes bounded"
        ),
    },
]


def build_research_complete_block() -> str:
    """Return the single honest `research-complete.yaml` block for .345."""

    finding = (
        "FULLY-LANDED MILESTONE: .345 had 11/11 tasks completed. The .344 skip "
        "cascade was recovered, Thesis-A closure was restored, G2/G3 gates were "
        "mechanized, package/CLI/MCP/distribution surfaces passed, the certified "
        "abstention point shipped, FR-11 v17 ran, PRM positioning landed, and "
        "KV260 terminal state held. The verifier product banked for ship, "
        "paper_ready TRUE (G1-G4), frozen FoVer 0.9131 unchanged, and both energy "
        "routes stayed bounded. .346 convergence agenda is active: settle P1 "
        "discrete-search v3, bank the verifier product surface, build "
        "Anomaly-Escalation, scaffold EDLM, continue self-learning, and re-grind "
        "nothing already bounded."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {yaml_string('V345 product-banking recovery fully landed and archived')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-04'",
        f"  finding: {yaml_string(finding)}",
        "  tasks:",
    ]
    for task in V345_TASKS:
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
    """Replace or append the `.345` archive block without duplicating it."""

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
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("- id: 2026.")
        ),
        len(lines),
    )
    return "\n".join([*lines[:start], *replacement, *lines[end:]]) + "\n"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    research_complete_yaml_parses: bool,
    publication_gate_report: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the Exp 3776 terminal artifact from checked-in evidence."""

    root_path = Path(root)
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    _ensure(active_milestone == ACTIVATED_MILESTONE, ".346 active milestone confirmation is required")

    roadmap_text = (root_path / active_roadmap_path).read_text(encoding="utf-8")
    design_text = (root_path / ROADMAP_DESIGN_REL_PATH).read_text(encoding="utf-8")
    roadmap_evidence = f"{roadmap_text}\n{design_text}".lower()
    _ensure(
        all(
            token in roadmap_evidence
            for token in ("2026.06.346", "p1", "anomaly-escalation", "edlm", "self-learning")
        ),
        ".346 roadmap evidence must record the convergence agenda",
    )
    _ensure(
        "re-grind" in roadmap_evidence or "regrind" in roadmap_evidence,
        ".346 roadmap evidence must record no bounded re-grind",
    )

    changelog_text = (root_path / CHANGELOG_REL_PATH).read_text(encoding="utf-8")
    _ensure(
        "Capstone .345" in changelog_text
        and ("11/11" in changelog_text or "fully landed" in changelog_text)
        and "2026.06.346" in changelog_text,
        "changelog must confirm .345 capstone and .346 planning evidence",
    )

    capstone = read_json_object(root_path / CAPSTONE_REL_PATH)
    v345_evidence = extract_v345_capstone_evidence(capstone)
    _ensure(v345_evidence["fully_landed"] is True, ".345 capstone must record fully landed 11/11 state")

    gate_report = (
        dict(publication_gate_report)
        if publication_gate_report is not None
        else evaluate_publication_gate(root_path)
    )
    paper_ready_evidence = extract_paper_ready_evidence(capstone, gate_report)
    _ensure(paper_ready_evidence["paper_ready"] is True, "publication gate must confirm paper_ready")

    paper_ready_preserved = (
        paper_ready_evidence["paper_ready"] is True
        and paper_ready_evidence["capstone_paper_ready"] is True
        and paper_ready_evidence["frozen_headline_unchanged"] is True
        and paper_ready_evidence["frozen_headline_auroc"] == 0.9131
        and all(paper_ready_evidence[gate] is True for gate in ("g1", "g2", "g3", "g4"))
    )
    both_energy_routes_still_bounded = (
        capstone.get("both_energy_routes_bounded") is True
        and capstone.get("energy_as_selector_status") == "honest-negative-bounded"
        and capstone.get("energy_as_generator_status") == "honest-negative-bounded"
    )
    _ensure(paper_ready_preserved, "paper_ready and frozen headline must be preserved")
    _ensure(both_energy_routes_still_bounded, "both energy routes must stay bounded")

    report = compact_verify_report(adversarial_report or {"flags": [], "flag_count": 0, "max_severity": -1})
    duration_s = duration_from(started_s, now_s)
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v345_to_v346_3776.v1",
        "experiment_id": "exp3776",
        "task_id": "exp3776-archive-v345-activate-v346",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "v346_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v345_outcome_recorded": V345_OUTCOME_RECORDED,
        "v346_focus_recorded": V346_FOCUS_RECORDED,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "paper_ready_preserved": paper_ready_preserved,
        "both_energy_routes_still_bounded": both_energy_routes_still_bounded,
        "n_tasks_archived": len(V345_TASKS),
        "adversarial_verify_clean": report_is_clean(report),
        "adversarial_verify_report": report,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "paper_ready_evidence": paper_ready_evidence,
        "v345_capstone_evidence": v345_evidence,
        "v346_activation_evidence": {
            "active_milestone": active_milestone,
            "active_roadmap_path": active_roadmap_path,
            "first_task": "exp3776-archive-v345-activate-v346",
            "convergence_focus": V346_FOCUS_RECORDED,
        },
        "source_artifact_checksums": [
            {"path": str(CAPSTONE_REL_PATH), "sha256": sha256_path(root_path / CAPSTONE_REL_PATH)}
        ],
        "source_document_checksums": [
            {"path": active_roadmap_path, "sha256": sha256_path(root_path / active_roadmap_path)},
            {
                "path": str(ROADMAP_DESIGN_REL_PATH),
                "sha256": sha256_path(root_path / ROADMAP_DESIGN_REL_PATH),
            },
            {"path": str(CHANGELOG_REL_PATH), "sha256": sha256_path(root_path / CHANGELOG_REL_PATH)},
            {"path": str(NORTH_STAR_REL_PATH), "sha256": sha256_path(root_path / NORTH_STAR_REL_PATH)},
        ],
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def run(
    root: Path | str = REPO_ROOT,
    *,
    publication_gate_report: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the honest `.345` archive and terminal Exp 3776 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    current_complete = complete_path.read_text(encoding="utf-8") if complete_path.exists() else ""
    rewritten = rewrite_research_complete(current_complete)
    _ensure(yaml_parses(rewritten), "rewritten research-complete.yaml must safe-load")
    complete_path.write_text(rewritten, encoding="utf-8")
    research_complete_parses = yaml_parses(complete_path.read_text(encoding="utf-8"))

    out_path = root_path / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_artifact(
        root_path,
        research_complete_yaml_parses=research_complete_parses,
        publication_gate_report=publication_gate_report,
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
    """Validate the required Exp 3776 archive/activation contract."""

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
    _ensure(artifact.get("v345_outcome_recorded") == V345_OUTCOME_RECORDED, ".345 outcome mismatch")
    _ensure(artifact.get("v346_focus_recorded") == V346_FOCUS_RECORDED, ".346 focus mismatch")
    _ensure(artifact.get("v346_active_confirmed") is True, ".346 active confirmation required")
    _ensure(artifact.get("research_complete_yaml_parses") is True, "safe-load confirmation required")
    _ensure(artifact.get("paper_ready_preserved") is True, "paper_ready must remain preserved")
    _ensure(
        artifact.get("both_energy_routes_still_bounded") is True,
        "both energy routes must remain bounded",
    )
    _ensure(artifact.get("n_tasks_archived") == 11, "n_tasks_archived must equal 11")
    _ensure(artifact.get("adversarial_verify_clean") is True, "adversarial_verify_clean must be true")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed must equal 3776")
    duration_s = artifact.get("duration_s")
    _ensure(
        isinstance(duration_s, int | float)
        and not isinstance(duration_s, bool)
        and float(duration_s) >= 0.0001,
        "duration_s must be numeric with the 0.0001s floor",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be a sha256 hex string")
    _ensure(checksum == payload_checksum(artifact), "reproducibility_checksum does not match artifact content")
    _ensure(report_is_clean(artifact.get("adversarial_verify_report", {"flags": []})), "adversarial report has critical flag")


def extract_v345_capstone_evidence(capstone: Mapping[str, Any]) -> JsonDict:
    """Extract the fully-landed `.345` facts from Exp 3775."""

    upstream_ids = capstone.get("headline_aggregation_experiment_ids")
    not_landed = capstone.get("not_landed_artifacts_recorded_honestly")
    flagged = capstone.get("flagged_artifacts_excluded")
    n_upstream = len(upstream_ids) if isinstance(upstream_ids, list) else 0
    fully_landed = bool(
        capstone.get("paper_ready_preserved") is True
        and capstone.get("both_energy_routes_bounded") is True
        and capstone.get("certified_abstention_point_status") == "shipped"
        and capstone.get("verifier_banked_for_ship") is True
        and capstone.get("frozen_headline_unchanged") is True
        and safe_point(capstone.get("frozen_fover_auroc")) == 0.9131
        and isinstance(not_landed, list)
        and not not_landed
        and isinstance(flagged, list)
        and not flagged
        and n_upstream == 10
    )
    return {
        "capstone_path": str(CAPSTONE_REL_PATH),
        "honest_verdict": capstone.get("honest_verdict"),
        "fully_landed": fully_landed,
        "n_upstream_tasks_landed": n_upstream,
        "n_tasks_with_capstone": 11 if fully_landed else n_upstream + 1,
        "certified_abstention_point_status": capstone.get("certified_abstention_point_status"),
        "verifier_product_banked": capstone.get("verifier_banked_for_ship") is True,
        "both_energy_routes_bounded": capstone.get("both_energy_routes_bounded") is True,
        "energy_as_selector_status": capstone.get("energy_as_selector_status"),
        "energy_as_generator_status": capstone.get("energy_as_generator_status"),
        "not_landed_artifacts_recorded_honestly": list(not_landed) if isinstance(not_landed, list) else None,
        "flagged_artifacts_excluded": list(flagged) if isinstance(flagged, list) else None,
    }


def extract_paper_ready_evidence(
    capstone: Mapping[str, Any],
    publication_gate_report: Mapping[str, Any],
) -> JsonDict:
    """Extract stable G1-G4 and frozen-headline evidence."""

    gates = publication_gate_report.get("gates")
    gates_map = gates if isinstance(gates, Mapping) else {}

    def gate_pass(name: str, fallback: str) -> bool:
        gate = gates_map.get(name)
        if isinstance(gate, Mapping):
            return gate.get("pass") is True
        return publication_gate_report.get(fallback) is True

    return {
        "publication_gate_source": str(publication_gate_report.get("__source__") or "provided_report"),
        "paper_ready": publication_gate_report.get("paper_ready") is True,
        "g1": gate_pass("G1", "g1"),
        "g2": gate_pass("G2", "g2"),
        "g3": gate_pass("G3", "g3"),
        "g4": gate_pass("G4", "g4"),
        "unmet_gates": list(publication_gate_report.get("unmet_gates") or []),
        "capstone_paper_ready": capstone.get("paper_ready_preserved") is True,
        "frozen_headline_auroc": safe_point(capstone.get("frozen_fover_auroc")),
        "frozen_headline_unchanged": capstone.get("frozen_headline_unchanged") is True,
    }


def evaluate_publication_gate(root: Path) -> JsonDict:
    """Run the stable publication gate and return its JSON report."""

    completed = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    _ensure(isinstance(report, dict), "publication_gate.py --json must return an object")
    report["__source__"] = "scripts/publication_gate.py --json"
    return report


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


def report_is_clean(report: Mapping[str, Any] | None) -> bool:
    """Return true when the adversarial report has no critical flag."""

    if not isinstance(report, Mapping):
        return True
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
