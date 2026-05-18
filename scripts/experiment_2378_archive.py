"""Generate the Exp 2378 archive artifact for milestone 2026.05.231.

Spec: REQ-REPORT-2378, SCENARIO-REPORT-2378.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "carnot.archive_activation.v1"
EXPERIMENT_ID = "exp2378-archive-and-activate"
ARCHIVE_MILESTONE = "2026.05.231"
ACTIVE_MILESTONE = "2026.05.232"
COMPLETED = "2026-05-18"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2378_archive.json")

MILESTONE_231_RESULT_SUMMARY = [
    {"id": "exp2364", "result": "archive_ready=true"},
    {
        "id": "exp2365",
        "result": "fst_live_validated=true (PATH C cached telemetry - first FST artifact!)",
    },
    {
        "id": "exp2366",
        "result": "nsvif_real_validated=true, nsvif_real_verification_pass_rate<1.0 (real data)",
    },
    {
        "id": "exp2367",
        "result": "eidoku_real_validated=true, ebm_cot_real_auroc<1.0 (real data)",
    },
    {
        "id": "exp2368",
        "result": "laab_k17_validated=true (Tier 0h LaaB logical consistency verifier)",
    },
    {
        "id": "exp2369",
        "result": "spilled_energy_k18_validated=true (Tier 0i k=18 variant)",
    },
    {
        "id": "exp2370",
        "result": "multi_verifier_comparison_complete=true (ensemble AUROC vs HalluScan 0.88)",
    },
    {"id": "exp2371", "result": "lagonn_validated=true (Kuramoto constraint satisfaction)"},
    {
        "id": "exp2372",
        "result": "rtl_lint_passed=true, lint_errors_count=0 (KV260 RTL CLEAN - first time!)",
    },
    {
        "id": "exp2373",
        "result": "nsvif_compliance_validated=true (financial regulatory text extension)",
    },
    {
        "id": "exp2374",
        "result": "kancl_hard_validated=true, kancl_hard_forgetting_reduction_pct<100.0",
    },
    {
        "id": "exp2375",
        "result": "fr11_real_csl_passed=true, cross_domain_retention_rate>=0.60 (real data)",
    },
    {"id": "exp2376", "result": "capstone_complete=true"},
    {"id": "exp2377", "result": "retro_complete=true"},
]

MILESTONE_231_ENTRY = """- id: 2026.05.231
  title: 'Real-Data Validation Sprint: Adversarial Stress Tests + FST Live Gen v11 + New Verifiers (k=17, k=18)'
  doc: openspec/change-proposals/research-roadmap-v231.md
  completed: '2026-05-18'
  finding: See conductor log for per-experiment results.
  tasks:
  - id: exp2364-archive-and-activate
    title: 'Phase 0: Archive .230 and activate .231'
    deliverable: results/experiment_2364_archive.json
    result: OK (conductor) - archive_ready=true
  - id: exp2365-fst-live-gen-v11
    title: 'Phase 1: FST Live Generation v11'
    deliverable: results/experiment_2365_fst_live_gen.json
    result: OK (conductor) - fst_live_validated=true (PATH C cached telemetry - first FST artifact!)
  - id: exp2366-nsvif-verge-real-data
    title: 'Phase 2: NSVIF+VERGE Real-Data Adversarial Stress Test'
    deliverable: results/experiment_2366_nsvif_verge_real.json
    result: OK (conductor) - nsvif_real_validated=true, nsvif_real_verification_pass_rate<1.0 (real data)
  - id: exp2367-eidoku-ebm-cot-real-data
    title: 'Phase 2: Eidoku CSP + EBM-CoT Real-Data Adversarial Stress Test'
    deliverable: results/experiment_2367_eidoku_ebm_cot_real.json
    result: OK (conductor) - eidoku_real_validated=true, ebm_cot_real_auroc<1.0 (real data)
  - id: exp2368-laab-k17-verifier
    title: 'Phase 3: LaaB Logical Consistency Verifier k=17'
    deliverable: results/experiment_2368_laab_k17.json
    result: OK (conductor) - laab_k17_validated=true (Tier 0h LaaB logical consistency verifier)
  - id: exp2369-spilled-energy-k18
    title: 'Phase 3: SpilledEnergy k=18 Variant'
    deliverable: results/experiment_2369_spilled_energy_k18.json
    result: OK (conductor) - spilled_energy_k18_validated=true (Tier 0i k=18 variant)
  - id: exp2370-multi-verifier-comparison
    title: 'Phase 3: Multi-Verifier Comparison'
    deliverable: results/experiment_2370_multi_verifier_comparison.json
    result: OK (conductor) - multi_verifier_comparison_complete=true (ensemble AUROC vs HalluScan 0.88)
  - id: exp2371-lagonn-constraint-satisfaction
    title: 'Phase 4: LagONN Deterministic Constraint Satisfaction'
    deliverable: results/experiment_2371_lagonn.json
    result: OK (conductor) - lagonn_validated=true (Kuramoto constraint satisfaction)
  - id: exp2372-kv260-rtl-lint-fix
    title: 'Phase 4: KV260 RTL Lint Fix'
    deliverable: results/experiment_2372_kv260_rtl_fix.json
    result: OK (conductor) - rtl_lint_passed=true, lint_errors_count=0 (KV260 RTL CLEAN - first time!)
  - id: exp2373-nsvif-compliance-domain
    title: 'Phase 4: NSVIF Compliance Domain Extension'
    deliverable: results/experiment_2373_nsvif_compliance.json
    result: OK (conductor) - nsvif_compliance_validated=true (financial regulatory text extension)
  - id: exp2374-kancl-hard-domains
    title: 'Phase 5: KAN-CL Hard Domain Adversarial Stress Test'
    deliverable: results/experiment_2374_kancl_hard_domains.json
    result: OK (conductor) - kancl_hard_validated=true, kancl_hard_forgetting_reduction_pct<100.0
  - id: exp2375-fr11-fst-real-csl
    title: 'Phase 5: FR-11 FST Real-Data Cross-Domain Retention'
    deliverable: results/experiment_2375_fr11_real_csl.json
    result: OK (conductor) - fr11_real_csl_passed=true, cross_domain_retention_rate>=0.60 (real data)
  - id: exp2376-capstone-v231
    title: 'Phase 6: Capstone v231'
    deliverable: results/experiment_2376_capstone.json
    result: OK (conductor) - capstone_complete=true
  - id: exp2377-retro-v231
    title: 'Phase 7: Milestone 2026.05.231 Operational Retrospective'
    deliverable: results/experiment_2377_retro.json
    result: OK (conductor) - retro_complete=true
"""

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix required for conductor reconciler. Must start complete:."
    ),
    "archive_ready": "Boolean gate: true only when .231 appears in research-complete.yaml.",
    "milestone_archived": (
        "Records which milestone was archived (2026.05.231) for audit traceability."
    ),
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _strip_yaml_scalar(value: str) -> str:
    return value.strip().strip("'\"")


def _roadmap_milestone(path: Path) -> str:
    lines = [line for line in _read_text(path).splitlines()[:5] if line.startswith("milestone:")]
    return _strip_yaml_scalar(lines[0].split(":", 1)[1]) if lines else ""


def _find_milestone_line(text: str, milestone: str) -> int | None:
    plain = f"- id: {milestone}"
    quoted = f"- id: '{milestone}'"
    double_quoted = f'- id: "{milestone}"'
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip() in {plain, quoted, double_quoted}:
            return line_number
    return None


def _append_milestone_entry(path: Path) -> None:
    original = _read_text(path)
    separator = "" if not original or original.endswith("\n") else "\n"
    path.write_text(original + separator + MILESTONE_231_ENTRY, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _honest_verdict(precondition_status: str, archive_ready: bool, observed_milestone: str) -> str:
    readiness = str(archive_ready).lower()
    if precondition_status == "blocked_roadmap_unexpected":
        token = "blocked_roadmap_unexpected"
    else:
        token = f"archive_ready={readiness}"
    return (
        f"complete: {token}; archive_ready={readiness}; "
        f"milestone_archived={ARCHIVE_MILESTONE}; observed_milestone={observed_milestone}"
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-2378: write the idempotent .231 archive activation artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    roadmap_path = root_path / "research-roadmap.yaml"
    roadmap_next_path = root_path / "research-roadmap-next.yaml"
    complete_path = root_path / "research-complete.yaml"

    observed_milestone = _roadmap_milestone(roadmap_path)
    complete_before = _read_text(complete_path)
    existing_line_before = _find_milestone_line(complete_before, ARCHIVE_MILESTONE)
    should_try_archive = observed_milestone in {ARCHIVE_MILESTONE, ACTIVE_MILESTONE}
    appended_this_run = False
    copied_this_run = False

    if should_try_archive and existing_line_before is None:
        _append_milestone_entry(complete_path)
        appended_this_run = True

    if observed_milestone == ARCHIVE_MILESTONE:
        shutil.copyfile(roadmap_next_path, roadmap_path)
        copied_this_run = True
        precondition_status = "archived_and_activated"
    elif observed_milestone == ACTIVE_MILESTONE:
        precondition_status = "already_activated"
    else:
        precondition_status = "blocked_roadmap_unexpected"

    complete_after = _read_text(complete_path)
    existing_line_after = _find_milestone_line(complete_after, ARCHIVE_MILESTONE)
    archive_ready = existing_line_after is not None
    active_milestone_after = _roadmap_milestone(roadmap_path)
    roadmap_next_present = roadmap_next_path.exists()

    artifact = {
        "id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": _honest_verdict(
            precondition_status, archive_ready, observed_milestone
        ),
        "archive_ready": archive_ready,
        "milestone_archived": ARCHIVE_MILESTONE,
        "completed": COMPLETED,
        "preconditions": {
            "roadmap_path": str(roadmap_path),
            "observed_milestone": observed_milestone,
            "expected_archive_milestone": ARCHIVE_MILESTONE,
            "expected_active_milestone": ACTIVE_MILESTONE,
            "status": precondition_status,
        },
        "archive": {
            "research_complete_path": str(complete_path),
            "research_complete_contains_2026_05_231": archive_ready,
            "existing_entry_line": existing_line_after,
            "existing_entry_line_before_run": existing_line_before,
            "appended_this_run": appended_this_run,
            "decision": (
                "Appended the .231 archive entry."
                if appended_this_run
                else "No duplicate archive entry was appended."
            ),
        },
        "activation": {
            "research_roadmap_path": str(roadmap_path),
            "research_roadmap_next_path": str(roadmap_next_path),
            "observed_active_milestone": active_milestone_after,
            "copied_this_run": copied_this_run,
            "research_roadmap_next_present": roadmap_next_present,
            "decision": (
                "Copied research-roadmap-next.yaml to activate .232."
                if copied_this_run
                else "No roadmap copy was performed for this precondition branch."
            ),
        },
        "milestone_231_result_summary": MILESTONE_231_RESULT_SUMMARY,
        "acceptance_gates": [
            {
                "condition": "archive_ready == true",
                "passed": archive_ready,
                "principle": (
                    "Ensures research-complete.yaml records .231 outcomes before .232 begins."
                ),
            }
        ],
        "notes": [
            "scripts/research_conductor.py was not modified.",
            "No push was performed.",
        ],
        "field_principles": FIELD_PRINCIPLES,
    }
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
