"""Tests for Exp 1456 GRPO/VPRM lineage consolidation and retirement.

Spec: REQ-REPORT-042, SCENARIO-REPORT-042.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.grpo_vprm_lineage_retirement import (
    EXCLUSION_MARKER,
    REQUIRED_ARTIFACT_FIELDS,
    _manifest_contains_block,
    _relative_path,
    build_artifact,
    ensure_manifest_block,
    render_consolidation_note,
    run,
    write_in_progress_artifact,
)


def _review_rows() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "exp1118",
            "title": "GRPO with ThinkPRM v2 Energy Reward",
            "verdict": "positive_improvement",
            "measured_positive": "+4pp on 25 live-GPU eval questions",
            "blocker": "",
            "lesson": "Energy-shaped process rewards can move small eval slices.",
        },
        {
            "experiment_id": "exp1208",
            "title": "GRPO v5 TinyV confidence abstention",
            "verdict": "improvement_below_v4",
            "measured_positive": "",
            "blocker": "TinyV abstained on 62.5% of rewards and regressed -35pp.",
            "lesson": "False-negative correction needs calibrated abstention thresholds.",
        },
        {
            "experiment_id": "exp1393",
            "title": "GRPO v8 NGRPO zero-reward fix",
            "verdict": "grpo_v8_ngrpo_no_improvement_all_unknown_retired",
            "measured_positive": "",
            "blocker": "All-UNKNOWN formal rewards produced 0pp held-out improvement.",
            "lesson": "Formal verifier rewards need non-UNKNOWN candidate diversity.",
        },
    ]


def test_scenario_report_042_builds_retirement_artifact_and_manifest_block() -> None:
    """SCENARIO-REPORT-042: Exp 1456 retires GRPO/VPRM scope."""

    manifest, block_added = ensure_manifest_block("retired_extras:\n")
    artifact = build_artifact(
        review_rows=_review_rows(),
        consolidation_note_path="ops/lineage-retirements/grpo_vprm_lineage_retired.md",
        manifest_text=manifest,
        manifest_block_added=block_added,
    )
    note = render_consolidation_note(_review_rows(), artifact)

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["lineage_name"] == "GRPO/VPRM"
    assert artifact["experiments_reviewed"] == ["exp1118", "exp1208", "exp1393"]
    assert artifact["grpo_lineage_retired"] is True
    assert artifact["exclusion_manifest_updated"] is True
    assert artifact["exclusion_manifest_block_added"] is True
    assert "false-negative correction" in " ".join(artifact["lessons_retained"]).lower()
    assert "operator explicitly reopens" in " ".join(artifact["future_reopen_conditions"])
    assert artifact["honest_verdict"] == "grpo_vprm_lineage_retired_no_v15_without_operator_reopen"

    assert EXCLUSION_MARKER in manifest
    assert "GRPO v15" in manifest
    assert "VPRM v15" in manifest
    assert "| exp1118 |" in note
    assert "Final Decision" in note


def test_req_report_042_manifest_block_is_idempotent() -> None:
    """REQ-REPORT-042: exclusion manifest receives one durable scope block."""

    manifest, first_added = ensure_manifest_block("retired_extras:\n")
    manifest_again, second_added = ensure_manifest_block(manifest)

    assert first_added is True
    assert second_added is False
    assert manifest_again == manifest
    assert manifest_again.count(EXCLUSION_MARKER) == 1
    assert _manifest_contains_block(manifest_again) is True
    assert _manifest_contains_block("retired_extras:\n") is False
    blank_manifest, blank_added = ensure_manifest_block("")
    assert blank_added is True
    assert blank_manifest.startswith("retired_extras:")


def test_req_report_042_run_writes_bootstrap_note_manifest_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-042: run writes the bootstrap and complete deliverables."""

    out_path = tmp_path / "results" / "experiment_1456_grpo_vprm_lineage_consolidation_retirement.json"
    note_path = tmp_path / "ops" / "lineage-retirements" / "grpo_vprm_lineage_retired.md"
    manifest_path = tmp_path / "ops" / "exclusion_manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("retired_extras:\n", encoding="utf-8")

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        note_path=note_path,
        manifest_path=manifest_path,
        review_rows=_review_rows(),
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    note = note_path.read_text(encoding="utf-8")
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["consolidation_note_path"] == (
        "ops/lineage-retirements/grpo_vprm_lineage_retired.md"
    )
    assert written["exclusion_manifest_updated"] is True
    assert EXCLUSION_MARKER in manifest
    assert "Repeated Blockers" in note
    assert _relative_path(note_path) == "ops/lineage-retirements/grpo_vprm_lineage_retired.md"
    assert _relative_path(tmp_path / "loose.md") == "loose.md"
