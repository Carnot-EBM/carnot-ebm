"""Tests for Exp 1457 WOPR puzzle-cartridge retirement.

Spec: REQ-REPORT-043, SCENARIO-REPORT-043.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.wopr_puzzle_cartridge_retirement import (
    EXCLUSION_MARKER,
    REQUIRED_ARTIFACT_FIELDS,
    _manifest_contains_block,
    _relative_path,
    build_artifact,
    ensure_manifest_block,
    render_retirement_note,
    run,
    write_in_progress_artifact,
)


def _review_rows() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "exp1141",
            "cartridge": "Slitherlink",
            "artifact_path": "results/experiment_1141_wopr_slitherlink_rescue.json",
            "outcome": "shipped",
            "evidence": "canonical puzzle reached E=0.0 in 1 iteration",
            "scope_reason": "demo-only CSP cartridge, not verify-repair or Phase-3 substrate signal",
        },
        {
            "experiment_id": "exp1175",
            "cartridge": "Connect Four",
            "artifact_path": "results/experiment_1175_wopr_connect_four_cartridge.json",
            "outcome": "shipped",
            "evidence": "42-spin gravity-valid cartridge shipped",
            "scope_reason": "gallery baseline rather than LLM verifier improvement",
        },
        {
            "experiment_id": "exp1188",
            "cartridge": "Hex",
            "artifact_path": "results/experiment_1188_wopr_hex_game_cartridge.json",
            "outcome": "shipped",
            "evidence": "7x7 Hex energy player operational",
            "scope_reason": "game-playing demo without thesis-critical repair signal",
        },
        {
            "experiment_id": "exp1214",
            "cartridge": "Nonogram",
            "artifact_path": "results/experiment_1214_wopr_nonogram_cartridge.json",
            "outcome": "shipped",
            "evidence": "run-length solution energy E=0",
            "scope_reason": "classic CSP cartridge, not current research trajectory",
        },
        {
            "experiment_id": "exp1227",
            "cartridge": "Futoshiki",
            "artifact_path": "results/experiment_1227_wopr_futoshiki_cartridge.json",
            "outcome": "shipped",
            "evidence": "valid solution E=0 and violations score positive",
            "scope_reason": "inequality puzzle demo with no direct verifier lift",
        },
        {
            "experiment_id": "exp1279",
            "cartridge": "Kakuro",
            "artifact_path": "results/experiment_1279_wopr_kakuro_v4_minimal.json",
            "outcome": "shipped",
            "evidence": "valid E=0.0 and invalid E=17.0",
            "scope_reason": "minimal gallery cartridge after repeated gate blocks",
        },
        {
            "experiment_id": "exp1280",
            "cartridge": "Masyu",
            "artifact_path": "results/experiment_1280_wopr_masyu_v3_minimal.json",
            "outcome": "shipped",
            "evidence": "valid E=0.0 and invalid E=3.0",
            "scope_reason": "loop-puzzle demo after repeated gate blocks",
        },
    ]


def _assets() -> list[str]:
    return [
        "python/carnot/games/connect_four.py",
        "python/carnot/games/hex.py",
        "python/carnot/games/nonogram.py",
        "python/carnot/games/futoshiki.py",
        "spaces/wopr-games/games/slitherlink.py",
        "spaces/wopr-games/games/kakuro.py",
        "spaces/wopr-games/games/masyu.py",
    ]


def test_scenario_report_043_builds_retirement_artifact_and_manifest_block() -> None:
    """SCENARIO-REPORT-043: Exp 1457 retires WOPR puzzle research scope."""

    manifest, block_added = ensure_manifest_block("retired_extras:\n")
    artifact = build_artifact(
        review_rows=_review_rows(),
        retirement_note_path="ops/lineage-retirements/wopr_puzzle_cartridges_retired.md",
        manifest_text=manifest,
        manifest_block_added=block_added,
        preserved_assets=_assets(),
    )
    note = render_retirement_note(_review_rows(), artifact)

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["cartridge_experiments_reviewed"] == [
        "exp1141",
        "exp1175",
        "exp1188",
        "exp1214",
        "exp1227",
        "exp1279",
        "exp1280",
    ]
    assert artifact["wopr_puzzle_lineage_retired"] is True
    assert artifact["exclusion_manifest_updated"] is True
    assert artifact["exclusion_manifest_block_added"] is True
    assert "python/carnot/games/hex.py" in artifact["preserved_assets"]
    assert "operator explicitly reopens" in " ".join(artifact["future_reopen_conditions"])
    assert artifact["honest_verdict"] == (
        "wopr_puzzle_lineage_retired_demo_assets_preserved_no_new_gallery_work"
    )

    assert EXCLUSION_MARKER in manifest
    assert "WOPR puzzle cartridge" in manifest
    assert "operator explicitly reopens gallery work" in manifest
    assert "Slitherlink" in note
    assert "exp1198" in note
    assert "Research Scope Decision" in note


def test_req_report_043_manifest_block_is_idempotent() -> None:
    """REQ-REPORT-043: exclusion manifest receives one durable WOPR scope block."""

    manifest, first_added = ensure_manifest_block("retired_extras:\n")
    manifest_again, second_added = ensure_manifest_block(manifest)
    blank_manifest, blank_added = ensure_manifest_block("")

    assert first_added is True
    assert second_added is False
    assert manifest_again == manifest
    assert manifest_again.count(EXCLUSION_MARKER) == 1
    assert _manifest_contains_block(manifest_again) is True
    assert _manifest_contains_block("retired_extras:\n") is False
    assert blank_added is True
    assert blank_manifest.startswith("retired_extras:")


def test_req_report_043_run_writes_bootstrap_note_manifest_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-043: run writes bootstrap, note, manifest block, and JSON."""

    out_path = tmp_path / "results" / "experiment_1457_wopr_puzzle_cartridge_retirement.json"
    note_path = tmp_path / "ops" / "lineage-retirements" / "wopr_puzzle_cartridges_retired.md"
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
        preserved_assets=_assets(),
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    note = note_path.read_text(encoding="utf-8")
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["retirement_note_path"] == (
        "ops/lineage-retirements/wopr_puzzle_cartridges_retired.md"
    )
    assert written["wopr_puzzle_lineage_retired"] is True
    assert EXCLUSION_MARKER in manifest
    assert "Preserved Demo Assets" in note
    assert _relative_path(note_path) == "ops/lineage-retirements/wopr_puzzle_cartridges_retired.md"
    assert (
        _relative_path(out_path) == "results/experiment_1457_wopr_puzzle_cartridge_retirement.json"
    )
    assert _relative_path(tmp_path / "loose.md") == "loose.md"
