"""Build the Exp 1457 WOPR puzzle-cartridge retirement artifact.

Spec: REQ-REPORT-043, SCENARIO-REPORT-043.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
SCHEMA = "wopr_puzzle_cartridge_retirement_v1"
EXPERIMENT = "1457_wopr_puzzle_cartridge_retirement"
EXCLUSION_MARKER = "wopr_puzzle_cartridge_research_scope_closed"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1457_wopr_puzzle_cartridge_retirement.json"
DEFAULT_NOTE_PATH = REPO_ROOT / "ops" / "lineage-retirements" / "wopr_puzzle_cartridges_retired.md"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "exclusion_manifest.yaml"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "cartridge_experiments_reviewed",
    "retirement_note_path",
    "wopr_puzzle_lineage_retired",
    "exclusion_manifest_updated",
    "preserved_assets",
    "future_reopen_conditions",
    "honest_verdict",
}

FUTURE_REOPEN_CONDITIONS = [
    (
        "An operator explicitly reopens gallery work and names why the new puzzle "
        "cartridge is research-critical rather than a demo expansion."
    ),
    (
        "The proposal states a direct verify-repair LLM thesis link, such as a "
        "measured reduction in verifier false accepts or repair failures on an "
        "LLM-output corpus."
    ),
    (
        "The proposal states a direct Phase-3 substrate link, such as a reusable "
        "hardware-acceleratable EBM primitive not already demonstrated by existing "
        "WOPR cartridges."
    ),
    (
        "The proposal includes a falsifiable acceptance gate and a retirement rule "
        "if it again produces only gallery/demo evidence."
    ),
]

PRESERVED_ASSETS = [
    "python/carnot/games/connect_four.py",
    "python/carnot/games/hex.py",
    "python/carnot/games/nonogram.py",
    "python/carnot/games/futoshiki.py",
    "tests/python/games/test_connect_four.py",
    "tests/python/games/test_hex.py",
    "tests/python/test_nonogram_cartridge.py",
    "tests/python/test_futoshiki_cartridge.py",
    "spaces/wopr-games/app.py",
    "spaces/wopr-games/wopr_shell.py",
    "spaces/wopr-games/README.md",
    "spaces/wopr-games/games/hashi.py",
    "spaces/wopr-games/games/kakuro.py",
    "spaces/wopr-games/games/masyu.py",
    "spaces/wopr-games/games/slitherlink.py",
    "spaces/wopr-games/games/sudoku.py",
    "spaces/wopr-games/games/lights_out.py",
    "spaces/wopr-games/games/nqueens.py",
    "spaces/wopr-games/tests/test_kakuro.py",
    "spaces/wopr-games/tests/test_masyu.py",
    "spaces/wopr-games/tests/test_slitherlink.py",
    "spaces/wopr_games/games/slitherlink.py",
]

CARTRIDGE_REVIEW_ROWS: list[dict[str, str]] = [
    {
        "experiment_id": "exp1059",
        "cartridge": "Sudoku Space",
        "artifact_path": "results/experiment_1059_wopr_spaces_sudoku_v1.json",
        "outcome": "code_complete_deploy_pending",
        "evidence": "HuggingFace Space demo scaffold created",
        "scope_reason": "useful WOPR demo substrate, not active verify-repair research",
    },
    {
        "experiment_id": "exp1069",
        "cartridge": "Sudoku HF deploy",
        "artifact_path": "results/experiment_1069_wopr_sudoku_hf_deploy.json",
        "outcome": "deployed_live",
        "evidence": "Sudoku Space deploy proved the gallery path",
        "scope_reason": "deployment artifact remains demo-only",
    },
    {
        "experiment_id": "exp1070",
        "cartridge": "Global Thermonuclear War",
        "artifact_path": "results/experiment_1070_wopr_gtw_cartridge_v2.json",
        "outcome": "cartridge_shipped",
        "evidence": "cultural-anchor cartridge shipped with final energy 0.0",
        "scope_reason": "thematic demo, not thesis-critical",
    },
    {
        "experiment_id": "exp1071",
        "cartridge": "Lights Out",
        "artifact_path": "results/experiment_1071_wopr_lights_out_cartridge_v2.json",
        "outcome": "cartridge_shipped",
        "evidence": "classic Ising ground-state demo shipped with final energy 0.0",
        "scope_reason": "early demo utility exhausted",
    },
    {
        "experiment_id": "exp1097",
        "cartridge": "N-Queens",
        "artifact_path": "results/experiment_1097_wopr_nqueens_cartridge.json",
        "outcome": "cartridge_shipped",
        "evidence": "N-Queens cartridge shipped after an earlier gate-blocked attempt",
        "scope_reason": "another CSP demo, not a new research direction",
    },
    {
        "experiment_id": "exp1102",
        "cartridge": "N-Queens gallery update",
        "artifact_path": "results/experiment_1102_hf_spaces_gallery_update.json",
        "outcome": "gallery_updated_n_queens_live",
        "evidence": "gallery update path proved live deployment once",
        "scope_reason": "gallery mechanics should not drive new research tasks",
    },
    {
        "experiment_id": "exp1124",
        "cartridge": "Hashi",
        "artifact_path": "results/experiment_1124_wopr_hashi_cartridge.json",
        "outcome": "e0_achieved",
        "evidence": "Hashi bridge-counting cartridge achieved E=0",
        "scope_reason": "CSP encoding lesson retained as demo",
    },
    {
        "experiment_id": "exp1125",
        "cartridge": "Hashi gallery update",
        "artifact_path": "results/experiment_1125_hf_spaces_gallery_update.json",
        "outcome": "deployed_live",
        "evidence": "HF Spaces gallery update deployed Hashi",
        "scope_reason": "deployment preserved, not an active line",
    },
    {
        "experiment_id": "exp1136",
        "cartridge": "Slitherlink",
        "artifact_path": "results/experiment_1136_wopr_slitherlink_cartridge.json",
        "outcome": "blocked_gate_check_failed",
        "evidence": "pre-gate blocked because prior_failures metadata was missing",
        "scope_reason": "precursor block shows gallery churn cost",
    },
    {
        "experiment_id": "exp1141",
        "cartridge": "Slitherlink rescue",
        "artifact_path": "results/experiment_1141_wopr_slitherlink_rescue.json",
        "outcome": "e0_achieved",
        "evidence": "canonical puzzle reached E=0.0 in 1 iteration with 24 spins",
        "scope_reason": "successful rescue remains a demo asset",
    },
    {
        "experiment_id": "exp1175",
        "cartridge": "Connect Four",
        "artifact_path": "results/experiment_1175_wopr_connect_four_cartridge.json",
        "outcome": "cartridge_shipped_e0_at_convergence",
        "evidence": "42-spin gravity-valid cartridge shipped",
        "scope_reason": "board-game gallery baseline rather than verifier improvement",
    },
    {
        "experiment_id": "exp1188",
        "cartridge": "Hex",
        "artifact_path": "results/experiment_1188_wopr_hex_game_cartridge.json",
        "outcome": "hex_operational_energy_player_wins",
        "evidence": "7x7 Hex cartridge operational with energy-player wins",
        "scope_reason": "game-playing demo without a Carnot repair-thesis link",
    },
    {
        "experiment_id": "exp1201",
        "cartridge": "Nonogram precursor",
        "artifact_path": "results/experiment_1201_wopr_nonogram_cartridge.json",
        "outcome": "blocked_gate_check_failed",
        "evidence": "pre-gate blocked by incomplete prior-failure metadata",
        "scope_reason": "another gallery-block iteration",
    },
    {
        "experiment_id": "exp1214",
        "cartridge": "Nonogram",
        "artifact_path": "results/experiment_1214_wopr_nonogram_cartridge.json",
        "outcome": "nonogram_shipped_e0_at_solution",
        "evidence": "run-length solution energy E=0",
        "scope_reason": "classic CSP cartridge, not current research trajectory",
    },
    {
        "experiment_id": "exp1227",
        "cartridge": "Futoshiki",
        "artifact_path": "results/experiment_1227_wopr_futoshiki_cartridge.json",
        "outcome": "futoshiki_shipped_e0_at_solution",
        "evidence": "valid solution E=0 and violations score positive",
        "scope_reason": "inequality puzzle demo with no direct verifier lift",
    },
    {
        "experiment_id": "exp1240",
        "cartridge": "Kakuro precursor",
        "artifact_path": "results/experiment_1240_wopr_kakuro_cartridge.json",
        "outcome": "blocked_gate_check_failed",
        "evidence": "pre-gate blocked by incomplete prior-failure metadata",
        "scope_reason": "repeated gallery-gate friction",
    },
    {
        "experiment_id": "exp1243",
        "cartridge": "Kakuro v2 skeleton",
        "artifact_path": "results/experiment_1243_wopr_kakuro_cartridge_v2.json",
        "outcome": "in_progress",
        "evidence": "stale in-progress skeleton before later minimal shipment",
        "scope_reason": "skeleton churn does not add research signal",
    },
    {
        "experiment_id": "exp1253",
        "cartridge": "Masyu precursor",
        "artifact_path": "results/experiment_1253_wopr_masyu_cartridge.json",
        "outcome": "blocked_gate_check_failed",
        "evidence": "pre-gate blocked by incomplete prior-failure metadata",
        "scope_reason": "gallery line kept recurring without thesis lift",
    },
    {
        "experiment_id": "exp1261",
        "cartridge": "Kakuro v3 skeleton",
        "artifact_path": "results/experiment_1261_wopr_kakuro_v3.json",
        "outcome": "in_progress",
        "evidence": "stale in-progress skeleton before v4 minimal shipment",
        "scope_reason": "skeleton churn should not be repeated",
    },
    {
        "experiment_id": "exp1262",
        "cartridge": "Masyu v2 skeleton",
        "artifact_path": "results/experiment_1262_wopr_masyu_v2.json",
        "outcome": "in_progress",
        "evidence": "known-issues named Masyu line before v3 minimal shipment",
        "scope_reason": "not enough to justify further variants",
    },
    {
        "experiment_id": "exp1279",
        "cartridge": "Kakuro v4 minimal",
        "artifact_path": "results/experiment_1279_wopr_kakuro_v4_minimal.json",
        "outcome": "shipped",
        "evidence": "valid E=0.0 and deterministic invalid E=17.0",
        "scope_reason": "minimal gallery cartridge after repeated gate blocks",
    },
    {
        "experiment_id": "exp1280",
        "cartridge": "Masyu v3 minimal",
        "artifact_path": "results/experiment_1280_wopr_masyu_v3_minimal.json",
        "outcome": "shipped_minimal_masyu_cartridge",
        "evidence": "valid E=0.0 and invalid E=3.0",
        "scope_reason": "minimal loop-puzzle demo after repeated gate blocks",
    },
]

KNOWN_ISSUES_ID_NOTES = [
    (
        "ops/known-issues.md names exp1198 as Connect Four in the scope-C sketch, "
        "but research-complete.yaml identifies exp1198 as FoVer v7; the actual "
        "Connect Four cartridge artifact is exp1175."
    ),
    (
        "ops/known-issues.md names exp1262 for Masyu; the terminal shipped Masyu "
        "artifact is exp1280 after exp1253/exp1262 precursor blocks and skeletons."
    ),
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        parts = path.parts
        for anchor in ("ops", "results"):
            if anchor in parts:
                return str(Path(*parts[parts.index(anchor) :]))
        return path.name


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-043: record the bootstrap state before terminal scoring."""

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "cartridge_experiments_reviewed": [],
            "retirement_note_path": _relative_path(DEFAULT_NOTE_PATH),
            "wopr_puzzle_lineage_retired": False,
            "exclusion_manifest_updated": False,
            "preserved_assets": [],
            "future_reopen_conditions": [],
            "honest_verdict": "in_progress",
        }
    )
    return _write_json(Path(out_path), artifact)


def _manifest_contains_block(manifest_text: str) -> bool:
    return (
        EXCLUSION_MARKER in manifest_text
        and "WOPR puzzle cartridge" in manifest_text
        and "operator explicitly reopens gallery work" in manifest_text
    )


def ensure_manifest_block(manifest_text: str) -> tuple[str, bool]:
    """REQ-REPORT-043: append a planner-visible WOPR puzzle block once."""

    if _manifest_contains_block(manifest_text):
        return manifest_text, False
    text = manifest_text.rstrip()
    if "retired_extras:" not in text:
        text = f"{text}\n\nretired_extras:" if text else "retired_extras:"
    block = f"""

  # Added by Exp 1457 - WOPR puzzle cartridge lineage consolidation and retirement.
  - id: {EXCLUSION_MARKER}
    reason: |
      WOPR puzzle cartridges are retired as active research scope after Hex,
      Connect Four, Nonogram, Futoshiki, Kakuro, Masyu, Slitherlink, and related
      gallery tasks proved useful as demos but did not map cleanly to Carnot's
      verify-repair LLM thesis or Phase-3 substrate trajectory. Future WOPR
      puzzle cartridge or gallery-update research tasks are blocked unless an
      operator explicitly reopens gallery work with a new root cause, direct
      thesis or substrate link, and falsifiable gate.
    experiment_ids:
      - exp1059
      - exp1069
      - exp1070
      - exp1071
      - exp1097
      - exp1102
      - exp1124
      - exp1125
      - exp1136
      - exp1141
      - exp1175
      - exp1188
      - exp1201
      - exp1214
      - exp1227
      - exp1240
      - exp1243
      - exp1253
      - exp1261
      - exp1262
      - exp1279
      - exp1280
    blocked_patterns:
      - "WOPR puzzle cartridge"
      - "WOPR game cartridge"
      - "WOPR games gallery"
      - "HF Spaces gallery update"
      - "new puzzle cartridge"
      - "Hex cartridge"
      - "Connect Four cartridge"
      - "Nonogram cartridge"
      - "Futoshiki cartridge"
      - "Kakuro cartridge"
      - "Masyu cartridge"
      - "Slitherlink cartridge"
    retired_milestone: "2026.04.112"
    retired_by_artifact: "results/experiment_1457_wopr_puzzle_cartridge_retirement.json"
    operator_reopen_required: true
    preserve_demo_assets: true
    retire_if_same_verdict: true
"""
    return text + block, True


def _md_cell(value: object) -> str:
    text = str(value) if value else "none"
    return text.replace("\n", " ").replace("|", "\\|")


def _experiment_ids(review_rows: Iterable[Mapping[str, object]]) -> list[str]:
    return [str(row["experiment_id"]) for row in review_rows]


def build_artifact(
    *,
    review_rows: Iterable[Mapping[str, object]],
    retirement_note_path: str,
    manifest_text: str,
    manifest_block_added: bool,
    preserved_assets: Iterable[str] = PRESERVED_ASSETS,
) -> dict[str, Any]:
    """REQ-REPORT-043: assemble the terminal WOPR retirement artifact."""

    rows = [dict(row) for row in review_rows]
    assets = list(dict.fromkeys(str(asset) for asset in preserved_assets))
    manifest_has_block = _manifest_contains_block(manifest_text)
    complete = bool(rows) and manifest_has_block and bool(assets)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if complete else "blocked",
        "cartridge_experiments_reviewed": _experiment_ids(rows),
        "cartridge_review_rows": rows,
        "retirement_note_path": retirement_note_path,
        "wopr_puzzle_lineage_retired": complete,
        "exclusion_manifest_updated": manifest_has_block,
        "exclusion_manifest_block_added": manifest_block_added,
        "preserved_assets": assets,
        "known_issues_id_notes": list(KNOWN_ISSUES_ID_NOTES),
        "future_reopen_conditions": list(FUTURE_REOPEN_CONDITIONS),
        "honest_verdict": (
            "wopr_puzzle_lineage_retired_demo_assets_preserved_no_new_gallery_work"
            if complete
            else "wopr_puzzle_lineage_retirement_blocked_missing_manifest_review_or_assets"
        ),
    }
    return artifact


def render_retirement_note(
    review_rows: Iterable[Mapping[str, object]],
    artifact: Mapping[str, Any],
) -> str:
    """SCENARIO-REPORT-043: render the operator-facing WOPR retirement note."""

    lines = [
        "# WOPR Puzzle Cartridge Lineage Retirement",
        "",
        f"Run date: `{RUN_DATE}`",
        f"Artifact: `results/experiment_1457_wopr_puzzle_cartridge_retirement.json`",
        "",
        "## Experiments Reviewed",
        "",
        "| experiment | cartridge | outcome | evidence | scope reason |",
        "|---|---|---|---|---|",
    ]
    for row in review_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row.get("experiment_id")),
                    _md_cell(row.get("cartridge")),
                    _md_cell(row.get("outcome")),
                    _md_cell(row.get("evidence")),
                    _md_cell(row.get("scope_reason")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Known-Issues ID Notes", ""])
    for note in artifact["known_issues_id_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Preserved Demo Assets", ""])
    for asset in artifact["preserved_assets"]:
        lines.append(f"- `{asset}`")
    lines.extend(["", "## Future Reopen Conditions", ""])
    for condition in artifact["future_reopen_conditions"]:
        lines.append(f"- {condition}")
    lines.extend(
        [
            "",
            "## Research Scope Decision",
            "",
            (
                "The WOPR puzzle-cartridge lineage is retired as active research scope. "
                "The existing code, tests, docs, and HuggingFace Spaces assets remain "
                "usable as demos, but new puzzle-cartridge or gallery-update research "
                "tasks are blocked unless an operator explicitly reopens the gallery "
                "under the conditions above."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    note_path: Path | str = DEFAULT_NOTE_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    review_rows: Iterable[Mapping[str, object]] = CARTRIDGE_REVIEW_ROWS,
    preserved_assets: Iterable[str] = PRESERVED_ASSETS,
) -> dict[str, Any]:
    """REQ-REPORT-043: write bootstrap, note, manifest block, and terminal JSON."""

    _ = Path(root)
    out = Path(out_path)
    note = Path(note_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(out)
    manifest_text = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    updated_manifest, block_added = ensure_manifest_block(manifest_text)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        updated_manifest + ("" if updated_manifest.endswith("\n") else "\n"), encoding="utf-8"
    )
    artifact = build_artifact(
        review_rows=review_rows,
        retirement_note_path=_relative_path(note),
        manifest_text=updated_manifest,
        manifest_block_added=block_added,
        preserved_assets=preserved_assets,
    )
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(render_retirement_note(review_rows, artifact), encoding="utf-8")
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
