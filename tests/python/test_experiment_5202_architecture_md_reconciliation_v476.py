"""Tests for Exp 5202 architecture.md reconciliation.

Spec refs: REQ-REPORT-5202, SCENARIO-REPORT-5202,
SCENARIO-REPORT-5202-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5202_architecture_md_reconciliation_v476 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARCHITECTURE_PATH = Path("_bmad/architecture.md")


def _wrapped(value: object, principle: str = "test principle") -> dict[str, object]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reconciled_architecture_text() -> str:
    legacy = "\n\n".join(f"{heading}\n\nlegacy section text" for heading in mod.LEGACY_SECTION_HEADINGS)
    added = "\n\n".join(
        [
            "## ARC-AGI-3 Harness Architecture\n\n"
            "scripts/arc_loop_solve.py dev twin, E3AgentPolicy scored cascade, "
            "arc_graph_explore.py best-first, arc_solver_kit.py reproduction gate, "
            "ops/arc_solve_registry.yaml knowledge capture.",
            "## PHASE D Lifecycle And Retirement\n\n"
            "Commissioned 2026-06-30; retired 2026-07-02/03 after seven null or "
            "marginal off-ARC external-text verifier milestones. "
            "hidden-state/internal-representation verifiers remain open.",
            "## Hidden-State Verifier Research Frontier\n\n"
            "TrajSelector-class and PHSV-class probes remain live; see "
            "ops/verifier_gaps.md GAP-4891 and GAP-4 plus Exp 5200.",
            "### 2026-07-03 Hardware Continuity Update\n\n"
            "KV260 terminal SSH continuity; PolarFire reachable but "
            "polarfire_workload_validated=false; GateMate jtag_protocol_level "
            "DirtyJTAG blocker.",
        ]
    )
    return (
        "# Carnot - Architecture\n\n"
        "**Last Reconciled:** 20260703\n\n"
        f"{legacy}\n\n"
        "## Verification Pipeline Tiers\n\n"
        "| Tier | Name | Class | Runtime role | Short-circuit / certificate behavior |\n"
        "|---|---|---|---|---|\n"
        "| 0 | JEPA fast paths | `JEPA_FAST_PATH` | skip low-risk responses | `skipped=True` |\n"
        "| 0c | NUP Probe | `NUP_PROBE_FAST_PATH` | skip low-energy responses | `skipped=True` |\n"
        "| Router | ODAR | `ODAR_FAST_PATH` | free-energy routing | fast path or fall through |\n"
        "| Advisory | AND-compose | `and_compose_k5` | k=5 ensemble | certificate only |\n\n"
        f"{added}\n"
    )


def make_repo(tmp_path: Path, *, reconciled: bool = True, omit_architecture: bool = False) -> Path:
    root = tmp_path
    for rel in (
        "_bmad",
        "ops",
        "results",
        "python/carnot/agentic",
        "python/carnot/pipeline",
        "python/carnot/verify",
        "scripts",
        "openspec/capabilities/research-reporting",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    if not omit_architecture:
        text = _reconciled_architecture_text()
        if not reconciled:
            text = text.replace("20260703", "2026-05-16")
        (root / mod.ARCHITECTURE_RELATIVE_PATH).write_text(text, encoding="utf-8")
    (root / "ops/north-star.md").write_text(
        "ARC-AGI-3 north star\n## 3. HARDWARE FOCUS\nKV260\n## 5. STRATEGIC REFRAME\nverifier moat\n",
        encoding="utf-8",
    )
    (root / "ops/exclusion_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "retired_extras": [
                    {
                        "id": "phase_d_external_text_scorer_retired_exp5163_v474",
                        "reason": "seven milestones; hidden-state/internal-representation verifiers outside scope",
                        "retired_milestone": "2026.07.474",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "ops/verifier_gaps.md").write_text(
        "GAP-4891 TrajSelector PHSV hidden-state\nGAP-4 same-shape rule application\n",
        encoding="utf-8",
    )
    (root / "ops/arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "reproducible_total_levels": 69,
                "reproducible_total_games": 24,
                "general_gotchas": [{"id": "offline_is_a_simulator"}],
                "live_submissions": [{"mode": "competition_kernel_live_agent", "public_leaderboard_score": 0.08}],
                "games": [{"game": "tr87", "levels_reproduced": 6}],
            }
        ),
        encoding="utf-8",
    )
    (root / "python/carnot/agentic/arc_competition_agent.py").write_text(
        "class E3AgentPolicy: pass\nSUBMITTED_AGENT_CONFIG = {'policy': 'E3AgentPolicy'}\n",
        encoding="utf-8",
    )
    (root / "python/carnot/pipeline/verify_repair.py").write_text(
        "def verify():\n    return 'JEPA_FAST_PATH NUP_PROBE_FAST_PATH ODAR_FAST_PATH and_compose_k5'\n",
        encoding="utf-8",
    )
    (root / "python/carnot/verify/__init__.py").write_text("# verify package\n", encoding="utf-8")
    (root / "scripts/research_conductor.py").write_text("# conductor\n", encoding="utf-8")
    (root / "openspec/capabilities/research-reporting/spec.md").write_text(
        SPEC_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_json(
        root / mod.HARDWARE_RESULT_RELATIVE_PATH,
        {
            "kv260_status": "reachable + hash-verified smoke",
            "polarfire_status": {"reachable": True, "polarfire_workload_validated": False},
            "gatemate_status": {"reachable": False, "narrowed_to": "jtag_protocol_level"},
            "gatemate_diagnostic_narrowed_to": "jtag_protocol_level",
            "boards_reachable_count": 2,
            "hardware_speedup_claimed": False,
        },
    )
    return root


def test_req_report_5202_spec_declares_reconciliation_contract() -> None:
    """REQ-REPORT-5202: OpenSpec declares the architecture reconciliation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5202") :]

    for marker in (
        "REQ-REPORT-5202",
        "SCENARIO-REPORT-5202",
        "SCENARIO-REPORT-5202-BLOCKED-PRECONDITION",
        str(mod.RESULT_RELATIVE_PATH),
        "`traceability_md_updated`",
        "aggregation_from_upstream_artifacts",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5202_builds_valid_reconciled_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5202: reconciled architecture inputs produce a terminal artifact."""

    root = make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        duration_s=1.25,
        run_date="20260703",
        tests_run=["unit fixture"],
    )

    mod.validate_artifact(artifact)
    assert artifact["sections_added"]["value"] == list(mod.NEW_SECTION_NAMES)
    assert artifact["sections_preserved_verbatim"]["value"] == len(mod.LEGACY_SECTION_HEADINGS)
    assert artifact["last_reconciled_date_updated"]["value"] is True
    assert artifact["traceability_md_updated"]["value"] is False
    assert artifact["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["arc_registry_summary"]["reproducible_total_levels"] == 69
    assert artifact["hardware_summary"]["gatemate_diagnostic_narrowed_to"] == "jtag_protocol_level"
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert json.loads((root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_report_5202_blocked_missing_architecture_is_honest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5202-BLOCKED-PRECONDITION: missing inputs block honestly."""

    root = make_repo(tmp_path, omit_architecture=True)
    artifact = mod.build_artifact(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        duration_s=0.5,
        run_date="20260703",
        tests_run=["unit fixture"],
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "blocked" in artifact["honest_verdict"]["value"]
    assert artifact["last_reconciled_date_updated"]["value"] is False
    assert artifact["sections_preserved_verbatim"]["value"] == 0
    assert artifact["traceability_md_updated"]["value"] is False
    assert artifact["failed_preconditions"]


def test_scenario_report_5202_real_architecture_contains_required_sections() -> None:
    """SCENARIO-REPORT-5202: the repository architecture document is reconciled."""

    text = ARCHITECTURE_PATH.read_text(encoding="utf-8")

    assert "**Last Reconciled:** 20260703" in text
    for section in mod.NEW_SECTION_NAMES:
        assert section in text
    for required in (
        "scripts/arc_loop_solve.py",
        "E3AgentPolicy",
        "arc_graph_explore.py",
        "arc_solver_kit.py",
        "ops/arc_solve_registry.yaml",
        "2026-06-30",
        "2026-07-02/03",
        "TrajSelector",
        "PHSV",
        "GAP-4891",
        "GAP-4",
        "JEPA_FAST_PATH",
        "ODAR_FAST_PATH",
        "jtag_protocol_level",
    ):
        assert required in text


def test_scenario_report_5202_input_failures_are_recorded(tmp_path: Path) -> None:
    """REQ-REPORT-5202: source parsing failures are visible preconditions."""

    failed: list[str] = []

    assert mod._load_json(tmp_path, Path("missing.json"), failed) == {}
    assert any(item.startswith("missing_or_unreadable:missing.json") for item in failed)

    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert mod._load_json(tmp_path, Path("bad.json"), failed) == {}
    assert any(item.startswith("malformed_json:bad.json") for item in failed)

    assert mod._load_yaml(tmp_path, Path("missing.yaml"), failed) == {}
    assert any(item.startswith("missing_or_unreadable:missing.yaml") for item in failed)

    (tmp_path / "bad.yaml").write_text(":\n", encoding="utf-8")
    assert mod._load_yaml(tmp_path, Path("bad.yaml"), failed) == {}
    assert any(item.startswith("malformed_yaml:bad.yaml") for item in failed)

    assert mod._find_dict_with_id({"items": [{"id": "present"}]}, "absent") == {}
    assert f"missing:{mod.VERIFY_DIR_RELATIVE_PATH}" in mod._missing_source_preconditions(tmp_path)


def test_scenario_report_5202_failed_preconditions_are_reported(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5202-BLOCKED-PRECONDITION: broken inputs stay blocked."""

    root = make_repo(tmp_path)
    (root / mod.RESEARCH_REPORTING_SPEC_RELATIVE_PATH).write_text("no REQ here\n", encoding="utf-8")
    (root / mod.NORTH_STAR_RELATIVE_PATH).write_text("no north star marker\n", encoding="utf-8")
    (root / mod.LIVE_AGENT_RELATIVE_PATH).write_text("class OtherPolicy: pass\n", encoding="utf-8")
    (root / mod.MANIFEST_RELATIVE_PATH).write_text("retired_extras: []\n", encoding="utf-8")
    architecture = (root / mod.ARCHITECTURE_RELATIVE_PATH).read_text(encoding="utf-8")
    (root / mod.ARCHITECTURE_RELATIVE_PATH).write_text(
        architecture.replace("jtag_protocol_level", "jtag protocol blocker"),
        encoding="utf-8",
    )

    artifact = mod.build_artifact(root=root, result_path=root / mod.RESULT_RELATIVE_PATH)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "blocked" in artifact["honest_verdict"]["value"]
    for marker in (
        "missing_spec_anchor:REQ-REPORT-5202",
        "north_star_missing_arc_agi3",
        "live_agent_missing_E3AgentPolicy",
        "architecture_required_topic_markers_missing",
        "phase_d_manifest_entry_missing",
    ):
        assert marker in artifact["failed_preconditions"]


def test_req_report_5202_validate_artifact_rejects_schema_errors(tmp_path: Path) -> None:
    """REQ-REPORT-5202: artifact validation rejects schema drift."""

    root = make_repo(tmp_path)
    valid = mod.build_artifact(root=root, result_path=root / mod.RESULT_RELATIVE_PATH)

    def clone() -> dict[str, object]:
        return json.loads(json.dumps(valid))

    def expect_error(payload: dict[str, object], message: str, *, fix_checksum: bool = True) -> None:
        if fix_checksum:
            payload["reproducibility_checksum"] = mod.payload_checksum(payload)
        with pytest.raises(AssertionError, match=message):
            mod.validate_artifact(payload)

    broken = clone()
    del broken["schema"]
    expect_error(broken, "missing required fields", fix_checksum=False)

    broken = clone()
    broken["schema"] = "wrong"
    expect_error(broken, "schema mismatch")

    broken = clone()
    broken["experiment_id"] = "wrong"
    expect_error(broken, "experiment_id mismatch")

    broken = clone()
    broken["inference_substrate"]["value"] = "manual_guess"
    expect_error(broken, "inference_substrate mismatch")

    broken = clone()
    broken["sections_added"] = []
    expect_error(broken, "must be principle-wrapped")

    broken = clone()
    broken["sections_added"]["principle"] = "wrong"
    expect_error(broken, "principle mismatch")

    broken = clone()
    broken["last_reconciled_date_updated"]["value"] = "true"
    expect_error(broken, "must be bool")

    broken = clone()
    broken["traceability_md_updated"]["value"] = True
    expect_error(broken, "must remain false")

    broken = clone()
    broken["honest_verdict"]["value"] = "blocked without terminal prefix"
    expect_error(broken, "must start")

    broken = clone()
    broken["reproducibility_checksum"] = "bad"
    expect_error(broken, "checksum mismatch", fix_checksum=False)

    broken = clone()
    broken["sections_added"]["value"] = []
    expect_error(broken, "sections_added mismatch")

    broken = clone()
    broken["sections_preserved_verbatim"]["value"] = 0
    expect_error(broken, "sections_preserved_verbatim mismatch")

    broken = clone()
    broken["last_reconciled_date_updated"]["value"] = False
    expect_error(broken, "must be true")


def test_scenario_report_5202_cli_writes_valid_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-5202: CLI writes and prints a validated artifact."""

    root = make_repo(tmp_path)
    result_path = root / "custom-result.json"

    assert mod.main(["--root", str(root), "--result-path", str(result_path), "--date", "20260703"]) == 0

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert printed == written
    assert written["result_path"] == str(mod.RESULT_RELATIVE_PATH)
    assert written["honest_verdict"]["value"].startswith("complete:")
