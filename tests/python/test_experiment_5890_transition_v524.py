"""Tests for the Exp5890 V524 transition receipt.

Spec refs: REQ-REPORT-5890, SCENARIO-REPORT-5890-EXACT-ARCHIVE,
SCENARIO-REPORT-5890-APPEND-ONCE, SCENARIO-REPORT-5890-UNACTIVATED-PROPOSAL,
SCENARIO-REPORT-5890-RANGE-COLLISION, SCENARIO-REPORT-5890-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5890_transition_v524 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5877-transition-v523": {
            "status": "complete",
            "honest_verdict": "complete: archived terminal .522 identities into .523",
            "next_range_collision_count": 0,
        },
        "exp5878-v523-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V523 source deltas",
            "accepted_finding_count": 0,
            "references_modified": False,
        },
        "exp5879-hardness-headroom-taxonomy-corrigendum": {
            "status": "blocked",
            "honest_verdict": "blocked: science_ready_but_unrelated_global_suite_debt",
            "hardness_surface_headroom_ready_score": 1.0,
        },
        "exp5881-one-to-one-grounding-acquisition-ab": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "upstream artifact not found for exp5880",
        },
    }
    return payloads[task_id]


def _completion_payload(*, include_523: bool = True, duplicate_510_blocks: int = 1) -> JsonDict:
    duplicate_block = {
        "id": "2026.07.510",
        "title": "Historical duplicate",
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": "2026-07-17",
        "finding": "fixture",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    milestones = [deepcopy(duplicate_block) for _ in range(duplicate_510_blocks)]
    if include_523:
        milestones.append(
            {
                "id": mod.MILESTONE_FROM,
                "title": mod.MILESTONE_FROM_TITLE,
                "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                "completed": "2026-07-24",
                "finding": "See conductor log for per-experiment results.",
                "tasks": [
                    {
                        "id": task_id,
                        "title": mod.ACTIVATED_TASK_TITLES[task_id],
                        "deliverable": rel_path.as_posix(),
                        "result": "OK (conductor)",
                    }
                    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
                ],
            }
        )
    return {"milestones": milestones}


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
            }
            for task_id, rel_path in mod.NEXT_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _vnext_doc() -> str:
    lines = [
        "# Research Roadmap vNEXT",
        "",
        "**Milestone:** 2026.07.524",
        "**Task range:** Exp5890-Exp5903",
        "",
        "Exp5883-Exp5889 were proposed only and never activated.",
    ]
    lines.extend(path.as_posix() for path in mod.NEXT_TASK_ARTIFACT_PATHS.values())
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-24 08:37 UTC | Exact terminal-boundary handoff from .522 into .52 | OK | 88 passed |",
            "| 2026-07-24 09:19 UTC | Dated evidence refresh after the V523 planner mark | OK | 124 passed |",
            "| 2026-07-24 09:40 UTC | Changed-taxonomy audit of nuisance controls versus | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-24 09:56 UTC | Changed-taxonomy audit of nuisance controls versus | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-24 10:14 UTC | Changed-taxonomy audit of nuisance controls versus | FAIL | artifact_not_updated_past_bootstrap |",
            "| 2026-07-24 10:16 UTC | Gated on Exp5879 headroom: exact constraint-satisf | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-24 10:18 UTC | Gated on Exp5879 headroom: exact constraint-satisf | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-24 10:20 UTC | Gated on Exp5879 headroom: exact constraint-satisf | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-24 10:16 UTC | Gated on Exp5880 fixture: one-to-one atom groundin | GATE_BLOCK | 1 of 1 gate(s) failed |",
            "| 2026-07-24 10:18 UTC | Gated on Exp5880 fixture: one-to-one atom groundin | GATE_BLOCK | 1 of 1 gate(s) failed |",
            "| 2026-07-24 10:21 UTC | Gated on Exp5880 fixture: one-to-one atom groundin | GATE_BLOCK | 1 of 1 gate(s) failed |",
            "| 2026-07-24 10:23 UTC | Gated on Exp5881 mechanism: prospective shortcut-r | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-24 11:22 UTC | Gated on Exp5881 mechanism: prospective shortcut-r | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-24 12:11 UTC | Gated on Exp5881 mechanism: prospective shortcut-r | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-07-24 13:01 UTC | Exact terminal-boundary handoff from .523 into .52 | FAIL | artifact_not_updated_past_bootstrap (deliverable=results/experiment_5890_transit |",
        ]
    )


def _make_root(
    root: Path,
    *,
    include_523_complete: bool = True,
    duplicate_510_blocks: int = 1,
) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id in mod.MISSING_DELIVERABLE_REASONS:
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(
            _completion_payload(
                include_523=include_523_complete,
                duplicate_510_blocks=duplicate_510_blocks,
            )
        ),
    )
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, _vnext_doc())
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.EVIDENCE_INDEX_RELATIVE_PATH, "# evidence index fixture\n")
    _write_text(root, mod.DOC_RECONCILE_RELATIVE_PATH, "# reconcile fixture\n")
    _write_text(root, mod.ADVERSARIAL_VERIFY_RELATIVE_PATH, "# verifier fixture\n")
    _write_text(
        root,
        "python/carnot/experiment_5890_transition_v524.py",
        "# owned exp5890 transition fixture\n",
    )
    _write_text(
        root,
        "tests/python/test_experiment_5890_transition_v524.py",
        "# owned exp5890 test fixture\n",
    )
    for rel_path in mod.PROTECTED_FILE_PATHS + (mod.SPEC_RELATIVE_PATH,):
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _receipt(task_id: str, *, flagged: bool = False) -> JsonDict:
    flags = (
        [{"kind": "METHODOLOGY_MISSING", "severity": "warn", "detail": "fixture"}]
        if flagged
        else []
    )
    stdout_json = {
        "reports": [
            {
                "path": mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
                "flag_count": len(flags),
                "flags": flags,
                "max_severity": 1 if flags else -1,
            }
        ],
        "flagged_count": 1 if flags else 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py --json "
            f"{mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()}"
        ),
        "exit_code": 1 if flagged else 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(
            task_id,
            flagged=task_id
            in {
                "exp5878-v523-source-delta-ingestion",
                "exp5879-hardness-headroom-taxonomy-corrigendum",
            },
        )
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id not in mod.MISSING_DELIVERABLE_REASONS
    }


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_5890_transition_v524.py -q",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.25,
    )


def test_req_report_5890_spec_declares_exact_transition_contract() -> None:
    """REQ-REPORT-5890: OpenSpec names activated identity, proposal, and range gates."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5890") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5890-EXACT-ARCHIVE" in section
    assert "SCENARIO-REPORT-5890-APPEND-ONCE" in section
    assert "SCENARIO-REPORT-5890-UNACTIVATED-PROPOSAL" in section
    assert "SCENARIO-REPORT-5890-RANGE-COLLISION" in section
    assert "Exp5890 through Exp5903" in section
    assert "ops/conductor-log.md" in section
    assert "next_range_collision_count=0" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5890_archives_terminal_v523_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5890-EXACT-ARCHIVE: activated classes stay disjoint."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.523",
        "destination_milestone": "2026.07.524",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    assert len(report["activated_task_and_deliverable_matrix"]) == 6
    exp5880 = report["activated_task_and_deliverable_matrix"][
        "exp5880-grounding-shortcut-fixture"
    ]
    assert exp5880["present"] is False
    assert exp5880["missing_recorded_explicitly"] is True
    assert exp5880["missing_reason"] == "upstream_retired_exp5879"
    assert exp5880["selection_policy"] == "exact_declared_deliverable"

    classes = report["outcome_classification"]
    assert classes["terminal_class_by_task_id"] == {
        "exp5877-transition-v523": "complete_transition",
        "exp5878-v523-source-delta-ingestion": "no_accepted_source_delta",
        "exp5879-hardness-headroom-taxonomy-corrigendum": "retired_science_ready",
        "exp5880-grounding-shortcut-fixture": "missing_upstream_retired",
        "exp5881-one-to-one-grounding-acquisition-ab": "gate_blocked",
        "exp5882-shortcut-resistant-continuous-self-learning": "missing_upstream_retired",
    }
    assert classes["retired_science_ready_task_ids"] == [
        "exp5879-hardness-headroom-taxonomy-corrigendum"
    ]
    assert classes["science_ready_task_ids"] == [
        "exp5879-hardness-headroom-taxonomy-corrigendum"
    ]
    assert classes["missing_declared_deliverable_task_ids"] == [
        "exp5880-grounding-shortcut-fixture",
        "exp5882-shortcut-resistant-continuous-self-learning",
    ]
    assert classes["gate_blocked_task_ids"] == [
        "exp5881-one-to-one-grounding-acquisition-ab"
    ]
    assert classes["verifier_warn_task_ids"] == [
        "exp5878-v523-source-delta-ingestion",
        "exp5879-hardness-headroom-taxonomy-corrigendum",
    ]
    assert report["retired_and_science_ready_preserved"] is True
    receipts = report["missing_and_gate_blocked_receipts"]
    assert receipts["missing_task_ids"] == [
        "exp5880-grounding-shortcut-fixture",
        "exp5882-shortcut-resistant-continuous-self-learning",
    ]
    assert receipts["gate_blocked_task_ids"] == [
        "exp5881-one-to-one-grounding-acquisition-ab"
    ]
    assert all(row["treated_as_success"] is False for row in receipts["receipts"])
    assert report["next_range_collision_count"] == 0
    assert {
        "path": "ops/conductor-log.md",
        "kind": "transition_owned_conductor_attempt_reference",
    } in report["preconditions_checked"]["range_collision_scan"]["allowed_references"]
    assert report["next_task_range"]["start"] == "exp5890"
    assert report["next_task_range"]["end"] == "exp5903"
    assert len(report["adversarial_verifier_receipts"]) == 4
    assert (
        "exp5880-grounding-shortcut-fixture"
        not in report["adversarial_verifier_receipts"]
    )
    mod.validate_artifact(report)


def test_scenario_report_5890_appends_completion_history_once_when_absent(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5890-APPEND-ONCE: absent .523 history is appended once."""

    _make_root(tmp_path, include_523_complete=False, duplicate_510_blocks=2)
    report = _build(tmp_path)
    after_first = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    v523_blocks = [row for row in after_first["milestones"] if row["id"] == mod.MILESTONE_FROM]

    assert report["research_complete_append_count"] == 1
    assert len(v523_blocks) == 1
    assert len(v523_blocks[0]["tasks"]) == 6
    assert not {
        row["id"] for row in v523_blocks[0]["tasks"]
    } & set(mod.UNACTIVATED_PROPOSAL_TASK_IDS)
    assert report["duplicate_history_amplification_count"] == 0
    assert report["preconditions_checked"]["duplicate_history"]["before_duplicate_block_count"] == 1
    assert report["preconditions_checked"]["duplicate_history"]["after_duplicate_block_count"] == 1
    mod.validate_artifact(report)

    second = _build(tmp_path)
    after_second = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    assert second["research_complete_append_count"] == 0
    assert [row["id"] for row in after_second["milestones"]].count(mod.MILESTONE_FROM) == 1
    assert second["duplicate_history_amplification_count"] == 0
    mod.validate_artifact(second)


def test_scenario_report_5890_unactivated_proposal_ids_are_not_completed(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5890-UNACTIVATED-PROPOSAL: proposal IDs stay out of history."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    receipt = report["unactivated_proposal_id_receipt"]
    assert receipt["task_ids"] == list(mod.UNACTIVATED_PROPOSAL_TASK_IDS)
    assert receipt["present_in_activated_completion_block"] == []
    assert receipt["appended_as_completed"] is False
    mod.validate_artifact(report)

    data = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    v523 = next(row for row in data["milestones"] if row["id"] == mod.MILESTONE_FROM)
    v523["tasks"].append(
        {
            "id": mod.UNACTIVATED_PROPOSAL_TASK_IDS[0],
            "deliverable": "results/experiment_5883_gguf_intermediate_layer_surface_preflight.json",
        }
    )
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(data))

    laundered = _build(tmp_path)
    assert laundered["status"] == "blocked"
    assert "unactivated_proposal_ids_laundered" in laundered["preconditions_checked"][
        "failed_preconditions"
    ]
    with pytest.raises(ValueError, match="unactivated proposal IDs"):
        mod.validate_artifact(laundered)


def test_scenario_report_5890_unexpected_range_reference_blocks_completion(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5890-RANGE-COLLISION: unexpected Exp5890-Exp5903 hits block."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5892_stale_collision.json", {"status": "stale"})
    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 1
    assert report["preconditions_checked"]["range_collision_scan"]["collision_count"] == 1
    assert report["preconditions_checked"]["range_collision_scan"]["collisions"][0]["path"] == (
        "results/experiment_5892_stale_collision.json"
    )
    mod.validate_artifact(report)

    conductor_root = tmp_path / "conductor-collision"
    _make_root(conductor_root)
    _write_text(
        conductor_root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        _conductor_log()
        + "\n| 2026-07-24 13:02 UTC | Exp5892 downstream task | FAIL | deliverable=results/experiment_5892_headroom_evidence_escrow.json |\n",
    )
    conductor_report = _build(conductor_root)
    assert conductor_report["status"] == "blocked"
    assert conductor_report["next_range_collision_count"] == 1
    assert conductor_report["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {"path": "ops/conductor-log.md", "kind": "unexpected_next_range_reference"}
    ]


def test_scenario_report_5890_schema_checksum_and_protection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5890-SCHEMA: required fields and protection are enforced."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"]["ops_status_md"] == "deferred_by_operator_stop_rule"
    assert report["docs_reconciled"]["ops_changelog_md"] == "deferred_by_operator_stop_rule"
    assert report["docs_reconciled"]["traceability_md"] == "deferred_by_operator_stop_rule"
    assert all(row["unchanged"] for row in report["protected_files_unchanged"].values())
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in report["field_provenance"]
        assert field in report["field_principles"]
    mod.validate_artifact(report)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (lambda artifact: artifact.update(status="bootstrap"), "status"),
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="mixed: ambiguous"), "honest_verdict"),
        (lambda artifact: artifact.update(next_range_collision_count="0"), "next_range_collision_count"),
        (
            lambda artifact: artifact.update(status="complete", next_range_collision_count=1),
            "next_range_collision_count must be zero",
        ),
        (lambda artifact: artifact.update(research_complete_append_count=2), "research_complete_append_count"),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact.update(retired_and_science_ready_preserved=False),
            "laundered",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"][
                mod.CONDUCTOR_RELATIVE_PATH.as_posix()
            ].update(unchanged=False),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (lambda artifact: artifact["field_provenance"].pop("status"), "field provenance missing"),
        (lambda artifact: artifact.update(outcome_classification=[]), "outcome_classification"),
        (
            lambda artifact: artifact["outcome_classification"][
                "gate_blocked_task_ids"
            ].append("exp5879-hardness-headroom-taxonomy-corrigendum"),
            "terminal classes",
        ),
        (
            lambda artifact: artifact.update(activated_task_and_deliverable_matrix=[]),
            "adversarial verifier receipt matrix",
        ),
        (
            lambda artifact: artifact["outcome_classification"].update(
                terminal_class_by_task_id=[]
            ),
            "terminal class map",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp5882-shortcut-resistant-continuous-self-learning"
            ),
            "exactly six",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5877-transition-v523": []}
            ),
            "malformed matrix row",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5877-transition-v523"
            ].update(identity=["2026.07.523", "exp5877-transition-v523", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5880-grounding-shortcut-fixture"
            ].update(present=True),
            "Exp5880 missing deliverable",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5879-hardness-headroom-taxonomy-corrigendum"
            ].update(science_ready_scalar=0.0),
            "Exp5879 science-ready scalar",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5877-transition-v523"
            ].pop("receipt_hash"),
            "missing adversarial verifier receipt fields",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5877-transition-v523"
            ].update(
                command=(
                    ".venv/bin/python scripts/adversarial_verify.py "
                    "--milestone-range 5877 5882 --json"
                )
            ),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["unactivated_proposal_id_receipt"].update(
                appended_as_completed=True
            ),
            "unactivated proposal IDs",
        ),
        (
            lambda artifact: artifact["missing_and_gate_blocked_receipts"][
                "receipts"
            ][0].update(treated_as_success=True),
            "missing and gate-blocked",
        ),
        (
            lambda artifact: artifact.update(missing_and_gate_blocked_receipts=[]),
            "missing and gate-blocked receipts missing",
        ),
        (
            lambda artifact: artifact["missing_and_gate_blocked_receipts"].update(
                missing_task_ids=["exp5880-grounding-shortcut-fixture"]
            ),
            "missing and gate-blocked receipts changed",
        ),
        (lambda artifact: artifact.update(next_task_range=[]), "next_task_range"),
        (
            lambda artifact: artifact["next_task_range"].update(end="exp5902"),
            "Exp5903",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["duration_s"] = 9.5
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)


def test_req_report_5890_helpers_cover_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5890: helper branches remain deterministic and auditable."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    _, bad_json_meta = mod._read_json_mapping(bad_json)
    assert bad_json_meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _, list_json_meta = mod._read_json_mapping(list_json)
    assert list_json_meta["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n", encoding="utf-8")
    _, bad_yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert bad_yaml_meta["error"].startswith("yaml_error:")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- a\n", encoding="utf-8")
    _, list_yaml_meta = mod._read_yaml_mapping(list_yaml)
    assert list_yaml_meta["error"] == "yaml_not_mapping"

    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    normalized = mod._normalize_adversarial_receipts(
        [_receipt("exp5877-transition-v523")],
        {"exp5877-transition-v523": {"present": True}},
    )
    assert normalized["exp5877-transition-v523"]["flag_count"] == 0
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._failed_required_test_commands(
        [
            {
                "command": ".venv/bin/python scripts/adversarial_verify.py --json x",
                "exit_code": 1,
            },
            {"command": "nonblocking failure", "exit_code": 1, "blocking": False},
            {"command": "real failure", "exit_code": 2},
        ]
    ) == ["real failure"]
    assert mod._failed_nonblocking_test_commands(
        [
            {"command": "nonblocking failure", "exit_code": 1, "blocking": False},
            {"command": "passing", "exit_code": 0, "blocking": False},
        ]
    ) == ["nonblocking failure"]
    assert mod._status_from_log(None) == "MISSING"
    assert mod._status_from_log("| x | UNKNOWN | y |") == "LOGGED"
    assert mod._task_signature({"tasks": "bad"}) == ()
    assert mod._completion_task_rows(tmp_path / "no-history") == []
    append_root = tmp_path / "append-missing-file"
    append_receipt = mod._append_completion_if_absent(append_root)
    assert append_receipt["append_count"] == 1
    assert (append_root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).exists()

    payloads = {task_id: {"status": "complete"} for task_id in mod.EXPECTED_TASK_IDS}
    metadata = {task_id: {"present": True} for task_id in mod.EXPECTED_TASK_IDS}
    metadata["exp5877-transition-v523"] = {"present": False}
    payloads["exp5878-v523-source-delta-ingestion"] = {"status": "mystery"}
    classes = mod._classify_outcomes(payloads, metadata, {})
    assert "exp5877-transition-v523" in classes["missing_declared_deliverable_task_ids"]
    assert "exp5878-v523-source-delta-ingestion" in classes["off_path_task_ids"]

    real_write_text = Path.write_text

    def broken_probe_write(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name.endswith(".tmp-probe"):
            raise OSError("fixture")
        return real_write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", broken_probe_write)
        receipt = mod._atomic_output_receipt(tmp_path / "artifact.json")
        assert receipt["ok"] is False
        assert receipt["error"].startswith("OSError:")

    _make_root(tmp_path / "missing-receipt")
    report = mod.build_report(
        tmp_path / "missing-receipt",
        adversarial_receipts={},
        tests_run=[{"command": "focused", "exit_code": 1}],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.0,
    )
    assert report["status"] == "blocked"
    assert report["preconditions_checked"]["failed_required_test_commands"] == ["focused"]
    assert "missing_adversarial_receipts" in report["preconditions_checked"][
        "failed_preconditions"
    ]
    with pytest.raises(ValueError, match="missing adversarial verifier receipt"):
        mod.validate_artifact(report)

    bad_root = tmp_path / "bad-preconditions"
    _make_root(bad_root)
    _write_text(bad_root, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    (bad_root / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()

    with monkeypatch.context() as patch:
        patch.setattr(
            mod,
            "_atomic_output_receipt",
            lambda path: {"ok": False, "declared_path": path.as_posix()},
        )
        patch.setattr(
            mod,
            "_resource_receipts",
            lambda root: {
                "disk": {"ok": False, "available_mb": 1, "required_mb": 512},
                "memory": {"ok": True, "available_mb": 1024, "required_mb": 512},
            },
        )
        bad_report = mod.build_report(
            bad_root,
            adversarial_receipts={},
            tests_run=[{"command": "focused", "exit_code": 0}],
            modification_overrides={
                **{rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
                mod.CONDUCTOR_RELATIVE_PATH: True,
            },
            duration_s=1.0,
        )
    assert {
        "active_roadmap_unloadable",
        "live_verifier_missing",
        "atomic_output_unavailable",
        "insufficient_resources",
        "research_complete_unparseable",
        "protected_file_modified",
    } <= set(bad_report["preconditions_checked"]["failed_preconditions"])

    amp_root = tmp_path / "duplicate-amplification"
    _make_root(amp_root)

    def amplified_append(root: Path) -> JsonDict:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "fixture",
            "before_sha256": "sha256:before",
            "after_sha256": "sha256:after",
            "before_duplicate_block_count": 0,
            "after_duplicate_block_count": 1,
            "duplicate_history_amplification_count": 1,
        }

    with monkeypatch.context() as patch:
        patch.setattr(mod, "_append_completion_if_absent", amplified_append)
        amp_report = mod.build_report(
            amp_root,
            adversarial_receipts=_receipts(),
            tests_run=[{"command": "focused", "exit_code": 0}],
            modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
            duration_s=1.0,
        )
    assert "duplicate_history_amplified" in amp_report["preconditions_checked"][
        "failed_preconditions"
    ]

    not_preserved_root = tmp_path / "not-preserved"
    _make_root(not_preserved_root)
    with monkeypatch.context() as patch:
        patch.setattr(mod, "_retired_and_science_ready_preserved", lambda classes, payloads: False)
        not_preserved = mod.build_report(
            not_preserved_root,
            adversarial_receipts=_receipts(),
            tests_run=[{"command": "focused", "exit_code": 0}],
            modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
            duration_s=1.0,
        )
    assert "terminal_outcomes_not_preserved" in not_preserved["preconditions_checked"][
        "failed_preconditions"
    ]

    mismatch_root = tmp_path / "declared-mismatch"
    _make_root(mismatch_root)
    data = yaml.safe_load((mismatch_root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    v523 = next(row for row in data["milestones"] if row["id"] == mod.MILESTONE_FROM)
    v523["tasks"][0]["deliverable"] = "results/experiment_5877_alias.json"
    _write_json(mismatch_root, "results/experiment_5877_alias.json", _artifact("exp5877-transition-v523"))
    _write_text(mismatch_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(data))
    mismatch_report = mod.build_report(
        mismatch_root,
        adversarial_receipts={
            **_receipts(),
            "exp5877-transition-v523": {
                **_receipt("exp5877-transition-v523"),
                "artifact_path": "results/experiment_5877_alias.json",
                "command": (
                    ".venv/bin/python scripts/adversarial_verify.py --json "
                    "results/experiment_5877_alias.json"
                ),
            },
        },
        tests_run=[{"command": "focused", "exit_code": 0}],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.0,
    )
    assert "declared_deliverable_mismatch" in mismatch_report["preconditions_checked"][
        "failed_preconditions"
    ]
