"""Tests for REQ-CAP-7049 and SCENARIO-CAP-7049-* branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7049_v617_capstone_disposition as mod


REPO = Path(__file__).resolve().parents[2]


def _roadmap() -> dict:
    tasks = []
    for index, task_id in enumerate(mod.EXPECTED_TASK_IDS):
        number = 7038 + index
        task = {
            "id": task_id,
            "title": f"Task {number}",
            "deliverable": mod.TASK_PATHS[task_id],
            "prompt": "REQUIRED ARTIFACT FIELDS: rows; verdict_class; honest_verdict.",
        }
        gates = mod.EXPECTED_GATES.get(task_id, ())
        if gates:
            task["gated_on"] = [
                {"upstream": upstream, "artifact_field": field, "op": "==", "value": 1}
                for upstream, field in gates
            ]
        tasks.append(task)
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _design(roadmap: dict) -> str:
    rows = []
    for order, task in enumerate(roadmap["tasks"], 1):
        gates = task.get("gated_on", [])
        prerequisites = "; ".join(
            f"Exp{mod.task_number(gate['upstream'])} `{gate['artifact_field']} == {gate['value']}`"
            for gate in gates
        ) or "None"
        rows.append(
            f"| {order} | `{task['id']}` | {task['title']} | "
            f"`{task['deliverable']}` | {prerequisites} |"
        )
    return (
        "# V617\n\n"
        f"**Milestone:** `{mod.MILESTONE}`\n\n"
        "## 7. Phases and Exact Task Contract\n\n"
        "| Order | Experiment | Title | Deliverable | Structured prerequisites |\n"
        "|---:|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n### Phase I\n"
    )


def _write_contract(root: Path) -> dict:
    roadmap = _roadmap()
    (root / "openspec/change-proposals").mkdir(parents=True)
    (root / "openspec/capabilities/capstone").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "ops").mkdir()
    (root / "_bmad").mkdir()
    (root / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap, sort_keys=False))
    (root / mod.DESIGN_PATH).write_text(_design(roadmap))
    (root / mod.EXCLUSION_PATH).write_text("retired: []\nretired_experiments: []\nretired_extras: []\n")
    (root / mod.SPEC_PATH).write_text("REQ-CAP-7049\n")
    (root / "ops/status.md").write_text("status\n")
    (root / "ops/changelog.md").write_text("changelog\n")
    (root / "_bmad/traceability.md").write_text("traceability\n")
    return roadmap


def _artifact(verdict_class: str = "positive", **fields: object) -> dict:
    verdicts = {
        "positive": "complete_positive_fixture",
        "null": "complete_null_fixture",
        "blocked": "complete_blocked_fixture",
        "disqualified": "complete_disqualified_fixture",
        "partial": "complete_partial_fixture",
        "circular_positive": "complete_circular_positive_fixture",
    }
    artifact = {
        "verdict_class": verdict_class,
        "honest_verdict": verdicts[verdict_class],
        "rows": [{"row_id": "fixture", "terminal": True}],
        "source_artifact_hashes": {},
        "verifier_is_oracle": False,
        **fields,
    }
    artifact["reproducibility_checksum"] = mod.upstream_checksum(artifact)
    return artifact


def _positive_artifacts() -> dict[str, dict]:
    values = {
        7038: {"v617_task_contract_conforms_score": 1},
        7039: {"arc_report_channel_forensics_ready_score": 1},
        7040: {"arc_typed_identity_bridge_ready_score": 1},
        7041: {"arc_identity_report_attack_audit_ready_score": 1},
        7042: {"belief_shadow_transport_ready_score": 1, "game_level_solve_claim": False},
        7043: {"belief_shadow_trace_audit_ready_score": 1, "game_level_solve_claim": False},
        7044: {
            "uniform_belief_live_ab_complete_score": 1,
            "uniform_belief_value_positive_score": 1,
            "game_level_solve_claim": False,
        },
        7045: {
            "uniform_belief_value_audit_complete_score": 1,
            "uniform_belief_value_audit_positive_score": 1,
        },
        7046: {"belief_frontier_curriculum_ready_score": 1},
        7047: {
            "selective_belief_policy_safety_score": 1,
            "selective_belief_policy_nontrivial_score": 1,
            "abstention_count": 2,
            "helpful_use_count": 8,
            "harmful_use_count": 0,
            "final_policy_hash": "sha256:policy",
            "continuous_self_learning_task": True,
        },
        7048: {
            "selective_belief_live_ab_complete_score": 1,
            "selective_belief_live_value_positive_score": 1,
            "final_policy_hash": "sha256:policy",
            "rollback_rows": [{"passed": True}],
            "game_level_solve_claim": False,
            "continuous_self_learning_task": True,
        },
    }
    return {mod.EXPECTED_TASK_IDS[n - 7038]: _artifact(**values[n]) for n in values}


def _write_artifacts(root: Path, artifacts: dict[str, dict]) -> None:
    for task_id, artifact in artifacts.items():
        path = root / mod.TASK_PATHS[task_id]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n")


def _refresh(artifact: dict) -> None:
    artifact["reproducibility_checksum"] = mod.upstream_checksum(artifact)


def _refresh_capstone(artifact: dict) -> None:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)


def test_req_cap_7049_is_specified_before_implementation() -> None:
    """REQ-CAP-7049: the capstone contract exists before its implementation."""
    text = (REPO / "openspec/capabilities/capstone/spec.md").read_text()
    assert "REQ-CAP-7049" in text
    assert "SCENARIO-CAP-7049-ALL-POSITIVE" in text
    assert "SCENARIO-CAP-7049-INTEGRITY" in text


def test_all_positive_releases_only_default_off_canary(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-ALL-POSITIVE: release stays bounded and reversible."""
    _write_contract(tmp_path)
    _write_artifacts(tmp_path, _positive_artifacts())

    artifact = mod.build_capstone(tmp_path, "20260906")
    decisions = {row["branch"]: row for row in artifact["release_hold_repair_retire_rows"]}

    assert mod.REQUIRED_FIELDS <= artifact.keys()
    assert artifact["v617_capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert decisions["identity"]["disposition"] == "release"
    assert decisions["uniform_belief"]["disposition"] == "release"
    assert decisions["selective_learning"]["disposition"] == "release"
    assert decisions["selective_learning"]["release_mode"] == "default_off_canary"
    assert decisions["selective_learning"]["policy_hash"] == "sha256:policy"
    assert decisions["selective_learning"]["rollback_ready"] is True
    assert artifact["production_default_unchanged"] is True
    assert len(artifact["next_handoff"]) == 1
    assert mod.validate_artifact(artifact) == []


def test_identity_block_is_a_repair_disposition(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-BRANCHES: a named identity defect requires repair."""
    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    identity = artifacts[mod.EXPECTED_TASK_IDS[2]]
    identity.update(
        verdict_class="blocked",
        honest_verdict="complete_blocked_identity_hash_mismatch",
        arc_typed_identity_bridge_ready_score=0,
    )
    _refresh(identity)
    for index in range(3, 11):
        blocked = artifacts[mod.EXPECTED_TASK_IDS[index]]
        blocked.update(verdict_class="blocked", honest_verdict="complete_blocked_identity_gate")
        for field in mod.OWN_GATE_FIELDS.get(mod.EXPECTED_TASK_IDS[index], ()):
            blocked[field] = 0
        _refresh(blocked)
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")
    decisions = {row["branch"]: row for row in artifact["release_hold_repair_retire_rows"]}

    assert decisions["identity"]["disposition"] == "repair"


def test_uniform_null_retires_only_exact_influence_scope(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-BRANCHES: a reproduced uniform null retires exact use."""
    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    for index in (6, 7):
        uniform = artifacts[mod.EXPECTED_TASK_IDS[index]]
        uniform.update(verdict_class="null", honest_verdict="complete_null_uniform_belief")
        for key in tuple(uniform):
            if "positive_score" in key:
                uniform[key] = 0
        _refresh(uniform)
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")
    decisions = {row["branch"]: row for row in artifact["release_hold_repair_retire_rows"]}

    assert decisions["uniform_belief"]["disposition"] == "retire"
    assert artifact["exclusion_manifest_rows"][0]["scope_key"] == mod.UNIFORM_SCOPE_KEY
    assert artifact["exclusion_manifest_rows"][0]["preserved_scopes"] == [
        "safe_belief_ledger",
        "bounded_belief_query_api",
        "exact_outcome_memory_substrate",
    ]
    assert artifact["verdict_class"] == "null"
    assert artifact["v617_capstone_complete_score"] == 1


def test_uniform_disqualification_and_tampered_hash_require_repair(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-INTEGRITY: invalid uniform evidence cannot promote."""
    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    uniform = artifacts[mod.EXPECTED_TASK_IDS[7]]
    uniform.update(
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_uniform_chronology",
        uniform_belief_value_audit_positive_score=0,
    )
    _refresh(uniform)
    artifacts[mod.EXPECTED_TASK_IDS[1]]["rows"].append({"tampered": True})
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")
    decisions = {row["branch"]: row for row in artifact["release_hold_repair_retire_rows"]}
    task_rows = {row["task_id"]: row for row in artifact["task_artifact_rows"]}

    assert decisions["uniform_belief"]["disposition"] == "repair"
    assert task_rows[mod.EXPECTED_TASK_IDS[1]]["effective_verdict_class"] == "disqualified"
    assert "reproducibility_checksum_mismatch" in task_rows[mod.EXPECTED_TASK_IDS[1]]["integrity_errors"]
    assert artifact["verdict_class"] == "disqualified"


def test_safe_all_abstain_is_null_and_keeps_live_transfer_blocked(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-BRANCHES: safe all-abstain learning is a null result."""
    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    learner = artifacts[mod.EXPECTED_TASK_IDS[9]]
    learner.update(
        verdict_class="null",
        honest_verdict="complete_null_safe_all_abstain",
        selective_belief_policy_nontrivial_score=0,
        abstention_count=30,
        helpful_use_count=0,
        harmful_use_count=0,
    )
    _refresh(learner)
    transfer = artifacts[mod.EXPECTED_TASK_IDS[10]]
    transfer.update(
        verdict_class="blocked",
        honest_verdict="complete_blocked_selective_nontrivial_gate",
        selective_belief_live_ab_complete_score=0,
        selective_belief_live_value_positive_score=0,
    )
    _refresh(transfer)
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["continuous_learning_claim_rows"][0]["claim_class"] == "null"
    assert artifact["continuous_learning_claim_rows"][0]["all_abstain"] is True
    assert artifact["selective_transfer_claim_rows"][0]["claim_class"] == "blocked"
    assert artifact["release_hold_repair_retire_rows"][2]["disposition"] == "hold"


def test_selective_gate_block_and_missing_artifact_are_terminal_blocked(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-MISSING: upstream absence is blocked, never partial."""
    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    learner = artifacts[mod.EXPECTED_TASK_IDS[9]]
    learner.update(
        verdict_class="blocked",
        honest_verdict="complete_blocked_curriculum_gate",
        selective_belief_policy_safety_score=0,
        selective_belief_policy_nontrivial_score=0,
    )
    _refresh(learner)
    artifacts.pop(mod.EXPECTED_TASK_IDS[6])
    artifacts.pop(mod.EXPECTED_TASK_IDS[10])
    for index in (7, 8):
        blocked = artifacts[mod.EXPECTED_TASK_IDS[index]]
        blocked.update(verdict_class="blocked", honest_verdict="complete_blocked_missing_uniform_input")
        for field in mod.OWN_GATE_FIELDS[mod.EXPECTED_TASK_IDS[index]]:
            blocked[field] = 0
        for field in mod.VALUE_POSITIVE_FIELDS.values():
            if field in blocked:
                blocked[field] = 0
        _refresh(blocked)
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["v617_capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert len(artifact["blocked_input_rows"]) >= 2
    assert all(row["effective_verdict_class"] != "partial" for row in artifact["blocked_input_rows"])
    assert artifact["release_hold_repair_retire_rows"][2]["disposition"] == "hold"


def test_markdown_yaml_mismatch_is_terminal_disqualification(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-CONTRACT: independent contract mismatch disqualifies."""
    _write_contract(tmp_path)
    _write_artifacts(tmp_path, _positive_artifacts())
    design = tmp_path / mod.DESIGN_PATH
    design.write_text(design.read_text().replace("Task 7044", "Changed task 7044"))

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["verdict_class"] == "disqualified"
    assert artifact["v617_capstone_complete_score"] == 0
    assert any(not row["matches"] for row in artifact["roadmap_contract_rows"])
    assert artifact["honest_verdict"].startswith("complete_disqualified")


def test_missing_design_blocks_capstone_input_but_still_inventories_rows(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-MISSING: a missing capstone input sets complete score zero."""
    _write_contract(tmp_path)
    _write_artifacts(tmp_path, _positive_artifacts())
    (tmp_path / mod.DESIGN_PATH).unlink()

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["verdict_class"] == "blocked"
    assert artifact["v617_capstone_complete_score"] == 0
    assert len(artifact["task_artifact_rows"]) == 12
    assert artifact["gate_check_summary"]["failed_check"] == "v617_markdown_readable"
    assert artifact["gate_check_summary"]["expected_value"] == "readable_nonempty_file"


def test_validator_and_cli_reject_mutation_and_write_atomically(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-INTEGRITY: final schema and checksum are self-checking."""
    _write_contract(tmp_path)
    _write_artifacts(tmp_path, _positive_artifacts())
    artifact = mod.build_capstone(tmp_path, "20260906")
    broken = dict(artifact)
    broken.pop("rows")
    assert mod.validate_artifact(broken) == ["missing_required_fields:rows"]
    changed = dict(artifact)
    changed["production_default_unchanged"] = False
    assert "production_default_changed" in mod.validate_artifact(changed)

    output = tmp_path / "results/out.json"
    assert mod.main(["--date", "20260906", "--root", str(tmp_path), "--output", str(output)]) == 0
    written = json.loads(output.read_text())
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert not list(output.parent.glob("*.tmp"))


def test_real_repository_preserves_current_terminal_block() -> None:
    """REQ-CAP-7049: current missing design is reported without changing defaults."""
    artifact = mod.build_capstone(REPO, "20260906")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "v617_markdown_readable"
    assert artifact["production_default_unchanged"] is True
    assert artifact["observed_id_order"] == list(mod.EXPECTED_TASK_IDS)


def test_defensive_parsers_and_unreadable_artifacts_remain_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CAP-7049-MISSING: malformed inputs remain explicit evidence."""
    assert mod._class_from_verdict("complete_circular-positive_fixture") == "circular_positive"
    assert mod._class_from_verdict("complete_partial_fixture") == "partial"
    assert mod._canonical_gates({"gated_on": "invalid"}) == []
    assert mod._load_yaml_contract(tmp_path / "missing.yaml")[2].startswith("FileNotFoundError")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("tasks: wrong\n")
    assert mod._load_yaml_contract(invalid)[2] == "active roadmap is not a task mapping"
    invalid.write_text("tasks:\n  - bad\n")
    assert mod._load_yaml_contract(invalid)[1][0]["id"] is None
    assert mod._source_hash_rows(tmp_path, {"source_artifact_hashes": []})[0]["matches"] is False

    probe = tmp_path / "probe"
    probe.write_text("x")

    def fail_read(_: Path) -> bytes:
        raise OSError("denied")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    assert mod._readable_file(probe)[1].startswith("OSError")
    monkeypatch.undo()

    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    _write_artifacts(tmp_path, artifacts)
    (tmp_path / mod.TASK_PATHS[mod.EXPECTED_TASK_IDS[0]]).write_text("{")
    (tmp_path / mod.TASK_PATHS[mod.EXPECTED_TASK_IDS[1]]).write_text("[]")
    rows, _ = mod._task_rows(tmp_path)
    assert rows[0]["artifact_state"] == "unreadable"
    assert rows[1]["integrity_errors"] == ["artifact_not_object"]


def test_integrity_failure_classes_and_hold_table(tmp_path: Path) -> None:
    """SCENARIO-CAP-7049-INTEGRITY: every declared field failure is checked."""
    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    cases = {
        0: {"verdict_class": "unknown"},
        1: {"honest_verdict": "complete_null_wrong_class"},
        2: {"rows": None},
        3: {"arc_identity_report_attack_audit_ready_score": None},
        4: {"belief_shadow_transport_ready_score": 0},
        5: {"verdict_class": "blocked", "honest_verdict": "complete_blocked_fixture", "belief_shadow_trace_audit_ready_score": 1},
        6: {"uniform_belief_value_positive_score": None},
        7: {"uniform_belief_value_audit_positive_score": 0},
    }
    for index, changes in cases.items():
        artifacts[mod.EXPECTED_TASK_IDS[index]].update(changes)
        _refresh(artifacts[mod.EXPECTED_TASK_IDS[index]])
    oracle = artifacts[mod.EXPECTED_TASK_IDS[8]]
    oracle["verifier_is_oracle"] = True
    _refresh(oracle)
    _write_artifacts(tmp_path, artifacts)

    rows, loaded = mod._task_rows(tmp_path)
    errors = [error for row in rows for error in row["integrity_errors"]]
    assert "verdict_class_outside_closed_enum" in errors
    assert "honest_verdict_class_mismatch" in errors
    assert "rows_missing_or_not_list" in errors
    assert errors.count("row_to_headline_mismatch") >= 4
    assert rows[8]["effective_verdict_class"] == "circular_positive"

    for row in rows[2:6]:
        row["effective_verdict_class"] = "blocked"
        row["honest_verdict"] = "complete_blocked_external_resource"
    decisions = mod._dispositions(rows, loaded)
    assert decisions[0]["disposition"] == "hold"
    assert mod._next_handoff(
        [
            {"branch": "identity", "disposition": "release"},
            {"branch": "uniform_belief", "disposition": "repair"},
            {"branch": "selective_learning", "disposition": "hold"},
        ],
        False,
    )[0]["action"].startswith("repair_the_uniform")
    assert mod._gate_replay({"tasks": ["invalid"]}, rows, loaded) == []


def test_partial_and_circular_outcomes_stay_distinct(tmp_path: Path) -> None:
    """REQ-CAP-7049: partial and circular inputs cannot become positive evidence."""
    _write_contract(tmp_path)
    artifacts = _positive_artifacts()
    partial = artifacts[mod.EXPECTED_TASK_IDS[0]]
    partial.update(verdict_class="partial", honest_verdict="complete_partial_fixture")
    _refresh(partial)
    _write_artifacts(tmp_path, artifacts)
    artifact = mod.build_capstone(tmp_path, "20260906")
    assert artifact["verdict_class"] == "partial"
    assert artifact["v617_capstone_complete_score"] == 0

    circular = artifacts[mod.EXPECTED_TASK_IDS[0]]
    circular.update(verdict_class="positive", honest_verdict="complete_positive_fixture", verifier_is_oracle=True)
    _refresh(circular)
    _write_artifacts(tmp_path, artifacts)
    artifact = mod.build_capstone(tmp_path, "20260906")
    assert artifact["verdict_class"] == "circular_positive"


def test_validator_covers_every_final_invariant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CAP-7049-INTEGRITY: final validation fails closed by field."""
    _write_contract(tmp_path)
    _write_artifacts(tmp_path, _positive_artifacts())
    valid = mod.build_capstone(tmp_path, "20260906")
    mutations = [
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("expected_task_count", 11, "expected_task_count_invalid"),
        ("expected_id_order", [], "expected_id_order_invalid"),
        ("random_seed", 0, "random_seed_invalid"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("honest_verdict", "not_terminal", "honest_verdict_invalid"),
        ("duration_s", True, "duration_s_invalid"),
        ("v617_capstone_complete_score", 2, "complete_score_invalid"),
        ("task_artifact_rows", [], "task_artifact_rows_invalid"),
        ("release_hold_repair_retire_rows", [], "disposition_rows_invalid"),
        ("next_handoff", [], "next_handoff_invalid"),
    ]
    for field, value, expected in mutations:
        changed = dict(valid)
        changed[field] = value
        _refresh_capstone(changed)
        assert expected in mod.validate_artifact(changed)

    partial = json.loads(json.dumps(valid))
    partial["task_artifact_rows"][0]["effective_verdict_class"] = "partial"
    _refresh_capstone(partial)
    assert "complete_score_contains_partial" in mod.validate_artifact(partial)
    blocked = dict(valid)
    blocked.update(verdict_class="blocked", honest_verdict="complete_blocked_fixture", gate_check_summary={})
    _refresh_capstone(blocked)
    assert "blocked_gate_summary_invalid" in mod.validate_artifact(blocked)

    with pytest.raises(argparse.ArgumentTypeError):
        mod._date_argument("2026-09-06")
    monkeypatch.setattr(mod, "build_capstone", lambda *args, **kwargs: {})
    with pytest.raises(SystemExit, match="invalid Exp7049 artifact"):
        mod.main(["--date", "20260906", "--root", str(tmp_path)])
