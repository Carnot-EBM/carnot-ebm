"""Tests for REQ-REPORT-7075 and SCENARIO-REPORT-7075-* evidence paths."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot import experiment_7075_v619_capstone as mod


REPO = Path(__file__).resolve().parents[2]


def _prompt(produced_fields: tuple[str, ...] = ()) -> str:
    """Build the minimum prompt contract used by the capstone fixture."""

    fields = "; ".join((*mod.COMMON_UPSTREAM_FIELDS, *produced_fields))
    return (
        f"REQUIRED ARTIFACT FIELDS: {fields}.\n"
        "Run command: fixture\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )


def _roadmap() -> dict[str, object]:
    """Create YAML data without sharing objects with the Markdown fixture."""

    tasks = []
    for expected in mod.EXPECTED_TASKS:
        task: dict[str, object] = {
            "id": expected["id"],
            "title": expected["title"],
            "deliverable": expected["deliverable"],
            "milestone": mod.MILESTONE,
            "per_unit_rows": True,
            "prompt": _prompt(mod.PRODUCER_FIELDS.get(expected["id"], ())),
        }
        gates = mod.EXPECTED_GATES.get(expected["id"], ())
        if gates:
            task["gated_on"] = [
                {
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": "==",
                    "value": value,
                }
                for upstream, field, value in gates
            ]
        priors = mod.EXPECTED_PRIORS.get(expected["id"], ())
        if priors:
            task["prior_failures"] = [deepcopy(row) for row in priors]
        tasks.append(task)
    return {
        "milestone": mod.MILESTONE,
        "milestone_doc": str(mod.DESIGN_PATH),
        "tasks": tasks,
    }


def _markdown() -> str:
    """Create a separate Markdown contract with its own task and gate text."""

    rows = [
        f"| {index} | `{task['id']}` | {task['title']} | `{task['deliverable']}` |"
        for index, task in enumerate(mod.EXPECTED_TASKS, 1)
    ]
    sections = []
    for task in mod.EXPECTED_TASKS:
        gates = mod.EXPECTED_GATES.get(task["id"], ())
        if not gates:
            continue
        lines = [f"### Exp{task['id'][3:7]} - {task['title']}", "", "**Gates:**", ""]
        lines.extend(f"- `{upstream}.{field} == {value}`" for upstream, field, value in gates)
        sections.append("\n".join(lines))
    return (
        "# V619 fixture\n\n"
        f"**Milestone:** `{mod.MILESTONE}`\n\n"
        "## Exact Task Contract\n\n"
        "| Order | ID | Title | Deliverable |\n"
        "|---:|---|---|---|\n" + "\n".join(rows) + "\n\n" + "\n\n".join(sections) + "\n"
    )


def _write_inputs(root: Path, *, roadmap: dict[str, object] | None = None) -> dict[str, object]:
    """Write only capstone inputs and keep all result writes under tmp_path."""

    data = roadmap or _roadmap()
    (root / "openspec/change-proposals").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / mod.ROADMAP_PATH).write_text(yaml.safe_dump(data, sort_keys=False))
    (root / mod.DESIGN_PATH).write_text(_markdown())
    (root / mod.CONDUCTOR_LOG_PATH).write_text("V619 fixture conductor log\n")
    (root / mod.EXCLUSION_PATH).write_text(
        "retired: []\nretired_experiments: []\nretired_extras: []\n"
    )
    return data


def _artifact(verdict_class: str = "positive", **updates: object) -> dict[str, object]:
    """Build one schema-complete upstream receipt for a named evidence class."""

    verdicts = {
        "positive": "complete_positive_fixture",
        "circular_positive": "complete_circular_positive_fixture",
        "null": "complete_null_fixture",
        "blocked": "blocked_fixture_resource",
        "disqualified": "complete_disqualified_fixture",
        "partial": "complete_partial_fixture",
    }
    artifact: dict[str, object] = {
        "field_principles": {
            field: "The field keeps the fixture falsifiable."
            for field in mod.COMMON_UPSTREAM_FIELDS
        },
        "preconditions_checked": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 1.0,
        "source_artifact_hashes": {},
        "rows": [],
        "random_seed": 1,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "passed": verdict_class != "blocked",
            "failed_check": "fixture_resource" if verdict_class == "blocked" else None,
            "expected_value": True,
            "observed_value": verdict_class != "blocked",
        },
        "verifier_is_oracle": verdict_class == "circular_positive",
        "verdict_class": verdict_class,
        "honest_verdict": verdicts[verdict_class],
        **updates,
    }
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    return artifact


def _positive_artifacts() -> dict[str, dict[str, object]]:
    """Supply positive terminal evidence and all gate-producing fields."""

    artifacts: dict[str, dict[str, object]] = {}
    for task in mod.EXPECTED_TASKS[:-1]:
        produced = {field: 1 for field in mod.PRODUCER_FIELDS.get(task["id"], ())}
        artifact = _artifact(**produced)
        if task["id"] in mod.COMPARISON_TASKS:
            artifact["rows"] = [
                {"unit_id": "u1", "treatment": 1.0, "control": 0.0},
                {"unit_id": "u2", "treatment": 1.0, "control": 0.0},
            ]
            artifact["comparative_headline"] = {
                "treatment_field": "treatment",
                "control_field": "control",
                "higher_is_better": True,
                "wins": 2,
                "losses": 0,
                "ties": 0,
                "comparable_units": 2,
            }
        gates = mod.EXPECTED_GATES.get(task["id"], ())
        if gates:
            artifact["upstream_gate_rows"] = [
                {
                    "check": field,
                    "expected_value": value,
                    "observed_value": value,
                    "passed": True,
                }
                for _, field, value in gates
            ]
        artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
        artifacts[task["id"]] = artifact
    return artifacts


def _write_artifacts(root: Path, artifacts: dict[str, dict[str, object]]) -> None:
    """Materialize fixture artifacts at their declared deliverable paths."""

    for task_id, artifact in artifacts.items():
        relative = next(row["deliverable"] for row in mod.EXPECTED_TASKS if row["id"] == task_id)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n")


def _replace_artifact(
    artifacts: dict[str, dict[str, object]], task_id: str, verdict_class: str, **updates: object
) -> None:
    """Replace one fixture while retaining its gate-producing fields."""

    produced = {field: updates.pop(field, 0) for field in mod.PRODUCER_FIELDS.get(task_id, ())}
    artifact = _artifact(verdict_class, **produced, **updates)
    gates = mod.EXPECTED_GATES.get(task_id, ())
    if gates:
        artifact["upstream_gate_rows"] = [
            {
                "check": field,
                "expected_value": value,
                "observed_value": value,
                "passed": True,
            }
            for _, field, value in gates
        ]
        artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    artifacts[task_id] = artifact


def test_req_report_7075_is_specified_before_implementation() -> None:
    """REQ-REPORT-7075: the reporting contract names all required scenarios."""

    text = (REPO / "openspec/capabilities/research-reporting/spec.md").read_text()
    section = text[text.index("### REQ-REPORT-7075") :]
    for scenario in ("CLASSES", "ROWS", "GATES", "CONTRACT", "PREFLIGHT", "HANDOFF"):
        assert f"SCENARIO-REPORT-7075-{scenario}" in section


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        (None, "missing"),
        (_artifact("blocked"), "blocked"),
        (_artifact("null"), "null"),
        (_artifact("circular_positive"), "circular_positive"),
        (_artifact("disqualified"), "disqualified"),
        (_artifact("partial"), "partial"),
        (_artifact("positive"), "positive"),
    ],
)
def test_all_evidence_classes_remain_distinct(
    fixture: dict[str, object] | None, expected: str
) -> None:
    """SCENARIO-REPORT-7075-CLASSES keeps six classes plus missing distinct."""

    row = mod.classify_artifact(fixture)
    assert row["effective_verdict_class"] == expected


def test_missing_artifacts_are_rows_and_do_not_block_capstone(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-CLASSES records absent tasks as evidence gaps."""

    _write_inputs(tmp_path)
    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["verdict_class"] == "null"
    assert artifact["milestone_release_ready_score"] == 1
    assert (
        sum(row["discovery_result"] == "missing" for row in artifact["artifact_discovery_rows"])
        == 12
    )
    assert {row["decision"] for row in artifact["branch_decision_rows"]} == {"blocked_resource"}


def test_blocked_and_null_fixtures_drive_exact_branch_decisions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-HANDOFF applies resource and repeated-null rules."""

    _write_inputs(tmp_path)
    artifacts = _positive_artifacts()
    _replace_artifact(artifacts, "exp7065-three-family-entrance-proposal-bank", "blocked")
    _replace_artifact(
        artifacts,
        "exp7070-bcit-prospective-self-learning",
        "null",
        bcit_comparison_complete_score=1,
    )
    _replace_artifact(artifacts, "exp7071-bcit-drift-rollback-audit", "null")
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["entrance_branch_decision"] == "blocked_resource"
    assert artifact["self_learning_branch_decision"] == "retire_null"
    self_retirements = [
        row
        for row in artifact["retirement_rows"]
        if row["task_id"]
        in {
            "exp7070-bcit-prospective-self-learning",
            "exp7071-bcit-drift-rollback-audit",
        }
    ]
    assert all(row["retirement_triggered"] for row in self_retirements)
    self_handoff = next(
        row for row in artifact["v620_handoff_rows"] if row["branch"] == "self_learning"
    )
    assert self_handoff["continuous_learning_follow_up"] is None


def test_circular_and_disqualified_evidence_cannot_release(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-CLASSES carries oracle and disqualification limits."""

    _write_inputs(tmp_path)
    artifacts = _positive_artifacts()
    _replace_artifact(
        artifacts,
        "exp7067-hopfield-entrance-energy-selection",
        "circular_positive",
        entrance_energy_comparison_complete_score=1,
    )
    _replace_artifact(artifacts, "exp7072-live-arc-compaction-ab", "disqualified")
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["entrance_branch_decision"] == "needs_independent_replication"
    assert artifact["arc_compaction_branch_decision"] == "retire_disqualified"
    assert artifact["milestone_release_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"


def test_positive_fixture_releases_clean_branches(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-HANDOFF releases only clean oracle-distinct evidence."""

    _write_inputs(tmp_path)
    _write_artifacts(tmp_path, _positive_artifacts())

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["milestone_release_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive")
    assert {row["decision"] for row in artifact["branch_decision_rows"]} == {"release"}
    assert mod.validate_artifact(artifact) == []


def test_row_contradiction_overrides_pooled_positive(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-ROWS rejects a headline contradicted by unit rows."""

    _write_inputs(tmp_path)
    artifacts = _positive_artifacts()
    target = artifacts["exp7072-live-arc-compaction-ab"]
    target["rows"] = [
        {"unit_id": "u1", "treatment": 0.0, "control": 1.0},
        {"unit_id": "u2", "treatment": 0.0, "control": 1.0},
    ]
    target["comparative_headline"] = {
        "treatment_field": "treatment",
        "control_field": "control",
        "higher_is_better": True,
        "wins": 2,
        "losses": 0,
        "ties": 0,
        "comparable_units": 2,
    }
    target["reproducibility_checksum"] = mod.artifact_checksum(target)
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")
    headline = next(
        row
        for row in artifact["row_headline_recomputation_rows"]
        if row["task_id"] == "exp7072-live-arc-compaction-ab"
    )

    assert headline["recomputed"] == {"wins": 0, "losses": 2, "ties": 0, "comparable_units": 2}
    assert headline["matches_claimed"] is False
    assert artifact["arc_compaction_branch_decision"] == "needs_independent_replication"
    assert artifact["milestone_release_ready_score"] == 0


def test_gate_mismatch_stays_on_schema_axis(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-GATES rejects a consumer gate that conflicts with its producer."""

    _write_inputs(tmp_path)
    artifacts = _positive_artifacts()
    consumer = artifacts["exp7065-three-family-entrance-proposal-bank"]
    consumer["upstream_gate_rows"] = [
        {
            "check": "entrance_fixture_ready_score",
            "expected_value": 1,
            "observed_value": 0,
            "passed": False,
        }
    ]
    consumer["reproducibility_checksum"] = mod.artifact_checksum(consumer)
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")
    schema = next(
        row
        for row in artifact["artifact_schema_result_rows"]
        if row["task_id"] == "exp7065-three-family-entrance-proposal-bank"
    )

    assert "gate_evidence_mismatch" in schema["errors"]
    assert artifact["milestone_release_ready_score"] == 0


def test_contract_mismatch_is_disqualified(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-CONTRACT keeps an active-YAML title drift visible."""

    roadmap = _roadmap()
    roadmap["tasks"][4]["title"] = "Changed title"
    _write_inputs(tmp_path, roadmap=roadmap)

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert artifact["milestone_release_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert any(not row["matches"] for row in artifact["contract_recomputation_rows"])


def test_missing_precondition_blocks_with_exact_summary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-PREFLIGHT blocks only on a capstone prerequisite."""

    _write_inputs(tmp_path)
    (tmp_path / mod.CONDUCTOR_LOG_PATH).unlink()

    artifact = mod.build_capstone(tmp_path, "20260906")

    assert mod.REQUIRED_FIELDS <= artifact.keys()
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked")
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "conductor_log_readable",
        "expected_value": "readable_nonempty_file",
        "observed_value": "missing",
    }
    assert mod.validate_artifact(artifact) == []


def test_terminal_checksum_and_artifact_validator_detect_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-7075 validates its own required fields, score, and terminal checksum."""

    _write_inputs(tmp_path)
    artifact = mod.build_capstone(tmp_path, "20260906")
    assert mod.validate_artifact(artifact) == []

    artifact["observed_task_count"] = 99
    assert "observed_task_count_mismatch" in mod.validate_artifact(artifact)
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(artifact)


def test_defensive_parsers_preserve_malformed_evidence(tmp_path: Path) -> None:
    """REQ-REPORT-7075 keeps malformed files and unusual fields explicit."""

    assert mod.sha256_file(tmp_path / "missing") is None
    assert mod._infer_verdict_class("not terminal") is None
    oracle_positive = _artifact("positive", verifier_is_oracle=True)
    assert mod.classify_artifact(oracle_positive)["effective_verdict_class"] == "circular_positive"
    assert mod._normalized_gates({"gated_on": "bad"}) == ()

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{")
    assert "JSONDecodeError" in str(mod._load_artifact(invalid_json)[1])
    list_json = tmp_path / "list.json"
    list_json.write_text("[]")
    assert mod._load_artifact(list_json)[1] == "artifact_root_not_mapping"

    source = tmp_path / "source.txt"
    source.write_text("source\n")
    source_rows = mod._source_hash_evidence(
        tmp_path,
        "task",
        {
            "source_artifact_hashes": {
                "files": [{"path": "source.txt", "sha256": mod.sha256_file(source)}]
            }
        },
    )
    assert source_rows[0]["matches"] is True

    class BrokenPath:
        """Expose an unreadable stat call without touching protected files."""

        @staticmethod
        def is_file() -> bool:
            return True

        @staticmethod
        def stat() -> object:
            raise OSError("fixture")

    assert mod._readable(BrokenPath())[0] is False


def test_model_headline_schema_and_gate_defenses_cover_bad_shapes() -> None:
    """SCENARIO-REPORT-7075-ROWS rejects absent identity and malformed rows."""

    live = _artifact("positive", inference_substrate="live_llm_inference")
    assert mod._model_identity_row("task", live)["valid"] is False
    live["model_identity_rows"] = [{"model_id": "m"}]
    assert mod._model_identity_row("task", live)["valid"] is True
    blocked = _artifact("blocked", inference_substrate="live_llm_inference", model_identity_rows=[])
    assert mod._model_identity_row("task", blocked)["result"] == "blocked_before_invocation"

    assert (
        mod._headline_row("task", {"comparative_headline": {}, "rows": None})["matches_claimed"]
        is False
    )
    headline = mod._headline_row(
        "task",
        {
            "comparative_headline": {"wins": 0, "losses": 0, "ties": 1, "comparable_units": 1},
            "rows": ["bad", {"treatment": True, "control": 0}, {"treatment": 1, "control": 1}],
        },
    )
    assert headline["matches_claimed"] is True

    malformed = _artifact("positive")
    malformed["honest_verdict"] = "complete_null_conflict"
    malformed["reproducibility_checksum"] = "bad"
    malformed["field_principles"] = []
    errors = mod._upstream_schema_errors(malformed, mod.classify_artifact(malformed))
    assert {
        "verdict_class_mismatch",
        "reproducibility_checksum_mismatch",
        "field_principles_not_mapping",
    } <= set(errors)
    bad_block = _artifact("blocked", gate_check_summary={})
    assert "blocked_gate_summary_incomplete" in mod._upstream_schema_errors(
        bad_block, mod.classify_artifact(bad_block)
    )
    bad_oracle = _artifact("positive", verifier_is_oracle=True)
    assert "positive_depends_on_oracle" in mod._upstream_schema_errors(
        bad_oracle, {"class_matches": True}
    )
    assert mod._gate_evidence_matches({}, "field", 1, 1) is False
    assert mod._gate_evidence_matches({"upstream_gate_rows": ["bad"]}, "field", 1, 1) is False
    assert (
        mod._gate_evidence_matches({"upstream_gate_rows": [{"check": "other"}]}, "field", 1, 1)
        is False
    )


def test_closed_branch_fallbacks_are_terminal() -> None:
    """SCENARIO-REPORT-7075-HANDOFF maps partial and unmatched targets safely."""

    def decision(branch: str, target_classes: list[str]) -> str:
        task_ids = mod.BRANCH_TASKS[branch]
        targets = mod.BRANCH_TARGETS[branch]
        artifacts = {task_id: {} for task_id in task_ids}
        classes = {task_id: {"effective_verdict_class": "positive"} for task_id in task_ids}
        for task_id, value in zip(targets, target_classes, strict=True):
            classes[task_id] = {"effective_verdict_class": value}
        valid = {task_id: True for task_id in task_ids}
        return mod._branch_decision(branch, artifacts, classes, valid)["decision"]

    assert decision("arc_compaction", ["partial"]) == "needs_independent_replication"
    assert decision("self_learning", ["null", "positive"]) == "needs_independent_replication"
    assert decision("arc_compaction", ["unexpected"]) == "needs_independent_replication"


def test_build_records_live_identity_and_source_hash_failures(tmp_path: Path) -> None:
    """REQ-REPORT-7075 carries model and source failures into artifact validity."""

    _write_inputs(tmp_path)
    artifacts = _positive_artifacts()
    target = artifacts["exp7072-live-arc-compaction-ab"]
    target["inference_substrate"] = "live_llm_inference"
    target["model_identity_rows"] = []
    target["source_artifact_hashes"] = {"missing-source": "sha256:" + "0" * 64}
    target["reproducibility_checksum"] = mod.artifact_checksum(target)
    _write_artifacts(tmp_path, artifacts)

    artifact = mod.build_capstone(tmp_path, "20260906")
    schema = next(
        row
        for row in artifact["artifact_schema_result_rows"]
        if row["task_id"] == "exp7072-live-arc-compaction-ab"
    )
    assert {"model_identity_invalid", "source_hash_mismatch"} <= set(schema["errors"])


def test_circular_capstone_class_is_explicit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7075-CLASSES labels an otherwise clean oracle target."""

    _write_inputs(tmp_path)
    artifacts = _positive_artifacts()
    _replace_artifact(
        artifacts,
        "exp7067-hopfield-entrance-energy-selection",
        "circular_positive",
        entrance_energy_comparison_complete_score=1,
    )
    _write_artifacts(tmp_path, artifacts)
    artifact = mod.build_capstone(tmp_path, "20260906")
    assert artifact["verdict_class"] == "circular_positive"


def test_validator_reports_every_closed_schema_error(tmp_path: Path) -> None:
    """REQ-REPORT-7075 validates principles, substrate, counts, decisions, and class."""

    _write_inputs(tmp_path)
    artifact = mod.build_capstone(tmp_path, "20260906")
    artifact["field_principles"] = {}
    artifact["inference_substrate"] = "bad"
    artifact["expected_task_count"] = 12
    artifact["verifier_is_oracle"] = True
    artifact["honest_verdict"] = "complete_positive_conflict"
    artifact["entrance_branch_decision"] = "bad"
    artifact["milestone_release_ready_score"] = 2
    errors = set(mod.validate_artifact(artifact))
    assert {
        "field_principles_incomplete",
        "inference_substrate_mismatch",
        "expected_task_count_mismatch",
        "verifier_is_oracle_must_be_false",
        "verdict_class_mismatch",
        "invalid_branch_decision:entrance_branch_decision",
        "milestone_release_ready_score_invalid",
    } <= errors

    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = "complete_blocked_fixture"
    artifact["gate_check_summary"] = {}
    assert "blocked_gate_summary_incomplete" in mod.validate_artifact(artifact)


def test_main_rejects_an_invalid_built_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7075 refuses to write when final self-validation fails."""

    _write_inputs(tmp_path)
    monkeypatch.setattr(mod, "validate_artifact", lambda artifact: ["fixture_error"])
    with pytest.raises(ValueError, match="fixture_error"):
        mod.main(["--root", str(tmp_path), "--output", "results/invalid.json"])


def test_module_entrypoint_runs_with_explicit_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7075 keeps the direct module entrypoint inside tmp_path."""

    _write_inputs(tmp_path)
    output = tmp_path / "results/direct.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_7075_v619_capstone.py", "--root", str(tmp_path), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("carnot.experiment_7075_v619_capstone", run_name="__main__")
    assert exc.value.code == 0
    assert output.is_file()


def test_cli_writes_requested_output(tmp_path: Path) -> None:
    """REQ-REPORT-7075 exposes one deterministic command-line artifact path."""

    _write_inputs(tmp_path)
    output = tmp_path / "results/capstone.json"
    assert mod.main(["--root", str(tmp_path), "--date", "20260906", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["expected_task_count"] == 13
