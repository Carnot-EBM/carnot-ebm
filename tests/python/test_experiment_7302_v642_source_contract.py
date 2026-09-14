"""Tests for the V642 source-contract receipt.

Spec refs: REQ-REPORT-7302 and SCENARIO-REPORT-7302-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any

import pytest
import yaml

from carnot import experiment_7302_v642_source_contract as mod


ROOT = Path(__file__).resolve().parents[2]


def _gate_text(gate: dict[str, Any]) -> str:
    """Render one Markdown input without using the production parser."""

    return f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} {json.dumps(gate['value'])}"


def _matching_markdown(roadmap: dict[str, Any]) -> str:
    """Build an independent literal contract fixture from explicit task fields."""

    lines = [
        "# V642 fixture",
        "",
        "**Milestone:** 2026.09.642",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Task ID | Exact title | Deliverable | Phase | Structured gate |",
        "|---|---|---|---|---|---|",
    ]
    for order, task in enumerate(roadmap["tasks"], 1):
        gates = "; ".join(_gate_text(gate) for gate in task.get("gated_on", [])) or "None"
        lines.append(
            f"| {order} | {task['id']} | {task['title']} | {task['deliverable']} | "
            f"{task['phase']} | {gates} |"
        )
    return "\n".join(lines) + "\n"


def _copy(root: Path, relative: Path) -> None:
    """Copy one real input so terminal tests cannot alter repository evidence."""

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


def _private_root(tmp_path: Path, *, exact_markdown: bool) -> Path:
    """Create a complete private repository view for lifecycle tests."""

    root = tmp_path / "repo"
    for relative in (*mod.INPUT_PATHS, *mod.CODE_PATHS, mod.HISTORY_PATH):
        _copy(root, relative)
    if exact_markdown:
        roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text())
        (root / mod.DESIGN_PATH).write_text(_matching_markdown(roadmap), encoding="utf-8")
    for relative in (mod.DEFAULT_OUTPUT_PATH.parent, mod.RAW_DIR, mod.CHECKPOINT_PATH.parent):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _fake_fetch(url: str) -> dict[str, Any]:
    """Return deterministic primary-page text without making a local claim."""

    records = {
        "2511.04108": ("Batch Prompting", "v4"),
        "2503.15551": ("BATCHSAFEBENCH", "v2"),
        "2509.24489": ("Interactive Constraint Refinement", "v1"),
        "2507.02092": ("Energy-Based Transformers", "v2"),
    }
    key = next(candidate for candidate in records if candidate in url)
    title, version = records[key]
    return {
        "ok": True,
        "status_code": 200,
        "body": f"<title>{title}</title> [version:{version}]",
        "error": None,
    }


def _passing_receipts() -> list[dict[str, Any]]:
    """Represent completed checks without recursively starting pytest."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "exit_code": 0,
            "duration_s": 0.001,
            "timed_out": False,
            "log_sha256": mod.sha256_bytes(b"fixture log"),
            "passed": True,
            "baseline_failure": False,
        }
        for name in mod.VALIDATION_COMMAND_NAMES
    ]


def test_scenario_report_7302_contract_rejects_all_named_mutations() -> None:
    """SCENARIO-REPORT-7302-CONTRACT covers count, order, ID, path, phase, and gate."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    markdown = _matching_markdown(roadmap)
    exact = mod.evaluate_contract(markdown, roadmap)
    assert exact["passed"] is True
    assert [row["unit_id"] for row in exact["contract_rows"]] == list(mod.EXPECTED_ID_ORDER)
    assert all(row["markdown"]["phase"] == row["yaml"]["phase"] for row in exact["contract_rows"])

    mutations = []
    short = deepcopy(roadmap)
    short["tasks"].pop()
    mutations.append(short)
    reordered = deepcopy(roadmap)
    reordered["tasks"][4], reordered["tasks"][5] = (
        reordered["tasks"][5],
        reordered["tasks"][4],
    )
    mutations.append(reordered)
    for field, value in (
        ("id", "exp9999-mutated"),
        ("deliverable", "results/mutated.json"),
        ("phase", 9),
    ):
        changed = deepcopy(roadmap)
        changed["tasks"][5][field] = value
        mutations.append(changed)
    changed = deepcopy(roadmap)
    changed["tasks"][5]["gated_on"][0]["artifact_field"] = "mutated_ready_score"
    mutations.append(changed)
    assert all(
        mod.evaluate_contract(markdown, candidate)["passed"] is False for candidate in mutations
    )


def test_req_report_7302_selects_only_v642_active_or_staged(tmp_path: Path) -> None:
    """REQ-REPORT-7302 binds selection to milestone and activation state."""

    root = tmp_path / "authority"
    root.mkdir()
    active = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(active, sort_keys=False))
    authority, roadmap, content, candidates = mod.select_yaml_authority(root)
    assert authority == mod.ACTIVE_ROADMAP_PATH
    assert roadmap is not None and content == (root / authority).read_bytes()
    assert candidates[0]["available"] is True

    active["milestone"] = "2026.09.641"
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(active, sort_keys=False))
    staged = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    (root / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged, sort_keys=False))
    authority, roadmap, _content, candidates = mod.select_yaml_authority(root)
    assert authority == mod.NEXT_ROADMAP_PATH and roadmap is not None
    assert [row["available"] for row in candidates] == [False, True]

    (root / mod.NEXT_ROADMAP_PATH).unlink()
    authority, roadmap, content, candidates = mod.select_yaml_authority(root)
    assert authority is roadmap is content is None
    assert all(row["available"] is False for row in candidates)


def test_scenario_report_7302_gates_use_real_reader_and_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7302-GATES keeps five controls and rejects bad producers."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    rows = mod.run_gate_replays(roadmap, tmp_path / "gates")
    assert len(rows) == 40
    assert mod.gate_replays_complete(rows) is True
    for edge_index in range(8):
        edge = [row for row in rows if row["edge_index"] == edge_index]
        assert [row["case"] for row in edge] == [
            "passing",
            "false",
            "missing_field",
            "missing_file",
            "quarantined",
        ]
        assert [row["conductor_passed"] for row in edge] == [True, False, False, False, True]
        assert [row["experiment_precondition_passed"] for row in edge] == [
            True,
            False,
            False,
            False,
            False,
        ]
    assert mod.gate_replays_complete(rows[:-1]) is False
    assert mod.upstream_precondition(None, "ready_score", 1)["passed"] is False
    assert (
        mod.upstream_precondition(
            {"status": "complete", "ready_score": 1, "flagged_adversarial": True},
            "ready_score",
            1,
        )["reason"]
        == "quarantined_upstream_rejected_before_field_consumption"
    )
    disqualified = mod.upstream_precondition(
        {
            "status": "complete",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_fixture",
            "ready_score": 1,
        },
        "ready_score",
        1,
    )
    assert disqualified["passed"] is False
    assert disqualified["reason"] == "disqualified_upstream_rejected_before_field_consumption"


def test_scenario_report_7302_sources_are_bounded_and_nonlocal() -> None:
    """SCENARIO-REPORT-7302-SOURCES preserves four decisions and access failures."""

    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _fake_fetch(url)

    rows, access = mod.collect_source_rows(fetch)
    assert len(calls) == len(rows) == len(access) == 4
    assert {row["method_family"] for row in rows} == {
        "batch_prompting",
        "batch_interference",
        "constraint_substructure_recovery",
        "selected_ebt_citation",
    }
    assert {row["disposition"] for row in rows} == {"adapt", "implement", "defer"}
    assert all(row["local_evidence_claimed"] is False for row in rows)
    assert all(row["observed_access"] == "http_success" for row in rows)
    assert all(row["publication_or_version_date"] for row in rows)
    assert all(row["adapted_mechanism"] and row["falsifier"] for row in rows)

    def unavailable(_url: str) -> dict[str, Any]:
        raise OSError("offline")

    offline, observations = mod.collect_source_rows(unavailable)
    assert all(row["observed_access"] == "access_failed" for row in offline)
    assert all(row["error"] for row in observations)
    assert mod.source_rows_complete(offline) is True


def test_scenario_report_7302_metadata_and_validation_commands() -> None:
    """SCENARIO-REPORT-7302-METADATA fixes no-model fields and command scope."""

    assert mod.ZERO_INVOCATION_COUNTS == {
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "model_loads_failed": 0,
        "model_loads_cancelled": 0,
        "model_loads_in_flight": 0,
        "generation_calls_attempted": 0,
        "generation_calls_completed": 0,
        "generation_calls_failed": 0,
        "generation_calls_cancelled": 0,
        "generation_calls_in_flight": 0,
        "usable_answers": 0,
    }
    commands = dict(
        mod.validation_commands(ROOT, ROOT / mod.RAW_CANDIDATE_PATH, mod.ACTIVE_ROADMAP_PATH)
    )
    assert "--no-cov" in commands["focused_pytest"]
    assert commands["focused_pytest"][commands["focused_pytest"].index("-n") + 1] == "0"
    assert "--strict" in commands["row_consistency"]
    assert commands["full_python_suite"][0].endswith("/.venv/bin/pytest")
    assert commands["full_python_suite"][1:3] == ["tests/python", "-q"]
    assert mod.REPOSITORY_HEALTH_NAMES == ("full_python_suite",)
    with pytest.raises(argparse.ArgumentTypeError):
        mod.date_argument("20260913")
    with pytest.raises(ValueError):
        mod.date_argument("not-a-date")
    assert mod.date_argument("20260914") == "20260914"
    assert mod._yaml_phase({"prompt": "Milestone 2026.09.642, phase 3"}) == 3
    assert mod._yaml_phase({"phase": True, "prompt": "no phase"}) is None


def test_scenario_report_7302_e2e_and_validation_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7302-ARTIFACT runs both authorities through the full reducer."""

    root = _private_root(tmp_path, exact_markdown=True)
    receipt = mod.e2e_contract_receipt(root, root / "e2e")
    assert receipt["passed"] is True
    (root / mod.ACTIVE_ROADMAP_PATH).unlink()
    assert mod.e2e_contract_receipt(root, root / "missing-e2e") == {
        "passed": False,
        "reason": "v642_yaml_authority_missing",
    }

    raw_dir = root / mod.RAW_DIR
    artifact = mod._base_artifact(
        mod.RUN_DATE, root / mod.CHECKPOINT_PATH, "2026-09-14T00:00:00+00:00"
    )
    artifact["yaml_authority_path"] = str(mod.ACTIVE_ROADMAP_PATH)
    calls: list[list[str]] = []

    def execute(
        _root: Path,
        _log_dir: Path,
        commands: tuple[tuple[str, list[str]], ...],
    ) -> list[dict[str, Any]]:
        names = [name for name, _command in commands]
        calls.append(names)
        return [
            {
                "name": name,
                "command": f"fixture:{name}",
                "exit_code": 0,
                "duration_s": 0.001,
                "timed_out": False,
                "log_sha256": mod.sha256_bytes(b"fixture log"),
                "passed": True,
                "baseline_failure": False,
            }
            for name in names
        ]

    monkeypatch.setattr(mod.shipped.base, "_execute_commands", execute)
    rows = mod.run_required_validation(root, raw_dir, artifact, time.monotonic())
    assert [row["name"] for row in rows] == list(mod.VALIDATION_COMMAND_NAMES)
    assert calls == [
        list(mod.AFFECTED_VALIDATION_NAMES[: mod.PRETERMINAL_VALIDATION_COUNT]),
        list(mod.AFFECTED_VALIDATION_NAMES[mod.PRETERMINAL_VALIDATION_COUNT :]),
        list(mod.REPOSITORY_HEALTH_NAMES),
    ]
    assert (raw_dir / mod.RAW_CANDIDATE_PATH.name).is_file()


def test_req_report_7302_history_and_sidecars_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7302 rejects malformed history and retains unavailable history."""

    root = tmp_path / "history"
    history_path = root / mod.HISTORY_PATH
    history_path.parent.mkdir(parents=True)
    raw_dir = root / mod.RAW_DIR
    history_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="historical artifact is not a mapping"):
        mod.build_sidecars(root, raw_dir, [], [])
    assert mod._historical_repository_failures(root) == []
    history_path.unlink()
    assert mod._historical_repository_failures(root) == []

    with mod._configured_shipped(runtime=True), mod.shipped._configured_runtime():
        preconditions, authority, _roadmap, _content = mod._preconditions(
            root,
            root / mod.DEFAULT_OUTPUT_PATH,
            root / mod.RAW_DIR,
            root / mod.CHECKPOINT_PATH,
        )
    requirement = next(row for row in preconditions if row["check"] == "driving_requirement")
    assert authority is None
    assert requirement["available"] is False
    assert requirement["observed_value"] == "missing"


@pytest.mark.parametrize(
    ("exact_markdown", "expected_class", "expected_score"),
    [(True, "circular_positive", 1), (False, "disqualified", 0)],
)
def test_scenario_report_7302_artifact_is_terminal_and_evidence_derived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exact_markdown: bool,
    expected_class: str,
    expected_score: int,
) -> None:
    """SCENARIO-REPORT-7302-ARTIFACT validates exact and stale authorities."""

    root = _private_root(tmp_path, exact_markdown=exact_markdown)
    monkeypatch.setattr(mod, "run_required_validation", lambda *_args: _passing_receipts())
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == expected_class
    assert artifact["contract_complete_score"] == expected_score
    assert artifact["rows"] == artifact["contract_rows"]
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["repository_health"]["passed"] is True
    if not exact_markdown:
        assert artifact["gate_check_summary"] == {
            "failed_check": "markdown_milestone",
            "upstream": str(mod.DESIGN_PATH),
            "field": "milestone",
            "expected_value": mod.MILESTONE,
            "observed_value": "2026.09.641",
            "passed": False,
        }
    assert mod.validate_artifact(artifact, root=root) == []
    assert json.loads((root / mod.DEFAULT_OUTPUT_PATH).read_text()) == artifact

    forged = deepcopy(artifact)
    forged["model_invoked"] = True
    assert "model_contract_invalid" in mod.validate_artifact(forged, root=root)

    if exact_markdown:
        assert mod.validate_artifact(None, root=root) == ["artifact_mapping_required"]
        missing = deepcopy(artifact)
        missing.pop("schema")
        assert mod.validate_artifact(missing, root=root) == ["missing_required_field:schema"]

        mutations = (
            ("schema", "bad", "schema_invalid"),
            ("experiment_id", "exp0-bad", "identity_invalid"),
            ("status", "running", "lifecycle_invalid"),
            ("field_principles", {}, "field_principles_invalid"),
            ("execution_venue", "gpu", "execution_invalid"),
            ("duration_s", 0, "duration_invalid"),
            ("random_seed", 0, "authority_invalid"),
            ("rows", [], "rows_invalid"),
            ("sample_size_budget", {}, "sample_size_budget_invalid"),
            ("source_artifact_hashes", {}, "source_hashes_invalid"),
            ("validation_receipts", [], "validation_receipts_invalid"),
            ("repository_health", {}, "repository_health_invalid"),
            ("honest_verdict", "complete_forged", "honest_verdict_invalid"),
            ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum_invalid"),
        )
        for field, value, expected_error in mutations:
            changed = deepcopy(artifact)
            changed[field] = value
            assert expected_error in mod.validate_artifact(changed, root=root)

        bad_times = deepcopy(artifact)
        bad_times["started_at_utc"] = "2026-09-14T00:00:00"
        bad_times["completed_at_utc"] = "not-a-time"
        time_errors = mod.validate_artifact(bad_times, root=root)
        assert "started_at_utc_invalid" in time_errors
        assert "completed_at_utc_invalid" in time_errors

        bad_hash_shape = deepcopy(artifact)
        bad_hash_shape["source_artifact_hashes"]["missing.input"] = "sha256:missing"
        assert "source_hash_mismatch" in mod.validate_artifact(bad_hash_shape, root=root)
        bad_hash = deepcopy(artifact)
        first_path = next(iter(bad_hash["source_artifact_hashes"]))
        bad_hash["source_artifact_hashes"][first_path] = "sha256:bad"
        assert "source_hash_mismatch" in mod.validate_artifact(bad_hash, root=root)

        bad_receipt = deepcopy(artifact)
        bad_receipt["validation_receipts"][0]["command"] = ""
        assert "validation_receipt_shape_invalid" in mod.validate_artifact(bad_receipt, root=root)

        bad_rows = deepcopy(artifact)
        bad_rows["contract_rows"] = []
        bad_rows["gate_control_rows"] = []
        bad_rows["source_dispositions"] = []
        bad_rows["raw_authority_rows"] = []
        row_errors = mod.validate_artifact(bad_rows, root=root)
        assert "contract_row_count" in row_errors
        assert "gate_control_reduction" in row_errors
        assert "source_disposition_reduction" in row_errors
        assert "raw_authority_reduction" in row_errors
        malformed_row = deepcopy(artifact)
        malformed_row["contract_rows"][0]["checks"] = []
        assert "contract_row_reduction" in mod.validate_artifact(malformed_row, root=root)

        blocked = deepcopy(artifact)
        blocked["preconditions_checked"].append(
            {
                "check": "missing_dependency",
                "upstream": "exp0-missing",
                "field": "ready_score",
                "expected_value": 1,
                "observed_value": None,
                "available": False,
                "blocking_external": True,
            }
        )
        blocked["validation_receipts"] = []
        mod._apply_terminal_state(blocked)
        assert blocked["verdict_class"] == "blocked"
        assert blocked["gate_check_summary"]["failed_check"] == "missing_dependency"

        parity = deepcopy(artifact)
        parity["contract_rows"][0]["checks"]["title"] = False
        parity["contract_rows"][0]["passed"] = False
        mod._apply_terminal_state(parity)
        assert parity["gate_check_summary"]["failed_check"] == "contract_parity"

        failed_validation = deepcopy(artifact)
        failed_validation["validation_receipts"][0]["passed"] = False
        mod._apply_terminal_state(failed_validation)
        assert failed_validation["gate_check_summary"]["failed_check"] == (
            "acceptance:affected_validation"
        )
