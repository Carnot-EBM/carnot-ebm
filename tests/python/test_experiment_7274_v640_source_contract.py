"""Tests for the V640 source ingestion and exact task contract.

Spec refs: REQ-REPORT-7274 and SCENARIO-REPORT-7274-*.
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

from carnot import experiment_7274_v640_source_contract as mod


ROOT = Path(__file__).resolve().parents[2]


def _copy(root: Path, relative: Path) -> None:
    """Copy one authenticated input into a private repository view."""

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


def _private_root(tmp_path: Path, *, milestone: str = mod.MILESTONE) -> Path:
    """Build a small complete repository without touching measured evidence."""

    root = tmp_path / "repo"
    for relative in (*mod.INPUT_PATHS, *mod.CODE_PATHS, mod.HISTORY_PATH):
        _copy(root, relative)
    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    roadmap["milestone"] = milestone
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap, sort_keys=False), encoding="utf-8"
    )
    for relative in (mod.DEFAULT_OUTPUT_PATH.parent, mod.RAW_DIR, mod.CHECKPOINT_PATH.parent):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _matching_markdown(roadmap: dict[str, Any]) -> str:
    """Render the literal table syntax that the independent parser consumes."""

    lines = [
        "# Carnot Research Roadmap v640",
        "",
        f"**Milestone:** {mod.MILESTONE}",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Task ID | Title | Deliverable | Structured gates |",
        "|---|---|---|---|---|",
    ]
    for index, task in enumerate(roadmap["tasks"], 1):
        gate_text = "; ".join(
            f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} {gate['value']}"
            for gate in task.get("gated_on", [])
        )
        lines.append(
            f"| {index} | {task['id']} | {task['title']} | {task['deliverable']} | "
            f"{gate_text or 'None'} |"
        )
    return "\n".join(lines) + "\n"


def _fake_fetch(url: str) -> dict[str, Any]:
    """Return stable arXiv-like records for the four bounded requests."""

    records = {
        "2607.17047": ("Solver-Hard Is Not Model-Hard", "v1"),
        "2601.03905": ("Current Agents Fail to Leverage World Model", "v2"),
        "2609.10873": ("When Validation Stops Learning", "v1"),
        "extropic.ai/writing/z1t": ("Z1T", "2026-09-04"),
    }
    source_id = next(value for value in records if value in url)
    title, version = records[source_id]
    return {
        "ok": True,
        "status_code": 200,
        "body": f"<title>{title}</title> [version:{version}]",
        "error": None,
    }


def _passing_receipts() -> list[dict[str, Any]]:
    """Avoid recursive validation commands inside artifact unit tests."""

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


def test_scenario_report_7274_contract_retains_stale_markdown() -> None:
    """SCENARIO-REPORT-7274-CONTRACT records the current V636 design mismatch."""

    authority, roadmap, yaml_bytes, candidates = mod.select_yaml_authority(ROOT)
    assert authority == mod.ACTIVE_ROADMAP_PATH
    assert roadmap is not None and yaml_bytes == (ROOT / authority).read_bytes()
    assert candidates[0]["available"] is True
    contract = mod.evaluate_contract((ROOT / mod.DESIGN_PATH).read_text(), roadmap)
    assert contract["passed"] is False
    assert contract["markdown_milestone"] == "2026.09.636"
    assert contract["yaml_milestone"] == mod.MILESTONE
    assert contract["observed_id_order"] == list(mod.EXPECTED_ID_ORDER)
    assert len(contract["contract_rows"]) == 14
    assert all(row["passed"] is False for row in contract["contract_rows"])


def test_req_report_7274_selects_only_v640_authority(tmp_path: Path) -> None:
    """REQ-REPORT-7274 falls back to staged V640 and fails closed without it."""

    root = _private_root(tmp_path, milestone="2026.09.638")
    staged = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    (root / mod.NEXT_ROADMAP_PATH).write_text(yaml.safe_dump(staged, sort_keys=False))
    authority, roadmap, _content, candidates = mod.select_yaml_authority(root)
    assert authority == mod.NEXT_ROADMAP_PATH and roadmap is not None
    assert [row["available"] for row in candidates] == [False, True]
    (root / mod.NEXT_ROADMAP_PATH).unlink()
    authority, roadmap, content, candidates = mod.select_yaml_authority(root)
    assert authority is roadmap is content is None
    assert all(row["available"] is False for row in candidates)


def test_scenario_report_7274_gate_e2e_has_distinct_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7274-GATES sends every case through the real evaluator."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    rows = mod.run_gate_replays(roadmap, tmp_path / "gate-e2e")
    assert len(rows) == 32
    assert mod.gate_replays_complete(rows) is True
    for edge_index in range(8):
        edge = [row for row in rows if row["edge_index"] == edge_index]
        assert [row["case"] for row in edge] == [
            "true",
            "false",
            "missing_field",
            "missing_file",
        ]
        assert [row["passed"] for row in edge] == [True, False, False, False]
        assert edge[1]["observed_value"] == 0
        assert len({json.dumps(row["observation"], sort_keys=True) for row in edge}) == 4
    assert mod.gate_replays_complete(rows[:-1]) is False
    changed = deepcopy(rows)
    changed[0]["case"] = "false"
    assert mod.gate_replays_complete(changed) is False


def test_scenario_report_7274_sources_are_bounded_and_nonlocal() -> None:
    """SCENARIO-REPORT-7274-SOURCES keeps adoption and source truth separate."""

    calls: list[str] = []

    def fetch(url: str) -> dict[str, Any]:
        calls.append(url)
        return _fake_fetch(url)

    rows, access = mod.collect_source_rows(fetch)
    assert len(calls) == len(access) == 4
    assert {row["method_family"] for row in rows} == {
        "source_controls",
        "world_model_consumption",
        "independent_update_admission",
        "acknowledgment_costs",
    }
    assert {row["disposition"] for row in rows} == {"implement", "adapt", "defer"}
    assert all(row["input_needs"] for row in rows)
    assert all(row["falsifying_controls"] for row in rows)
    assert all(row["retirement_overlap"] for row in rows)
    assert all(row["refresh_date"] == "2026-09-13" for row in rows)
    assert all("access_error" in row for row in rows)
    assert all(row["local_evidence_claimed"] is False for row in rows)
    assert all(row["source_delta"] == "no_version_delta" for row in rows)

    def unavailable(_url: str) -> dict[str, Any]:
        raise OSError("offline")

    offline_rows, offline_access = mod.collect_source_rows(unavailable)
    assert all(row["access_status"] == "access_failed_refresh_retained" for row in offline_access)
    assert all(row["source_delta"] == "access_failed" for row in offline_rows)
    assert mod.observed_version("[version:v1] [version:v3]") == "v3"
    assert mod.observed_version("no marker") is None


def test_scenario_report_7274_sidecars_hash_history_and_fixtures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7274-ARTIFACT isolates historical models and controls."""

    root = _private_root(tmp_path)
    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text())
    gates = mod.run_gate_replays(roadmap, tmp_path / "gate-inputs")
    _rows, access = mod.collect_source_rows(_fake_fetch)
    receipts = mod.build_sidecars(root, root / mod.RAW_DIR, gates, access)
    assert set(receipts) == {"history", "negative_fixtures", "source_access"}
    assert all(
        receipt["sha256"] == mod.sha256(root / receipt["path"]) for receipt in receipts.values()
    )
    history = json.loads((root / receipts["history"]["path"]).read_text())
    assert history["current_invocation"]["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    assert history["historical_artifacts"][0]["path"] == str(mod.HISTORY_PATH)
    assert history["historical_artifacts"][0]["accepted_for_current_values"] is False
    negative = json.loads((root / receipts["negative_fixtures"]["path"]).read_text())
    assert len(negative["fixture_rows"]) == 24
    assert all(row["research_result"] is False for row in negative["fixture_rows"])


def test_scenario_report_7274_artifact_derives_terminal_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7274-ARTIFACT recomputes mismatch and exact agreement."""

    root = _private_root(tmp_path)
    monkeypatch.setattr(
        mod, "run_required_validation", lambda *_args, **_kwargs: _passing_receipts()
    )
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["source_contract_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "markdown_milestone"
    assert artifact["rows"] == artifact["contract_rows"]
    assert artifact["source_dispositions"]
    assert len(artifact["gate_control_rows"]) == 32
    assert all(
        {"unit_id", "arm", "seed", "error", "abstention", "cost", "metric", "censored"}
        <= row.keys()
        for row in artifact["rows"]
    )
    assert artifact["verifier_is_oracle"] is True
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert all(
        {"expected", "observed", "passed", "principle"} <= row.keys()
        for row in artifact["acceptance_gate_results"]
    )
    assert mod.validate_artifact(artifact, root=root) == []

    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text())
    (root / mod.DESIGN_PATH).write_text(_matching_markdown(roadmap))
    exact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / "results/exact.json",
        raw_dir=root / "results/raw/exact",
        checkpoint_path=root / "results/checkpoints/exact.json",
        fetcher=_fake_fetch,
    )
    assert exact["source_contract_complete_score"] == 1
    assert exact["verdict_class"] == "circular_positive"
    assert all(row["passed"] for row in exact["acceptance_gate_results"])
    assert mod.validate_artifact(exact, root=root) == []

    mutations = (
        lambda value: value.__setitem__("schema", "wrong"),
        lambda value: value.__setitem__("status", "running"),
        lambda value: value.__setitem__("run_date", "20260912"),
        lambda value: value.__setitem__("duration_s", 0),
        lambda value: value.__setitem__("model_invoked", True),
        lambda value: value.__setitem__("verifier_is_oracle", False),
        lambda value: value.__setitem__("rows", []),
        lambda value: value.__setitem__("source_dispositions", []),
        lambda value: value.__setitem__("gate_control_rows", []),
        lambda value: value.__setitem__("source_artifact_hashes", {}),
        lambda value: value.__setitem__("source_contract_complete_score", 0),
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"),
    )
    for mutate in mutations:
        changed = deepcopy(exact)
        mutate(changed)
        assert mod.validate_artifact(changed, root=root)
    assert mod.validate_artifact([], root=root) == ["artifact_mapping_required"]


def test_req_report_7274_missing_or_quarantined_upstream_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7274 uses blocked only for an unavailable external input."""

    root = _private_root(tmp_path)
    (root / mod.HISTORY_PATH).unlink()
    monkeypatch.setattr(
        mod, "run_required_validation", lambda *_args, **_kwargs: _passing_receipts()
    )
    artifact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["upstream"] == str(mod.HISTORY_PATH)
    assert artifact["gate_check_summary"]["field"] == "status|quarantine"
    assert artifact["source_contract_complete_score"] == 0
    assert mod.validate_artifact(artifact, root=root) == []


def test_req_report_7274_validation_scope_and_independent_reducer(tmp_path: Path) -> None:
    """REQ-REPORT-7274 forbids full-suite validation and rejects row mutations."""

    commands = mod.validation_commands(
        ROOT,
        ROOT / mod.RAW_CANDIDATE_PATH,
        mod.ACTIVE_ROADMAP_PATH,
    )
    assert [name for name, _command in commands] == list(mod.VALIDATION_COMMAND_NAMES)
    assert "full_python_suite" not in dict(commands)
    assert "-n" in dict(commands)["focused_tests"]
    assert dict(commands)["adversarial"][-1].endswith(str(mod.RAW_CANDIDATE_PATH))
    assert dict(commands)["row_consistency"][-1].endswith(str(mod.RAW_CANDIDATE_PATH))

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    gates = mod.run_gate_replays(roadmap, tmp_path / "reducer-gates")
    contract = mod.evaluate_contract(_matching_markdown(roadmap), roadmap)
    rows, _access = mod.collect_source_rows(_fake_fetch)
    payload = {
        "contract_rows": contract["contract_rows"],
        "gate_control_rows": gates,
        "source_dispositions": rows,
        "raw_authority_rows": [
            {"source_sha256": "sha256:a", "raw_sha256": "sha256:a"},
            {"source_sha256": "sha256:b", "raw_sha256": "sha256:b"},
        ],
    }
    assert mod.independent_reduce(payload) == []
    changed = deepcopy(payload)
    changed["contract_rows"][0]["passed"] = False
    assert "contract_row_reduction" in mod.independent_reduce(changed)
    changed = deepcopy(payload)
    changed["source_dispositions"][0]["disposition"] = "claim"
    assert "source_disposition_reduction" in mod.independent_reduce(changed)
    assert set(mod.independent_reduce({})) == {
        "contract_row_count",
        "gate_control_reduction",
        "source_disposition_count",
        "raw_authority_reduction",
    }


def test_req_report_7274_defensive_validation_and_command_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7274 covers malformed controls, receipts, and prerequisites."""

    roadmap = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text())
    gates = mod.run_gate_replays(roadmap, tmp_path / "defensive-gates")
    for mutate in (
        lambda value: value[0].__setitem__("passed", False),
        lambda value: value[1].__setitem__("observed_value", 2),
        lambda value: value[2].__setitem__("observation", value[1]["observation"]),
    ):
        changed = deepcopy(gates)
        mutate(changed)
        assert mod.gate_replays_complete(changed) is False

    root = _private_root(tmp_path / "invalid-history")
    (root / mod.HISTORY_PATH).write_text("[]")
    _source_rows, access = mod.collect_source_rows(_fake_fetch)
    with pytest.raises(ValueError, match="not a mapping"):
        mod.build_sidecars(root, root / mod.RAW_DIR, gates, access)

    root = _private_root(tmp_path / "missing-spec")
    (root / mod.SPEC_PATH).unlink()
    preconditions, *_authority = mod._preconditions(
        root,
        root / mod.DEFAULT_OUTPUT_PATH,
        root / mod.RAW_DIR,
        root / mod.CHECKPOINT_PATH,
    )
    requirement = next(row for row in preconditions if row["check"] == "driving_requirement")
    assert requirement["available"] is False
    assert mod._failed_precondition({"preconditions_checked": None}) is None

    contract_failure = mod._base_artifact(
        mod.RUN_DATE,
        tmp_path / "contract-checkpoint.json",
        "2026-09-13T00:00:00+00:00",
    )
    contract_failure["preconditions_checked"] = []
    contract_failure["markdown_milestone"] = mod.MILESTONE
    contract_failure["contract_rows"] = [
        {
            "unit_id": mod.FIRST_TASK_ID,
            "yaml": {"id": mod.FIRST_TASK_ID},
            "markdown": {"id": "stale"},
            "passed": False,
        }
    ]
    assert mod._failure_summary(contract_failure)["failed_check"] == "contract_parity"

    root = _private_root(tmp_path / "valid")
    monkeypatch.setattr(
        mod, "run_required_validation", lambda *_args, **_kwargs: _passing_receipts()
    )
    roadmap = yaml.safe_load((root / mod.ACTIVE_ROADMAP_PATH).read_text())
    (root / mod.DESIGN_PATH).write_text(_matching_markdown(roadmap))
    exact = mod.build_artifact(
        root,
        mod.RUN_DATE,
        output_path=root / mod.DEFAULT_OUTPUT_PATH,
        raw_dir=root / mod.RAW_DIR,
        checkpoint_path=root / mod.CHECKPOINT_PATH,
        fetcher=_fake_fetch,
    )
    mutations = (
        lambda value: value.pop("schema"),
        lambda value: value.__setitem__("experiment_id", "wrong"),
        lambda value: value.__setitem__("completed_at_utc", "invalid"),
        lambda value: value.__setitem__("started_at_utc", "2026-09-13T00:00:00"),
        lambda value: value.__setitem__("field_principles", {}),
        lambda value: value.__setitem__("execution_host", "wrong"),
        lambda value: value.__setitem__("random_seed", 0),
        lambda value: value.__setitem__("sample_size_budget", {}),
        lambda value: value.__setitem__("source_artifact_hashes", {"missing": "sha256:bad"}),
        lambda value: value.__setitem__("validation_receipts", []),
        lambda value: value["validation_receipts"][0].__setitem__("command", ""),
    )
    expected_errors = (
        "missing_required_field:",
        "identity_invalid",
        "completed_at_utc_invalid",
        "started_at_utc_invalid",
        "field_principles_invalid",
        "execution_invalid",
        "random_seed_invalid",
        "sample_size_budget_invalid",
        "source_hash_mismatch",
        "validation_receipts_invalid",
        "validation_receipt_shape_invalid",
    )
    for mutate, expected_error in zip(mutations, expected_errors, strict=True):
        changed = deepcopy(exact)
        mutate(changed)
        assert any(
            error.startswith(expected_error) for error in mod.validate_artifact(changed, root=root)
        )

    monkeypatch.undo()
    command_rows = mod._execute_commands(
        tmp_path,
        tmp_path / "logs",
        (("probe", ["/bin/echo", "measured"]),),
    )
    assert command_rows[0]["passed"] is True
    assert command_rows[0]["log_sha256"] == mod.sha256(tmp_path / "logs/probe.log")
    assert command_rows[0]["inference_substrate"] == "cpu_exact_solver_or_simulator"

    candidate_root = tmp_path / "candidate-root"
    candidate_raw = candidate_root / "results/raw/experiment_7274"
    candidate_raw.mkdir(parents=True)
    candidate = mod._base_artifact(
        mod.RUN_DATE,
        candidate_root / mod.CHECKPOINT_PATH,
        "2026-09-13T00:00:00+00:00",
    )
    candidate["yaml_authority_path"] = str(mod.ACTIVE_ROADMAP_PATH)

    def execute_fixture(
        _root: Path,
        _log_dir: Path,
        commands: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "command": "fixture",
                "exit_code": 0,
                "duration_s": 0.001,
                "timed_out": False,
                "log_sha256": mod.sha256_bytes(b"fixture"),
                "passed": True,
                "baseline_failure": False,
            }
            for name, _command in commands
        ]

    monkeypatch.setattr(mod, "_execute_commands", execute_fixture)
    receipts = mod.run_required_validation(
        candidate_root,
        candidate_raw,
        candidate,
        time.monotonic(),
    )
    assert [row["name"] for row in receipts] == list(mod.VALIDATION_COMMAND_NAMES)
    raw_candidate = candidate_raw / mod.RAW_CANDIDATE_PATH.name
    assert raw_candidate.is_file()
    assert json.loads(raw_candidate.read_text())["status"] == "complete"

    assert mod.date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(argparse.ArgumentTypeError):
        mod.date_argument("20260912")
    with pytest.raises(ValueError):
        mod.date_argument("not-a-date")
