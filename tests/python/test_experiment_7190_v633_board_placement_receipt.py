"""Tests for REQ-ISING-7190 and its board-placement receipt scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import experiment_7190_v633_board_placement_receipt as exp


ROOT = Path(__file__).resolve().parents[2]


def _task() -> dict[str, Any]:
    """Return the exact SCENARIO-ISING-7190-PREFLIGHT roadmap row."""

    return {
        "id": exp.TASK_ID,
        "title": "KV260, GateMate, and PolarFire continuity with sparse placement limits",
        "track": "hardware",
        "requires_gpu": False,
        "per_unit_rows": True,
        "milestone": exp.MILESTONE,
        "deliverable": exp.RESULT_PATH.as_posix(),
        "prior_failures": [
            {
                "experiment_id": "exp7146-gatemate-changed-state-continuity",
                "verdict": "blocked_no_new_operator_physical_state_receipt_after_exp6559",
                "retire_if_same_verdict": True,
            }
        ],
        "prompt": (
            "REQUIRED ARTIFACT FIELDS:\n"
            '- inference_substrate_class: principle: "Use aggregation when the declared work runs."\n'
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one isolated fixture without touching the checked-in results tree."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build the smallest complete SCENARIO-ISING-7190 fixture tree."""

    root = tmp_path / "repo"
    required = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("openspec/capabilities/ising-backend/spec.md"),
        Path("research-roadmap.yaml"),
    )
    monkeypatch.setattr(exp, "REQUIRED_SOURCE_PATHS", required)
    for relative in required[:2]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("instructions\n", encoding="utf-8")
    spec = root / exp.SPEC_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("### REQ-ISING-7190\n", encoding="utf-8")
    roadmap = root / exp.ROADMAP_PATH
    roadmap.write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": [_task()]}, sort_keys=False),
        encoding="utf-8",
    )

    raw = b"kv260 raw board transcript\n"
    raw_path = root / exp.KV260_TRANSCRIPT_PATH
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    _write_json(
        root / exp.KV260_GRADUATION_PATH,
        {
            "kv260_terminal_condition_confirmed": True,
            "kv260_terminal_transcript_path": exp.KV260_TRANSCRIPT_PATH.as_posix(),
            "kv260_terminal_transcript_sha256": exp.sha256_bytes(raw).removeprefix("sha256:"),
            "kv260_overlay_loaded": True,
            "honest_verdict": "complete: kv260 terminal",
        },
    )
    _write_json(
        root / exp.LATEST_BOARD_STATE_PATH,
        {
            "run_date": "20260723",
            "board_capability_matrix": {
                "kv260": {
                    "reachability": "cached_ssh_reachable",
                    "programmed_image": "cached_carnot_ising_v4_alias_carnot_ising_v2_n64",
                },
                "polarfire": {
                    "reachability": "cached_ssh_reachable",
                    "programmed_image": "board-local Linux workload path only",
                    "prior_workload_validated": True,
                },
            },
        },
    )
    _write_json(
        root / exp.GATEMATE_STATE_PATH,
        {
            "run_date": "20260908",
            "physical_state_receipt": {"exists": False, "latest_candidate_date": "20260811"},
            "receipt_cutoff_experiment": {
                "experiment": "Exp6559",
                "run_date": "20260823",
            },
            "hardware_command_count": 0,
            "honest_verdict": "blocked_no_new_operator_physical_state_receipt_after_exp6559",
        },
    )
    _write_json(
        root / exp.GATEMATE_CUTOFF_PATH,
        {"run_date": "20260823", "status": "blocked_missing_new_physical_receipt"},
    )
    _write_json(
        root / exp.POLARFIRE_DISPATCH_PATH,
        {
            "polarfire_workload_validated": True,
            "result_hash_match": True,
            "board_result_sha256": "a" * 64,
            "inference_substrate": "hardware_smoke",
            "command_transcript": [{"stage": "remote_ising_eval", "returncode": 0}],
        },
    )
    return root


def test_req_ising_7190_spec_precedes_implementation() -> None:
    """REQ-ISING-7190 fixes the fields, boundaries, and named scenarios."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-ISING-7190", 1)[1]
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for scenario in ("PREFLIGHT", "BOARDS", "GATEMATE", "PLACEMENT", "ARTIFACT"):
        assert f"SCENARIO-ISING-7190-{scenario}" in section


def test_scenario_ising_7190_graph_metrics_preserve_every_edge() -> None:
    """SCENARIO-ISING-7190-PLACEMENT computes degree without pruning."""

    edges = ((0, 1, 1.0), (1, 2, -0.5), (2, 3, 0.25), (0, 3, 0.75))
    assert exp.graph_metrics(4, edges) == {"edge_count": 4, "maximum_degree": 2}
    for invalid in (
        ((0, 0, 1.0),),
        ((0, 4, 1.0),),
        ((1, 0, 1.0),),
        ((0, 1, 1.0), (0, 1, 2.0)),
    ):
        with pytest.raises(ValueError):
            exp.graph_metrics(4, invalid)


def test_scenario_ising_7190_fallback_is_fixed_and_topology_unknown() -> None:
    """SCENARIO-ISING-7190-PLACEMENT keeps the deterministic n=16 fallback honest."""

    first = exp.synthetic_fallback_row()
    second = exp.synthetic_fallback_row()
    assert first == second
    assert first["n"] == 16
    assert first["claim_scope"] == "compatibility_only"
    assert first["maximum_degree"] <= 16
    assert first["degree_limit_necessary_passed"] is True
    assert first["topology_fit"] == "topology_unknown"
    assert first["explicit_parent_graph_mapping_present"] is False
    assert first["edges_dropped"] == 0


def test_scenario_ising_7190_real_upstream_contracts_are_additional_rows() -> None:
    """SCENARIO-ISING-7190-PLACEMENT reads every Exp7187/7188 graph contract."""

    rows = exp.build_placement_rows(ROOT)
    slice_rows = [row for row in rows if row["source_experiment"] == "Exp7187"]
    quantized_rows = [row for row in rows if row["source_experiment"] == "Exp7188"]
    assert len(slice_rows) == 26
    assert len(quantized_rows) == 18
    assert {row["field_width_bits"] for row in slice_rows} == {64}
    assert {row["field_width_bits"] for row in quantized_rows} == {4, 8, 16}
    assert all(row["edge_count"] > 0 for row in rows)
    assert all(row["maximum_degree"] <= 16 for row in rows)
    assert all(row["topology_fit"] == "topology_unknown" for row in rows)
    assert all(row["edges_dropped"] == 0 for row in rows)
    corrected = [row for row in quantized_rows if row["field_width_bits"] == 4]
    assert corrected and all(
        row["host_correction_cost"]["full_energy_calls"] == 497 for row in corrected
    )


def test_scenario_ising_7190_current_board_rows_keep_evidence_kinds_distinct() -> None:
    """SCENARIO-ISING-7190-BOARDS preserves the current three-board boundary."""

    rows, operator_receipt = exp.build_board_rows(ROOT)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["disposition"] == "graduated_preserved"
    assert by_board["KV260"]["programmable_logic_sampling_observed"] is True
    assert by_board["KV260"]["raw_transcript_hash"].startswith("sha256:")
    assert by_board["GateMate"]["disposition"] == "blocked_inherited_no_new_physical_state"
    assert by_board["GateMate"]["hardware_command_count"] == 0
    assert by_board["PolarFire"]["ssh_reachability_observed"] is True
    assert by_board["PolarFire"]["board_cpu_work_observed"] is True
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert by_board["PolarFire"]["raw_transcript_hash"] is None
    assert by_board["PolarFire"]["disposition"] == "blocked_missing_raw_dispatch_transcript"
    assert operator_receipt["newer_than_exp6559"] is False


def test_scenario_ising_7190_gatemate_change_only_authorizes_later_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7190-GATEMATE never performs the newly authorized action."""

    root = _fixture_root(tmp_path, monkeypatch)
    state_path = root / exp.GATEMATE_STATE_PATH
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["physical_state_receipt"] = {
        "exists": True,
        "receipt_date": "20260910",
        "source": "operator receipt",
        "changed_physical_fields": ["power"],
    }
    _write_json(state_path, state)
    rows, receipt = exp.build_board_rows(root)
    gate = next(row for row in rows if row["board"] == "GateMate")
    assert receipt["newer_than_exp6559"] is True
    assert gate["disposition"] == "authorized_later_action"
    assert gate["hardware_command_count"] == 0
    assert "later task" in gate["exact_next_prerequisite"]


def test_req_ising_7190_preconditions_bind_sources_tools_and_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7190-PREFLIGHT records expected and observed values."""

    root = _fixture_root(tmp_path, monkeypatch)
    output = root / exp.RESULT_PATH
    checkpoint = root / exp.CHECKPOINT_PATH
    checks, hashes = exp.collect_preconditions(root, output, checkpoint)
    assert all(row["passed"] for row in checks)
    by_check = {row["check"]: row for row in checks}
    assert by_check["driving_capability_spec"]["observed_value"]["req_present"] is True
    assert by_check["same_milestone_gate_fields"]["expected_value"] == exp.EXPECTED_TASK_CONTRACT
    assert by_check["same_milestone_gate_fields"]["observed_value"] == exp.EXPECTED_TASK_CONTRACT
    assert by_check["python_tool"]["passed"] is True
    assert set(hashes) == {path.as_posix() for path in exp.REQUIRED_SOURCE_PATHS}

    (root / "CODEX.md").unlink()
    failed, failed_hashes = exp.collect_preconditions(root, output, checkpoint)
    missing = next(row for row in failed if row["check"] == "required_source:CODEX.md")
    assert missing["passed"] is False
    assert failed_hashes["CODEX.md"] is None


def test_scenario_ising_7190_artifact_complete_with_blocked_board_rows(tmp_path: Path) -> None:
    """SCENARIO-ISING-7190-ARTIFACT treats visibility as completion, not readiness."""

    artifact = exp.build_artifact(
        ROOT,
        exp.RUN_DATE,
        output_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("positive:")
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["board_placement_receipt_complete_score"] == 1
    assert artifact["hardware_execution_claimed"] is False
    assert artifact["hardware_command_count"] == 0
    assert len(artifact["board_rows"]) == 3
    assert artifact["rows"] == artifact["board_rows"] + artifact["placement_rows"]


def test_scenario_ising_7190_external_failure_is_terminal_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7190-PREFLIGHT writes a precise blocked result."""

    root = _fixture_root(tmp_path, monkeypatch)
    (root / "CODEX.md").unlink()
    artifact = exp.build_artifact(
        root,
        exp.RUN_DATE,
        output_path=root / exp.RESULT_PATH,
        checkpoint_path=root / exp.CHECKPOINT_PATH,
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["board_rows"] == artifact["placement_rows"] == artifact["rows"] == []
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "required_source:CODEX.md",
        "upstream": "CODEX.md",
        "field": "bytes",
        "expected_value": "readable_nonempty_file",
        "observed_value": "missing",
    }


@pytest.mark.parametrize(
    ("mutator", "error"),
    (
        (lambda item: item.pop("rows"), "missing_required_fields"),
        (lambda item: item["field_principles"].pop("rows"), "field_principles_invalid"),
        (lambda item: item.update(hardware_execution_claimed=True), "hardware_claim_invalid"),
        (lambda item: item.update(hardware_command_count=1), "hardware_command_invalid"),
        (lambda item: item["board_rows"].pop(), "board_rows_invalid"),
        (
            lambda item: item["placement_rows"][0].update(edges_dropped=1),
            "edge_preservation_invalid",
        ),
        (
            lambda item: item["placement_rows"][0].update(topology_fit="fits_parent_graph"),
            "topology_claim_invalid",
        ),
        (
            lambda item: item.update(board_placement_receipt_complete_score=0),
            "completion_score_invalid",
        ),
        (
            lambda item: item.update(
                honest_verdict="positive_complete_board_visibility_and_host_compatibility_only"
            ),
            "honest_verdict_invalid",
        ),
        (lambda item: item.update(reproducibility_checksum="sha256:bad"), "checksum_invalid"),
    ),
)
def test_req_ising_7190_validator_rejects_inflated_or_incomplete_evidence(
    tmp_path: Path, mutator: Any, error: str
) -> None:
    """SCENARIO-ISING-7190-ARTIFACT fails closed under receipt mutations."""

    artifact = exp.build_artifact(
        ROOT,
        exp.RUN_DATE,
        output_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
    )
    changed = deepcopy(artifact)
    mutator(changed)
    if error != "checksum_invalid":
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert error in exp.validate_artifact(changed)


def test_req_ising_7190_atomic_writer_and_cli_use_only_requested_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7190 validates the file-to-parser-to-gate path in isolation."""

    output = tmp_path / "artifact.json"
    artifact = exp.run_experiment(
        ROOT,
        exp.RUN_DATE,
        output_path=output,
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--validate", str(output)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    output.write_text("{}\n", encoding="utf-8")
    assert exp.main(["--validate", str(output)]) == 2
    assert exp.main(["--date", "20260909", "--output", str(tmp_path / "bad.json")]) == 2


def test_req_ising_7190_parser_and_preflight_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7190-PREFLIGHT rejects malformed bytes and task shapes."""

    with pytest.raises(ValueError, match="nonfinite"):
        exp.canonical_json({"value": float("nan")})
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert exp._read_json(malformed) is None
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp._read_json(array) is None
    assert exp._read_json(tmp_path / "missing.json") is None

    root = tmp_path / "contracts"
    root.mkdir()
    assert exp._task_contract(root) is None
    roadmap = root / exp.ROADMAP_PATH
    for value in (
        [],
        {"milestone": "wrong", "tasks": []},
        {"milestone": exp.MILESTONE, "tasks": "wrong"},
        {"milestone": exp.MILESTONE, "tasks": []},
    ):
        roadmap.write_text(yaml.safe_dump(value), encoding="utf-8")
        assert exp._task_contract(root) is None
    bad_prior = _task()
    bad_prior["prior_failures"] = ["wrong"]
    roadmap.write_text(
        yaml.safe_dump({"milestone": exp.MILESTONE, "tasks": [bad_prior]}),
        encoding="utf-8",
    )
    assert exp._task_contract(root)["prior_failure"]["experiment_id"] is None

    def fail_tempfile(*_args: object, **_kwargs: object) -> object:
        raise OSError("read-only")

    monkeypatch.setattr(exp.tempfile, "NamedTemporaryFile", fail_tempfile)
    assert exp._directory_receipt(tmp_path / "blocked") == (False, "OSError: read-only")
    monkeypatch.undo()

    root = _fixture_root(tmp_path / "missing-spec", monkeypatch)
    (root / exp.SPEC_PATH).unlink()
    checks, _hashes = exp.collect_preconditions(
        root,
        root / exp.RESULT_PATH,
        root / exp.CHECKPOINT_PATH,
    )
    assert (
        next(row for row in checks if row["check"] == "driving_capability_spec")["passed"] is False
    )


def test_req_ising_7190_malformed_optional_receipts_stay_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7190-BOARDS never upgrades malformed optional evidence."""

    root = _fixture_root(tmp_path, monkeypatch)
    _write_json(root / exp.LATEST_BOARD_STATE_PATH, {"board_capability_matrix": []})
    _write_json(
        root / exp.GATEMATE_STATE_PATH,
        {
            "run_date": "20260908",
            "receipt_cutoff_experiment": [],
            "physical_state_receipt": [],
            "honest_verdict": "blocked",
        },
    )
    rows, receipt = exp.build_board_rows(root)
    assert all(row["disposition"].startswith("blocked_") for row in rows)
    assert receipt["newer_than_exp6559"] is False


def test_req_ising_7190_defensive_graph_and_upstream_parsers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7190-PLACEMENT rejects invalid graph data and falls back."""

    with pytest.raises(ValueError, match="positive"):
        exp.graph_metrics(0, ())
    with pytest.raises(ValueError, match="finite"):
        exp.graph_metrics(2, ((0, 1, float("inf")),))
    assert (
        exp._unique_graph_contracts(
            {"finite_law_rows": "wrong", "benchmark_rows": [None, {"n": "wrong"}]}
        )
        == []
    )

    root = _fixture_root(tmp_path, monkeypatch)
    _write_json(root / exp.SLICE_ARTIFACT_PATH, {"slice_sampler_ready_score": 0})
    _write_json(
        root / exp.QUANTIZED_ARTIFACT_PATH,
        {
            "quantized_audit_complete_score": 1,
            "cost_rows": [],
            "quantizer_rows": [None, {"n": "wrong"}],
        },
    )
    assert exp.build_placement_rows(root) == [exp.synthetic_fallback_row()]
    with pytest.raises(ValueError, match=exp.RUN_DATE):
        exp.build_artifact(
            root,
            "20260909",
            output_path=root / exp.RESULT_PATH,
            checkpoint_path=root / exp.CHECKPOINT_PATH,
        )


def test_req_ising_7190_validator_and_writer_defensive_paths(tmp_path: Path) -> None:
    """SCENARIO-ISING-7190-ARTIFACT covers each independent claim boundary."""

    artifact = exp.build_artifact(
        ROOT,
        exp.RUN_DATE,
        output_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
    )
    cases: list[tuple[dict[str, Any], str]] = []

    bad_board = deepcopy(artifact)
    bad_board["board_rows"][0]["hardware_command_count"] = 1
    bad_board["rows"] = bad_board["board_rows"] + bad_board["placement_rows"]
    cases.append((bad_board, "board_command_boundary_invalid"))

    bad_placement = deepcopy(artifact)
    bad_placement["placement_rows"][0] = "wrong"
    bad_placement["rows"] = bad_placement["board_rows"] + bad_placement["placement_rows"]
    cases.append((bad_placement, "placement_row_invalid"))

    bad_claim = deepcopy(artifact)
    bad_claim["placement_rows"][0]["hardware_execution_claimed"] = True
    bad_claim["rows"] = bad_claim["board_rows"] + bad_claim["placement_rows"]
    cases.append((bad_claim, "placement_hardware_claim_invalid"))

    for changed, expected in cases:
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["run_date"] = "20260909"
    assert "checksum_invalid" in exp.validate_artifact(changed)

    broken_block = deepcopy(artifact)
    broken_block.update(
        verdict_class="blocked",
        status="complete",
        inference_substrate_class="aggregation",
        board_rows=[],
        placement_rows=[],
        rows=[],
        board_placement_receipt_complete_score=0,
        gate_check_summary={},
    )
    broken_block["reproducibility_checksum"] = exp.artifact_checksum(broken_block)
    assert "blocked_state_invalid" in exp.validate_artifact(broken_block)

    with pytest.raises(ValueError, match="invalid Exp7190"):
        exp.atomic_write(tmp_path / "invalid.json", {"status": "wrong"})


def test_req_ising_7190_blocked_run_and_main_success_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ISING-7190 publishes both terminal branches with visible progress."""

    root = _fixture_root(tmp_path, monkeypatch)
    (root / "CODEX.md").unlink()
    blocked_output = root / "results/blocked.json"
    blocked = exp.run_experiment(
        root,
        exp.RUN_DATE,
        output_path=blocked_output,
        checkpoint_path=root / exp.CHECKPOINT_PATH,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked_output.is_file()

    (root / "CODEX.md").write_text("instructions\n", encoding="utf-8")
    assert (
        exp.main(
            [
                "--root",
                str(root),
                "--output",
                "results/main.json",
                "--checkpoint",
                "results/checkpoints/main.json",
            ]
        )
        == 0
    )
    assert (root / "results/main.json").is_file()
    assert "experiment_complete" in capsys.readouterr().out

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        exp.run_experiment(
            root,
            exp.RUN_DATE,
            output_path=root / "results/forced.json",
            checkpoint_path=root / exp.CHECKPOINT_PATH,
        )
