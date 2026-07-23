"""Tests for Exp5861 attached-board state receipts.

Spec refs: REQ-HW-5861, SCENARIO-HW-5861.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5861_attached_board_state_receipts as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/hardware/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5861_attached_board_state_receipts.py")


class LocalVersionRunner:
    """REQ-HW-5861 fake local runner; board commands must never reach it."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.LocalCommandReceipt:
        assert timeout_s > 0.0
        rendered = mod.command_to_string(command)
        assert "ssh kria" not in rendered
        assert "ssh polarfire" not in rendered
        assert "openFPGALoader -c dirtyJtag --detect" not in rendered
        assert "/dev/mmcblk" not in rendered
        assert "/dev/disk" not in rendered
        self.commands.append(command)
        return mod.LocalCommandReceipt(
            command=command,
            exit_code=0,
            stdout=f"{command[0]} fixture version\n",
            stderr="",
            duration_s=0.001,
        )


def _test_exit_codes() -> dict[str, int]:
    return {
        TEST_PATH.as_posix(): 0,
        ".venv/bin/pytest tests/python -q": 0,
        ".venv/bin/python scripts/adversarial_verify.py "
        "results/experiment_5861_attached_board_state_receipts.json": 0,
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_5861_attached_board_state_receipts.py": 0,
        ".venv/bin/python scripts/root_clutter_sweep.py --check": 0,
        '.venv/bin/python -c "from pathlib import Path; '
        "assert Path('scripts/research_conductor.py').exists()\"": 0,
    }


def _artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260723",
        duration_s=1.25,
        local_command_runner=LocalVersionRunner(),
        test_exit_codes=_test_exit_codes(),
    )


def test_req_hw_5861_spec_declares_attached_board_receipt_contract() -> None:
    """REQ-HW-5861: OpenSpec names required fields, principles, and no-repeat gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5861") : spec.index("### SCENARIO-HW-5861")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-HW-5861",
        "SCENARIO-HW-5861",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "adaptive_state_microkernel_ready_score == 1.0",
        "no board commands",
        "host `/dev/mmcblk*`",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_hw_5861_no_change_receipt_avoids_board_commands() -> None:
    """SCENARIO-HW-5861: unchanged/blocked routes emit receipts without probes."""

    runner = LocalVersionRunner()
    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260723",
        duration_s=1.25,
        local_command_runner=runner,
        test_exit_codes=_test_exit_codes(),
    )

    assert {command[0] for command in runner.commands} >= {"python", "ssh", "openFPGALoader"}
    assert artifact["status"] == "no_change_no_authenticated_state_operation_execution"
    assert artifact["spec_refs"] == ["REQ-HW-5861", "SCENARIO-HW-5861"]
    assert artifact["random_seed"] == 5861
    assert artifact["run_date"] == "20260723"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["authenticated_physical_execution_receipts"] == []
    assert artifact["authenticated_state_operation_parity_score"] == 0.0
    assert isinstance(artifact["authenticated_state_operation_parity_score"], float)
    assert artifact["same_input_state_and_hash_parity"]["physical_execution_observed"] is False
    assert artifact["software_fallback_disclosed"]["cpu_reference_is_not_board_execution"] is True
    assert artifact["honest_verdict"].startswith("no-change:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_5861_preconditions_hash_prior_receipts_and_resources_before_board_use() -> None:
    """REQ-HW-5861: exact prior receipts, tools, resources, identities, and output are hashed."""

    artifact = _artifact()
    preconditions = artifact["preconditions_checked"]
    prior = artifact["prior_receipt_hashes"]

    assert preconditions["recorded_before_any_board_command"] is True
    assert preconditions["board_commands_run_during_precondition_collection"] == []
    assert preconditions["atomic_output"]["ok"] is True
    assert preconditions["resources"]["disk"]["ok"] is True
    assert preconditions["resources"]["ram"]["ok"] is True
    assert set(preconditions["board_identities"]) == {"kv260", "polarfire", "gatemate"}
    assert preconditions["access_state"]["kv260"]["interface"] == "ssh:kria"
    assert preconditions["access_state"]["polarfire"]["interface"] == "ssh:polarfire"
    assert preconditions["access_state"]["gatemate"]["interface"] == "dirtyjtag"
    assert all(path in prior for path in mod.PRIOR_RECEIPT_PATHS)
    assert prior["results/experiment_5859_adaptive_state_microkernel_parity.json"][
        "sha256"
    ] == mod.file_sha256(REPO / "results/experiment_5859_adaptive_state_microkernel_parity.json")
    assert not mod.contains_retired_kv260_precondition(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_5861_per_board_matrix_does_not_launder_execution() -> None:
    """REQ-HW-5861: compile, reachability, and prior execution stay separate per board."""

    artifact = _artifact()
    matrix = artifact["board_capability_matrix"]
    access = artifact["per_board_access_and_toolchain_receipts"]

    assert set(matrix) == {"kv260", "polarfire", "gatemate"}
    assert matrix["kv260"]["capability_class"] == "programmed_image"
    assert matrix["kv260"]["authenticated_state_operation_execution"] is False
    assert matrix["kv260"]["measured_state_update_dynamics"] is False
    assert matrix["kv260"]["route_changed_since_prior_receipt"] is False

    assert matrix["polarfire"]["capability_class"] == "authenticated_physical_execution"
    assert matrix["polarfire"]["prior_workload_validated"] is True
    assert matrix["polarfire"]["authenticated_state_operation_execution"] is False

    assert matrix["gatemate"]["capability_class"] == "unreachable"
    assert matrix["gatemate"]["toolchain_only"] is True
    assert matrix["gatemate"]["authenticated_state_operation_execution"] is False

    assert access["kv260"]["physical_reachability"] == "cached_ssh_reachable"
    assert access["polarfire"]["physical_reachability"] == "cached_ssh_reachable"
    assert access["gatemate"]["physical_reachability"] == "cached_dirtyjtag_no_gm1ax_idcode"
    assert access["gatemate"]["toolchain_receipts"]["openFPGALoader"]["available"] is True
    mod.validate_artifact(artifact)


def test_req_hw_5861_exp5859_block_prevents_mapping_and_parity_score() -> None:
    """REQ-HW-5861: Exp5859 readiness is required before same-input board parity."""

    artifact = _artifact()
    exp5859 = artifact["exp5859_input_receipt"]

    assert exp5859["present"] is True
    assert exp5859["adaptive_state_microkernel_ready_score"] == 0.0
    assert exp5859["mapping_allowed"] is False
    assert artifact["bounded_operation_mapping"]["status"] == "not_mapped_exp5859_not_ready"
    assert artifact["bounded_operation_mapping"]["unsupported_operations"] == list(
        mod.ADAPTIVE_STATE_OPERATIONS
    )
    assert artifact["cpu_reference_receipts"]["status"] == "not_run_exp5859_not_ready"
    assert artifact["cpu_reference_receipts"]["software_authority_only"] is True
    assert artifact["same_input_state_and_hash_parity"]["parity_within_exact_tolerance"] is None
    assert artifact["same_input_state_and_hash_parity"]["score_reason"] == (
        "no same-input authenticated physical execution occurred"
    )
    mod.validate_artifact(artifact)


def test_req_hw_5861_dynamics_capacity_and_timing_are_auditable_but_not_claims() -> None:
    """REQ-HW-5861: dynamics, backend semantics, raw logs, and timing stay bounded."""

    artifact = _artifact()
    dynamics = artifact["requested_vs_programmed_vs_observed_dynamics"]
    semantics = artifact["capacity_precision_stochasticity_and_observability"]
    timing = artifact["timing_source_and_raw_logs"]
    avoided = artifact["unchanged_precondition_actions_avoided"]

    assert dynamics["requested_operation"] == "adaptive_state_microkernel_same_input_parity"
    assert dynamics["kv260"]["requested_topology_is_execution"] is False
    assert dynamics["polarfire"]["compile_or_reachability_is_execution"] is False
    assert dynamics["gatemate"]["programmed_image_observed"] is False
    assert semantics["kv260"]["stochastic_update_capability"] == "not_authenticated_for_state_ops"
    assert semantics["polarfire"]["observability"] == "prior_ssh_stdout_hash_only"
    assert semantics["gatemate"]["supported_operations"] == []
    assert timing["speedup_claimed"] is False
    assert timing["board_timing_claimed"] is False
    assert timing["new_board_commands"] == []
    assert {row["board"] for row in avoided} == {"kv260", "polarfire", "gatemate"}
    assert all(row["external_action_required"] for row in avoided)
    assert artifact["prohibited_claims_absent"]["all_absent"] is True
    mod.validate_artifact(artifact)


def test_req_hw_5861_schema_rejects_overclaims_and_fallback_drift() -> None:
    """REQ-HW-5861: schema blocks fake hardware success, score drift, and unsafe probes."""

    base = _artifact()
    mutations = [
        (lambda a: a.update(inference_substrate="hardware_smoke"), "inference_substrate"),
        (lambda a: a.update(spec_refs=["REQ-HW-5861"]), "spec_refs"),
        (lambda a: a.update(authenticated_state_operation_parity_score=1.0), "score requires"),
        (
            lambda a: a["software_fallback_disclosed"].update(  # type: ignore[index,union-attr]
                cpu_reference_is_not_board_execution=False
            ),
            "software fallback",
        ),
        (
            lambda a: a["prohibited_claims_absent"].update(  # type: ignore[index,union-attr]
                speedup_claim_absent=False
            ),
            "prohibited claims",
        ),
        (
            lambda a: a["board_capability_matrix"]["kv260"].update(  # type: ignore[index,union-attr]
                authenticated_state_operation_execution=True
            ),
            "physical execution receipt",
        ),
        (
            lambda a: a["preconditions_checked"]["access_state"]["kv260"].update(  # type: ignore[index,union-attr]
                command="ls /dev/mmcblk0"
            ),
            "retired KV260",
        ),
        (lambda a: a.update(field_principles={}), "field_principles"),
        (lambda a: a.update(honest_verdict="complete: speedup=true"), "honest_verdict"),
        (lambda a: a["field_provenance"].pop("status"), "field provenance"),  # type: ignore[index,union-attr]
    ]

    for mutate, needle in mutations:
        artifact = deepcopy(base)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        assert any(needle in error for error in mod.artifact_schema_errors(artifact))

    artifact = deepcopy(base)
    artifact["status"] = "tampered"
    assert any("checksum" in error for error in mod.artifact_schema_errors(artifact))

    artifact = deepcopy(base)
    del artifact["status"]
    assert any("missing required fields" in error for error in mod.artifact_schema_errors(artifact))

    with pytest.raises(ValueError, match="score requires"):
        invalid = deepcopy(base)
        invalid["authenticated_state_operation_parity_score"] = 1.0
        invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
        mod.validate_artifact(invalid)


def test_req_hw_5861_schema_accepts_real_same_input_physical_parity_fixture() -> None:
    """REQ-HW-5861: score 1.0 is legal only with same-input physical parity evidence."""

    artifact = _artifact()
    artifact["status"] = "parity_authenticated_physical_state_operation_execution"
    artifact["honest_verdict"] = "parity: same-input physical state hash parity observed"
    artifact["authenticated_physical_execution_receipts"] = [
        {
            "board": "kv260",
            "board_identity": "fixture-kv260",
            "input_hash": "sha256:" + "1" * 64,
            "output_hash": "sha256:" + "2" * 64,
            "state_hash": "sha256:" + "3" * 64,
            "cpu_state_hash": "sha256:" + "3" * 64,
            "raw_log_sha256": "sha256:" + "4" * 64,
            "timing_source": "fixture_counter",
            "exact_tolerance": "canonical_json_and_state_hash_identical",
            "parity": True,
        }
    ]
    artifact["same_input_state_and_hash_parity"] = {
        "physical_execution_observed": True,
        "exact_tolerance": "canonical_json_and_state_hash_identical",
        "parity_within_exact_tolerance": True,
        "matched_receipt_count": 1,
        "mismatches": [],
        "score_reason": "same-input physical execution matched CPU state hash",
    }
    artifact["authenticated_state_operation_parity_score"] = 1.0
    artifact["board_capability_matrix"]["kv260"]["authenticated_state_operation_execution"] = True  # type: ignore[index,union-attr]
    artifact["board_capability_matrix"]["kv260"]["measured_state_update_dynamics"] = True  # type: ignore[index,union-attr]
    artifact["bounded_operation_mapping"]["status"] = "mapped_bounded_same_input_operations"  # type: ignore[index,union-attr]
    artifact["cpu_reference_receipts"]["status"] = "same_input_reference_recorded"  # type: ignore[index,union-attr]
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    mod.validate_artifact(artifact)


def test_req_hw_5861_run_experiment_writes_valid_json(tmp_path: Path) -> None:
    """REQ-HW-5861: run_experiment writes the deliverable JSON atomically."""

    artifact = _artifact()
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert saved == artifact
    assert saved["test_exit_codes"] == _test_exit_codes()
    mod.validate_artifact(saved)

    live_path = mod.run_experiment(
        repo_root=REPO,
        run_date="20260723",
        duration_s=1.25,
        local_command_runner=LocalVersionRunner(),
        test_exit_codes=_test_exit_codes(),
    )
    assert live_path == REPO / mod.RESULT_RELATIVE_PATH
    live = json.loads(live_path.read_text(encoding="utf-8"))
    assert live["reproducibility_checksum"] == mod.payload_checksum(live)
    mod.validate_artifact(live)


def test_req_hw_5861_helper_edges_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HW-5861: helper failure edges and CLI parsing stay deterministic."""

    assert mod.unwrap_field({"value": "wrapped"}) == "wrapped"
    assert mod.unwrap_field("bare") == "bare"
    assert mod.file_receipt(REPO / "missing-exp5861-file", REPO)["present"] is False
    assert mod.sha256_text("x").startswith("sha256:")
    assert mod.is_sha256("sha256:" + "a" * 64)
    assert not mod.is_sha256("bad")

    missing_exp5859 = mod.exp5859_receipt(tmp_path)
    assert missing_exp5859["present"] is False
    assert missing_exp5859["mapping_allowed"] is False

    list_json = tmp_path / "bad.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object expected"):
        mod.read_json(list_json)
    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_test_results_json("[]")

    def fake_run_experiment(**kwargs: object) -> Path:
        assert kwargs["run_date"] == "20260723"
        assert kwargs["test_exit_codes"] == _test_exit_codes()
        return Path("results/fake5861.json")

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert (
        mod.main(["--date", "20260723", "--test-results-json", json.dumps(_test_exit_codes())]) == 0
    )
    assert "results/fake5861.json" in capsys.readouterr().out


def test_req_hw_5861_local_command_mapping_and_parity_helper_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-5861: local helpers cover ready-route and command failure branches."""

    class Completed:
        returncode = 7
        stdout = "out\n"
        stderr = "err\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Completed())
    receipt = mod.run_local_command(("tool", "--version"), 1.0)
    assert receipt.exit_code == 7
    assert receipt.stdout == "out\n"

    def raise_os_error(*args: object, **kwargs: object) -> None:
        raise OSError("missing tool")

    monkeypatch.setattr(mod.subprocess, "run", raise_os_error)
    missing = mod.run_local_command(("missing-tool",), 1.0)
    assert missing.exit_code == 127
    assert "missing tool" in missing.stderr

    base = _artifact()
    ready = {
        "mapping_allowed": True,
        "operation_trace_hash": "sha256:" + "5" * 64,
    }
    matrix = deepcopy(base["board_capability_matrix"])
    no_route = mod.bounded_operation_mapping(ready, matrix)  # type: ignore[arg-type]
    assert no_route["status"] == "not_mapped_no_changed_authenticated_route"
    assert mod.cpu_reference_receipts(ready, no_route)["status"] == (
        "not_run_no_changed_authenticated_route"
    )

    matrix["kv260"]["route_changed_since_prior_receipt"] = True  # type: ignore[index]
    mapped = mod.bounded_operation_mapping(ready, matrix)  # type: ignore[arg-type]
    assert mapped["status"] == "mapped_bounded_same_input_operations"
    assert mapped["capacity_bound"] == 64
    assert mod.cpu_reference_receipts(ready, mapped)["status"] == "same_input_reference_recorded"

    mismatch = mod.same_input_state_and_hash_parity([{"board": "kv260", "parity": False}])
    assert mismatch["physical_execution_observed"] is True
    assert mismatch["parity_within_exact_tolerance"] is False

    live_path = mod.run_experiment(
        repo_root=tmp_path,
        run_date="20260723",
        duration_s=None,
        local_command_runner=LocalVersionRunner(),
        test_exit_codes=_test_exit_codes(),
    )
    assert live_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(live_path.read_text(encoding="utf-8"))["duration_s"] >= 0.0


def test_req_hw_5861_schema_guard_defensive_error_branches() -> None:
    """REQ-HW-5861: defensive schema branches are asserted, not decorative."""

    base = _artifact()
    mutations = [
        (lambda a: a.update(schema="bad"), "schema mismatch"),
        (lambda a: a.update(experiment_id="bad"), "experiment_id mismatch"),
        (lambda a: a.update(random_seed=0), "random_seed mismatch"),
        (lambda a: a.update(preconditions_checked=[]), "preconditions_checked invalid"),
        (
            lambda a: a["preconditions_checked"].update(  # type: ignore[index,union-attr]
                recorded_before_any_board_command=False
            ),
            "preconditions must precede",
        ),
        (
            lambda a: a["preconditions_checked"].update(  # type: ignore[index,union-attr]
                board_commands_run_during_precondition_collection=["ssh kria true"]
            ),
            "board commands ran",
        ),
        (lambda a: a.update(prior_receipt_hashes={}), "prior_receipt_hashes"),
        (lambda a: a.update(board_capability_matrix=[]), "board_capability_matrix"),
        (
            lambda a: a.update(authenticated_physical_execution_receipts="bad"),
            "authenticated_physical_execution_receipts",
        ),
        (
            lambda a: a["board_capability_matrix"].update(kv260="bad"),  # type: ignore[index,union-attr]
            "kv260 matrix row invalid",
        ),
        (
            lambda a: a["authenticated_physical_execution_receipts"].append("bad"),  # type: ignore[index,union-attr]
            "physical execution receipt invalid",
        ),
        (
            lambda a: a["authenticated_physical_execution_receipts"].append(  # type: ignore[index,union-attr]
                {
                    "board": "kv260",
                    "input_hash": "bad",
                    "output_hash": "sha256:" + "2" * 64,
                    "state_hash": "sha256:" + "3" * 64,
                    "cpu_state_hash": "sha256:" + "3" * 64,
                    "raw_log_sha256": "sha256:" + "4" * 64,
                    "exact_tolerance": mod.STATE_OPERATION_EXACT_TOLERANCE,
                    "parity": True,
                }
            ),
            "physical execution input_hash invalid",
        ),
        (
            lambda a: a["authenticated_physical_execution_receipts"].append(  # type: ignore[index,union-attr]
                {
                    "board": "kv260",
                    "input_hash": "sha256:" + "1" * 64,
                    "output_hash": "sha256:" + "2" * 64,
                    "state_hash": "sha256:" + "3" * 64,
                    "cpu_state_hash": "sha256:" + "3" * 64,
                    "raw_log_sha256": "sha256:" + "4" * 64,
                    "exact_tolerance": "loose",
                    "parity": True,
                }
            ),
            "exact_tolerance",
        ),
        (
            lambda a: a.update(authenticated_state_operation_parity_score=0),
            "bare 0.0 or 1.0 float",
        ),
        (
            lambda a: a.update(same_input_state_and_hash_parity=[]),
            "same_input_state_and_hash_parity",
        ),
        (
            lambda a: a["same_input_state_and_hash_parity"].update(  # type: ignore[index,union-attr]
                parity_within_exact_tolerance=False
            ),
            "zero no-execution parity",
        ),
        (lambda a: a.update(prohibited_claims_absent=[]), "prohibited claims absence"),
        (lambda a: a.update(field_provenance=[]), "field provenance invalid"),
    ]

    for mutate, needle in mutations:
        artifact = deepcopy(base)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        assert any(needle in error for error in mod.artifact_schema_errors(artifact))
