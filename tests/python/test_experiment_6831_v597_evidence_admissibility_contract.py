"""REQ-CONSTRAINT-6831 and REQ-CL-6831 evidence-contract tests."""

from __future__ import annotations

import copy
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6831_v597_evidence_admissibility_contract as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def clone_json(value: object) -> Any:
    """Clone large artifact fixtures through the optimized JSON codec."""
    return json.loads(json.dumps(value))


@pytest.fixture(scope="module")
def source_bundle() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Load the immutable source set once; individual tests copy before mutation."""
    payloads: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for name, relative_path in exp.SOURCE_PATHS.items():
        raw = (REPO_ROOT / relative_path).read_bytes()
        payloads[name] = json.loads(raw)
        hashes[name] = exp.sha256_bytes(raw)
    return payloads, hashes


@pytest.fixture()
def phase_clocks() -> dict[str, dict[str, object]]:
    """Give artifact construction an ordered task-owned clock fixture."""
    return {
        name: {"utc": f"2026-08-31T00:00:0{index}Z", "elapsed_s": index / 10}
        for index, name in enumerate(exp.PHASE_NAMES)
    }


@pytest.fixture()
def runtime_receipts() -> tuple[dict[str, object], dict[str, object]]:
    launch = {
        "process_id": 6831,
        "parent_process_id": 1,
        "executable": ".venv/bin/python",
        "working_directory": str(REPO_ROOT),
        "command": [
            ".venv/bin/python",
            "scripts/experiments/experiment_6831_v597_evidence_admissibility_contract.py",
            "--date",
            "20260831",
        ],
    }
    accelerator = {
        "accelerator_credit_claimed": False,
        "process_device_handles": [],
        "loaded_accelerator_libraries": [],
    }
    return launch, accelerator


def make_contract(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> dict[str, object]:
    payloads, hashes = source_bundle
    launch, accelerator = runtime_receipts
    return exp.build_contract(
        payloads,
        hashes.copy(),
        hashes.copy(),
        run_date="20260831",
        phase_clocks=phase_clocks,
        duration_s=0.5,
        launch_receipts=launch,
        accelerator_samples=accelerator,
    )


def test_scenario_6831_preconditions_missing_input_blocks(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """SCENARIO-CONSTRAINT-6831-PRECONDITIONS fails before row reduction."""
    payloads = dict(source_bundle[0])
    hashes = source_bundle[1].copy()
    del payloads["exp6825"]
    del hashes["exp6825"]
    artifact = exp.build_contract(
        payloads,
        hashes,
        hashes,
        run_date="20260831",
        phase_clocks=phase_clocks,
        duration_s=0.5,
        launch_receipts=runtime_receipts[0],
        accelerator_samples=runtime_receipts[1],
    )
    assert artifact["honest_verdict"] == "complete_blocked_v597_evidence_admissibility"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "source_exp6825_readable"


def test_scenario_6831_hash_stability_blocks_drift(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """SCENARIO-CONSTRAINT-6831-HASH-STABILITY rejects changed source bytes."""
    payloads, hashes = source_bundle
    changed = hashes.copy()
    changed["exp6827"] = "sha256:" + "0" * 64
    artifact = exp.build_contract(
        payloads,
        hashes.copy(),
        changed,
        run_date="20260831",
        phase_clocks=phase_clocks,
        duration_s=0.5,
        launch_receipts=runtime_receipts[0],
        accelerator_samples=runtime_receipts[1],
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "source_exp6827_hash_stable"


@pytest.mark.parametrize(
    ("source_name", "mutation", "failed_check"),
    [
        (
            "exp6813",
            lambda payload: payload.__setitem__("status", "running"),
            "source_exp6813_terminal",
        ),
        (
            "exp6813",
            lambda payload: payload.__setitem__("verdict_class", "unknown"),
            "source_exp6813_verdict_closed",
        ),
        (
            "exp6813",
            lambda payload: payload.__setitem__("honest_verdict", "running"),
            "source_exp6813_honest_terminal",
        ),
        (
            "exp6813",
            lambda payload: payload["rows"].__setitem__(-1, payload["rows"][0]),
            "source_exp6813_row_identity_unique",
        ),
        (
            "exp6813",
            lambda payload: payload.__setitem__("selective_arbiter_ab_completed", False),
            "source_exp6813_selective_arbiter_ab_completed",
        ),
        (
            "exp6824",
            lambda payload: payload["source_artifact_hashes"]["exp6813"].__setitem__(
                "sha256", "sha256:" + "0" * 64
            ),
            "source_exp6824_seal_exp6813",
        ),
        (
            "exp6826",
            lambda payload: payload.__setitem__("flagged_adversarial", False),
            "source_exp6826_flagged_receipt",
        ),
    ],
)
def test_req_6831_terminal_metadata_and_seals_fail_closed(
    source_name: str,
    mutation: Any,
    failed_check: str,
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """REQ-CONSTRAINT-6831 checks every terminal identity before reduction."""
    payloads = dict(source_bundle[0])
    payloads[source_name] = clone_json(payloads[source_name])
    mutation(payloads[source_name])
    artifact = exp.build_contract(
        payloads,
        source_bundle[1],
        source_bundle[1],
        run_date="20260831",
        phase_clocks=phase_clocks,
        duration_s=0.5,
        launch_receipts=runtime_receipts[0],
        accelerator_samples=runtime_receipts[1],
    )
    assert artifact["gate_check_summary"]["failed_check"] == failed_check
    assert artifact["rows"] == []


@pytest.mark.parametrize("source_name", ["exp6813", "exp6824", "exp6825", "exp6826", "exp6827"])
def test_scenario_6831_incomplete_source_rows_block(
    source_name: str,
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """REQ-CONSTRAINT-6831 requires the exact per-unit source roster."""
    payloads = dict(source_bundle[0])
    hashes = source_bundle[1]
    payloads[source_name] = clone_json(payloads[source_name])
    payloads[source_name]["rows"].pop()
    artifact = exp.build_contract(
        payloads,
        hashes,
        hashes,
        run_date="20260831",
        phase_clocks=phase_clocks,
        duration_s=0.5,
        launch_receipts=runtime_receipts[0],
        accelerator_samples=runtime_receipts[1],
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == f"source_{source_name}_row_count"


def test_scenario_6831_fresh_reduction_and_disqualified_authority(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
) -> None:
    """SCENARIO-CONSTRAINT-6831-FRESH-REDUCTION fails closed on an attack row."""
    payloads = dict(source_bundle[0])
    decisions, rows = exp.reduce_selective_arbiter(
        payloads["exp6813"], payloads["exp6824"], payloads["exp6825"]
    )
    assert decisions["utility"]["paired_progress_delta"] == pytest.approx(0.125)
    assert decisions["adoption"]["decision"] == "enable"
    assert len(rows) == exp.AUTHORITY_CRITERION_ROW_COUNT

    payloads["exp6825"] = clone_json(payloads["exp6825"])
    payloads["exp6825"]["rows"][0]["passed"] = False
    decisions, _ = exp.reduce_selective_arbiter(
        payloads["exp6813"], payloads["exp6824"], payloads["exp6825"]
    )
    assert decisions["hard_safety"]["finding"] == "disqualified"
    assert decisions["adoption"]["decision"] == "redesign"


@pytest.mark.parametrize(
    ("findings", "expected"),
    [
        (("positive",) * 4, "enable"),
        (("positive", "positive", "positive", "null"), "keep_shadow"),
        (("positive", "positive", "positive", "harmful"), "retire"),
        (("positive", "positive", "partial", "positive"), "insufficient"),
        (("positive", "blocked", "positive", "positive"), "insufficient"),
        (("positive", "disqualified", "positive", "positive"), "redesign"),
    ],
)
def test_scenario_6831_conservative_table(findings: tuple[str, ...], expected: str) -> None:
    """SCENARIO-CONSTRAINT-6831-CONSERVATIVE-TABLE has closed propagation."""
    assert exp.derive_adoption(findings) == expected


def test_scenario_6831_admissibility_does_not_require_positive_utility() -> None:
    """SCENARIO-CONSTRAINT-6831-ADMISSIBILITY separates effect from procedure."""
    findings = {
        "hard_safety": "positive",
        "safe_action_identity": "positive",
        "certificate_truth": "positive",
        "utility": "null",
    }
    assert exp.receipt_is_admissible(findings, rows_complete=True)
    findings["hard_safety"] = "disqualified"
    assert not exp.receipt_is_admissible(findings, rows_complete=True)
    assert not exp.receipt_is_admissible(
        {**findings, "hard_safety": "positive"}, rows_complete=False
    )


@pytest.mark.parametrize(
    ("metrics", "finding"),
    [
        ({"harmful_selection_count": 1, "paired_progress_delta": 1.0}, "harmful"),
        ({"harmful_selection_count": 0, "paired_progress_delta": 0.0}, "null"),
        (
            {
                "harmful_selection_count": 0,
                "paired_progress_delta": 0.1,
                "held_selective_rows": 1,
                "pair_count": 1,
                "legal_support_count": 1,
            },
            "partial",
        ),
    ],
)
def test_utility_findings_cover_harm_null_and_partial(
    metrics: dict[str, object], finding: str
) -> None:
    assert exp._finding_from_utility(metrics) == finding


def test_scenario_6831_quarantines_flagged_comparator(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> None:
    """SCENARIO-CONSTRAINT-6831-QUARANTINE leaves Exp6826 nonauthoritative."""
    artifact = make_contract(source_bundle, phase_clocks, runtime_receipts)
    disposition = artifact["flagged_receipt_disposition"]
    assert disposition["source"] == "exp6826"
    assert disposition["disposition"] == "quarantined_comparator_only"
    assert disposition["authority_consumed"] is False
    assert artifact["selective_arbiter_receipt_admissible"] is True


def test_scenario_6831_clock_capture_is_ordered(
    phase_clocks: dict[str, dict[str, object]],
) -> None:
    """SCENARIO-CONSTRAINT-6831-CLOCKS checks every task-owned phase."""
    assert exp.validate_phase_clocks(phase_clocks, duration_s=0.5) == []
    broken = copy.deepcopy(phase_clocks)
    broken["verify"]["elapsed_s"] = 0.2
    assert "phase clocks are not ordered" in exp.validate_phase_clocks(broken, duration_s=0.5)
    del broken["write"]
    assert "phase clock names differ from the required phases" in exp.validate_phase_clocks(
        broken, duration_s=0.5
    )
    nonnumeric = copy.deepcopy(phase_clocks)
    nonnumeric["read"]["elapsed_s"] = "later"
    assert "phase clocks contain a nonnumeric elapsed value" in exp.validate_phase_clocks(
        nonnumeric, duration_s=0.5
    )
    assert "duration does not cover the verify phase" in exp.validate_phase_clocks(
        phase_clocks, duration_s=0.1
    )


def test_bad_clock_blocks_an_otherwise_complete_contract(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> None:
    broken = copy.deepcopy(phase_clocks)
    broken["verify"]["elapsed_s"] = 1.0
    artifact = exp.build_contract(
        source_bundle[0],
        source_bundle[1],
        source_bundle[1],
        run_date="20260831",
        phase_clocks=broken,
        duration_s=0.5,
        launch_receipts=runtime_receipts[0],
        accelerator_samples=runtime_receipts[1],
    )
    assert artifact["gate_check_summary"]["failed_check"] == "task_owned_phase_clocks"


def _criterion_map(validation: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["criterion"]: row for row in validation["rows"]}


@pytest.mark.parametrize(
    ("criterion", "mutate"),
    [
        (
            "row_hash_identity",
            lambda stream: stream["rows"][0].__setitem__("receipt_sha256", "bad"),
        ),
        ("order_identity", lambda stream: stream["order_hashes"].__setitem__("order_1", "bad")),
        (
            "split_identity",
            lambda stream: stream["split_manifest"].__setitem__("development_count_per_family", 71),
        ),
        (
            "canonical_transactions",
            lambda stream: stream["rows"][0].__setitem__("new_state_sha256", "sha256:" + "0" * 64),
        ),
        (
            "family_rotation",
            lambda stream: stream["split_manifest"]["rotations"][0].__setitem__(
                "held_out_family", stream["split_manifest"]["rotations"][1]["held_out_family"]
            ),
        ),
        ("sealed_fields", lambda stream: stream["feature_denylist"].pop()),
        (
            "nonzero_headroom",
            lambda stream: stream["headroom_metrics"].__setitem__("conflict_event_count", 0),
        ),
        ("learning_preconditions", lambda stream: stream.__setitem__("honest_verdict", "complete")),
    ],
)
def test_req_cl_6831_stream_checks_detect_mutation(
    criterion: str,
    mutate: object,
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
) -> None:
    """REQ-CL-6831 validators independently reject each readiness mutation."""
    stream = clone_json(source_bundle[0]["exp6827"])
    mutate(stream)
    validation = exp.validate_csl_stream(stream)
    assert _criterion_map(validation)[criterion]["passed"] is False
    assert validation["admissible"] is False


def test_req_cl_6831_stream_is_ready_without_learning(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
) -> None:
    """SCENARIO-CL-6831-NO-LEARNING validates inputs without a training claim."""
    validation = exp.validate_csl_stream(source_bundle[0]["exp6827"])
    assert validation["admissible"] is True
    assert validation["learning_ran"] is False
    assert all(row["passed"] for row in validation["rows"])
    assert len(validation["rows"]) == exp.STREAM_CRITERION_ROW_COUNT


def test_stream_validator_defensive_branches(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
) -> None:
    """REQ-CL-6831 rejects malformed identities at each structural layer."""
    stream = source_bundle[0]["exp6827"]
    rows = stream["rows"]
    malformed_id = dict(rows[0])
    malformed_id["row_id"] = "wrong"
    assert exp._digest_fields_valid([malformed_id])[0] is False
    assert exp._order_identity_valid({"order_hashes": {}, "split_manifest": {}}, []) is False
    assert exp._order_identity_valid(stream, []) is False

    for field, value in (
        ("held_future_count_per_family", 23),
        ("development_count_per_family", 72),
    ):
        changed = clone_json(stream)
        changed["split_manifest"][field] = value
        if field == "development_count_per_family":
            family = next(iter(changed["split_manifest"]["by_family"]))
            changed["split_manifest"]["by_family"][family]["development"].pop()
        assert exp._split_identity_valid(changed, changed["rows"]) is False
    changed = clone_json(stream)
    family = next(iter(changed["split_manifest"]["by_family"]))
    changed["split_manifest"]["by_family"][family]["held_future"].pop()
    assert exp._split_identity_valid(changed, changed["rows"]) is False
    changed = clone_json(stream)
    family = next(iter(changed["split_manifest"]["by_family"]))
    changed["split_manifest"]["by_family"][family]["hard_case"].pop()
    assert exp._split_identity_valid(changed, changed["rows"]) is False

    rejected = dict(next(row for row in rows if row["operation_admitted"] is False))
    rejected["new_state_sha256"] = "sha256:" + "0" * 64
    assert exp._transaction_validation([rejected])[0] is False
    mutating = dict(
        next(
            row
            for row in rows
            if row["operation_admitted"] is True
            and row["operation_kind"] in exp.MUTATING_OPERATION_KINDS
        )
    )
    mutating["new_state_sha256"] = mutating["parent_state_sha256"]
    assert exp._transaction_validation([mutating])[0] is False

    changed = clone_json(stream)
    changed["split_manifest"]["rotations"][0]["development_families"] = []
    assert exp._rotation_valid(changed) is False


def test_accelerator_sampling_records_handles_and_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accelerator observations remain explicit even when samples are present."""

    class FakeEntry:
        def __init__(self, target: str | None) -> None:
            self.target = target

        def resolve(self) -> str:
            if self.target is None:
                raise OSError("closed descriptor")
            return self.target

    class FakePath:
        def __init__(self, value: str) -> None:
            self.value = value

        def is_dir(self) -> bool:
            return self.value.endswith("fd")

        def iterdir(self) -> list[FakeEntry]:
            return [FakeEntry(None), FakeEntry("/dev/nvidia0")]

        def is_file(self) -> bool:
            return self.value.endswith("maps")

        def read_text(self, **_kwargs: object) -> str:
            return "00-01 r--p /usr/lib/libnvidia.so\n"

    monkeypatch.setattr(exp, "Path", FakePath)
    samples = exp._accelerator_samples()
    assert samples["process_device_handles"] == ["/dev/nvidia0"]
    assert samples["loaded_accelerator_libraries"] == ["/usr/lib/libnvidia.so"]


def test_terminal_contract_has_complete_schema_and_checksum(
    source_bundle: tuple[dict[str, dict[str, object]], dict[str, str]],
    phase_clocks: dict[str, dict[str, object]],
    runtime_receipts: tuple[dict[str, object], dict[str, object]],
) -> None:
    artifact = make_contract(source_bundle, phase_clocks, runtime_receipts)
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["v597_contract_ready"] is True
    assert artifact["csl_inputs_admissible"] is True
    assert artifact["verifier_is_oracle"] is False
    assert set(artifact["field_principles"]) == set(artifact) - {"field_principles"}
    assert exp.validate_artifact(artifact) == []

    changed = copy.deepcopy(artifact)
    changed["rows"][0]["passed"] = False
    assert "reproducibility checksum mismatch" in exp.validate_artifact(changed)

    bad_rows = copy.deepcopy(artifact)
    bad_rows["rows"].pop()
    exp._apply_checksum(bad_rows)
    assert (
        "ready contract does not have one row per criterion source unit"
        in exp.validate_artifact(bad_rows)
    )
    bad_components = copy.deepcopy(artifact)
    bad_components["csl_inputs_admissible"] = False
    exp._apply_checksum(bad_components)
    assert "ready contract disagrees with component admissibility" in exp.validate_artifact(
        bad_components
    )


def test_execute_writes_and_validates_task_owned_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command path captures clocks, rechecks hashes, and writes one result."""
    for relative_path in exp.SOURCE_PATHS.values():
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative_path).read_bytes())
    output = tmp_path / exp.OUTPUT_PATH
    artifact = exp.execute(tmp_path, "20260831", output)
    assert json.loads(output.read_text()) == artifact
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", str(output)]) == 0
    cli_output = tmp_path / "cli-result.json"
    assert (
        exp.main(["--root", str(tmp_path), "--date", "20260831", "--output", str(cli_output)]) == 0
    )
    assert cli_output.exists()
    monkeypatch.setattr(sys, "argv", [str(exp.MODULE_PATH), "--validate", str(output)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO_ROOT / exp.MODULE_PATH), run_name="__main__")
    assert exit_info.value.code == 0


def test_execute_missing_sources_still_writes_blocked_artifact(tmp_path: Path) -> None:
    """A preflight failure remains a terminal auditable result."""
    output = tmp_path / exp.OUTPUT_PATH
    artifact = exp.execute(tmp_path, "20260831", output)
    assert output.exists()
    assert artifact["honest_verdict"] == "complete_blocked_v597_evidence_admissibility"
    assert exp.main(["--validate", str(output)]) == 0
    malformed = copy.deepcopy(artifact)
    malformed["rows"] = [{}]
    exp._apply_checksum(malformed)
    assert "blocked precondition contract contains reduced rows" in exp.validate_artifact(malformed)


def test_main_rejects_invalid_artifact_and_date(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}\n")
    assert exp.main(["--validate", str(bad)]) == 1
    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.execute(tmp_path, "2026-08-31", tmp_path / "out.json")
