"""REQ-REPORT-7977 and REQ-HW-7977: estimates preserve historical limits."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest
from coverage import CoverageData

from carnot.reporting import experiment_7977_v691_hardware_evidence as q
from carnot.reporting import validation_7977 as plan
from carnot.reporting import publication_7977 as publication
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7977_v691_hardware_evidence as cli


def authority(private: Path) -> Path:
    """Copy bound inputs so rejection tests cannot rewrite historical evidence."""
    prior = json.loads((q.ROOT / q.PRIOR).read_text())
    current = json.loads((q.ROOT / q.SERVICE).read_text())
    old_terminal = json.loads(Path(prior["terminal_validation_sidecar_path"]).read_text())
    current_terminal = json.loads(Path(current["terminal_validation_sidecar_path"]).read_text())
    report = json.loads(Path(current_terminal["sidecar_path"]).read_text())
    refs = [
        *prior["source_artifact_hashes"].values(),
        *current["source_artifact_hashes"],
        *current["code_config_hashes"],
        current["input_checkpoint"],
    ]
    labels = {
        q.PRIOR,
        q.SERVICE,
        prior["terminal_validation_sidecar_path"],
        current["terminal_validation_sidecar_path"],
        current_terminal["sidecar_path"],
        *prior["terminal_receipt_hashes"],
    }
    labels.update(ref["path"] for ref in refs if ref.get("sha256"))
    labels.update(r["log_path"] for r in [*old_terminal["reports"], *report["report"]["receipts"]])
    for label in labels:
        source = q.ROOT / label
        if source.is_file():
            target = q.bound_path(private, str(source))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return private


def test_custody_and_current_work(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7977-CUSTODY, SCENARIO-HW-7977-SCOPE: keep all obligations."""
    value = q.read_evidence(authority(tmp_path), "20261001")
    assert value["hardware_evidence_ready_score"] == 1
    assert value["workload_attachment_available"]
    assert [r["board"] for r in value["board_rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert value["board_rows"][0]["k_max"] == 5
    assert value["board_rows"][1]["processor_class"] == "linux_cpu"
    assert value["board_rows"][2]["blocker"] == "0xffffffff"
    assert value["verdict_class"] == "null" and value["current_device_execution_count"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["execution_date"] == "20261001" and value["milestone"] == "2026.10.691"
    assert value["workload_branch_readiness"] == {
        "durable_learning": 0,
        "qwen_calibration": 1,
        "source_energy": 0,
    }
    assert {r["target"] for r in value["workload_placement_rows"]} == {
        "CPU",
        "Rust",
        "GPU",
        "LUT/FPGA",
        "TSU",
    }
    assert {r["branch"] for r in value["workload_placement_rows"]} == {"qwen_calibration"}
    assert value["measured_host_costs"]["transfer_bytes"] == 0
    assert value["modeled_acceleration_bounds"]["modeled_100x_bound"] == 1
    assert not value["wishlist_decision"]["priorities_changed"]
    assert set(value) <= set(value["field_principles"])
    assert q.cold_reduce(tmp_path, value)["row_count"] == 3
    with pytest.raises(ValueError, match="run_date_mismatch"):
        q.read_evidence(tmp_path, "20260930")


@pytest.mark.parametrize(
    "kind", ["missing", "malformed", "stub", "terminal", "board_receipt", "retired"]
)
def test_required_custody_failure(tmp_path: Path, kind: str) -> None:
    """SCENARIO-REPORT-7977-CUSTODY: external failures are terminal blocked."""
    root = authority(tmp_path)
    path = root / q.PRIOR
    if kind == "missing":
        path.unlink()
    elif kind == "malformed":
        path.write_text("[]")
    elif kind == "stub":
        atomic_json(path, {"experiment_id": 7964})
    elif kind == "retired":
        (root / "ops").mkdir(exist_ok=True)
        (root / "ops/exclusion_manifest.yaml").write_text(
            "retired_experiments:\n- experiment_id: 7964\n"
        )
    else:
        old = json.loads(path.read_text())
        label = (
            old["terminal_validation_sidecar_path"]
            if kind == "terminal"
            else old["board_rows"][0]["source_path"]
        )
        q.bound_path(root, label).unlink()
    value = q.read_evidence(root, "20261001")
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["hardware_evidence_ready_score"] == 0 and len(value["board_rows"]) == 3
    assert value["gate_check_summary"] and value["typed_blockers"]
    assert all(r["placement_constraint"] is None for r in value["operation_map"])
    assert q.cold_reduce(root, value)["row_count"] == 3


@pytest.mark.parametrize(
    "kind", ["missing", "malformed", "stub", "checkpoint", "terminal", "retired"]
)
def test_optional_service_failure(tmp_path: Path, kind: str) -> None:
    """SCENARIO-REPORT-7977-WORKLOAD: optional absence cannot erase board custody."""
    root = authority(tmp_path)
    path = root / q.SERVICE
    if kind == "missing":
        path.unlink()
    elif kind == "malformed":
        path.write_text("bad json")
    elif kind == "stub":
        atomic_json(path, {"experiment_id": 7976, "service_measurement_ready_score": 1})
    elif kind == "retired":
        (root / "ops").mkdir(exist_ok=True)
        (root / "ops/exclusion_manifest.yaml").write_text("retired:\n- experiment_id: 7976\n")
    else:
        old = json.loads(path.read_text())
        label = (
            old["input_checkpoint"]["path"]
            if kind == "checkpoint"
            else old["terminal_validation_sidecar_path"]
        )
        q.bound_path(root, label).write_text("{}")
    value = q.read_evidence(root, "20261001")
    assert value["hardware_evidence_ready_score"] == 1 and value["verdict_class"] == "null"
    assert not value["workload_attachment_available"] and not value["workload_placement_rows"]
    assert value["workload_attachment_operands"]


def test_malformed_spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7977-WORKLOAD: the upstream cold reader must succeed."""
    root = authority(tmp_path)

    def reject(value: Any) -> None:
        raise ValueError("span_drift")

    monkeypatch.setattr(q.service, "replay", reject)
    value = q.read_evidence(root, "20261001")
    assert not value["workload_attachment_available"]
    assert any(r["observed"] == "span_drift" for r in value["workload_attachment_operands"])


@pytest.mark.parametrize(
    "field",
    [
        "board_rows",
        "workload_placement_rows",
        "source_receipt_hashes",
        "sample_size_budget",
        "gate_check_summary",
        "hardware_evidence_ready_score",
    ],
)
def test_cold_claim_drift(tmp_path: Path, field: str) -> None:
    """SCENARIO-REPORT-7977-VALIDATION: primitive reduction rejects claim drift."""
    root = authority(tmp_path)
    value = q.read_evidence(root, "20261001")
    value[field] = "changed"
    with pytest.raises(ValueError, match="claims_changed"):
        q.cold_reduce(root, value)


def test_owned_failure_reduction(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7977-VALIDATION: owned failures need actual failed receipts."""
    root = authority(tmp_path)
    value = q.read_evidence(root, "20261001")
    value.update(verdict_class="disqualified", hardware_evidence_ready_score=0)
    with pytest.raises(ValueError, match="unsubstantiated_disqualification"):
        q.cold_reduce(root, value)
    value["validation_receipts"]["checks"] = [{"classification": "required", "passed": False}]
    value["acceptance_gate_results"]["readiness"] = 0
    assert q.cold_reduce(root, value)["row_count"] == 3


def test_manifest_and_shards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7977-VALIDATION: dates and coverage inputs stay frozen."""
    commands = plan.manifest(tmp_path, "20261001")
    for spec in commands:
        assert spec["deadline_s"] > 0
        for arg in spec["argv"]:
            if arg.startswith("--include="):
                assert arg == plan.INCLUDE
        if spec["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            assert spec["argv"][spec["argv"].index("--date") + 1] == "20260929"
    assert len(plan.terminal_manifest(tmp_path)) == 3
    with pytest.raises(ValueError, match="private_tmp_required"):
        plan.manifest(q.ROOT / "results", "20261001")
    with pytest.raises(ValueError, match="empty_coverage"):
        q.check_coverage_shards([])
    for mode in ("missing", "empty", "foreign", "valid"):
        path = tmp_path / (mode + ".coverage")
        if mode != "missing":
            data = CoverageData(basename=str(path))
            data.add_lines(
                {
                    str(q.ROOT / plan.MODULE) if mode != "foreign" else "/tmp/foreign.py": {1}
                    if mode == "valid"
                    else set()
                }
            )
            data.write()
        if mode == "valid":
            q.check_coverage_shards([path])
        else:
            with pytest.raises(ValueError, match="empty_coverage"):
                q.check_coverage_shards([path])


@pytest.mark.parametrize("mode", ["passing", "validator_failure", "reader_failure"])
def test_private_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-REPORT-7977-VALIDATION: actual readers select checked bytes."""

    def qualified(root: Path, date: str, output: Path, raw: Path, **kwargs: Any) -> int:
        assert kwargs["evidence"] is q and kwargs["validation"] is plan
        value = q.read_evidence(root, date)
        value["validation_receipts"]["required_checks_passed"] = True
        atomic_json(output, value)
        return 0

    def child(
        spec: dict[str, Any], private: Path, durable: Path, started: float, units: int
    ) -> dict[str, Any]:
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"flagged_count": 0}))
        return {
            **spec,
            "passed": mode != "validator_failure",
            "log_path": str(log),
            "log_sha256": sha256_file(log),
            "exit_code": 0,
            "timed_out": False,
        }

    monkeypatch.setattr(publication.runner, "qualify", qualified)
    monkeypatch.setattr(publication.runner, "run_child", child)
    output = tmp_path / "results/experiment_7977_v691_hardware_evidence.json"
    raw = output.parent / "raw" / output.stem
    if mode == "reader_failure":
        monkeypatch.setattr(publication, "reader_receipt", lambda *a, **k: {"passed": False})
    if mode != "passing":
        with pytest.raises(ValueError, match="candidate_rejected|reader_identity"):
            q.qualify(q.ROOT, "20261001", output, raw)
        return
    assert q.qualify(q.ROOT, "20261001", output, raw) == 0
    value = json.loads(output.read_text())
    receipt = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_text())
    assert receipt["passed"] and receipt["gate_sha256"] == receipt[
        "document_sha256"
    ] == sha256_file(output)
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert terminal["candidate_sha256"] == sha256_file(output)
    assert cli.main(["--output", str(output), "--raw-root", str(raw)]) == 0


def test_cli_private_routes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7977-VALIDATION: success, blocked and replay stay private."""
    output = tmp_path / "success/experiment_7977_evidence.json"
    assert cli.main(["--evidence-only", "--output", str(output)]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260929"])
    value = deepcopy(json.loads(output.read_text()))
    value["rows"] = []
    atomic_json(tmp_path / "changed.json", value)
    with pytest.raises(ValueError, match="claims_changed"):
        cli.main(["--cold-replay", str(tmp_path / "changed.json")])
