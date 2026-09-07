"""Focused tests for REQ-REPORT-7123 and SCENARIO-REPORT-7123-*.

The fixtures use temporary GGUF and trace files. They never load a model or
write the repository result, so a unit test cannot alter measurement evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7123_v625_arc_loo_shard_a as exp


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _registry(*, first: str = "r11l", positive_receipt: bool = False) -> dict[str, object]:
    """Return registry-only inputs in a stable eligibility order."""

    receipt = (
        [{"mechanism": exp.MECHANISM_ID, "reached_level": 1}]
        if positive_receipt
        else []
    )
    return {
        "games": [
            {
                "game": first,
                "reproducibility": "reproduced",
                "levels_reproduced": 6,
                "full_game_clear": True,
                "adapter_withheld_receipts": receipt,
            },
            {
                "game": "second",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
                "full_game_clear": True,
            },
        ]
    }


def _model(tmp_path: Path) -> dict[str, object]:
    """Create one small file with the same identity fields as the real cache row."""

    path = tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    path.write_bytes(b"GGUF" + b"model-fixture")
    return exp.resolve_headline_model(
        lambda **_kwargs: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp.HEADLINE_REPO_ID,
                "gpu": 0,
                "model_path": str(path),
            },
            {
                "name": "unused-second-cache-row",
                "hf_id": "unused/second",
                "gpu": 1,
                "model_path": str(tmp_path / "unused.gguf"),
            },
        ]
    )


def _attempt(
    tmp_path: Path,
    arm: str,
    *,
    level: int = 0,
    parse_valid: bool = True,
    forbidden: bool = False,
) -> dict[str, object]:
    """Build one content-addressed attempt without model or environment access."""

    raw_path = tmp_path / arm / "attempt-1.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "prompt": "frame-only prompt",
        "output": '{"action":1,"data":null}',
        "before": {"level": 0, "frame": [[0]]},
        "after": {"level": level, "frame": [[level]]},
    }
    raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    return exp.make_attempt_receipt(
        arm=arm,
        attempt_index=1,
        prompt=raw["prompt"],
        output=raw["output"],
        proposed_action={"action": 1, "data": None} if parse_valid else None,
        executed_action={"action": 1, "data": None} if parse_valid else None,
        before=raw["before"],
        after=raw["after"],
        forecast={"source": "E3AgentPolicy", "predicted_change": False},
        verifier={"accepted": True, "oracle": False},
        signal_consumed=False,
        prompt_tokens=4,
        output_tokens=6,
        duration_s=0.25,
        worker_pid=4100 if arm == "adapter_withheld" else 4200,
        raw_trace_path=raw_path,
        isolation={"fresh_process": True, "adapter_visible": arm != "adapter_withheld"},
        forbidden_access_rows=(
            [{"kind": "import", "target": exp.FORBIDDEN_MODULE_PREFIXES[0], "attempted": True}]
            if forbidden
            else []
        ),
        action_source="E3AgentPolicy.next_move",
    )


def _artifact(tmp_path: Path, *, withheld_level: int = 0, control_level: int = 0) -> dict[str, object]:
    """Build a complete paired artifact through the public aggregation functions."""

    registry = _registry()
    registry_hash = exp.sha256_json(registry)
    artifact = exp.initial_artifact(
        run_date="20260907",
        output_path=tmp_path / "artifact.json",
        raw_paths={
            "adapter_withheld": tmp_path / "withheld",
            "adapter_visible_control": tmp_path / "control",
        },
    )
    artifact = exp.apply_selection(
        artifact,
        exp.select_rank_one(registry),
        registry_hash=registry_hash,
    )
    model = _model(tmp_path)
    artifact["MODEL_SPECS_rows"] = [model]
    artifact["resolved_model_paths"] = [model["resolved_model_path"]]
    artifact["resolved_model_hashes"] = [model["model_hash"]]
    artifact["preconditions_checked"] = [exp.gate_row("all_runtime_prerequisites", True, True)]
    return exp.finalize_artifact(
        artifact,
        arm_results=[
            exp.arm_result("adapter_withheld", [_attempt(tmp_path, "adapter_withheld", level=withheld_level)]),
            exp.arm_result(
                "adapter_visible_control",
                [_attempt(tmp_path, "adapter_visible_control", level=control_level)],
            ),
        ],
        registry_hash_after=registry_hash,
        duration_s=12.0,
    )


def test_req_report_7123_spec_precedes_implementation() -> None:
    """REQ-REPORT-7123 owns all required fields and the named scenarios."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7123") :]
    for name in ("INLINE-BLOCK", "ZERO-GPU", "ISOLATION", "EXECUTION", "PAIR", "ZERO-LEVEL", "ADVERSARIAL", "NONCLAIM"):
        assert f"SCENARIO-REPORT-7123-{name}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


@pytest.mark.parametrize(("raw", "expected"), [(0, 0), ("0", 0), (" 0 % ", 0), (None, None), ("N/A", None)])
def test_scenario_report_7123_zero_gpu_is_not_missing(raw: object, expected: int | None) -> None:
    """SCENARIO-REPORT-7123-ZERO-GPU preserves numeric idle utilization."""

    assert exp.parse_gpu_utilization(raw) == expected


def test_scenario_report_7123_inline_block_missing_cache_file(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-INLINE-BLOCK stops on a missing cached headline file."""

    with pytest.raises(exp.PrerequisiteFailure) as error:
        exp.resolve_headline_model(lambda **_kwargs: None)
    assert error.value.check == "cached_headline_model"
    assert error.value.expected_value == exp.HEADLINE_REPO_ID
    assert error.value.observed_value is None

    with pytest.raises(exp.PrerequisiteFailure):
        exp.resolve_headline_model(
            lambda **_kwargs: [{"hf_id": exp.HEADLINE_REPO_ID, "model_path": str(tmp_path / "missing.gguf")}]
        )


def test_scenario_report_7123_inline_block_releases_first_lease_on_second_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-INLINE-BLOCK fails closed when either GPU lease is unavailable."""

    class Lease:
        def __init__(self, index: int) -> None:
            self.index = index
            self.closed = False

        def owner_receipt(self) -> dict[str, object]:
            return {"lease_id": f"lease-{self.index}", "device_uuid": f"GPU-{self.index}"}

        def close(self) -> None:
            self.closed = True

    made: list[Lease] = []

    def factory(**_kwargs: object) -> Lease:
        if made:
            raise RuntimeError("busy")
        lease = Lease(0)
        made.append(lease)
        return lease

    with pytest.raises(exp.PrerequisiteFailure) as error:
        exp.acquire_gpu_leases(
            [{"gpu_uuid": "GPU-0", "memory_used_mb": 0}, {"gpu_uuid": "GPU-1", "memory_used_mb": 0}],
            _model(tmp_path),
            runtime_dir=tmp_path / "leases",
            lease_factory=factory,
        )
    assert error.value.check == "both_gpu_leases"
    assert made[0].closed is True


def test_scenario_report_7123_isolation_blocks_import_and_source_read(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-ISOLATION denies adapter imports and game-source reads."""

    monitor = exp.IsolationMonitor(Path.cwd())
    with pytest.raises(exp.IsolationViolation):
        monitor.check_import("carnot.agentic.arc_game_adapters")
    with pytest.raises(exp.IsolationViolation):
        monitor.check_read(tmp_path / "environment_files" / "r11l" / "game.py")
    assert [row["kind"] for row in monitor.receipts()] == ["import", "read"]


def test_scenario_report_7123_execution_rejects_invalid_or_unexecuted_output(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-EXECUTION requires the exact E3 proposal to reach the environment."""

    assert exp.parse_action_output('{"action":6,"data":{"x":3,"y":4}}') == {"action": 6, "data": {"x": 3, "y": 4}}
    for raw in ("move left", '{"action":9,"data":null}', '{"action":1,"data":null} trailing'):
        assert exp.parse_action_output(raw) is None

    artifact = _artifact(tmp_path)
    artifact["action_rows"][0]["executed_action"] = {"action": 2, "data": None}
    artifact["reproducibility_checksum"] = exp.payload_checksum(artifact)
    assert "executed_action_mismatch" in exp.validate_artifact(artifact)


def test_scenario_report_7123_adversarial_rejects_rank_drift_and_registry_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-ADVERSARIAL binds rank one and immutable registry bytes."""

    artifact = _artifact(tmp_path)
    drifted = _registry(first="changed")
    assert "eligibility_rank_drift" in exp.validate_artifact(artifact, registry_document=drifted)

    mutated = deepcopy(artifact)
    mutated["arc_registry_hash_after"] = "sha256:" + "f" * 64
    mutated["reproducibility_checksum"] = exp.payload_checksum(mutated)
    assert "registry_mutation" in exp.validate_artifact(mutated)


def test_scenario_report_7123_adversarial_rejects_time_provenance_and_forbidden_access(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-ADVERSARIAL rejects overrun, lost provenance, and access attempts."""

    overrun = _artifact(tmp_path / "overrun")
    overrun["duration_s"] = exp.MEASURED_CAP_S + 0.01
    overrun["reproducibility_checksum"] = exp.payload_checksum(overrun)
    assert "measured_time_cap_exceeded" in exp.validate_artifact(overrun)

    provenance = _artifact(tmp_path / "provenance")
    provenance["solve_provenance"] = None
    provenance["reproducibility_checksum"] = exp.payload_checksum(provenance)
    assert "solve_provenance_invalid" in exp.validate_artifact(provenance)

    forbidden = _artifact(tmp_path / "forbidden")
    forbidden["forbidden_access_rows"] = [{"attempted": True, "kind": "read", "target": "adapter"}]
    forbidden["reproducibility_checksum"] = exp.payload_checksum(forbidden)
    assert "forbidden_access_detected" in exp.validate_artifact(forbidden)


def test_scenario_report_7123_adversarial_rejects_raw_trace_change(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-ADVERSARIAL detects evidence changed after aggregation."""

    artifact = _artifact(tmp_path)
    Path(artifact["raw_trace_receipts"][0]["path"]).write_text("changed", encoding="utf-8")
    assert "raw_trace_hash_mismatch" in exp.validate_artifact(artifact, verify_raw_traces=True)


def test_scenario_report_7123_zero_level_is_complete_terminal_null(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-ZERO-LEVEL accepts zero reached levels as a complete result."""

    artifact = _artifact(tmp_path)
    assert artifact["arc_loo_shard_complete_score"] == 1
    assert artifact["adapter_withheld_any_level_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert str(artifact["honest_verdict"]).startswith("complete_null:")
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert exp.validate_artifact(artifact, registry_document=_registry(), verify_raw_traces=True) == []


def test_scenario_report_7123_nonclaim_survives_positive_measurement(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-NONCLAIM never converts a measured level into solve credit."""

    artifact = _artifact(tmp_path, withheld_level=1, control_level=1)
    assert artifact["adapter_withheld_any_level_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["offline_reproduced"] is False
    assert artifact["headline_solve_eligible"] is False
    assert artifact["arc_registry_delta"] == 0
    assert artifact["verifier_is_oracle"] is False


def test_scenario_report_7123_pair_rejects_mismatched_limits_and_processes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7123-PAIR requires matching budgets and fresh arm processes."""

    artifact = _artifact(tmp_path)
    artifact["adapter_visible_control_rows"][0]["request_limit"] -= 1
    artifact["process_rows"][1]["worker_pid"] = artifact["process_rows"][0]["worker_pid"]
    artifact["reproducibility_checksum"] = exp.payload_checksum(artifact)
    errors = set(exp.validate_artifact(artifact))
    assert {"arm_budget_mismatch", "arm_process_not_isolated"} <= errors

