"""Tests for the Exp6762 procedural-versus-trajectory memory comparison.

Spec refs: REQ-CL-6762, SCENARIO-CL-6762-CHRONOLOGY,
SCENARIO-CL-6762-READ-ONLY, SCENARIO-CL-6762-CAPACITY,
SCENARIO-CL-6762-RETRIEVAL-ACTION, SCENARIO-CL-6762-TRANSACTIONS,
SCENARIO-CL-6762-REDUCERS, SCENARIO-CL-6762-RESTART,
SCENARIO-CL-6762-BLOCKED, REQ-REPORT-6762,
SCENARIO-REPORT-6762-ATOMIC, and SCENARIO-REPORT-6762-BLOCKED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import runpy

import pytest

from carnot import experiment_6762_procedural_vs_trace_csl_ab as mod


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / mod.FIXTURE_RELATIVE_PATH
SPEC = REPO / mod.SPEC_RELATIVE_PATH
REPORT_SPEC = REPO / mod.REPORT_SPEC_RELATIVE_PATH


class FakeRunner:
    """Return deterministic actions that expose representation effects."""

    def __init__(self, spec: dict[str, object]) -> None:
        self.spec = spec
        self.loaded = False

    def load(self) -> dict[str, object]:
        self.loaded = True
        return {
            "model_id": self.spec["hf_id"],
            "model_path": self.spec["model_path"],
            "loaded": True,
            "cuda_offload": True,
            "process_id": 6762,
            "gpu_before": [{"index": 0, "free_mib": 24000}],
            "gpu_after": [{"index": 0, "free_mib": 16000}],
            "load_duration_s": 0.25,
        }

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> dict[str, object]:
        event_id = re.search(r"EVENT=([a-z]\d+)", prompt).group(1)  # type: ignore[union-attr]
        arm = re.search(r"ARM=([a-z_]+)", prompt).group(1)  # type: ignore[union-attr]
        candidate = int(re.search(r"CANDIDATE=(\d+)", prompt).group(1))  # type: ignore[union-attr]
        family = re.search(r"FAMILY=([a-z_]+)", prompt).group(1)  # type: ignore[union-attr]
        expected = mod.ACTION_BY_FAMILY[family]
        memory_match = re.search(r'"memory_id":"([^"]+)"', prompt)
        difficulty = re.search(r"DIFFICULTY=([a-z]+)", prompt).group(1)  # type: ignore[union-attr]
        action = expected
        memory_id = "none"
        if difficulty == "hard" and arm == "no_memory":
            action = "no_action"
        elif difficulty == "hard" and arm == "detailed_trajectory" and memory_match:
            action = "no_action"
            memory_id = memory_match.group(1)
        elif memory_match:
            memory_id = memory_match.group(1)
        if candidate == 1 and arm == "detailed_trajectory":
            action = "no_action"
        text = json.dumps({"action": action, "memory_id": memory_id}, separators=(",", ":"))
        return {
            "text": text,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(text.split()),
            "latency_s": 0.01,
            "seed": seed,
            "max_tokens": max_tokens,
        }

    def close(self) -> dict[str, object]:
        self.loaded = False
        return {
            "model_id": self.spec["hf_id"],
            "closed": True,
            "vram_released": True,
            "gpu_after_close": [{"index": 0, "free_mib": 24000}],
        }


class FakeLease:
    """Provide owner-bound lease evidence without touching a real GPU lock."""

    def __init__(self) -> None:
        self.closed = False

    def owner_receipts(self) -> list[dict[str, object]]:
        return [{"task_id": mod.EXPERIMENT_ID, "owner_bound": True, "device_uuid": "GPU-test"}]

    def heartbeat(self) -> None:
        assert self.closed is False

    def start_inference(self, resident_mib: int) -> None:
        assert resident_mib >= 0

    def complete(self) -> dict[str, object]:
        self.closed = True
        return {"released": True, "terminal": True, "owner_bound": True}

    def block(self) -> dict[str, object]:
        self.closed = True
        return {"released": True, "terminal": True, "owner_bound": True}


@pytest.fixture
def fake_model_specs(tmp_path: Path) -> list[dict[str, object]]:
    specs = []
    for index, base in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"model-{index}".encode())
        specs.append({**base, "model_path": str(path)})
    return specs


@pytest.fixture
def completed_artifact(
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
) -> dict[str, object]:
    return mod.run_experiment(
        fixture_path=FIXTURE,
        state_root=tmp_path / "state",
        model_specs=fake_model_specs,
        runner_factory=FakeRunner,
        lease_factory=lambda *_args, **_kwargs: FakeLease(),
        precondition_overrides=mod.TEST_PRECONDITION_OVERRIDES,
        duration_s=2.0,
    )


def test_req_cl_6762_specs_and_manifest_freeze_all_controls() -> None:
    """REQ-CL-6762: specs and the manifest freeze the complete comparison."""

    learning = SPEC.read_text(encoding="utf-8").split("## REQ-CL-6762", 1)[1]
    reporting = REPORT_SPEC.read_text(encoding="utf-8").split("### REQ-REPORT-6762", 1)[1]
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest = mod.freeze_manifest(fixture)

    assert "SCENARIO-CL-6762-CHRONOLOGY" in learning
    assert "SCENARIO-CL-6762-RETRIEVAL-ACTION" in learning
    assert "SCENARIO-CL-6762-BLOCKED" in learning
    assert "SCENARIO-REPORT-6762-ATOMIC" in reporting
    assert "SCENARIO-REPORT-6762-BLOCKED" in reporting
    assert manifest["frozen_before_first_model_load"] is True
    assert manifest["arms"] == list(mod.ARMS)
    assert manifest["model_specs"] == mod.MODEL_SPECS
    assert len(manifest["orders"]) == 6
    assert len({row["order_hash"] for row in manifest["orders"]}) == 6
    assert manifest["candidate_count_k"] == 2
    assert manifest["max_tokens_per_candidate"] == mod.MAX_TOKENS
    assert (
        manifest["memory_contract"]["detailed_trajectory"]
        == manifest["memory_contract"]["procedural_constraint"]
    )
    assert manifest["arm_rotations"]
    assert set(manifest["retention_anchors"]) == {"t01", "t02"}


def test_scenario_cl_6762_blocked_artifact_is_complete_and_row_free(
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
) -> None:
    """SCENARIO-CL-6762-BLOCKED: a failed gate has no fallback rows."""

    artifact = mod.run_experiment(
        fixture_path=FIXTURE,
        state_root=tmp_path / "blocked",
        model_specs=fake_model_specs,
        runner_factory=FakeRunner,
        lease_factory=lambda *_args, **_kwargs: FakeLease(),
        precondition_overrides={
            **mod.TEST_PRECONDITION_OVERRIDES,
            "procedural_memory_stream_ready": False,
            "task_owned_lease": False,
        },
        duration_s=0.5,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_blocked_procedural_csl_ab"
    assert artifact["honest_verdict"].startswith("complete_blocked_procedural_csl_ab:")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["live_model_invoked"] is False
    assert artifact["prospective_csl_completed"] is False
    assert artifact["gate_check_summary"]["failed_checks"] == [
        "procedural_memory_stream_ready",
        "task_owned_lease",
    ]
    assert mod.validate_artifact(artifact) == []


def test_scenarios_cl_6762_rows_are_complete_isolated_and_read_only(
    completed_artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-6762-CHRONOLOGY/READ-ONLY/CAPACITY: all rows are isolated."""

    rows = completed_artifact["rows"]
    assert len(rows) == mod.PLANNED_ROW_COUNT == 1080
    assert len({row["row_key"] for row in rows}) == len(rows)
    assert completed_artifact["prospective_csl_completed"] is True
    assert completed_artifact["live_model_invoked"] is True
    assert len(completed_artifact["gpu_receipts"]) == 2
    assert all(row["closed"] is True for row in completed_artifact["teardown_receipts"])
    assert completed_artifact["model_specific_answer_traces_shared"] is False
    assert all(row["snapshot_immutable"] is True for row in rows)
    assert all(row["active_episode_write_count"] == 0 for row in rows)
    assert all(row["current_evidence_visible"] is False for row in rows)
    assert all(row["future_evidence_visible"] is False for row in rows)
    assert all(len(row["candidates"]) == mod.CANDIDATE_COUNT_K for row in rows)
    assert all(row["candidate_seeds"] == row["paired_no_memory_candidate_seeds"] for row in rows)

    no_memory = [row for row in rows if row["arm"] == "no_memory"]
    assert all(row["retrieved_ids"] == [] for row in no_memory)
    assert all(row["retrieval_scores"] == [] for row in no_memory)
    assert all(row["memory_read_count"] == 0 for row in no_memory)
    assert all(row["memory_write_count"] == 0 for row in no_memory)
    assert all(row["commit_status"] == "not_applicable" for row in no_memory)

    manifest = completed_artifact["frozen_manifest"]
    for model in mod.MODEL_SPECS:
        for order in manifest["orders"]:
            for arm in mod.ARMS:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["model_id"] == model["hf_id"]
                        and row["order_id"] == order["order_id"]
                        and row["arm"] == arm
                    ),
                    key=lambda row: row["order_position"],
                )
                assert [row["event_id"] for row in selected] == order["event_ids"]
                assert all(
                    row["visible_event_ids"] == order["event_ids"][:position]
                    for position, row in enumerate(selected)
                )


def test_scenarios_cl_6762_retrieval_action_and_transactions_are_observed(
    completed_artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-6762-RETRIEVAL-ACTION/TRANSACTIONS: behavior and writes occur."""

    rows = completed_artifact["rows"]
    memory_rows = [row for row in rows if row["arm"] != "no_memory"]
    retrieved = [row for row in memory_rows if row["actual_retrieval"] is True]
    influenced = [row for row in memory_rows if row["action_influenced"] is True]

    assert retrieved
    assert influenced
    assert all(len(row["retrieved_ids"]) == len(row["retrieval_scores"]) for row in retrieved)
    assert all(row["context_bytes"] > 0 and row["context_tokens"] > 0 for row in retrieved)
    assert all(row["before_action_fingerprint"] for row in rows)
    assert all(row["after_action_fingerprint"] for row in rows)
    assert any(row["memory_cited"] is True for row in retrieved)
    assert any(row["operational_memory_use"] is True for row in retrieved)
    assert completed_artifact["commits_by_arm"] == {
        "no_memory": 0,
        "detailed_trajectory": 144,
        "procedural_constraint": 144,
    }
    assert completed_artifact["rejects_by_arm"] == {
        "no_memory": 0,
        "detailed_trajectory": 144,
        "procedural_constraint": 144,
    }
    assert all(row["committed"] is False for row in completed_artifact["poison_receipts"])
    assert all(row["restart_passed"] is True for row in memory_rows)
    assert all(row["rollback_passed"] is True for row in memory_rows)


def test_scenario_cl_6762_reducers_and_positive_gate_are_row_derived(
    completed_artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-6762-REDUCERS: cold rows reproduce every required aggregate."""

    reduced = mod.reduce_rows(completed_artifact["rows"])
    for field in mod.ROW_DERIVED_FIELDS:
        assert completed_artifact[field] == reduced[field]
    assert completed_artifact["cold_aggregate_recomputation_passed"] is True
    assert completed_artifact["procedural_over_no_memory_order_lcb"] > 0.0
    assert completed_artifact["procedural_over_trace_order_lcb"] > 0.0
    assert completed_artifact["positive_credit_checks"]["nonzero_commits_and_rejects"] is True
    assert completed_artifact["positive_credit_checks"]["zero_poison"] is True
    assert completed_artifact["verdict_class"] == "positive"
    assert completed_artifact["honest_verdict"].startswith("complete_positive:")


def test_scenario_cl_6762_restart_store_retrieval_and_read_only(tmp_path: Path) -> None:
    """SCENARIO-CL-6762-RESTART: state reopens, retrieves, and rolls back exactly."""

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    event = next(
        row
        for row in fixture["stream_manifest"]["events"]
        if row["expected_transaction_class"] == "accept"
    )
    pair = mod.source_representation_pair(fixture, event["event_id"])
    store = mod.ArmMemoryStore(tmp_path / "store", "procedural_constraint")
    snapshot = store.begin_episode(event["event_id"], [])

    with pytest.raises(mod.ReadOnlyEpisodeError, match="active episode is read-only"):
        store.transact(event, pair, 0, "order_test")

    assert store.state_hash() == snapshot["state_hash"]
    store.end_episode()
    receipt = store.transact(event, pair, 0, "order_test")
    assert receipt["committed"] is True
    assert receipt["restart_receipt"]["hash_match"] is True
    retrieval = store.retrieve(event, top_k=3, position=1)
    assert retrieval[0]["memory_id"] == event["event_id"]
    rollback = store.rollback(receipt)
    assert rollback["byte_identical"] is True
    store.reapply(receipt)
    reopened = mod.ArmMemoryStore(tmp_path / "store", "procedural_constraint")
    assert reopened.state_hash() == receipt["state_hash"]


def test_helpers_cover_resolution_expiry_context_and_candidate_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6762: bounded helper branches retain fail-closed behavior."""

    snapshot_model = tmp_path / "models--owner--repo" / "snapshots" / "revision-1" / "model.gguf"
    snapshot_model.parent.mkdir(parents=True)
    snapshot_model.write_bytes(b"gguf")
    receipt = mod.model_file_receipt({**mod.MODEL_SPECS[0], "model_path": str(snapshot_model)})
    assert receipt["revision"] == "revision-1"

    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args, **_kwargs: str(snapshot_model))
    assert [row["model_path"] for row in mod.resolve_model_specs()] == [
        str(snapshot_model),
        str(snapshot_model),
    ]
    with pytest.raises(ValueError, match="memory arm required"):
        mod.ArmMemoryStore(tmp_path / "invalid", "no_memory")

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    event = next(
        row
        for row in fixture["stream_manifest"]["events"]
        if row["expected_transaction_class"] == "accept"
    )
    expiring = {**event, "ttl_events": 0}
    pair = mod.source_representation_pair(fixture, event["event_id"])
    store = mod.ArmMemoryStore(tmp_path / "expiry", "procedural_constraint")
    expiry_receipt = store.transact(expiring, pair, 0, "expiry")
    assert expiry_receipt["committed"] is True
    assert store.retrieve(event, top_k=3, position=1) == []

    class HugeTokenRunner(FakeRunner):
        def count_tokens(self, text: str) -> int:
            return 999 if text != "[]" else 0

    runner = HugeTokenRunner({**mod.MODEL_SPECS[0], "model_path": str(snapshot_model)})
    context = mod.render_context(
        fixture,
        "procedural_constraint",
        [{"memory_id": event["event_id"], "score": 3.0}],
        runner,
    )
    assert context == ("[]", 0, 0)

    malformed = mod.evaluate_candidate({"text": "not-json", "seed": 1, "latency_s": 0.0}, event)
    failed = mod.failed_candidate(2, RuntimeError("generation failed"))
    assert malformed["abstained"] is True
    assert failed["failed"] is True
    assert "generation failed" in failed["error"]
    assert mod.order_level_lcb([]) == 0.0
    assert mod.order_level_lcb([0.25]) == 0.25


def test_blocked_live_lease_is_released(
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
) -> None:
    """SCENARIO-CL-6762-BLOCKED: an acquired but rejected lease closes."""

    lease = FakeLease()
    artifact = mod.run_experiment(
        fixture_path=FIXTURE,
        state_root=tmp_path / "lease-block",
        model_specs=fake_model_specs,
        runner_factory=FakeRunner,
        lease_factory=lambda *_args, **_kwargs: lease,
        precondition_overrides={
            **mod.TEST_PRECONDITION_OVERRIDES,
            "task_owned_lease": False,
        },
        duration_s=0.1,
    )
    assert artifact["verdict_class"] == "blocked"
    assert lease.closed is True


def test_internal_post_run_validation_failure_is_terminal(
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6762: a final internal validation error cannot publish."""

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced-invalid"])
    with pytest.raises(ValueError, match="forced-invalid"):
        mod.run_experiment(
            fixture_path=FIXTURE,
            state_root=tmp_path / "forced-invalid",
            model_specs=fake_model_specs,
            runner_factory=FakeRunner,
            lease_factory=lambda *_args, **_kwargs: FakeLease(),
            precondition_overrides=mod.TEST_PRECONDITION_OVERRIDES,
            duration_s=0.1,
        )


def test_validation_rejects_row_aggregate_completion_and_checksum_tampering(
    completed_artifact: dict[str, object],
) -> None:
    """REQ-REPORT-6762: validation fails closed on every acceptance surface."""

    missing = deepcopy(completed_artifact)
    missing.pop("rows")
    assert "required field set mismatch" in mod.validate_artifact(missing)

    aggregate = deepcopy(completed_artifact)
    aggregate["commits_by_arm"]["procedural_constraint"] += 1
    aggregate["reproducibility_checksum"] = mod.reproducibility_checksum(aggregate)
    assert "row-derived metrics mismatch" in mod.validate_artifact(aggregate)

    chronology = deepcopy(completed_artifact)
    chronology["rows"][0]["future_evidence_visible"] = True
    chronology["reproducibility_checksum"] = mod.reproducibility_checksum(chronology)
    assert "completion gates mismatch" in mod.validate_artifact(chronology)

    checksum = deepcopy(completed_artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(checksum)

    substrate = deepcopy(completed_artifact)
    substrate["inference_substrate"] = "remote"
    principles = deepcopy(completed_artifact)
    principles["field_principles"].pop("rows")
    oracle = deepcopy(completed_artifact)
    oracle["verifier_is_oracle"] = True
    shared = deepcopy(completed_artifact)
    shared["model_specific_answer_traces_shared"] = True
    weights = deepcopy(completed_artifact)
    weights["model_weights_mutated"] = True
    no_rows = deepcopy(completed_artifact)
    no_rows["rows"] = []
    blocked_rows = deepcopy(completed_artifact)
    blocked_rows["verdict_class"] = "blocked"
    assert "inference_substrate mismatch" in mod.validate_artifact(substrate)
    assert "field_principles coverage mismatch" in mod.validate_artifact(principles)
    assert "verifier_is_oracle must be false" in mod.validate_artifact(oracle)
    assert "model answer traces shared" in mod.validate_artifact(shared)
    assert "model weights mutated" in mod.validate_artifact(weights)
    assert "completed artifact has no rows" in mod.validate_artifact(no_rows)
    assert "blocked artifact contains rows" in mod.validate_artifact(blocked_rows)
    assert mod._chronology_passes([], completed_artifact["frozen_manifest"]) is False


def test_scenario_report_6762_atomic_writer_and_cli_validation(
    tmp_path: Path,
    completed_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6762-ATOMIC: writer and both CLI paths validate JSON."""

    result = tmp_path / "result.json"
    receipt = mod.write_artifact(result, completed_artifact)
    assert receipt["atomic_rename"] is True
    assert receipt["sha256"] == mod.sha256_file(result)
    assert mod.main(["--validate", "--result-path", str(result)]) == 0

    generated = tmp_path / "generated.json"
    monkeypatch.setattr(mod, "run_experiment", lambda **_kwargs: completed_artifact)
    assert mod.main(["--date", mod.RUN_DATE, "--result-path", str(generated)]) == 0
    assert json.loads(generated.read_text(encoding="utf-8"))["schema"] == mod.SCHEMA
    with pytest.raises(ValueError, match="planning date"):
        mod.main(["--date", "19000101", "--result-path", str(generated)])

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="required field set mismatch"):
        mod.main(["--validate", "--result-path", str(invalid)])
    with pytest.raises(ValueError, match="required field set mismatch"):
        mod.write_artifact(tmp_path / "must-not-write.json", {"schema": "bad"})

    wrapper = REPO / mod.SCRIPT_RELATIVE_PATH
    monkeypatch.setattr(mod, "main", lambda _argv=None: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert raised.value.code == 0
