"""Tests for the Exp6749 prospective transactional-memory comparison.

Spec refs: REQ-CL-6749, SCENARIO-CL-6749-PROSPECTIVE-ORDER,
SCENARIO-CL-6749-SNAPSHOT, SCENARIO-CL-6749-EXACT-ADMISSION,
SCENARIO-CL-6749-ARM-ISOLATION, SCENARIO-CL-6749-SUPPORT,
SCENARIO-CL-6749-NO-WEIGHT-WRITES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_6749_prospective_support_preserving_csl_ab as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "results/experiment_6748_transactional_constraint_memory_fixture.json"


class FakeRunner:
    """Return deterministic candidate shapes while preserving the runner contract."""

    expected = {
        "e01": "clamp_upper_bound",
        "e02": "normalize_even_parity",
        "e03": "require_schema_field",
        "e04": "none",
        "e05": "clamp_upper_bound",
        "e06": "clamp_upper_bound",
        "e07": "clamp_lower_bound",
        "e08": "normalize_even_parity",
        "e09": "require_schema_field",
        "e10": "clamp_upper_bound",
        "e11": "normalize_modulo",
        "e12": "reject_unsafe",
    }

    def __init__(self, spec: dict[str, object]) -> None:
        self.spec = spec
        self.closed = False

    def load(self) -> dict[str, object]:
        return {
            "model_id": self.spec["hf_id"],
            "model_path": self.spec["model_path"],
            "loaded": True,
            "cuda_offload": True,
            "gpu_before": [{"index": 0, "free_mib": 24000}],
            "gpu_after": [{"index": 0, "free_mib": 18000}],
            "load_duration_s": 0.25,
        }

    def generate(self, prompt: str, *, seed: int, max_tokens: int) -> dict[str, object]:
        event_id = re.search(r"EVENT=(e\d+)", prompt).group(1)  # type: ignore[union-attr]
        arm = re.search(r"ARM=([a-z_]+)", prompt).group(1)  # type: ignore[union-attr]
        candidate_index = int(re.search(r"CANDIDATE=(\d+)", prompt).group(1))  # type: ignore[union-attr]
        expected = self.expected[event_id]
        reusable_bootstrap = event_id in {"e01", "e02", "e03", "e04", "e11", "e12"}
        answer = expected if arm == "transactional_memory" or reusable_bootstrap else "none"
        text = answer if candidate_index == 0 else f"Answer: {answer}"
        return {
            "text": text,
            "prompt_tokens": 20,
            "completion_tokens": 2,
            "latency_s": 0.01,
            "seed": seed,
            "max_tokens": max_tokens,
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_model_specs(tmp_path: Path) -> list[dict[str, object]]:
    specs = []
    for index, base in enumerate(mod.MODEL_SPECS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"read-only-test-weight")
        specs.append({**base, "model_path": str(path)})
    return specs


@pytest.fixture
def completed_artifact(
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
) -> dict[str, object]:
    return mod.run_experiment(
        fixture_path=FIXTURE_PATH,
        state_root=tmp_path / "state",
        model_specs=fake_model_specs,
        runner_factory=FakeRunner,
        precondition_overrides=mod.TEST_PRECONDITION_OVERRIDES,
        duration_s=1.0,
    )


def test_req_cl_6749_freezes_protocol_before_rows() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    protocol = mod.freeze_protocol(fixture)

    assert protocol["frozen_before_first_episode"] is True
    assert protocol["candidate_count_k"] == 2
    assert len(protocol["orders"]) == 6
    assert protocol["order_hashes"] == [
        row["order_hash"] for row in fixture["stream_manifest"]["orders"]
    ]
    assert protocol["retention_anchors"] == ["e05"]
    assert protocol["target_family_future_evidence_allowed"] is False
    assert set(protocol["support_definitions"]) == {
        "pass_at_1",
        "best_at_k",
        "effective_rewardable_support",
        "joint_correct_constraint_support",
    }


def test_scenario_cl_6749_blocked_gate_has_complete_schema(
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["transaction_memory_ready"] = False
    blocked_fixture = tmp_path / "blocked-fixture.json"
    blocked_fixture.write_text(json.dumps(fixture), encoding="utf-8")

    artifact = mod.run_experiment(
        fixture_path=blocked_fixture,
        state_root=tmp_path / "state",
        model_specs=fake_model_specs,
        runner_factory=FakeRunner,
        precondition_overrides=mod.TEST_PRECONDITION_OVERRIDES,
        duration_s=0.5,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_blocked_prospective_csl"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["prospective_csl_completed"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failures"] == [
        {"check": "exp6748_transaction_memory_ready", "expected": True, "observed": False}
    ]
    assert mod.validate_artifact(artifact) == []


def test_scenarios_order_snapshot_admission_isolation_and_weights(
    completed_artifact: dict[str, object],
) -> None:
    rows = completed_artifact["rows"]
    assert len(rows) == mod.PLANNED_ROW_COUNT == 288
    assert len({row["row_key"] for row in rows}) == 288
    assert completed_artifact["prospective_csl_completed"] is True
    assert completed_artifact["live_model_invoked"] is True
    assert completed_artifact["model_weights_mutated"] is False

    no_memory = [row for row in rows if row["arm"] == "no_memory"]
    transactional = [row for row in rows if row["arm"] == "transactional_memory"]
    assert all(row["memory_read_count"] == 0 for row in no_memory)
    assert all(row["memory_write_count"] == 0 for row in no_memory)
    assert all(row["commit_status"] == "not_applicable" for row in no_memory)
    assert all(row["snapshot_immutable"] is True for row in transactional)
    assert all(row["active_episode_write_count"] == 0 for row in transactional)
    assert all(len(row["candidates"]) == mod.CANDIDATE_COUNT_K for row in rows)

    qwen_transactional = [
        row
        for row in transactional
        if row["model_id"] == mod.MODEL_SPECS[0]["hf_id"]
    ]
    admitted = [row for row in qwen_transactional if row["commit_status"] == "committed"]
    rejected = [row for row in qwen_transactional if row["commit_status"] == "rejected"]
    assert admitted
    assert rejected
    assert all(row["exact_result_known_before_commit"] is True for row in admitted + rejected)
    assert all(all(row["admission_checks"].values()) for row in admitted)
    assert all(not all(row["admission_checks"].values()) for row in rejected)

    gemma_rows = [row for row in transactional if row["model_role"] == "held_dense_transfer"]
    assert all(row["memory_source_model_id"] == mod.MODEL_SPECS[0]["hf_id"] for row in gemma_rows)
    assert all(row["target_family_future_evidence_count"] == 0 for row in gemma_rows)

    counts = completed_artifact["commit_reject_rollback_counts"]
    assert counts["totals"]["commits"] > 0
    assert counts["totals"]["rejects"] > 0
    assert counts["totals"]["rollbacks"] == counts["totals"]["commits"]
    assert counts["totals"]["rollback_failures"] == 0


def test_scenario_cl_6749_support_metrics_are_row_derived(
    completed_artifact: dict[str, object],
) -> None:
    reduced = mod.reduce_rows(
        completed_artifact["rows"],
        completed_artifact["commit_reject_rollback_counts"]["by_order"],
    )

    for field in mod.ROW_DERIVED_FIELDS:
        assert completed_artifact[field] == reduced[field]
    assert (
        completed_artifact["best_at_k_by_arm"]["transactional_memory"]["rate"]
        >= completed_artifact["best_at_k_by_arm"]["no_memory"]["rate"]
    )
    assert completed_artifact["effective_rewardable_support_by_arm"][
        "transactional_memory"
    ]["denominator"] == len(completed_artifact["rows"]) // 2 * mod.CANDIDATE_COUNT_K


def test_candidate_evaluation_keeps_failure_abstention_and_format_support() -> None:
    event = {"certified_repair": "clamp_upper_bound"}
    candidates = [
        mod.evaluate_candidate("clamp_upper_bound", event, 1, 2, 3, 0.1),
        mod.evaluate_candidate("Answer: clamp_upper_bound", event, 2, 2, 3, 0.1),
        mod.failed_candidate(3, 2, "timeout"),
        mod.evaluate_candidate("I abstain", event, 4, 2, 0, 0.1),
    ]
    metrics = mod.candidate_metrics(candidates)

    assert metrics == {
        "pass_at_1": 1,
        "best_at_k": 1,
        "effective_rewardable_support": 0.5,
        "joint_correct_constraint_support": 0.25,
    }
    assert candidates[1]["constraint_following"] is False
    assert candidates[2]["failed"] is True
    assert candidates[3]["abstained"] is True


def test_validation_rejects_aggregate_snapshot_and_weight_tampering(
    completed_artifact: dict[str, object],
) -> None:
    missing = deepcopy(completed_artifact)
    missing.pop("rows")
    assert "required field set mismatch" in mod.validate_artifact(missing)

    bad_metric = deepcopy(completed_artifact)
    bad_metric["best_at_k_by_arm"]["no_memory"]["rate"] = 9.0
    assert "row-derived metrics mismatch" in mod.validate_artifact(bad_metric)

    bad_snapshot = deepcopy(completed_artifact)
    bad_snapshot["rows"][1]["snapshot_immutable"] = False
    bad_snapshot["reproducibility_checksum"] = mod.reproducibility_checksum(bad_snapshot)
    assert "completion gates mismatch" in mod.validate_artifact(bad_snapshot)

    bad_weight = deepcopy(completed_artifact)
    bad_weight["model_weights_mutated"] = True
    bad_weight["reproducibility_checksum"] = mod.reproducibility_checksum(bad_weight)
    assert "model weights mutated" in mod.validate_artifact(bad_weight)


def test_live_llama_runner_uses_chat_generation_and_cuda_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"gguf")

    class Llama:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["n_gpu_layers"] == -1

        def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["seed"] == 17
            return {
                "choices": [{"message": {"content": "none"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            }

    class LlamaGrammar:
        @staticmethod
        def from_string(value: str) -> str:
            assert "clamp_upper_bound" in value
            return value

    fake_llama_cpp = SimpleNamespace(
        Llama=Llama,
        LlamaGrammar=LlamaGrammar,
        llama_cpp=SimpleNamespace(llama_supports_gpu_offload=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    snapshots = iter(
        [
            [{"index": 0, "free_mib": 24000}],
            [{"index": 0, "free_mib": 23000}],
        ]
    )
    monkeypatch.setattr(mod, "gpu_snapshot", lambda: next(snapshots))
    runner = mod.LiveLlamaRunner(
        {**mod.MODEL_SPECS[0], "model_path": str(model_path)},
    )

    receipt = runner.load()
    result = runner.generate("prompt", seed=17, max_tokens=5)
    runner.close()

    assert receipt["loaded"] is True
    assert receipt["cuda_offload"] is True
    assert result["text"] == "none"
    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] == 1


def test_write_validate_and_main_validate(
    tmp_path: Path,
    completed_artifact: dict[str, object],
) -> None:
    path = tmp_path / "artifact.json"
    receipt = mod.write_artifact(path, completed_artifact)

    assert receipt["atomic_rename"] is True
    assert receipt["path"] == str(path)
    assert mod.main(["--validate", "--result-path", str(path)]) == 0

    invalid = deepcopy(completed_artifact)
    invalid["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        mod.write_artifact(tmp_path / "bad.json", invalid)


def test_environment_and_defensive_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
    completed_artifact: dict[str, object],
) -> None:
    resolved_paths = iter(["/cache/qwen.gguf", "/cache/gemma.gguf"])
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args, **_kwargs: next(resolved_paths))
    assert [row["model_path"] for row in mod.resolve_model_specs()] == [
        "/cache/qwen.gguf",
        "/cache/gemma.gguf",
    ]

    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setattr(mod, "gpu_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("gpu")))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    preconditions = mod.check_preconditions(fixture, fake_model_specs, tmp_path / "checks")
    assert preconditions["checks"]["llama_cpp_cuda_offload"]["passed"] is False
    assert preconditions["checks"]["gpu_count_at_least_two"]["passed"] is False

    runner = mod.LiveLlamaRunner(fake_model_specs[0])
    with pytest.raises(RuntimeError, match="model is not loaded"):
        runner.generate("prompt", seed=1, max_tokens=1)

    class RaisingRunner(FakeRunner):
        def generate(self, prompt: str, *, seed: int, max_tokens: int) -> dict[str, object]:
            raise TimeoutError("candidate timeout")

    failed = mod._run_candidates(
        RaisingRunner(fake_model_specs[0]),
        fixture["stream_manifest"]["events"][0],
        "no_memory",
        None,
        0,
        0,
        0,
    )
    assert all(row["failed"] is True for row in failed)

    bad_rows = deepcopy(completed_artifact["rows"])
    bad_rows[0]["event_id"] = "wrong"
    assert mod._chronology_passes(bad_rows, completed_artifact["frozen_protocol"]) is False


def test_temporary_state_validation_failures_and_main_run_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_model_specs: list[dict[str, object]],
    completed_artifact: dict[str, object],
) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture["transaction_memory_ready"] = False
    blocked_fixture = tmp_path / "blocked.json"
    blocked_fixture.write_text(json.dumps(fixture), encoding="utf-8")
    blocked = mod.run_experiment(
        fixture_path=blocked_fixture,
        state_root=None,
        model_specs=fake_model_specs,
        runner_factory=FakeRunner,
        precondition_overrides=mod.TEST_PRECONDITION_OVERRIDES,
        duration_s=0.1,
    )
    assert blocked["verdict_class"] == "blocked"

    original_validate = mod.validate_artifact
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced validation failure"])
    with pytest.raises(ValueError, match="forced validation failure"):
        mod.run_experiment(
            fixture_path=FIXTURE_PATH,
            state_root=tmp_path / "forced",
            model_specs=fake_model_specs,
            runner_factory=FakeRunner,
            precondition_overrides=mod.TEST_PRECONDITION_OVERRIDES,
            duration_s=0.1,
        )
    monkeypatch.setattr(mod, "validate_artifact", original_validate)

    invalid = deepcopy(completed_artifact)
    invalid["inference_substrate"] = "wrong"
    invalid["verdict_class"] = "other"
    invalid["field_principles"] = {}
    errors = mod.validate_artifact(invalid)
    assert "inference_substrate mismatch" in errors
    assert "verdict_class outside closed enum" in errors
    assert "field_principles coverage mismatch" in errors

    no_rows = deepcopy(completed_artifact)
    no_rows["rows"] = []
    no_rows["prospective_csl_completed"] = True
    no_rows["reproducibility_checksum"] = mod.reproducibility_checksum(no_rows)
    assert "completed artifact has no rows" in mod.validate_artifact(no_rows)

    invalid_path = tmp_path / "invalid-main.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="inference_substrate mismatch"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    output = tmp_path / "main-run.json"
    monkeypatch.setattr(mod, "run_experiment", lambda **_kwargs: completed_artifact)
    writes: list[Path] = []
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda path, _artifact: writes.append(Path(path)) or {"atomic_rename": True},
    )
    assert mod.main(["--result-path", str(output), "--state-root", str(tmp_path / "main")]) == 0
    assert writes == [output]
