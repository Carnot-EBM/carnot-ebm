"""Tests for the V620 three-family entrance proposal bank.

Spec refs: REQ-INFRA-7080, REQ-VERIFY-7080,
SCENARIO-INFRA-7080-ROSTER, SCENARIO-INFRA-7080-MATCHED,
SCENARIO-INFRA-7080-RAW-FIRST, SCENARIO-INFRA-7080-RESUME,
SCENARIO-INFRA-7080-LEASE-LOSS, SCENARIO-INFRA-7080-BLOCKED,
SCENARIO-INFRA-7080-CLEANUP,
SCENARIO-VERIFY-7080-LABELS, SCENARIO-VERIFY-7080-PREFIX,
SCENARIO-VERIFY-7080-COMPLETE, SCENARIO-VERIFY-7080-BLOCKED, and
SCENARIO-VERIFY-7080-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_7080_v620_three_family_entrance_bank as mod


REPO = Path(__file__).resolve().parents[2]


def _lease_receipts(models: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "model_id": model,
            "phase": phase,
            "lease_id": f"lease:{model}:{phase}",
            "owner_preserved": True,
            "phase_history": [
                "preflight",
                "admitted",
                "loading",
                "resident",
                "inferencing",
                "unloading",
                "validating",
                "terminal_complete",
            ],
            "released": True,
            "lease_lost": False,
        }
        for phase in ("proposal", "forced_prefix")
        for model in models
    ]


def _execution_receipts(models: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "model_id": model,
            "phase": phase,
            "terminal_state": "complete",
            "raw_row_count": 1,
            "offloaded_layers": 48,
            "used_both_gpus": True,
            "cleanup_passed": True,
        }
        for phase in ("proposal", "forced_prefix")
        for model in models
    ]


def _checkpoint_receipts(models: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "model_id": model,
            "phase": phase,
            "sha256": "sha256:" + "1" * 64,
            "manifest_hash": "sha256:" + "2" * 64,
            "row_count": 1,
        }
        for phase in ("proposal", "forced_prefix")
        for model in models
    ]


def _vram_receipts(models: list[str]) -> list[dict[str, Any]]:
    return [
        {"model_id": model, "phase": phase, "passed": True}
        for phase in ("proposal", "forced_prefix")
        for model in models
    ]


def test_req_infra_7080_spec_precedes_implementation() -> None:
    """REQ-INFRA-7080 and REQ-VERIFY-7080 exist before implementation."""

    infra = (REPO / "openspec/capabilities/llm-ebm-inference/spec.md").read_text(
        encoding="utf-8"
    )
    verify = (REPO / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )
    for anchor in (
        "REQ-INFRA-7080",
        "SCENARIO-INFRA-7080-ROSTER",
        "SCENARIO-INFRA-7080-MATCHED",
        "SCENARIO-INFRA-7080-RAW-FIRST",
        "SCENARIO-INFRA-7080-RESUME",
        "SCENARIO-INFRA-7080-LEASE-LOSS",
        "SCENARIO-INFRA-7080-BLOCKED",
        "SCENARIO-INFRA-7080-CLEANUP",
    ):
        assert anchor in infra
    for anchor in (
        "REQ-VERIFY-7080",
        "SCENARIO-VERIFY-7080-LABELS",
        "SCENARIO-VERIFY-7080-PREFIX",
        "SCENARIO-VERIFY-7080-COMPLETE",
        "SCENARIO-VERIFY-7080-BLOCKED",
        "SCENARIO-VERIFY-7080-MUTATION",
    ):
        assert anchor in verify


def _model_files(tmp_path: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for model_id in mod.REQUIRED_MODEL_IDS:
        path = tmp_path / f"{model_id.rsplit('/', 1)[-1]}.Q4_K_M.gguf"
        path.write_bytes(model_id.encode("utf-8"))
        paths[model_id] = str(path)
    return paths


def _resolved_specs(tmp_path: Path) -> list[dict[str, Any]]:
    paths = _model_files(tmp_path)

    def pair(**kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs == {"gpu_indices": (0, 1), "model_indices": (0, 2)}
        return [
            {
                "name": "qwen",
                "hf_id": mod.REQUIRED_MODEL_IDS[0],
                "model_path": paths[mod.REQUIRED_MODEL_IDS[0]],
            },
            {
                "name": "dense",
                "hf_id": mod.REQUIRED_MODEL_IDS[1],
                "model_path": paths[mod.REQUIRED_MODEL_IDS[1]],
            },
        ]

    return mod.resolve_model_specs(
        cached_pair_func=pair,
        resolver=lambda model_id, _quant: paths.get(model_id),
    )


def _visible_units() -> list[dict[str, Any]]:
    return [
        {
            "unit_id": "held-0",
            "numbers": [1, 2, 3, 4, 5, 6],
            "target": 21,
            "formatting_rules": ["one branch"],
        },
        {
            "unit_id": "held-1",
            "numbers": [2, 3, 4, 5, 6, 7],
            "target": 42,
            "formatting_rules": ["one branch"],
        },
    ]


def _raw_row(
    *,
    model_id: str = "model-a",
    unit_id: str = "unit-a",
    seed: int = 11,
    raw_text: str = '{"operand_pair":[2,3],"operator":"*"}',
    arm: str = "proposal",
) -> dict[str, Any]:
    row = {
        "raw_key": f"{model_id}|{arm}|{unit_id}|{seed}",
        "model_id": model_id,
        "unit_id": unit_id,
        "seed": seed,
        "arm": arm,
        "prompt": "prompt bytes",
        "prompt_hash": mod.sha256_text("prompt bytes"),
        "raw_text": raw_text,
        "raw_output_hash": mod.sha256_text(raw_text),
        "token_scores": [{"token": "{", "logprob": -0.1}],
        "timings": {"duration_s": 0.25},
        "terminal_state": "complete",
        "finish_reason": "stop",
        "generation_config": deepcopy(mod.GENERATION_CONFIG),
        "raw_persisted_before_parse": True,
        "parsed_at_write_time": False,
        "labeled_at_write_time": False,
    }
    return row


def test_req_infra_7080_resolves_exact_cached_model_order(tmp_path: Path) -> None:
    """REQ-INFRA-7080 and SCENARIO-INFRA-7080-ROSTER."""

    specs = _resolved_specs(tmp_path)

    assert [row["hf_id"] for row in specs] == list(mod.REQUIRED_MODEL_IDS)
    assert specs[0]["resolution_method"].startswith("cached_sota_pair")
    assert specs[1]["resolution_method"].startswith("cached_sota_pair")
    assert specs[2]["resolution_method"] == "resolve_cached_gguf exact family extension"
    assert mod.model_spec_errors(specs) == []
    assert all(row["tokenizer_source"] == "embedded_gguf" for row in specs)


def test_req_infra_7080_cache_miss_and_identity_mismatch_fail_closed(tmp_path: Path) -> None:
    """REQ-INFRA-7080 rejects missing paths and mismatched GGUF identity."""

    specs = _resolved_specs(tmp_path)
    missing = deepcopy(specs)
    missing[2]["model_path"] = ""
    assert "model_path_missing:unsloth/gemma-4-26B-A4B-it-GGUF" in mod.model_spec_errors(missing)

    identity_rows = [
        {"model_id": row["hf_id"], "identity_matches": True, "tokenizer_source": "embedded_gguf"}
        for row in specs
    ]
    identity_rows[1]["identity_matches"] = False
    assert mod.model_identity_errors(specs, identity_rows) == [
        "model_identity_mismatch:unsloth/gemma-4-31B-it-GGUF"
    ]


def test_req_infra_7080_schedule_is_matched_and_reproducibly_randomized(tmp_path: Path) -> None:
    """REQ-INFRA-7080 and SCENARIO-INFRA-7080-MATCHED."""

    specs = _resolved_specs(tmp_path)
    rows_a = mod.build_proposal_schedule(specs, _visible_units(), ["held-0", "held-1"], seed=7080)
    rows_b = mod.build_proposal_schedule(specs, _visible_units(), ["held-0", "held-1"], seed=7080)

    assert rows_a == rows_b
    assert len(rows_a) == 3 * 2 * len(mod.PROPOSAL_SEEDS)
    assert len({row["execution_index"] for row in rows_a}) == len(rows_a)
    assert {row["seed"] for row in rows_a} == set(mod.PROPOSAL_SEEDS)
    assert mod.matched_schedule_errors(rows_a) == []
    assert len({row["prompt_hash"] for row in rows_a if row["unit_id"] == "held-0"}) == 1
    assert [row["model_id"] for row in rows_a] != sorted(row["model_id"] for row in rows_a)


def test_req_infra_7080_prompt_mismatch_is_rejected(tmp_path: Path) -> None:
    """REQ-INFRA-7080 rejects one model receiving changed prompt bytes."""

    rows = mod.build_proposal_schedule(
        _resolved_specs(tmp_path), _visible_units(), ["held-0", "held-1"], seed=7080
    )
    mutated_unit = rows[0]["unit_id"]
    rows[0]["prompt"] += " leaked label"

    assert "prompt_hash_mismatch" in mod.matched_schedule_errors(rows)
    assert f"cross_model_prompt_mismatch:{mutated_unit}" in mod.matched_schedule_errors(rows)


def test_req_infra_7080_raw_is_durable_before_parser_runs(tmp_path: Path) -> None:
    """REQ-INFRA-7080 and SCENARIO-INFRA-7080-RAW-FIRST."""

    path = tmp_path / "raw.jsonl"
    observed: dict[str, Any] = {}

    def labeler(row: dict[str, Any]) -> dict[str, Any]:
        observed["bytes"] = path.read_bytes()
        observed["row"] = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        return {"legal": True, "raw_key": row["raw_key"]}

    raw = _raw_row()
    labeled = mod.persist_raw_then_label(path, raw, labeler)

    assert observed["row"] == raw
    assert observed["bytes"] == path.read_bytes()
    assert labeled == {"legal": True, "raw_key": raw["raw_key"]}
    assert "legal" not in json.loads(path.read_text(encoding="utf-8"))


def test_req_infra_7080_checkpoint_resume_preserves_rows_and_rejects_drift(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-7080 and SCENARIO-INFRA-7080-RESUME."""

    path = tmp_path / "checkpoint.json"
    first = _raw_row()
    receipt = mod.checkpoint_raw_row(path, "manifest-a", first)
    assert receipt["written"] is True
    assert mod.checkpoint_raw_row(path, "manifest-a", first)["written"] is False
    assert mod.load_checkpoint(path, "manifest-a") == [first]

    with pytest.raises(ValueError, match="manifest_mismatch"):
        mod.load_checkpoint(path, "manifest-b")
    changed = deepcopy(first)
    changed["raw_text"] = "changed"
    with pytest.raises(ValueError, match="row_mismatch"):
        mod.checkpoint_raw_row(path, "manifest-a", changed)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["rows"].append(deepcopy(first))
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        mod.load_checkpoint(path, "manifest-a")


def test_req_verify_7080_proposals_are_labeled_after_raw_capture() -> None:
    """REQ-VERIFY-7080 and SCENARIO-VERIFY-7080-LABELS."""

    entrances = [
        {
            "unit_id": "unit-a",
            "entrance_id": "unit-a:2:3:multiply",
            "operand_pair": [2, 3],
            "operator": "*",
            "reachable": True,
        }
    ]
    raws = [
        _raw_row(seed=1),
        _raw_row(seed=2),
        _raw_row(seed=3, raw_text="not json"),
    ]
    before = deepcopy(raws)
    labels = mod.label_proposal_rows(raws, entrances)

    assert raws == before
    assert labels[0]["legal"] is True and labels[0]["reachable"] is True
    assert labels[0]["duplicate"] is False
    assert labels[1]["duplicate"] is True
    assert labels[2]["parse_failure"] is True
    assert labels[2]["legal"] is False and labels[2]["reachable"] is False


def test_req_verify_7080_forced_prefix_is_reachable_and_initially_unselected() -> None:
    """REQ-VERIFY-7080 and SCENARIO-VERIFY-7080-PREFIX."""

    units = [{"unit_id": "unit-a", "numbers": [2, 3, 4], "target": 20}]
    entrances = [
        {
            "unit_id": "unit-a",
            "entrance_id": "unit-a:2:3:add",
            "operand_pair": [2, 3],
            "operator": "+",
            "left": 2,
            "right": 3,
            "result": 5,
            "reachable": True,
            "continuation_witness": [{"left": 4, "right": 5, "operator": "*", "result": 20}],
        },
        {
            "unit_id": "unit-a",
            "entrance_id": "unit-a:2:3:multiply",
            "operand_pair": [2, 3],
            "operator": "*",
            "left": 2,
            "right": 3,
            "result": 6,
            "reachable": False,
            "continuation_witness": None,
        },
    ]
    proposals = [
        {
            "model_id": "model-a",
            "unit_id": "unit-a",
            "entrance_id": "unit-a:2:3:multiply",
            "parse_failure": False,
        }
    ]

    selected = mod.select_forced_prefixes(
        ["model-a"], units, entrances, proposals, ["unit-a"], count_per_model=1
    )
    assert selected[0]["entrance_id"] == "unit-a:2:3:add"
    assert selected[0]["reachable"] is True
    assert selected[0]["initially_unselected"] is True
    assert "reachable" not in selected[0]["original_proposal_prompt"]

    continuation = '[{"left":4,"right":5,"operator":"*","result":20}]'
    assert mod.continuation_succeeds(units[0], selected[0], continuation) is True
    assert mod.continuation_succeeds(units[0], selected[0], "[]") is False


def test_req_infra_7080_worker_preserves_token_scores_and_backend_failures(tmp_path: Path) -> None:
    """REQ-INFRA-7080 keeps scored and failed attempts as raw terminal evidence."""

    class FakeLlama:
        metadata = {"general.name": "Qwen3.6-35B-A3B"}

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create_completion(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            del prompt, kwargs
            return {
                "choices": [
                    {
                        "text": '{"operand_pair":[2,3],"operator":"*"}',
                        "finish_reason": "stop",
                        "logprobs": {
                            "tokens": ["{"],
                            "token_logprobs": [-0.1],
                            "text_offset": [0],
                            "top_logprobs": [{"{": -0.1}],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 9},
                "timings": {"predicted_ms": 10.0},
            }

        def close(self) -> None:
            return None

    row = mod.worker_generate_one(
        {**_raw_row(), "model_path": str(tmp_path / "qwen.gguf")},
        llama_factory=FakeLlama,
        clock=iter([1_000_000_000, 1_250_000_000]).__next__,
    )
    assert row["terminal_state"] == "complete"
    assert row["token_scores"][0]["logprob"] == -0.1
    assert row["raw_output_hash"] == mod.sha256_text(row["raw_text"])
    assert row["timings"]["duration_s"] == 0.25
    assert row["model_close_called"] is True

    def broken_factory(**_kwargs: Any) -> Any:
        raise RuntimeError("load failed")

    failed = mod.worker_generate_one(
        {**_raw_row(), "model_path": "missing.gguf"},
        llama_factory=broken_factory,
        clock=iter([1, 2]).__next__,
    )
    assert failed["terminal_state"] == "failed"
    assert failed["exception_type"] == "RuntimeError"
    assert failed["raw_output_hash"] == mod.sha256_text("")


def test_req_infra_7080_cleanup_accepts_only_owned_release() -> None:
    """REQ-INFRA-7080 ownership and cleanup scenarios never credit foreign signals."""

    clean = {
        "model_id": mod.REQUIRED_MODEL_IDS[0],
        "process_owned": True,
        "owned_process_absent": True,
        "port_release_confirmed": True,
        "gpu_leases_released": True,
        "vram_release_passed": True,
        "signals_sent": ["SIGTERM"],
        "signaled_pid_owned": True,
    }
    assert mod.cleanup_passes([clean]) is True
    foreign = deepcopy(clean)
    foreign["signaled_pid_owned"] = False
    assert mod.cleanup_passes([foreign]) is False
    unattributed = mod.unattributed_resource_gate("server", {"pid": 99})
    assert unattributed["passed"] is False
    assert unattributed["observed_value"]["signals_sent"] == []


def test_req_verify_7080_completeness_does_not_depend_on_quality() -> None:
    """REQ-VERIFY-7080 and SCENARIO-VERIFY-7080-COMPLETE."""

    models = list(mod.REQUIRED_MODEL_IDS)
    units = ["held-a"]
    proposals = [
        {
            **_raw_row(model_id=model, unit_id="held-a", seed=seed, raw_text="bad"),
            "parse_failure": True,
            "legal": False,
            "reachable": False,
        }
        for model in models
        for seed in mod.PROPOSAL_SEEDS
    ]
    forced = [
        {
            **_raw_row(
                model_id=model,
                unit_id="held-a",
                seed=mod.FORCED_PREFIX_SEED,
                raw_text="bad",
                arm="forced_prefix",
            ),
            "reachable": True,
            "initially_unselected": True,
            "continuation_success": False,
        }
        for model in models
    ]
    evidence = {
        "model_identity_rows": [{"model_id": model, "identity_matches": True} for model in models],
        "model_file_hash_rows": [
            {"model_id": model, "sha256": f"sha256:{'1' * 64}"} for model in models
        ],
        "cleanup_rows": [{"model_id": model, "passed": True} for model in models],
        "cuda_layer_offload_confirmed": True,
        "raw_output_hashes": [row["raw_output_hash"] for row in proposals + forced],
        "gpu_lease_rows": _lease_receipts(models),
        "model_execution_rows": _execution_receipts(models),
        "checkpoint_rows": _checkpoint_receipts(models),
        "vram_release_rows": _vram_receipts(models),
    }
    assert (
        mod.completion_errors(
            proposal_rows=proposals,
            forced_prefix_rows=forced,
            held_unit_ids=units,
            forced_unit_ids=units,
            evidence=evidence,
        )
        == []
    )

    proposals.pop()
    assert "proposal_key_set_mismatch" in mod.completion_errors(
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        held_unit_ids=units,
        forced_unit_ids=units,
        evidence=evidence,
    )
    proposals, forced, _preconditions, _hashes, _phases = _complete_artifact_inputs()
    forced.pop()
    evidence["raw_output_hashes"] = [row["raw_output_hash"] for row in proposals + forced]
    assert "forced_prefix_key_set_mismatch" in mod.completion_errors(
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        held_unit_ids=units,
        forced_unit_ids=units,
        evidence=evidence,
    )


def test_req_verify_7080_blocked_artifact_has_exact_gate_and_full_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-7080 and SCENARIO-VERIFY-7080-BLOCKED."""

    specs = _resolved_specs(tmp_path)
    preconditions = {
        "all_passed": False,
        "checks": [
            mod.gate_row("entrance_fixture_ready_score", 1, 0, False),
            mod.gate_row("unattributed_server_count", 0, [{"pid": 99}], False),
        ],
    }
    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.5,
        model_specs=specs,
        preconditions=preconditions,
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["entrance_proposal_bank_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "entrance_fixture_ready_score"
    assert artifact["gate_check_summary"]["expected_value"] == 1
    assert artifact["gate_check_summary"]["observed_value"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert mod.validate_artifact(artifact) == []


def test_req_verify_7080_mutations_and_wrong_terminal_prefix_are_rejected(tmp_path: Path) -> None:
    """REQ-VERIFY-7080 and SCENARIO-VERIFY-7080-MUTATION."""

    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        model_specs=_resolved_specs(tmp_path),
        preconditions={
            "all_passed": False,
            "checks": [mod.gate_row("cache", True, False, False)],
        },
    )
    changed = deepcopy(artifact)
    changed["prompt_hash"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed)

    bad_prefix = deepcopy(artifact)
    bad_prefix["honest_verdict"] = "complete_wrong"
    bad_prefix["reproducibility_checksum"] = mod.artifact_checksum(bad_prefix)
    assert "blocked_verdict_mismatch" in mod.validate_artifact(bad_prefix)


def test_req_infra_7080_preconditions_cover_source_hash_stop_authority_and_paths(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-7080 preflight records exact upstream and local resource failures."""

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps({"entrance_fixture_ready_score": 0}), encoding="utf-8")
    preflight = mod.collect_preconditions(
        fixture_path=fixture_path,
        expected_fixture_hash="sha256:not-the-file",
        model_specs=_resolved_specs(tmp_path),
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": True, "devices": [], "processes": [{"pid": 99}]},
        llama_probe=lambda: {"importable": True, "gpu_offload": True, "version": "test"},
        lease_probe=lambda _devices: [],
        stop_authority_probe=lambda: {"passed": False, "observed": "candidate found"},
        identity_probe=lambda row: {
            "model_id": row["hf_id"],
            "identity_matches": True,
            "tokenizer_source": "embedded_gguf",
        },
    )
    failed = {row["check"] for row in preflight["checks"] if not row["passed"]}

    assert "entrance_fixture_ready_score" in failed
    assert "entrance_fixture_source_hash" in failed
    assert "owned_idle_rtx_3090_devices" in failed
    assert "unattributed_gpu_processes" in failed
    assert "clean_stop_authority" in failed
    assert preflight["all_passed"] is False


def test_req_infra_7080_model_and_schedule_mutation_matrix(tmp_path: Path) -> None:
    """REQ-INFRA-7080 rejects every roster, budget, seed, and tokenizer mutation."""

    specs = _resolved_specs(tmp_path)
    changed = deepcopy(specs)
    changed.reverse()
    changed[0].update(
        model_path="mmproj.bin",
        gpu_indices=[0],
        tokenizer_source="remote",
        remote_allowed=True,
        headline_eligible=False,
    )
    errors = mod.model_spec_errors(changed)
    assert "model_ids_mismatch" in errors
    assert any(value.startswith("model_path_not_primary_gguf:") for value in errors)
    assert any(value.startswith("gpu_indices_mismatch:") for value in errors)
    assert any(value.startswith("tokenizer_source_mismatch:") for value in errors)
    assert any(value.startswith("headline_policy_mismatch:") for value in errors)

    identities = [
        {"model_id": row["hf_id"], "identity_matches": True, "tokenizer_source": "embedded_gguf"}
        for row in specs
    ]
    identities[0]["tokenizer_source"] = "remote"
    assert mod.model_identity_errors(specs, identities) == [
        f"model_tokenizer_mismatch:{mod.REQUIRED_MODEL_IDS[0]}"
    ]

    schedule = mod.build_proposal_schedule(specs, _visible_units(), ["held-0"], seed=9)
    schedule[0]["generation_config"]["top_p"] = 0.1
    schedule[1]["seed"] = -1
    schedule_errors = mod.matched_schedule_errors(schedule)
    assert "cross_model_budget_mismatch:held-0" in schedule_errors
    assert "proposal_seed_mismatch:held-0" in schedule_errors


def test_req_infra_7080_worker_schedule_resumes_without_regeneration(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7080-RESUME covers the worker's raw-only checkpoint path."""

    class FakeLlama:
        metadata = {"general.name": "test-model"}

        def __init__(self, **_kwargs: Any) -> None:
            self.closed = False

        def create_completion(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "text": '{"operand_pair":[1,2],"operator":"+"}',
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {},
            }

        def close(self) -> None:
            self.closed = True

    model = {
        "hf_id": "test/model",
        "model_path": str(tmp_path / "model.gguf"),
    }
    schedule = mod.build_proposal_schedule([model], _visible_units(), ["held-0"], seed=12)
    payload = {
        "model_id": model["hf_id"],
        "model_path": model["model_path"],
        "phase": "proposal",
        "schedule_rows": schedule,
        "raw_path": str(tmp_path / "raw.jsonl"),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "manifest_hash": "manifest",
    }

    first = mod.worker_run_schedule(payload, llama_factory=FakeLlama)
    raw_before = Path(payload["raw_path"]).read_bytes()
    second = mod.worker_run_schedule(payload, llama_factory=FakeLlama)

    assert first["row_count"] == len(schedule)
    assert len(first["checkpoint_receipts"]) == len(schedule)
    assert second["row_count"] == len(schedule)
    assert second["checkpoint_receipts"] == []
    assert Path(payload["raw_path"]).read_bytes() == raw_before
    assert first["metadata_hash"] == mod.sha256_text(mod.canonical_json(first["metadata"]))


def test_req_infra_7080_forced_worker_budget_uses_embedded_tokenizer(tmp_path: Path) -> None:
    """REQ-INFRA-7080 subtracts forced-prefix tokens from the matched total budget."""

    class FakeLlama:
        metadata: dict[str, Any] = {}

        def __init__(self, **_kwargs: Any) -> None:
            return None

        def tokenize(self, _value: bytes, *, add_bos: bool) -> list[int]:
            assert add_bos is False
            return [1, 2, 3]

        def create_completion(self, _prompt: str, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["max_tokens"] == mod.GENERATION_CONFIG["completion_budget_tokens"] - 3
            return {"choices": [{"text": "[]", "finish_reason": "stop"}], "usage": {}}

        def close(self) -> None:
            return None

    raw = _raw_row(arm="forced_prefix")
    raw.update(model_path=str(tmp_path / "model.gguf"), prefix_json="{}")
    result = mod.worker_generate_one(
        raw,
        llama_factory=FakeLlama,
        clock=iter([1, 2]).__next__,
    )
    assert result["prefix_token_count"] == 3
    assert result["effective_completion_budget_tokens"] == 61


def test_req_verify_7080_parser_and_forced_selection_fail_closed() -> None:
    """REQ-VERIFY-7080 rejects malformed branch and continuation shapes."""

    assert mod.parse_entrance("prefix {bad} then {}") is None
    assert mod.parse_entrance('{"operand_pair":[1,"2"],"operator":"+"}') is None
    assert mod.parse_entrance('{"operand_pair":[1,2],"operator":"%"}') is None

    units = [
        {"unit_id": "u0", "numbers": [1, 2, 3], "target": 6},
        {"unit_id": "u1", "numbers": [1, 2, 3], "target": 6},
    ]
    entrances = [
        {
            "unit_id": "u0",
            "entrance_id": "u0:1:2:add",
            "operand_pair": [1, 2],
            "operator": "+",
            "left": 1,
            "right": 2,
            "result": 3,
            "reachable": True,
        }
    ]
    selected = mod.select_forced_prefixes(
        ["model"], units, entrances, [], ["u1", "u0", "u1"], count_per_model=1
    )
    assert len(selected) == 1
    assert mod.continuation_succeeds(units[0], selected[0], '[{"operator":"+"}]') is False
    assert mod.continuation_succeeds(units[0], selected[0], '["not an operation"]') is False
    assert mod.continuation_succeeds(units[0], selected[0], "[broken") is False

    forced_schedule = mod.build_forced_schedule(
        selected,
        [{"hf_id": "model", "model_path": "/tmp/model.gguf"}],
    )
    assert forced_schedule[0]["arm"] == "forced_prefix"
    labeled = mod.label_forced_rows([{**forced_schedule[0], "raw_text": "not json"}], units)
    assert labeled[0]["parse_failure"] is True
    assert labeled[0]["continuation_success"] is False
    assert mod.cleanup_passes([]) is False


def test_req_infra_7080_unwritable_probe_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-7080 retains an operating-system write-probe failure."""

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("read only")

    monkeypatch.setattr(mod.tempfile, "mkstemp", fail)
    assert mod._writable(tmp_path / "result.json") is False


def _complete_artifact_inputs() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    models = list(mod.REQUIRED_MODEL_IDS)
    proposals = [
        {
            **_raw_row(model_id=model, unit_id="held-a", seed=seed, raw_text="bad"),
            "parse_failure": True,
            "legal": False,
            "reachable": False,
            "duplicate": False,
        }
        for model in models
        for seed in mod.PROPOSAL_SEEDS
    ]
    forced = [
        {
            **_raw_row(
                model_id=model,
                unit_id="held-a",
                seed=mod.FORCED_PREFIX_SEED,
                raw_text="bad",
                arm="forced_prefix",
            ),
            "reachable": True,
            "initially_unselected": True,
            "parse_failure": True,
            "continuation_success": False,
        }
        for model in models
    ]
    preconditions = {
        "all_passed": True,
        "checks": [mod.gate_row("all_preconditions", True, True, True)],
        "fixture_hash": mod.PINNED_FIXTURE_SHA256,
        "model_identity_rows": [{"model_id": model, "identity_matches": True} for model in models],
        "runner_build_rows": [{"version": "test", "gpu_offload": True}],
    }
    hashes = [
        {"model_id": model, "path": f"/{index}.gguf", "sha256": f"sha256:{index + 1:064x}"}
        for index, model in enumerate(models)
    ]
    phases = [
        {
            "model_id": model,
            "phase": phase,
            "terminal_state": "complete",
            "raw_rows": [{"raw_key": f"{model}|{phase}"}],
            "offloaded_layers": 48,
            "used_both_gpus": True,
            "gpu_lease_rows": [
                next(
                    row
                    for row in _lease_receipts(models)
                    if row["model_id"] == model and row["phase"] == phase
                )
            ],
            "gpu_sample_rows": [{"model_id": model, "gpu_uuid": "GPU-test"}],
            "port_lease": {"model_id": model, "released": True},
            "cleanup": {"model_id": model, "passed": True},
            "vram_release": {"passed": True},
        }
        for phase in ("proposal", "forced_prefix")
        for model in models
    ]
    return proposals, forced, preconditions, hashes, phases


def test_req_verify_7080_complete_and_partial_artifacts_recompute_quality_blind() -> None:
    """SCENARIO-VERIFY-7080-COMPLETE covers positive and partial reductions."""

    proposals, forced, preconditions, hashes, phases = _complete_artifact_inputs()
    specs = [
        {
            "hf_id": model,
            "model_path": f"/{index}.gguf",
        }
        for index, model in enumerate(mod.REQUIRED_MODEL_IDS)
    ]
    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=10,
        model_specs=specs,
        preconditions=preconditions,
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        held_unit_ids=["held-a"],
        forced_unit_ids=["held-a"],
        selected_model_specs=specs,
        model_file_hash_rows=hashes,
        checkpoint_rows=_checkpoint_receipts(list(mod.REQUIRED_MODEL_IDS)),
        phase_rows=phases,
    )
    assert artifact["entrance_proposal_bank_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["per_game_results"][0]["legal_count"] == 0
    assert artifact["per_game_results"][0]["forced_continuation_success_count"] == 0
    assert mod.validate_artifact(artifact) == []

    partial = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=10,
        model_specs=specs,
        preconditions=preconditions,
        proposal_rows=proposals[:-1],
        forced_prefix_rows=forced,
        held_unit_ids=["held-a"],
        forced_unit_ids=["held-a"],
        selected_model_specs=specs,
        model_file_hash_rows=hashes,
        checkpoint_rows=_checkpoint_receipts(list(mod.REQUIRED_MODEL_IDS)),
        phase_rows=phases,
    )
    assert partial["entrance_proposal_bank_complete_score"] == 0
    assert partial["verdict_class"] == "partial"
    assert mod.validate_artifact(partial) == []


def test_req_verify_7080_completion_evidence_mutation_matrix() -> None:
    """REQ-VERIFY-7080 completeness requires raw, identity, file, cleanup, and CUDA evidence."""

    proposals, forced, preconditions, hashes, phases = _complete_artifact_inputs()
    base = {
        "model_identity_rows": preconditions["model_identity_rows"],
        "model_file_hash_rows": hashes,
        "cleanup_rows": [row["cleanup"] for row in phases],
        "cuda_layer_offload_confirmed": True,
        "raw_output_hashes": [row["raw_output_hash"] for row in proposals + forced],
        "gpu_lease_rows": _lease_receipts(list(mod.REQUIRED_MODEL_IDS)),
        "model_execution_rows": _execution_receipts(list(mod.REQUIRED_MODEL_IDS)),
        "checkpoint_rows": _checkpoint_receipts(list(mod.REQUIRED_MODEL_IDS)),
        "vram_release_rows": _vram_receipts(list(mod.REQUIRED_MODEL_IDS)),
    }

    def errors(evidence: dict[str, Any], rows: list[dict[str, Any]] = proposals) -> list[str]:
        return mod.completion_errors(
            proposal_rows=rows,
            forced_prefix_rows=forced,
            held_unit_ids=["held-a"],
            forced_unit_ids=["held-a"],
            evidence=evidence,
        )

    changed_rows = deepcopy(proposals)
    changed_rows[0]["terminal_state"] = "failed"
    assert "raw_terminal_or_hash_mismatch" in errors(base, changed_rows)
    changed = deepcopy(base)
    changed["raw_output_hashes"] = []
    assert "raw_output_hash_projection_mismatch" in errors(changed)
    changed = deepcopy(base)
    changed["model_identity_rows"] = []
    assert "model_identity_incomplete" in errors(changed)
    changed = deepcopy(base)
    changed["model_file_hash_rows"] = []
    assert "model_file_hash_incomplete" in errors(changed)
    changed = deepcopy(base)
    changed["cleanup_rows"] = []
    assert "cleanup_incomplete" in errors(changed)
    changed = deepcopy(base)
    changed["cuda_layer_offload_confirmed"] = False
    assert "cuda_offload_unconfirmed" in errors(changed)


def test_req_verify_7080_validator_rejects_schema_projection_and_class_mutations(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7080-MUTATION covers each cold-validator branch."""

    base = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        model_specs=_resolved_specs(tmp_path),
        preconditions={
            "all_passed": False,
            "checks": [mod.gate_row("cache", True, False, False)],
        },
    )

    mutations = [
        (lambda value: value.pop("rows"), "missing_field:rows"),
        (lambda value: value["field_principles"].pop("rows"), "field_principles_mismatch"),
        (lambda value: value.update(schema="wrong"), "schema_mismatch"),
        (lambda value: value.update(run_date="wrong"), "run_date_mismatch"),
        (lambda value: value.update(inference_substrate="remote"), "inference_substrate_mismatch"),
        (lambda value: value.update(verifier_is_oracle=True), "verifier_is_oracle_mismatch"),
        (
            lambda value: value.update(entrance_proposal_bank_complete_score=False),
            "completion_score_not_bare_int",
        ),
        (lambda value: value.update(model_specs=[]), "model_specs_projection_mismatch"),
        (
            lambda value: value.update(raw_output_hashes=["bad"]),
            "raw_output_hash_projection_mismatch",
        ),
        (lambda value: value.update(gate_check_summary={}), "blocked_gate_summary_incomplete"),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(base)
        mutate(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)

    positive = deepcopy(base)
    positive["preconditions_checked"]["all_passed"] = True
    positive["entrance_proposal_bank_complete_score"] = 1
    positive["verdict_class"] = "positive"
    positive["honest_verdict"] = "wrong"
    positive["reproducibility_checksum"] = mod.artifact_checksum(positive)
    assert "positive_verdict_mismatch" in mod.validate_artifact(positive)

    partial = deepcopy(base)
    partial["preconditions_checked"]["all_passed"] = True
    partial["verdict_class"] = "null"
    partial["honest_verdict"] = "complete_null"
    partial["reproducibility_checksum"] = mod.artifact_checksum(partial)
    assert "partial_verdict_mismatch" in mod.validate_artifact(partial)


def test_req_infra_7080_real_fixture_precondition_can_pass_with_injected_resources(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-7080 recomputes and validates the real frozen fixture bytes."""

    devices = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GeForce RTX 3090",
            "utilization_gpu_pct": 0,
        }
        for index in (0, 1)
    ]
    specs = _resolved_specs(tmp_path)
    preflight = mod.collect_preconditions(
        fixture_path=mod.FIXTURE_PATH,
        expected_fixture_hash=mod.PINNED_FIXTURE_SHA256,
        model_specs=specs,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": True, "devices": devices, "processes": []},
        llama_probe=lambda: {"importable": True, "gpu_offload": True, "version": "test"},
        lease_probe=lambda rows: [
            {"device_uuid": row["uuid"], "classification": "available"} for row in rows
        ],
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        identity_probe=lambda row: {
            "model_id": row["hf_id"],
            "identity_matches": True,
            "tokenizer_source": "embedded_gguf",
        },
    )
    assert preflight["all_passed"] is True
    assert preflight["fixture_hash"] == mod.PINNED_FIXTURE_SHA256


def test_req_infra_7080_requires_cold_audit_hash_and_score(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7080-BLOCKED checks the Exp7079 boundary first."""

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps({"gpu_lease_cold_audit_ready_score": 0}), encoding="utf-8"
    )
    devices = [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GeForce RTX 3090",
            "utilization_gpu_pct": 0,
        }
        for index in (0, 1)
    ]
    result = mod.collect_preconditions(
        lease_audit_path=audit_path,
        expected_lease_audit_hash=mod.sha256_file(audit_path),
        fixture_path=mod.FIXTURE_PATH,
        expected_fixture_hash=mod.PINNED_FIXTURE_SHA256,
        model_specs=_resolved_specs(tmp_path),
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": True, "devices": devices, "processes": []},
        llama_probe=lambda: {"importable": True, "gpu_offload": True},
        lease_probe=lambda rows: [
            {"device_uuid": row["uuid"], "classification": "available"} for row in rows
        ],
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        identity_probe=lambda row: {
            "model_id": row["hf_id"],
            "identity_matches": True,
            "tokenizer_source": "embedded_gguf",
        },
    )
    failed = {row["check"] for row in result["checks"] if row["passed"] is not True}
    assert failed == {"gpu_lease_cold_audit_ready_score"}

    audit_path.write_text(
        json.dumps({"gpu_lease_cold_audit_ready_score": 1}), encoding="utf-8"
    )
    stale = mod.collect_preconditions(
        lease_audit_path=audit_path,
        expected_lease_audit_hash="sha256:" + "0" * 64,
        fixture_path=mod.FIXTURE_PATH,
        expected_fixture_hash=mod.PINNED_FIXTURE_SHA256,
        model_specs=_resolved_specs(tmp_path),
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        gpu_probe=lambda: {"query_ok": True, "devices": devices, "processes": []},
        llama_probe=lambda: {"importable": True, "gpu_offload": True},
        lease_probe=lambda rows: [
            {"device_uuid": row["uuid"], "classification": "available"} for row in rows
        ],
        stop_authority_probe=lambda: {"passed": True, "observed": "clean"},
        identity_probe=lambda row: {
            "model_id": row["hf_id"],
            "identity_matches": True,
            "tokenizer_source": "embedded_gguf",
        },
    )
    assert stale["checks"][0]["check"] == "gpu_lease_cold_audit_source_hash"
    assert stale["checks"][0]["passed"] is False


def test_req_infra_7080_lease_loss_removes_completion_credit() -> None:
    """SCENARIO-INFRA-7080-LEASE-LOSS rejects changed ownership and phases."""

    proposals, forced, preconditions, hashes, phases = _complete_artifact_inputs()
    evidence = {
        "model_identity_rows": preconditions["model_identity_rows"],
        "model_file_hash_rows": hashes,
        "cleanup_rows": [row["cleanup"] for row in phases],
        "cuda_layer_offload_confirmed": True,
        "raw_output_hashes": [row["raw_output_hash"] for row in proposals + forced],
        "gpu_lease_rows": _lease_receipts(list(mod.REQUIRED_MODEL_IDS)),
        "model_execution_rows": _execution_receipts(list(mod.REQUIRED_MODEL_IDS)),
        "checkpoint_rows": _checkpoint_receipts(list(mod.REQUIRED_MODEL_IDS)),
        "vram_release_rows": _vram_receipts(list(mod.REQUIRED_MODEL_IDS)),
    }
    assert mod.completion_errors(
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        held_unit_ids=["held-a"],
        forced_unit_ids=["held-a"],
        evidence=evidence,
    ) == []

    evidence["gpu_lease_rows"][0]["owner_preserved"] = False
    assert "lease_identity_or_release_incomplete" in mod.completion_errors(
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        held_unit_ids=["held-a"],
        forced_unit_ids=["held-a"],
        evidence=evidence,
    )


def test_req_verify_7080_required_schema_and_raw_derived_separation(tmp_path: Path) -> None:
    """REQ-VERIFY-7080 keeps raw, token, parse, and exact rows separate."""

    raw = _raw_row(model_id=mod.REQUIRED_MODEL_IDS[0])
    labeled = {
        **raw,
        "parsed_entrance": {"operand_pair": [2, 3], "operator": "*"},
        "entrance_id": "unit-a:2:3:multiply",
        "parse_failure": False,
        "legal": True,
        "reachable": True,
        "duplicate": False,
        "label_source": "experiment_7064_exhaustive_enumerator",
    }
    rows = mod.proposal_evidence_rows([labeled])
    assert set(rows) == {
        "raw_proposal_rows",
        "token_score_rows",
        "parse_rows",
        "exact_label_rows",
    }
    assert "legal" not in rows["raw_proposal_rows"][0]
    assert "parsed_entrance" not in rows["raw_proposal_rows"][0]
    assert "raw_text" not in rows["exact_label_rows"][0]
    assert rows["token_score_rows"][0]["raw_output_hash"] == raw["raw_output_hash"]

    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        model_specs=_resolved_specs(tmp_path),
        preconditions={
            "all_passed": False,
            "checks": [mod.gate_row("cache", True, False, False)],
        },
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == "live_local_sota_gguf_cuda_llamacpp"
    assert mod.validate_artifact(artifact) == []


def test_req_infra_7080_unreadable_audit_and_live_lease_projection(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7080-BLOCKED and LEASE-LOSS cover raw host evidence."""

    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text("{}", encoding="utf-8")
    common = {
        "fixture_path": fixture_path,
        "expected_fixture_hash": mod.sha256_file(fixture_path),
        "model_specs": _resolved_specs(tmp_path),
        "result_path": tmp_path / "result.json",
        "checkpoint_path": tmp_path / "checkpoint.json",
        "gpu_probe": lambda: {"query_ok": True, "devices": [], "processes": []},
        "llama_probe": lambda: {"importable": True, "gpu_offload": True},
        "lease_probe": lambda _rows: [],
        "stop_authority_probe": lambda: {"passed": True, "observed": "clean"},
        "identity_probe": lambda row: {
            "model_id": row["hf_id"],
            "identity_matches": True,
            "tokenizer_source": "embedded_gguf",
        },
    }
    missing = mod.collect_preconditions(
        lease_audit_path=tmp_path / "missing.json",
        expected_lease_audit_hash="sha256:" + "0" * 64,
        **common,
    )
    assert missing["lease_audit"]["read_error"].startswith("FileNotFoundError:")

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    malformed = mod.collect_preconditions(
        lease_audit_path=list_path,
        expected_lease_audit_hash=mod.sha256_file(list_path),
        **common,
    )
    assert malformed["lease_audit"]["read_error"] == "ValueError: lease_audit_not_object"

    owner = {"pid": 10, "pid_start_ticks": 20}
    history = [{"phase": phase} for phase in mod.lease_api.COMPLETE_PHASE_SEQUENCE]
    normalized = mod.normalize_gpu_lease_rows(
        [
            {
                "model_id": mod.REQUIRED_MODEL_IDS[0],
                "phase": "proposal",
                "gpu_lease_rows": [
                    {
                        "device_uuid": "GPU-test",
                        "lease_id": "lease:test",
                        "journal_after_acquisition": {
                            "lease_id": "lease:test",
                            "owner": owner,
                        },
                        "journal_after_release": {
                            "lease_id": "lease:test",
                            "owner": owner,
                            "phase_history": history,
                            "released": True,
                            "checksum": "sha256:journal",
                        },
                        "release_receipt": {"lease_id": "lease:test", "released": True},
                        "signals_sent": [],
                    }
                ],
            }
        ]
    )
    assert normalized[0]["owner_preserved"] is True
    assert normalized[0]["released"] is True
    assert normalized[0]["lease_lost"] is False


def test_req_infra_7080_identity_probe_runs_in_disposable_process(tmp_path: Path) -> None:
    """SCENARIO-INFRA-7080-ROSTER prevents parent CUDA context contamination."""

    model = _resolved_specs(tmp_path)[0]

    def runner(command: list[str], **kwargs: Any) -> Any:
        assert kwargs["check"] is False
        payload = Path(command[command.index("--identity-payload") + 1])
        output = Path(command[command.index("--identity-output") + 1])
        assert json.loads(payload.read_text(encoding="utf-8"))["hf_id"] == model["hf_id"]
        output.write_text(
            json.dumps(
                {
                    "model_id": model["hf_id"],
                    "identity_matches": True,
                    "tokenizer_source": "embedded_gguf",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    row = mod._identity_probe(model, runner=runner)
    assert row["identity_matches"] is True
    assert row["probe_process_exit_code"] == 0
    assert row["isolated_process"] is True


def test_req_verify_7080_validator_rejects_separated_projection_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7080-MUTATION checks every separated row projection."""

    artifact = mod.build_artifact(
        run_date=mod.RUN_DATE,
        duration_s=0.1,
        model_specs=_resolved_specs(tmp_path),
        preconditions={
            "all_passed": False,
            "checks": [mod.gate_row("cache", True, False, False)],
        },
    )
    mutations = (
        ("raw_proposal_rows", "raw_proposal_projection_mismatch"),
        ("parse_rows", "parse_projection_mismatch"),
        ("exact_label_rows", "exact_label_projection_mismatch"),
        ("token_score_rows", "token_score_projection_mismatch"),
    )
    for field, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = [{"forged": True}]
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)
