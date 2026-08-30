"""Focused tests for the paired proof-transport experiment.

Spec refs: REQ-VERIFY-6770 and SCENARIO-VERIFY-6770-*.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from carnot import experiment_6770_dccd_environment_grammar_ab_v2 as exp


def _panel() -> dict:
    """Read the frozen upstream panel without changing the research record."""

    return json.loads((exp.REPO_ROOT / exp.PANEL_PATH).read_text(encoding="utf-8"))


def _grammar_artifact() -> dict:
    """Read the invoked grammar receipt that gates this experiment."""

    return json.loads((exp.REPO_ROOT / exp.GRAMMAR_PATH).read_text(encoding="utf-8"))


def _models() -> list[dict]:
    """Return resolved-looking model records for deterministic unit tests."""

    return [
        {
            **deepcopy(model),
            "model_path": f"/models/{model['family_id']}.gguf",
            "model_sha256": f"sha256:{index:064x}",
            "model_size_bytes": 8_000_000_000,
            "resolved": True,
            "tokenizer": {
                "source": "llama.cpp_embedded_gguf",
                "loadable": True,
                "detail": "unit-test tokenizer",
            },
            "context_limit": exp.CONTEXT_LIMIT,
            "total_output_token_ceiling": exp.TOTAL_OUTPUT_TOKENS,
        }
        for index, model in enumerate(exp.MODEL_DEFINITIONS, start=1)
    ]


def _passing_preconditions(models: list[dict] | None = None) -> dict:
    """Build the smallest complete preflight receipt used by reducers."""

    rows = [
        {"check": name, "expected": True, "observed": True, "passed": True}
        for name in exp.PRECONDITION_NAMES
    ]
    return {
        "all_passed": True,
        "checks": rows,
        "models": deepcopy(models or _models()),
        "device_selection_receipt": {
            "selected_device": {
                "index": 0,
                "uuid": "GPU-unit",
                "memory_total_mb": 24_576,
                "memory_used_mb": 128,
                "memory_free_mb": 24_448,
            }
        },
        "ports": [47_700, 47_701, 47_702],
        "remote_inference_allowed": False,
        "legacy_headline_fallback_allowed": False,
    }


def _gpu_receipts(models: list[dict] | None = None) -> list[dict]:
    """Return one complete ownership and teardown receipt per model."""

    return [
        {
            "model_family_id": model["family_id"],
            "model_hf_id": model["hf_id"],
            "device": {"index": 0, "uuid": "GPU-unit"},
            "lease_owner": {"task_id": f"exp6770-{model['family_id']}"},
            "cuda_offload": True,
            "gpu_layers": {"requested": -1, "offloaded": 64, "total": 64},
            "peak_vram_mb": 12_000,
            "live_model_invoked": True,
            "first_token_observed": True,
            "process_exit": {"exit_code": 0, "absent_after_exit": True},
            "lease_release": {"released": True},
            "vram_recovery": {"passed": True, "owned_pid_present": False},
            "teardown_passed": True,
            "errors": [],
        }
        for model in (models or _models())
    ]


def _complete_rows(manifest: dict, models: list[dict] | None = None) -> list[dict]:
    """Create attributable rows with controlled paired outcomes."""

    rows = []
    for model in models or _models():
        for index, instance in enumerate(manifest["instances"]):
            for arm in exp.ARMS:
                exact = arm != "repaired_direct" or index % 2 == 0
                semantic = exact or index % 3 == 0
                row = {
                    "schema": exp.ROW_SCHEMA,
                    "row_id": f"{model['family_id']}|{instance['instance_id']}|{arm}",
                    "model_family_id": model["family_id"],
                    "model_hf_id": model["hf_id"],
                    "instance_id": instance["instance_id"],
                    "family": instance["family"],
                    "size": instance["size"],
                    "error_class": instance["error_class"],
                    "source_role": instance["source_role"],
                    "relabel_role": instance["relabel_role"],
                    "arm": arm,
                    "generation_seed": instance["generation_seed"],
                    "total_output_token_ceiling": exp.TOTAL_OUTPUT_TOKENS,
                    "exact_check_budget": exp.EXACT_CHECK_BUDGET,
                    "draft_render_split": deepcopy(exp.DCCD_SPLIT)
                    if arm == "dccd_environment"
                    else None,
                    "raw_output": "SAT x1=1",
                    "raw_output_sha256": exp.sha256_text("SAT x1=1"),
                    "parsed_proof": {"parser_status": "parseable", "claim": "SAT"},
                    "encoder_a": {"attempted": True},
                    "encoder_b": {"attempted": True},
                    "exact_result": {"attempted": True, "valid": exact},
                    "exact_valid": exact,
                    "semantic_correct": semantic,
                    "parseable": True,
                    "abstained": False,
                    "invalid_reference": False,
                    "invalid_domain": False,
                    "support_contracted": arm == "dccd_environment" and index % 4 == 0,
                    "generated_tokens": 8 if arm != "dccd_environment" else 16,
                    "latency_s": 0.2 if arm != "dccd_environment" else 0.4,
                    "device": {"index": 0, "uuid": "GPU-unit"},
                    "peak_vram_mb": 12_000,
                    "seed": instance["generation_seed"],
                    "stop_reason": "eos",
                    "runtime_grammar": {
                        "requested": arm != "repaired_direct",
                        "passed_to_runtime": arm != "repaired_direct",
                        "policy_calls": 0 if arm == "repaired_direct" else 1,
                        "post_hoc_filter_used": False,
                        "fixture_used": False,
                        "answer_conditioned": False,
                        "substituted_model": False,
                        "qualified": True,
                    },
                    "failure": None,
                    "solver_conflicts": None,
                    "row_sha256": "",
                }
                row["row_sha256"] = exp.row_checksum(row)
                rows.append(row)
    return rows


def test_scenario_verify_6770_pairing_freezes_an_exact_balanced_denominator() -> None:
    """SCENARIO-VERIFY-6770-PAIRING freezes each quota and arm rotation."""

    manifest = exp.build_frozen_manifest(_panel())
    instances = manifest["instances"]

    assert len(instances) == exp.MINIMUM_INSTANCES == 36
    assert Counter(row["family"] for row in instances) == {
        "expander_tseitin": 12,
        "ladder_tseitin": 12,
        "pigeonhole_anchor": 12,
    }
    assert Counter(row["size"] for row in instances) == {"small": 18, "medium": 18}
    assert set(Counter(row["error_class"] for row in instances).values()) == {6}
    assert Counter(row["source_role"] for row in instances) == {"SAT": 18, "UNSAT": 18}
    assert Counter(row["relabel_role"] for row in instances) == {"base": 18, "relabel": 18}
    assert Counter(tuple(row["arm_order"]) for row in instances) == {
        tuple(exp.ARMS): 12,
        tuple(exp.ARMS[1:] + exp.ARMS[:1]): 12,
        tuple(exp.ARMS[2:] + exp.ARMS[:2]): 12,
    }
    assert manifest["planned_row_count"] == 36 * 3 * 3
    assert manifest["manifest_sha256"] == exp.manifest_checksum(manifest)


def test_scenario_verify_6770_budget_is_equal_for_all_three_arms() -> None:
    """SCENARIO-VERIFY-6770-BUDGET keeps matched ceilings and checks."""

    manifest = exp.build_frozen_manifest(_panel())
    budgets = manifest["arm_budgets"]

    assert {row["total_output_tokens"] for row in budgets.values()} == {exp.TOTAL_OUTPUT_TOKENS}
    assert {row["exact_check_budget"] for row in budgets.values()} == {exp.EXACT_CHECK_BUDGET}
    assert sum(exp.DCCD_SPLIT.values()) == exp.TOTAL_OUTPUT_TOKENS
    assert all(row["context_limit"] == exp.CONTEXT_LIMIT for row in budgets.values())


def test_scenario_verify_6770_runtime_passes_real_grammar_objects() -> None:
    """SCENARIO-VERIFY-6770-RUNTIME sends static and dynamic grammars live."""

    instance = exp.build_frozen_manifest(_panel())["instances"][0]
    calls = []

    def generate(**kwargs: object) -> dict:
        calls.append(kwargs)
        prompt = str(kwargs["prompt"])
        output = instance["before_certificate"]
        if "semantic draft" in prompt:
            output = f"Reason briefly.\nFINAL: {instance['before_certificate']}"
        return {
            "raw_output": output,
            "prompt_tokens": 20,
            "generated_tokens": 10,
            "latency_s": 0.1,
            "stop_reason": "eos",
            "failure": None,
        }

    rows = exp.execute_instance_arms(
        generate,
        instance,
        _models()[0],
        {"index": 0, "uuid": "GPU-unit"},
        peak_vram_mb=12_000,
    )

    assert [row["arm"] for row in rows] == instance["arm_order"]
    direct_calls = [call for call in calls if call["stage"] == "direct"]
    static_calls = [call for call in calls if call["stage"] == "static"]
    draft_calls = [call for call in calls if call["stage"] == "draft"]
    render_calls = [call for call in calls if call["stage"] == "render"]
    assert direct_calls[0]["grammar"] is None
    assert draft_calls[0]["grammar"] is None
    assert static_calls[0]["grammar"] is not None
    assert render_calls[0]["grammar"] is not None
    assert (
        rows[instance["arm_order"].index("static_grammar")]["runtime_grammar"]["policy_calls"] == 1
    )
    dccd = rows[instance["arm_order"].index("dccd_environment")]
    assert dccd["runtime_grammar"]["policy_calls"] == 1
    assert dccd["draft_receipt"]["raw_output"].startswith("Reason briefly")
    assert all(row["row_sha256"] == exp.row_checksum(row) for row in rows)


def test_scenario_verify_6770_runtime_disqualifies_non_live_constraints() -> None:
    """SCENARIO-VERIFY-6770-RUNTIME rejects every forbidden fallback."""

    base = {
        "requested": True,
        "passed_to_runtime": True,
        "policy_calls": 1,
        "post_hoc_filter_used": False,
        "fixture_used": False,
        "answer_conditioned": False,
        "substituted_model": False,
    }
    assert exp.qualify_runtime_grammar(base) is True
    for field, value in (
        ("passed_to_runtime", False),
        ("policy_calls", 0),
        ("post_hoc_filter_used", True),
        ("fixture_used", True),
        ("answer_conditioned", True),
        ("substituted_model", True),
    ):
        changed = {**base, field: value}
        assert exp.qualify_runtime_grammar(changed) is False


def test_req_verify_6770_model_resolution_calls_pair_then_third(tmp_path: Path) -> None:
    """REQ-VERIFY-6770 resolves all exact GGUFs and embedded tokenizers."""

    paths = []
    for index, model in enumerate(exp.MODEL_DEFINITIONS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(model["family_id"].encode())
        paths.append(path)
    pair_calls = []
    third_calls = []

    def pair_resolver(**kwargs: object) -> list[dict]:
        pair_calls.append(kwargs)
        return [
            {"hf_id": exp.MODEL_DEFINITIONS[0]["hf_id"], "model_path": str(paths[0])},
            {"hf_id": exp.MODEL_DEFINITIONS[1]["hf_id"], "model_path": str(paths[1])},
        ]

    def single_resolver(hf_id: str, _quant: str) -> str:
        third_calls.append(hf_id)
        return str(paths[2])

    resolved = exp.resolve_models(
        pair_resolver=pair_resolver,
        single_resolver=single_resolver,
        tokenizer_probe=lambda path: (True, f"embedded:{Path(path).name}"),
    )

    assert pair_calls == [{"gpu_indices": (0, 0), "model_indices": (0, 2)}]
    assert third_calls == [exp.MODEL_DEFINITIONS[2]["hf_id"]]
    assert [row["hf_id"] for row in resolved] == [row["hf_id"] for row in exp.MODEL_DEFINITIONS]
    assert all(row["resolved"] and row["tokenizer"]["loadable"] for row in resolved)
    assert all(row["model_sha256"].startswith("sha256:") for row in resolved)


def test_scenario_verify_6770_blocked_artifact_keeps_full_schema() -> None:
    """SCENARIO-VERIFY-6770-BLOCKED retains the failed observed value."""

    manifest = exp.build_frozen_manifest(_panel())
    preflight = _passing_preconditions()
    preflight["all_passed"] = False
    preflight["checks"][3] = {
        "check": "llama_cpp_cuda_offload",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    artifact = exp.build_blocked_artifact(
        date="20260830",
        duration_s=0.25,
        manifest=manifest,
        models=_models(),
        preconditions=preflight,
    )

    assert artifact["status"] == "complete_blocked_proof_transport_ab_v2"
    assert artifact["rows"] == []
    assert artifact["live_model_invoked"] is False
    assert artifact["proof_transport_ab_completed"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_proof_transport_ab_v2")
    assert artifact["gate_check_summary"]["failed_check"] == "llama_cpp_cuda_offload"
    assert set(artifact) == set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []


def test_scenario_verify_6770_teardown_controls_completion() -> None:
    """SCENARIO-VERIFY-6770-TEARDOWN requires release and VRAM recovery."""

    receipts = _gpu_receipts()
    assert all(exp.teardown_receipt_passes(row) for row in receipts)
    for path in (
        ("teardown_passed", False),
        ("cuda_offload", False),
        ("first_token_observed", False),
    ):
        changed = deepcopy(receipts[0])
        changed[path[0]] = path[1]
        assert exp.teardown_receipt_passes(changed) is False
    changed = deepcopy(receipts[0])
    changed["lease_release"]["released"] = False
    assert exp.teardown_receipt_passes(changed) is False
    changed = deepcopy(receipts[0])
    changed["vram_recovery"]["passed"] = False
    assert exp.teardown_receipt_passes(changed) is False
    changed = deepcopy(receipts[0])
    changed["process_exit"]["absent_after_exit"] = False
    assert exp.teardown_receipt_passes(changed) is False


def test_scenario_verify_6770_cold_reducer_recomputes_headline_and_groups() -> None:
    """SCENARIO-VERIFY-6770-COLD derives paired effects from retained rows."""

    models = _models()
    manifest = exp.build_frozen_manifest(_panel())
    rows = _complete_rows(manifest, models)
    reduction = exp.recompute_aggregates(rows, manifest, models, _gpu_receipts(models))

    assert reduction["proof_transport_ab_completed"] is True
    assert reduction["runtime_mask_invocations_by_arm"] == {
        "repaired_direct": 0,
        "static_grammar": 108,
        "dccd_environment": 108,
    }
    assert reduction["exact_valid_rate_by_arm"]["static_grammar"] == 1.0
    assert reduction["exact_valid_rate_by_arm"]["dccd_environment"] == 1.0
    assert reduction["exact_valid_rate_by_arm"]["repaired_direct"] == 0.5
    assert reduction["paired_exact_valid_deltas"]["static_grammar-minus-repaired_direct"] == 0.5
    assert reduction["paired_exact_valid_deltas"]["dccd_environment-minus-static_grammar"] == 0.0
    assert set(reduction["group_metrics"]) == {
        "model",
        "family",
        "size",
        "error_class",
        "relabel_role",
    }
    assert reduction["row_consistency_errors"] == []

    rows[0]["total_output_token_ceiling"] -= 1
    rows[0]["row_sha256"] = exp.row_checksum(rows[0])
    drift = exp.recompute_aggregates(rows, manifest, models, _gpu_receipts(models))
    assert drift["proof_transport_ab_completed"] is False
    assert "budget_mismatch" in drift["row_consistency_errors"]


def test_req_verify_6770_complete_artifact_rejects_row_and_aggregate_drift() -> None:
    """REQ-VERIFY-6770 validates cold reduction and content hashes."""

    models = _models()
    manifest = exp.build_frozen_manifest(_panel())
    rows = _complete_rows(manifest, models)
    artifact = exp.build_artifact(
        date="20260830",
        duration_s=61.0,
        manifest=manifest,
        models=models,
        rows=rows,
        gpu_receipts=_gpu_receipts(models),
        preconditions=_passing_preconditions(models),
    )

    assert artifact["proof_transport_ab_completed"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith("complete:")
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["exact_valid"] = not changed["rows"][0]["exact_valid"]
    assert set(exp.validate_artifact(changed)) >= {
        "row_checksum_mismatch",
        "aggregate_recomputation_mismatch",
        "reproducibility_checksum_mismatch",
    }
    changed = deepcopy(artifact)
    changed["verdict_class"] = "wrong"
    changed["verifier_is_oracle"] = True
    changed["claim_boundary"] = "parseability"
    changed["field_principles"].pop("rows")
    assert set(exp.validate_artifact(changed)) >= {
        "verdict_class_invalid",
        "verifier_is_oracle_mismatch",
        "claim_boundary_mismatch",
        "field_principles_mismatch",
        "reproducibility_checksum_mismatch",
    }


def test_req_verify_6770_preconditions_fail_closed_by_named_check(tmp_path: Path) -> None:
    """REQ-VERIFY-6770 records upstream, model, CUDA, lease, and host gates."""

    manifest = exp.build_frozen_manifest(_panel())
    models = _models()
    devices = [
        {
            "index": 0,
            "uuid": "GPU-unit",
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mb": 24_576,
            "memory_used_mb": 128,
            "memory_free_mb": 24_448,
            "active_compute_processes": [],
        }
    ]
    kwargs = {
        "panel": _panel(),
        "grammar_artifact": _grammar_artifact(),
        "models": models,
        "manifest": manifest,
        "llama_receipt": {"cuda_offload": True},
        "devices": devices,
        "ports": [47_700, 47_701, 47_702],
        "ports_free": [True, True, True],
        "lease_probe": {"available": True, "error": None},
        "host_resources": {
            "ram_available_bytes": exp.MIN_RAM_BYTES + 1,
            "disk_free_bytes": exp.MIN_DISK_BYTES + 1,
        },
        "exact_authority_ready": True,
    }
    ready = exp.evaluate_preconditions(**kwargs)
    assert ready["all_passed"] is True
    assert [row["check"] for row in ready["checks"]] == list(exp.PRECONDITION_NAMES)

    cases = {
        "dynamic_proof_grammar_ready": ("grammar_artifact", {"dynamic_proof_grammar_ready": False}),
        "all_model_specs_resolved": ("models", [{**models[0], "resolved": False}, *models[1:]]),
        "llama_cpp_cuda_offload": ("llama_receipt", {"cuda_offload": False}),
        "task_owned_lease": ("lease_probe", {"available": False, "error": "busy"}),
        "ports_free": ("ports_free", [True, False, True]),
        "exact_authority": ("exact_authority_ready", False),
        "ram_available": (
            "host_resources",
            {"ram_available_bytes": 0, "disk_free_bytes": exp.MIN_DISK_BYTES + 1},
        ),
        "disk_available": (
            "host_resources",
            {"ram_available_bytes": exp.MIN_RAM_BYTES + 1, "disk_free_bytes": 0},
        ),
    }
    for expected, (key, value) in cases.items():
        changed = deepcopy(kwargs)
        changed[key] = value
        receipt = exp.evaluate_preconditions(**changed)
        assert exp.first_failed_check(receipt)["check"] == expected


def test_req_verify_6770_run_writes_blocked_artifact_without_session(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6770-BLOCKED stops before a model session."""

    manifest = exp.build_frozen_manifest(_panel())
    preflight = _passing_preconditions()
    preflight["all_passed"] = False
    preflight["checks"][0]["observed"] = False
    preflight["checks"][0]["passed"] = False
    session_calls = []
    result = tmp_path / "result.json"
    artifact = exp.run(
        date="20260830",
        result_path=result,
        panel=_panel(),
        grammar_artifact=_grammar_artifact(),
        models=_models(),
        manifest=manifest,
        preconditions=preflight,
        session_runner=lambda *_args, **_kwargs: session_calls.append(True),
        clock=iter((1_000_000_000, 1_250_000_000)).__next__,
    )

    assert session_calls == []
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["duration_s"] == 0.25
    assert exp.validate_artifact(artifact) == []


def test_req_verify_6770_run_stops_after_failed_teardown(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6770-TEARDOWN prevents a second resident model."""

    models = _models()
    manifest = exp.build_frozen_manifest(_panel())
    calls = []

    def session(
        model: dict, _instances: list[dict], _device: dict, _port: int
    ) -> tuple[list, dict]:
        calls.append(model["family_id"])
        receipt = _gpu_receipts([model])[0]
        receipt["teardown_passed"] = False
        return [], receipt

    artifact = exp.run(
        date="20260830",
        result_path=tmp_path / "partial.json",
        panel=_panel(),
        grammar_artifact=_grammar_artifact(),
        models=models,
        manifest=manifest,
        preconditions=_passing_preconditions(models),
        session_runner=session,
        clock=iter((2_000_000_000, 3_000_000_000)).__next__,
    )

    assert calls == [models[0]["family_id"]]
    assert artifact["proof_transport_ab_completed"] is False
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("complete_partial")
    assert exp.validate_artifact(artifact) == []


def test_req_verify_6770_live_session_adapter_and_atomic_writer(tmp_path: Path) -> None:
    """REQ-VERIFY-6770 adapts llama.cpp receipts and writes atomically."""

    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs: object) -> dict:
            assert kwargs["messages"][0]["role"] == "user"
            return {
                "choices": [{"message": {"content": "SAT x1=1"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }

        def close(self) -> None:
            self.closed = True

    session = exp.LiveLlamaSession(
        _models()[0],
        {"index": 0, "uuid": "GPU-unit"},
        llama_factory=FakeLlama,
        grammar_factory=lambda text: {"grammar": text},
        clock=iter((10.0, 10.5)).__next__,
    )
    receipt = session.generate(
        prompt="solve",
        max_tokens=16,
        seed=7,
        temperature=0.0,
        stop=["\n"],
        grammar='root ::= "ABSTAIN"',
        stage="static",
    )
    session.close()

    assert receipt == {
        "raw_output": "SAT x1=1",
        "prompt_tokens": 7,
        "generated_tokens": 3,
        "latency_s": 0.5,
        "stop_reason": "stop",
        "failure": None,
        "stage": "static",
        "grammar_passed": True,
    }
    destination = tmp_path / "nested" / "artifact.json"
    exp.write_json_atomic(destination, {"ok": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"ok": True}


def test_req_verify_6770_parser_and_cli_defaults() -> None:
    """REQ-VERIFY-6770 keeps the planning date and wrapper contract fixed."""

    assert exp.extract_draft_certificate("work\nFINAL: UNSAT c1,c2") == "UNSAT c1,c2"
    assert exp.extract_draft_certificate("no final certificate") == ""
    assert exp.parse_args([]).date == "20260830"
    assert exp.main(["--validate-only", "--date", "20260830"]) == 0


def test_req_verify_6770_defensive_parser_and_manifest_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6770 covers fail-closed manifest, grammar, and checker paths."""

    with pytest.raises(ValueError, match="source_role_missing"):
        exp._source_role({"source_stream_row_id": "role-free"})
    panel = _panel()
    panel["rows"] = [row for row in panel["rows"] if row["row_id"] != exp.FROZEN_INSTANCE_IDS[0]]
    with pytest.raises(ValueError, match="frozen_instance_missing"):
        exp.build_frozen_manifest(panel)

    cnf = {"n_vars": 1, "clauses": [[1]]}
    assert '"SAT "' in exp._environment_grammar(cnf, "SAT")
    assert '"UNSAT "' in exp._environment_grammar(cnf, "UNSAT")
    assert exp._environment_grammar(cnf, None).startswith('root ::= ("ABSTAIN")')
    parsed, first, second, exact = exp._normalized_checks("ABSTAIN", cnf)
    assert parsed["parser_status"] == "abstention"
    assert not first["attempted"] and not second["attempted"] and not exact["valid"]

    monkeypatch.setattr(
        exp.encoder_a,
        "encode_certificate",
        lambda _parsed: (_ for _ in ()).throw(ValueError("bad")),
    )
    parsed, first, second, exact = exp._normalized_checks("SAT x1=1", cnf)
    assert parsed["parser_status"] == "parseable"
    assert first["error"] == second["error"] == exact["reason"] == "bad"

    assert exp._cnf_is_sat(cnf) is True
    assert exp._cnf_is_sat({"n_vars": 1, "clauses": [[1], [-1]]}) is False
    assert exp._reference_diagnosis("UNSAT c2", {"claim": "UNSAT", "terms": ["c2"]}, cnf) == (
        True,
        False,
    )
    assert exp._reference_diagnosis("SAT x2=7", {"claim": None, "terms": []}, cnf) == (
        True,
        True,
    )


def test_req_verify_6770_reducer_reports_all_attribution_drift() -> None:
    """SCENARIO-VERIFY-6770-COLD reports every retained-row consistency fault."""

    models = _models()
    manifest = exp.build_frozen_manifest(_panel())
    rows = _complete_rows(manifest, models)
    rows.append(deepcopy(rows[1]))
    rows[0]["row_id"] = "unknown-row"
    rows[0]["raw_output_sha256"] = "sha256:wrong"
    rows[0]["instance_id"] = "unknown"
    rows[1]["model_hf_id"] = "substituted/model"
    rows[1]["seed"] = 0
    rows[1]["runtime_grammar"]["passed_to_runtime"] = False
    reduction = exp.recompute_aggregates(rows, manifest, models, _gpu_receipts(models))

    assert set(reduction["row_consistency_errors"]) >= {
        "duplicate_row_id",
        "planned_row_set_mismatch",
        "row_checksum_mismatch",
        "raw_output_hash_mismatch",
        "row_attribution_mismatch",
        "model_substitution",
        "seed_mismatch",
        "runtime_grammar_disqualified",
    }
    assert exp.first_failed_check(_passing_preconditions()) == {
        "check": None,
        "expected": True,
        "observed": True,
        "passed": True,
    }


def test_req_verify_6770_validator_and_host_probe_reject_tampering() -> None:
    """REQ-VERIFY-6770 exercises terminal schema and local port defenses."""

    manifest = exp.build_frozen_manifest(_panel())
    preflight = _passing_preconditions()
    preflight["all_passed"] = False
    preflight["checks"][0]["passed"] = False
    blocked = exp.build_blocked_artifact(
        date="20260830",
        duration_s=0.1,
        manifest=manifest,
        models=_models(),
        preconditions=preflight,
    )
    changed = deepcopy(blocked)
    changed.pop("title")
    changed["inference_substrate"] = "remote"
    changed["honest_verdict"] = "not terminal"
    changed["frozen_manifest"]["manifest_sha256"] = "sha256:wrong"
    changed["rows"] = [{"row_sha256": "wrong"}]
    changed["live_model_invoked"] = True
    changed["proof_transport_ab_completed"] = True
    assert set(exp.validate_artifact(changed)) >= {
        "artifact_fields_mismatch",
        "inference_substrate_mismatch",
        "honest_verdict_prefix_invalid",
        "manifest_checksum_mismatch",
        "row_checksum_mismatch",
        "blocked_artifact_live_evidence",
        "blocked_artifact_completed",
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        assert exp._port_is_free(port) is False
    assert exp._port_is_free(port) is True
    assert exp._exact_authority_smoke({"rows": []}) is False
    assert exp._exact_authority_smoke(_panel()) is True


def test_req_verify_6770_complete_run_and_cli_terminal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6770 runs every sequential model and checks CLI failures."""

    models = _models()
    manifest = exp.build_frozen_manifest(_panel())
    all_rows = _complete_rows(manifest, models)

    def session(
        model: dict, _instances: list[dict], _device: dict, _port: int
    ) -> tuple[list, dict]:
        rows = [row for row in all_rows if row["model_family_id"] == model["family_id"]]
        return rows, _gpu_receipts([model])[0]

    artifact = exp.run(
        date="20260830",
        result_path=tmp_path / "complete.json",
        panel=_panel(),
        grammar_artifact=_grammar_artifact(),
        models=models,
        manifest=manifest,
        preconditions=_passing_preconditions(models),
        session_runner=session,
        clock=iter((4_000_000_000, 5_000_000_000)).__next__,
    )
    assert artifact["proof_transport_ab_completed"] is True

    monkeypatch.setattr(exp, "run", lambda *_args, **_kwargs: artifact)
    assert exp.main(["--date", "20260830"]) == 0
    invalid = deepcopy(artifact)
    invalid["claim_boundary"] = "syntax"
    monkeypatch.setattr(exp, "run", lambda *_args, **_kwargs: invalid)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.main(["--date", "20260830"])
