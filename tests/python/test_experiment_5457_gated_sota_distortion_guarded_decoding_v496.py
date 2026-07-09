"""Tests for Exp5457 distortion-guarded SOTA guided decoding.

Spec refs: REQ-SAFE-5457, SCENARIO-SAFE-5457.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5443_verifier_potential_prefix_fixture_v495 as exp5443
from carnot import experiment_5457_gated_sota_distortion_guarded_decoding_v496 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5457_gated_sota_distortion_guarded_decoding_v496.py -q"
)


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF-fixture")
    return path


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        hf_id: _minimal_gguf(tmp_path / f"{hf_id.split('/')[-1]}.gguf")
        for hf_id in mod.MANDATED_HF_IDS
    }


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, _quantization: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolve


def _runtime_receipt(*, offload: bool = True) -> dict[str, Any]:
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "llama_cpp_import_ok": True,
        "cuda_visible": True,
        "gpu_offload_supported": True,
        "n_gpu_layers": -1,
        "offload_evidence": offload,
        "gpu_memory_delta_mb": 8192 if offload else 0,
        "blocked_preconditions": [] if offload else ["gpu_offload_evidence_missing"],
    }


def _fixture_artifact() -> dict[str, Any]:
    return exp5443.build_artifact(tests_run=[TEST_COMMAND])


def _corrigendum_clean() -> dict[str, Any]:
    return {"guided_decoding_corrigendum_clean": True, "honest_verdict": "complete: clean"}


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _valid_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    family = fixture["constraint_family"]
    if family == "schema_only_trap":
        return {"kind": "claim", "payload": {"answer": "yes"}}
    if family == "semantic_contradiction":
        return {
            "subject": "door-17",
            "relation": "state",
            "object": "open",
            "negated_object": "closed",
        }
    if family == "unreachable_tool_action":
        return {"tool": "cancel_order", "order_state": "paid", "lock_active": False}
    if family in {"factual_conflict_cocoa", "strict_constraint_distortion_guard"}:
        return {
            "claim": fixture["expected_claim"],
            "answer": fixture["expected_answer"],
            "evidence_ids": [fixture["supporting_evidence_ids"][0]],
        }
    raise AssertionError(f"unexpected fixture family in test: {family}")


def _fake_generation(**kwargs: Any) -> dict[str, Any]:
    fixture = kwargs["fixture"]
    condition = kwargs["condition"]
    if condition == "unconstrained":
        text = "No structured answer."
    elif condition == "grammar_lcd_only":
        text = _json_text(_valid_payload(fixture))
    elif condition == "verifier_potential_guided":
        text = _json_text(_valid_payload(fixture))
    else:
        raise AssertionError(f"unexpected condition: {condition}")
    return {
        "output_text": text,
        "duration_s": 0.2,
        "generated_token_count": 8,
        "backend_details": {"mocked_live_runtime": True},
    }


def _distorting_generation(**kwargs: Any) -> dict[str, Any]:
    fixture = kwargs["fixture"]
    condition = kwargs["condition"]
    if condition == "grammar_lcd_only" and fixture["constraint_family"] in {
        "factual_conflict_cocoa",
        "strict_constraint_distortion_guard",
    }:
        text = _json_text(
            {
                "claim": fixture["distorted_claim"],
                "answer": fixture["contradicted_answers"][0],
                "evidence_ids": [fixture["evidence_spans"][0]["id"]],
            }
        )
    elif condition == "unconstrained":
        text = "No structured answer."
    else:
        text = _json_text(_valid_payload(fixture))
    return {
        "output_text": text,
        "duration_s": 0.2,
        "generated_token_count": 8,
        "backend_details": {"mocked_live_runtime": True},
    }


def _complete_artifact(tmp_path: Path, generation_runner=_fake_generation) -> dict[str, Any]:
    return mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        row_results_path=tmp_path / mod.ROW_RESULTS_RELATIVE_PATH,
        claim_attribution_receipts_path=tmp_path / mod.CLAIM_ATTRIBUTION_RELATIVE_PATH,
        fixture_artifact=_fixture_artifact(),
        corrigendum_artifact=_corrigendum_clean(),
        model_resolver=_resolver(_paths(tmp_path)),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=generation_runner,
        max_base_fixtures=3,
        tests_run=[TEST_COMMAND],
        write=True,
    )


def test_req_safe_5457_spec_declares_distortion_guarded_contract() -> None:
    """REQ-SAFE-5457: OpenSpec anchors the SOTA distortion-guarded artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5457") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5457",
        "SCENARIO-SAFE-5457",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.ROW_RESULTS_RELATIVE_PATH),
        str(mod.CLAIM_ATTRIBUTION_RELATIVE_PATH),
        "strict-constraint distortion",
        "CoCoA-style context conflict",
        "chance_risk_bound",
        "lcd_bias_check_passed",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_HF_IDS:
        assert hf_id in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5457_blocks_before_generation_when_preconditions_fail(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5457: failed gates emit blocked artifact without generation."""

    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        row_results_path=tmp_path / "blocked_rows.jsonl",
        claim_attribution_receipts_path=tmp_path / "blocked_receipts.jsonl",
        fixture_artifact=_fixture_artifact(),
        corrigendum_artifact={"guided_decoding_corrigendum_clean": False},
        model_resolver=lambda _hf_id, _quantization: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(offload=False),
        generation_runner=lambda **kwargs: calls.append(kwargs["condition"]) or {},
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert calls == []
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is False
    assert artifact["verifier_guided_decoding_ready"] is False
    assert artifact["fixture_count"] == 0
    assert artifact["honest_verdict"].startswith("blocked:")
    assert not (tmp_path / "blocked_rows.jsonl").exists()
    assert not (tmp_path / "blocked_receipts.jsonl").exists()
    mod.validate_artifact(artifact)


def test_scenario_safe_5457_complete_run_writes_rows_and_attribution_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5457: complete panel uses exact final authority and receipts."""

    artifact = _complete_artifact(tmp_path)
    rows = [
        json.loads(line)
        for line in Path(artifact["row_results_path"]).read_text(encoding="utf-8").splitlines()
    ]
    receipts = [
        json.loads(line)
        for line in Path(artifact["claim_attribution_receipts_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"] == "llama_cpp_python_cuda_gguf"
    assert artifact["condition_names"] == list(mod.CONDITION_NAMES)
    assert artifact["exact_final_authority"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_guided_decoding_ready"] is True
    assert artifact["metric_independence_checks_passed"] is True
    assert artifact["lcd_bias_check_passed"] is True
    assert artifact["chance_risk_bound"] < 0.5
    assert artifact["factual_distortion_rate"] == pytest.approx(0.0)
    assert artifact["semantic_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["guided_validity_delta_vs_unconstrained"] > 0.0
    assert artifact["guided_validity_delta_vs_lcd_only"] == pytest.approx(0.0)
    assert len(rows) == artifact["fixture_count"] * len(mod.CONDITION_NAMES)
    assert len(receipts) == len(rows)
    assert {row["constraint_family"] for row in rows}.issuperset(
        {"factual_conflict_cocoa", "strict_constraint_distortion_guard"}
    )

    for row in rows:
        assert row["model_hf_id"] in mod.MANDATED_HF_IDS
        assert row["model_path"].endswith(".gguf")
        assert row["runtime_backend"] == artifact["runtime_backend"]
        assert row["n_gpu_layers"] == -1
        assert row["prompt_hash"]
        assert row["token_budget"] == mod.DEFAULT_TOKEN_BUDGET
        assert row["acceptance_threshold"] == pytest.approx(mod.ACCEPTANCE_THRESHOLD)
        assert row["exact_final_verdict"]["authority"] == "exact_final_verifier"
        assert row["final_authority_bypassed"] is False
        assert row["claim_attribution_receipt_id"]

    for receipt in receipts:
        assert receipt["row_id"]
        assert receipt["receipt_checksum"] == mod.receipt_checksum(receipt)
        if receipt["constraint_family"] in {
            "factual_conflict_cocoa",
            "strict_constraint_distortion_guard",
        } and receipt["condition"] != "unconstrained":
            assert receipt["claim_attributed"] is True
            assert receipt["evidence_span_ids"]


def test_req_safe_5457_distortion_metrics_detect_lcd_bias(tmp_path: Path) -> None:
    """REQ-SAFE-5457: factual distortion and LCD bias derive from row evidence."""

    artifact = _complete_artifact(tmp_path, generation_runner=_distorting_generation)

    mod.validate_artifact(artifact)
    assert artifact["factual_distortion_rate"] > 0.0
    assert artifact["lcd_bias_check_passed"] is False
    assert artifact["verifier_guided_decoding_ready"] is False
    assert artifact["metric_details"]["lcd_bias_indicators"]["lcd_factual_distortion_count"] > 0
    assert artifact["metric_details"]["chance_risk_failure_count"] > 0


def test_req_safe_5457_validation_rejects_model_drift_cpu_headline_and_bad_authority(
    tmp_path: Path,
) -> None:
    """REQ-SAFE-5457: validation fails closed for model drift and bad readiness."""

    artifact = _complete_artifact(tmp_path)

    missing_model = deepcopy(artifact)
    missing_model["model_specs"] = [
        spec
        for spec in missing_model["model_specs"]
        if spec["hf_id"] != "unsloth/Qwen3.6-35B-A3B-GGUF"
    ]
    with pytest.raises(ValueError, match="mandated model_specs"):
        mod.validate_artifact(missing_model)

    cpu_headline = deepcopy(artifact)
    for spec in cpu_headline["model_specs"]:
        if spec["ran_headline"]:
            spec["gpu_offload_verified"] = False
    with pytest.raises(ValueError, match="CPU-only headline"):
        mod.validate_artifact(cpu_headline)

    legacy_headline = deepcopy(artifact)
    legacy_headline["model_specs"][0]["hf_id"] = "google/gemma-4-E4B-it"
    legacy_headline["model_specs"][0]["ran_headline"] = True
    legacy_headline["model_specs"][0]["legacy_smoke_only"] = True
    with pytest.raises(ValueError, match="legacy smoke"):
        mod.validate_artifact(legacy_headline)

    row_tampered = deepcopy(artifact)
    row_tampered["row_results"][0]["exact_final_verdict"]["authority"] = "model_self_verdict"
    row_tampered["row_results"][0]["final_authority_bypassed"] = True
    with pytest.raises(ValueError, match="exact final authority"):
        mod.validate_artifact(row_tampered)

    bad_receipt = deepcopy(artifact)
    bad_receipt["claim_attribution_receipts"][0]["row_id"] = "wrong"
    with pytest.raises(ValueError, match="claim attribution receipts"):
        mod.validate_artifact(bad_receipt)

    self_validating = deepcopy(artifact)
    self_validating["metric_dependency_graph"]["verifier_guided_decoding_ready"] = [
        "verifier_guided_decoding_ready"
    ]
    with pytest.raises(ValueError, match="self-validating readiness"):
        mod.validate_artifact(self_validating)


def test_req_safe_5457_defensive_helpers_and_write_false_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAFE-5457: helper branches stay fail-closed and covered."""

    kwargs: dict[str, Any] = {
        "root": tmp_path,
        "artifact_path": tmp_path / "default.json",
        "row_results_path": tmp_path / "default_rows.jsonl",
        "claim_attribution_receipts_path": tmp_path / "default_receipts.jsonl",
        "fixture_artifact": _fixture_artifact(),
        "corrigendum_artifact": _corrigendum_clean(),
        "model_resolver": _resolver(_paths(tmp_path)),
        "runtime_probe": lambda **_kwargs: _runtime_receipt(),
        "max_base_fixtures": 1,
        "tests_run": [TEST_COMMAND],
    }

    class FakeLiveRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.load_receipt = {"offload_evidence": True, "gpu_memory_delta_mb": 1024}

        def __call__(self, **call_kwargs: Any) -> dict[str, Any]:
            return _fake_generation(**call_kwargs)

    monkeypatch.setattr(mod, "LlamaCppGenerationRunner", FakeLiveRunner)
    artifact = mod.run(**kwargs, write=True)
    assert artifact["gpu_offload_verified"] is True
    assert mod.run(**kwargs, write=False)["metric_independence_checks_passed"] is True

    class CpuOnlyRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.load_receipt = {"offload_evidence": False, "gpu_memory_delta_mb": 0}

    monkeypatch.setattr(mod, "LlamaCppGenerationRunner", CpuOnlyRunner)
    cpu_only = mod.run(**(kwargs | {"artifact_path": tmp_path / "cpu.json"}), write=True)
    assert cpu_only["honest_verdict"].startswith("blocked:")
    assert "gpu_offload_not_observed_after_load" in cpu_only["precondition_details"][
        "blocked_preconditions"
    ]

    class RaisingRunner:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    monkeypatch.setattr(mod, "LlamaCppGenerationRunner", RaisingRunner)
    failed_load = mod.run(**(kwargs | {"artifact_path": tmp_path / "raise.json"}), write=True)
    assert "llama_cpp_model_load_failed" in failed_load["honest_verdict"]

    assert mod.select_panel_fixtures({"fixture_rows": "bad"}) == []
    assert mod._normalise_tests_run([])[0]["outcome"] == "not_recorded"  # noqa: SLF001
    assert mod._normalise_test_run({"command": "cmd", "outcome": "passed"}) == {  # noqa: SLF001
        "command": "cmd",
        "outcome": "passed",
    }
    assert mod._destination(tmp_path, None, Path("x.json")) == tmp_path / "x.json"  # noqa: SLF001
    assert mod._float_close("bad", 1.0) is False  # noqa: SLF001
    assert mod.wilson_upper_bound(0, 0) == pytest.approx(1.0)

    broken = deepcopy(artifact)
    broken["row_results_path"] = ""
    assert "row_results_path" in "; ".join(mod.artifact_schema_errors(broken))


def test_req_safe_5457_fail_closed_validation_branches(tmp_path: Path) -> None:
    """REQ-SAFE-5457: defensive validation branches report concrete blockers."""

    artifact = _complete_artifact(tmp_path)

    preconditions = mod.evaluate_preconditions(
        fixture_payload={},
        corrigendum_payload={},
        model_specs=[],
        selected_model=None,
        runtime_receipt={"cuda_visible": False, "offload_evidence": False},
    )
    assert set(preconditions["blocked_preconditions"]) >= {
        "exp5456_guided_decoding_corrigendum_not_clean",
        "exp5443_verifier_potential_fixture_not_ready",
        "mandated_model_specs_missing",
        "non_empty_mandated_model_paths_missing",
        "no_mandated_local_gguf_model_path",
        "cuda_not_visible",
        "gpu_offload_evidence_missing",
    }

    assert mod.select_panel_fixtures(
        {
            "fixture_rows": [
                "bad",
                {"exact_final_verdict": {"verified": False}},
                _fixture_artifact()["fixture_rows"][0],
            ]
        },
        max_base_fixtures=2,
    )
    with pytest.raises(ValueError, match="unknown condition"):
        mod.build_prompt(artifact["row_results"][0], condition="bad")

    schema_cases: list[tuple[str, Any, str]] = [
        ("field_principles", {}, "field_principles"),
        ("preconditions_checked", "yes", "preconditions_checked"),
        ("model_specs", "bad", "model_specs"),
        ("headline_required_any_of", [], "headline_required_any_of"),
        ("condition_names", [], "condition_names"),
        ("exact_final_authority", False, "exact_final_authority"),
        ("gpu_offload_verified", "yes", "gpu_offload_verified"),
        ("lcd_bias_check_passed", "yes", "lcd_bias_check_passed"),
        ("metric_independence_checks_passed", "yes", "metric_independence_checks_passed"),
        ("verifier_guided_decoding_ready", "yes", "verifier_guided_decoding_ready"),
        ("runtime_backend", 7, "runtime_backend"),
        ("fixture_count", -1, "fixture_count"),
        ("honest_verdict", "done\n", "honest_verdict"),
        ("research_conductor_modified", True, "research_conductor.py"),
        ("row_results", "bad", "row_results"),
        ("claim_attribution_receipts", "bad", "claim_attribution_receipts"),
    ]
    for field, value, expected in schema_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    missing = deepcopy(artifact)
    del missing["runtime_backend"]
    assert "missing required fields" in "; ".join(mod.artifact_schema_errors(missing))

    ready_cases: list[tuple[str, Any, str]] = [
        ("status", "blocked", "complete status"),
        ("inference_substrate", "blocked_preconditions_no_live_llm", "live_llm_inference"),
        ("gpu_offload_verified", False, "gpu_offload_verified"),
        ("lcd_bias_check_passed", False, "lcd_bias_check_passed"),
        ("chance_risk_bound", 1.0, "bounded chance risk"),
        ("row_results", [], "row and attribution evidence"),
    ]
    for field, value, expected in ready_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    no_headline = deepcopy(artifact)
    for spec in no_headline["model_specs"]:
        spec["ran_headline"] = False
    assert "at least one mandated model ran" in "; ".join(mod.artifact_schema_errors(no_headline))

    aggregate_cases: list[tuple[str, Any, str]] = [
        ("chance_risk_bound", 1.0, "chance_risk_bound"),
        ("factual_distortion_rate", 1.0, "factual_distortion_rate"),
        ("semantic_false_accept_rate", 1.0, "semantic_false_accept_rate"),
        ("guided_validity_delta_vs_unconstrained", -99.0, "guided_validity_delta_vs_unconstrained"),
        ("guided_validity_delta_vs_lcd_only", -99.0, "guided_validity_delta_vs_lcd_only"),
        ("accepted_validity_rate", -1.0, "accepted_validity_rate"),
        ("abstention_rate", -1.0, "abstention_rate"),
        ("condition_metrics", {}, "condition_metrics"),
        ("metric_details", {}, "metric_details"),
        ("constraint_family_counts", {}, "constraint_family_counts"),
        ("reward_evaluation_budget", {}, "reward_evaluation_budget"),
        ("row_checksums", [], "row_checksums"),
        ("receipt_checksums", [], "receipt_checksums"),
        ("fixture_count", artifact["fixture_count"] + 1, "fixture_count"),
    ]
    for field, value, expected in aggregate_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    row_cases: list[tuple[str, Any, str]] = [
        ("model_hf_id", "legacy/not-headline", "row model_hf_id"),
        ("model_path", "/tmp/model.bin", "row model_path"),
        ("gpu_offload_evidence", "yes", "row gpu_offload_evidence"),
        ("condition", "unknown", "row condition"),
    ]
    for field, value, expected in row_cases:
        bad = deepcopy(artifact)
        bad["row_results"][0][field] = value
        assert expected in "; ".join(mod.artifact_schema_errors(bad))

    bad_checksum = deepcopy(artifact)
    bad_checksum["row_results"][0]["row_checksum"] = "0" * 64
    assert "row checksum" in "; ".join(mod.artifact_schema_errors(bad_checksum))

    bad_receipt_count = deepcopy(artifact)
    bad_receipt_count["claim_attribution_receipts"] = bad_receipt_count[
        "claim_attribution_receipts"
    ][:-1]
    assert "claim attribution receipts must match row count" in "; ".join(
        mod.artifact_schema_errors(bad_receipt_count)
    )

    bad_receipt_checksum = deepcopy(artifact)
    bad_receipt_checksum["claim_attribution_receipts"][0]["receipt_checksum"] = "0" * 64
    assert "valid checksums" in "; ".join(mod.artifact_schema_errors(bad_receipt_checksum))

    bad_receipt_ids = deepcopy(artifact)
    for index, receipt in enumerate(bad_receipt_ids["claim_attribution_receipts"]):
        row = bad_receipt_ids["row_results"][index]
        if receipt["constraint_family"] in mod.FACTUAL_FAMILIES and row["parse_status"] == "parsed":
            receipt["evidence_span_ids"] = "bad"
            break
    assert "evidence span ids" in "; ".join(mod.artifact_schema_errors(bad_receipt_ids))

    missing_rows = deepcopy(artifact)
    missing_rows["row_results_path"] = str(tmp_path / "missing.jsonl")
    assert "row_results_path must point" in "; ".join(mod.artifact_schema_errors(missing_rows))

    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("{bad}\n", encoding="utf-8")
    unreadable = deepcopy(artifact)
    unreadable["row_results_path"] = str(invalid_jsonl)
    assert "row_results_path is unreadable" in "; ".join(mod.artifact_schema_errors(unreadable))

    mismatch = deepcopy(artifact)
    mismatch_path = tmp_path / "mismatch.jsonl"
    mismatch_path.write_text(json.dumps({"row": "wrong"}) + "\n", encoding="utf-8")
    mismatch["claim_attribution_receipts_path"] = str(mismatch_path)
    assert "claim_attribution_receipts_path contents" in "; ".join(
        mod.artifact_schema_errors(mismatch)
    )

    bad_graph = deepcopy(artifact)
    bad_graph["metric_dependency_graph"] = "bad"
    assert "metric_dependency_graph must be a dict" in "; ".join(
        mod.artifact_schema_errors(bad_graph)
    )

    missing_graph_deps = deepcopy(artifact)
    missing_graph_deps["metric_dependency_graph"] = {}
    assert "readiness dependencies" in "; ".join(mod.artifact_schema_errors(missing_graph_deps))

    assert (
        "metric_independence_failed"
        in mod._honest_verdict(  # noqa: SLF001
            False,
            artifact["row_results"],
            {},
            {
                "lcd_bias_check_passed": True,
                "chance_risk_bound": 0.0,
                "metric_independence_checks_passed": False,
            },
        )
    )
    assert mod._list_of_mappings("bad") == []  # noqa: SLF001
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001
    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("7", encoding="utf-8")
    assert mod._read_json(scalar_json) == {}  # noqa: SLF001
