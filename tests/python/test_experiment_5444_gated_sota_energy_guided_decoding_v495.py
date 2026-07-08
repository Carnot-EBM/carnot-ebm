"""Tests for Exp5444 gated SOTA verifier-potential decoding.

Spec refs: REQ-SAFE-5444, SCENARIO-SAFE-5444.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5443_verifier_potential_prefix_fixture_v495 as exp5443
from carnot import experiment_5444_gated_sota_energy_guided_decoding_v495 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5444_gated_sota_energy_guided_decoding_v495.py -q"
)


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF-fixture")
    return path


def _runtime_receipt() -> dict[str, Any]:
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "llama_cpp_import_ok": True,
        "cuda_visible": True,
        "gpu_offload_supported": True,
        "n_gpu_layers": -1,
        "offload_evidence": True,
        "gpu_memory_delta_mb": 8192,
        "blocked_preconditions": [],
    }


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, _quantization: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolve


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
    raise AssertionError(f"unexpected fixture family in test: {family}")


def _grammar_only_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    family = fixture["constraint_family"]
    if fixture["fixture_id"] == "schema_extra_admin_override":
        return {
            "kind": "claim",
            "payload": {"answer": "yes"},
            "admin_override": True,
        }
    if family == "schema_only_trap":
        return {"kind": "claim", "payload": {"answer": "yes"}}
    if family == "semantic_contradiction":
        return {
            "subject": "door-17",
            "relation": "state",
            "object": "open",
            "negated_object": "open",
        }
    if family == "unreachable_tool_action":
        return {"tool": "cancel_order", "order_state": "paid", "lock_active": True}
    raise AssertionError(f"unexpected fixture family in test: {family}")


def _fake_generation(**kwargs: Any) -> dict[str, Any]:
    fixture = kwargs["fixture"]
    condition = kwargs["condition"]
    if condition == "unconstrained":
        text = "The answer is probably valid, but I will not emit JSON."
    elif condition == "grammar_only":
        text = _json_text(_grammar_only_payload(fixture))
    elif condition == "verifier_potential_guided":
        text = _json_text(_valid_payload(fixture))
    else:
        raise AssertionError(f"unexpected condition: {condition}")
    return {
        "output_text": text,
        "duration_s": 0.25,
        "generated_token_count": 12,
        "backend_details": {"mocked_live_runtime": True},
    }


def _complete_artifact(tmp_path: Path) -> dict[str, Any]:
    gguf = _minimal_gguf(tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf")
    paths = {"unsloth/gemma-4-26B-A4B-it-GGUF": gguf}
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        row_results_path=tmp_path / mod.ROW_RESULTS_RELATIVE_PATH,
        fixture_artifact=exp5443.build_artifact(tests_run=[TEST_COMMAND]),
        model_resolver=_resolver(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=_fake_generation,
        max_fixtures=4,
        tests_run=[TEST_COMMAND],
        write=True,
    )
    return artifact


def test_req_safe_5444_spec_declares_sota_guided_decoding_contract() -> None:
    """REQ-SAFE-5444: OpenSpec anchors the gated SOTA decoding artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5444") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5444",
        "SCENARIO-SAFE-5444",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.ROW_RESULTS_RELATIVE_PATH),
        "unconstrained decoding",
        "grammar-only constrained decoding",
        "verifier-potential guided prefix/particle decoding",
        "guided accepted rate minus the respective baseline rate",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_HF_IDS:
        assert hf_id in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5444_blocks_before_generation_when_fixture_gate_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5444: failed Exp5443 gate emits blocked artifact without generation."""

    calls: list[str] = []
    fixture_artifact = exp5443.build_artifact(tests_run=[TEST_COMMAND])
    fixture_artifact["verifier_potential_fixture_ready"] = False

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        row_results_path=tmp_path / "blocked_rows.jsonl",
        fixture_artifact=fixture_artifact,
        model_resolver=lambda _hf_id, _quantization: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=lambda **kwargs: calls.append(kwargs["condition"]) or {},
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert calls == []
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"] is True
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["verifier_guided_decoding_ready"] is False
    assert artifact["fixture_count"] == 0
    assert artifact["row_results_path"] == str(tmp_path / "blocked_rows.jsonl")
    assert not (tmp_path / "blocked_rows.jsonl").exists()
    mod.validate_artifact(artifact)


def test_scenario_safe_5444_complete_run_writes_row_level_exact_evidence(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5444: complete pilot compares three conditions under exact authority."""

    artifact = _complete_artifact(tmp_path)
    rows_path = Path(artifact["row_results_path"])
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"] == "llama_cpp_python_cuda_gguf"
    assert artifact["fixture_count"] == 4
    assert artifact["condition_names"] == list(mod.CONDITION_NAMES)
    assert artifact["exact_final_authority"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_guided_decoding_ready"] is True
    assert len(rows) == artifact["fixture_count"] * len(mod.CONDITION_NAMES)

    by_condition = artifact["condition_metrics"]
    assert by_condition["unconstrained"]["accepted_validity_rate"] == pytest.approx(0.0)
    assert by_condition["grammar_only"]["accepted_validity_rate"] == pytest.approx(0.25)
    assert by_condition["verifier_potential_guided"]["accepted_validity_rate"] == pytest.approx(1.0)
    assert artifact["guided_validity_delta_vs_unconstrained"] == pytest.approx(1.0)
    assert artifact["guided_validity_delta_vs_grammar_only"] == pytest.approx(0.75)
    assert artifact["guided_validity_delta_vs_unconstrained"] != pytest.approx(
        by_condition["unconstrained"]["accepted_validity_rate"]
    )
    assert artifact["semantic_false_accept_rate"] > 0.0
    assert artifact["action_unreachability_rate"] > 0.0
    assert artifact["abstention_rate"] > 0.0
    assert artifact["unsafe_false_accept_rate"] > 0.0
    assert artifact["reward_evaluation_budget"]["total_reward_evaluations"] > len(rows)

    for row in rows:
        assert row["model_hf_id"] in mod.MANDATED_HF_IDS
        assert row["model_path"].endswith(".gguf")
        assert row["runtime_backend"] == artifact["runtime_backend"]
        assert row["n_gpu_layers"] == -1
        assert row["prompt_hash"]
        assert row["token_budget"] == mod.DEFAULT_TOKEN_BUDGET
        assert row["exact_final_verdict"]["authority"] == "exact_final_verifier"
        assert row["final_authority_bypassed"] is False


def test_req_safe_5444_validation_rejects_delta_final_authority_and_model_drift(
    tmp_path: Path,
) -> None:
    """REQ-SAFE-5444: validation fails for copied deltas, bypassed authority, or missing specs."""

    artifact = _complete_artifact(tmp_path)

    bad_delta = deepcopy(artifact)
    bad_delta["guided_validity_delta_vs_unconstrained"] = bad_delta["condition_metrics"][
        "unconstrained"
    ]["accepted_validity_rate"]
    with pytest.raises(ValueError, match="guided_validity_delta_vs_unconstrained"):
        mod.validate_artifact(bad_delta)

    bad_authority = deepcopy(artifact)
    bad_authority["exact_final_authority"] = False
    with pytest.raises(ValueError, match="exact_final_authority"):
        mod.validate_artifact(bad_authority)

    row_tampered = deepcopy(artifact)
    first_row = row_tampered["row_results"][0]
    first_row["exact_final_verdict"]["authority"] = "model_self_verdict"
    first_row["final_authority_bypassed"] = True
    with pytest.raises(ValueError, match="exact final authority"):
        mod.validate_artifact(row_tampered)

    missing_model = deepcopy(artifact)
    missing_model["model_specs"] = [
        spec
        for spec in missing_model["model_specs"]
        if spec["hf_id"] != "unsloth/Qwen3.6-35B-A3B-GGUF"
    ]
    with pytest.raises(ValueError, match="mandated model_specs"):
        mod.validate_artifact(missing_model)

    bad_rows_path = deepcopy(artifact)
    bad_rows_path["row_results_path"] = str(tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="row_results_path"):
        mod.validate_artifact(bad_rows_path)


def test_req_safe_5444_metric_independence_recomputes_from_rows(tmp_path: Path) -> None:
    """REQ-SAFE-5444: aggregate metrics are recomputed from row predicates."""

    artifact = _complete_artifact(tmp_path)
    metrics = mod.derive_metrics(artifact["row_results"])

    assert metrics["condition_metrics"] == artifact["condition_metrics"]
    assert metrics["guided_validity_delta_vs_grammar_only"] == pytest.approx(0.75)
    assert metrics["metric_independence_checks_passed"] is True

    bad_semantic = deepcopy(artifact)
    bad_semantic["semantic_false_accept_rate"] = 0.0
    with pytest.raises(ValueError, match="semantic_false_accept_rate"):
        mod.validate_artifact(bad_semantic)

    bad_budget = deepcopy(artifact)
    bad_budget["reward_evaluation_budget"]["total_reward_evaluations"] += 1
    with pytest.raises(ValueError, match="reward_evaluation_budget"):
        mod.validate_artifact(bad_budget)

    bad_ready = deepcopy(artifact)
    bad_ready["metric_independence_checks_passed"] = False
    with pytest.raises(ValueError, match="metric_independence_checks_passed"):
        mod.validate_artifact(bad_ready)


def test_req_safe_5444_default_runner_and_write_false_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAFE-5444: default runner load receipts gate complete and blocked paths."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf")
    kwargs = {
        "root": tmp_path,
        "artifact_path": tmp_path / "default.json",
        "row_results_path": tmp_path / "default_rows.jsonl",
        "fixture_artifact": exp5443.build_artifact(tests_run=[TEST_COMMAND]),
        "model_resolver": _resolver({"unsloth/gemma-4-26B-A4B-it-GGUF": gguf}),
        "runtime_probe": lambda **_kwargs: _runtime_receipt(),
        "max_fixtures": 1,
        "tests_run": [TEST_COMMAND],
    }

    class FakeLiveRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.load_receipt = {"offload_evidence": True, "gpu_memory_delta_mb": 1024}

        def __call__(self, **call_kwargs: Any) -> dict[str, Any]:
            fixture = call_kwargs["fixture"]
            return {
                "output_text": _json_text(_valid_payload(fixture)),
                "duration_s": 0.1,
                "generated_token_count": 5,
            }

    monkeypatch.setattr(mod, "LlamaCppGenerationRunner", FakeLiveRunner)
    artifact = mod.run(**kwargs, write=True)
    assert artifact["verifier_guided_decoding_ready"] is True

    # Existing row JSONL lets write=False still validate row_results_path.
    second = mod.run(**kwargs, write=False)
    assert second["verifier_guided_decoding_ready"] is True

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


def test_req_safe_5444_defensive_validation_and_helper_branches(tmp_path: Path) -> None:
    """REQ-SAFE-5444: defensive helpers fail closed on malformed inputs."""

    artifact = _complete_artifact(tmp_path)
    validation_cases: list[tuple[str, Any, str]] = [
        ("field_principles", {}, "field_principles"),
        ("preconditions_checked", "yes", "preconditions_checked"),
        ("model_specs", "bad", "model_specs"),
        ("headline_required_any_of", [], "headline_required_any_of"),
        ("condition_names", [], "condition_names"),
        ("gpu_offload_verified", "yes", "gpu_offload_verified"),
        ("runtime_backend", 7, "runtime_backend"),
        ("fixture_count", -1, "fixture_count"),
        ("honest_verdict", "done\n", "honest_verdict"),
        ("research_conductor_modified", True, "research_conductor.py"),
        ("row_results", "bad", "row_results"),
        ("row_results_path", "", "row_results_path"),
    ]
    for field, value, expected in validation_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    del missing["runtime_backend"]
    assert "missing required fields" in "; ".join(mod.artifact_schema_errors(missing))

    aggregate_cases: list[tuple[str, Any, str]] = [
        ("accepted_validity_rate", 0.0, "accepted_validity_rate"),
        ("condition_metrics", {}, "condition_metrics"),
        ("metric_details", {}, "metric_details"),
        ("constraint_family_counts", {}, "constraint_family_counts"),
        ("row_checksums", [], "row_checksums"),
        ("fixture_count", artifact["fixture_count"] + 1, "fixture_count"),
    ]
    for field, value, expected in aggregate_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    ready_status = deepcopy(artifact)
    ready_status["status"] = "blocked"
    with pytest.raises(ValueError, match="complete status"):
        mod.validate_artifact(ready_status)

    ready_substrate = deepcopy(artifact)
    ready_substrate["inference_substrate"] = "blocked_preconditions_no_live_llm"
    with pytest.raises(ValueError, match="live_llm_inference"):
        mod.validate_artifact(ready_substrate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["row_results"][0]["row_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="row checksum"):
        mod.validate_artifact(bad_checksum)

    bad_row_model = deepcopy(artifact)
    bad_row_model["row_results"][0]["model_hf_id"] = "legacy/not-headline"
    with pytest.raises(ValueError, match="row model_hf_id"):
        mod.validate_artifact(bad_row_model)

    bad_row_path = deepcopy(artifact)
    bad_row_path["row_results"][0]["model_path"] = "/tmp/model.bin"
    with pytest.raises(ValueError, match="row model_path"):
        mod.validate_artifact(bad_row_path)

    unreadable_rows = deepcopy(artifact)
    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("{not json}\n", encoding="utf-8")
    unreadable_rows["row_results_path"] = str(invalid_jsonl)
    with pytest.raises(ValueError, match="row_results_path is unreadable"):
        mod.validate_artifact(unreadable_rows)

    mismatched_rows = deepcopy(artifact)
    mismatch_path = tmp_path / "mismatch.jsonl"
    mismatch_path.write_text(json.dumps({"row": "wrong"}) + "\n", encoding="utf-8")
    mismatched_rows["row_results_path"] = str(mismatch_path)
    with pytest.raises(ValueError, match="contents must match"):
        mod.validate_artifact(mismatched_rows)

    assert mod.extract_json_object("prefix {bad json") == (
        None,
        "JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
    )
    assert mod.extract_json_object("no object here") == (None, "no_json_object")
    with pytest.raises(ValueError, match="unknown condition"):
        mod.build_prompt(artifact["row_results"][0], condition="bad")

    fixture_rows = exp5443.build_fixture_rows()
    assert mod.select_fixture_rows({"fixture_rows": "bad"}) == []
    assert (
        mod.select_fixture_rows(
            {"fixture_rows": ["bad", {}, {"exact_final_verdict": {"verified": False}}]}
        )
        == []
    )
    assert mod._best_guided_prefix({"prefixes": []}) is None  # noqa: SLF001
    for row in fixture_rows:
        assert mod._family_instruction(row, row["constraint_family"])  # noqa: SLF001
    assert mod._family_instruction({}, "unknown") == "Satisfy the deterministic row constraints."  # noqa: SLF001

    preconditions = mod.evaluate_preconditions(
        fixture_payload={},
        model_specs=[],
        selected_model=None,
        runtime_receipt={"cuda_visible": False, "offload_evidence": False},
    )
    assert set(preconditions["blocked_preconditions"]) >= {
        "exp5443_verifier_potential_fixture_not_ready",
        "mandated_model_specs_missing",
        "no_mandated_local_gguf_model_path",
        "cuda_not_visible",
        "gpu_offload_evidence_missing",
    }

    assert mod._honest_verdict(False, {}, {"metric_independence_checks_passed": False}) == (  # noqa: SLF001
        "blocked: metric independence checks failed"
    )
    assert mod._honest_verdict(False, {}, {"metric_independence_checks_passed": True}) == (  # noqa: SLF001
        "blocked: verifier-guided decoding readiness checks failed"
    )
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001
    assert mod._normalise_tests_run([])[0]["outcome"] == "not_recorded"  # noqa: SLF001
    assert mod._normalise_test_run({"command": "cmd", "outcome": "passed"}) == {  # noqa: SLF001
        "command": "cmd",
        "outcome": "passed",
    }
    assert mod._destination(tmp_path, None, Path("x.json")) == tmp_path / "x.json"  # noqa: SLF001
    assert mod._destination(tmp_path, Path("rel.json"), Path("x.json")) == tmp_path / "rel.json"  # noqa: SLF001
    assert mod._float_close("bad", 1.0) is False  # noqa: SLF001
    assert mod._max_memory_delta_mb(  # noqa: SLF001
        [{"index": 0, "memory_used_mb": 10}],
        [{"index": 0, "memory_used_mb": 25}, {"index": 1, "memory_used_mb": 4}],
    ) == 15
