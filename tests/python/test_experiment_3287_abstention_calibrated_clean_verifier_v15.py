"""Tests for Exp 3287 abstention-calibrated clean verifier rerun v15.

Spec refs: REQ-VERIFY-3287, SCENARIO-VERIFY-3287.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import abstention_calibrated_clean_verifier_v15 as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "abstention_calibrated_clean_verifier_v15_ready",
    "clean_verifier_rerun_ready",
    "repair_gate_input_clean_enough",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "n_eval",
    "exact_checkable_row_count",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "coverage_rate",
    "abstention_reason_counts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_exp3286(root: Path, *, identified: bool = True) -> None:
    _write_json(
        root,
        mod.EXP3286_REL_PATH,
        {
            "experiment_id": "exp3286",
            "abstention_root_cause_identified": identified,
            "dominant_root_cause": (
                "model_output_parser_contract_mismatch" if identified else "unknown"
            ),
            "calibrated_rerun_plan": {
                "experiment_id": "exp3287",
                "root_cause_to_address": "model_output_parser_contract_mismatch",
                "acceptance_criteria": {
                    "minimum_decision_coverage": 0.5,
                    "target_false_accept_rate": 0.0,
                    "target_max_abstention_rate": 0.5,
                },
            },
            "honest_verdict": "complete: clean verifier abstention root cause identified",
        },
    )


def _write_exp3268(root: Path, *, eligible: bool = True) -> None:
    _write_json(
        root,
        mod.EXP3268_REL_PATH,
        {
            "experiment_id": "exp3268",
            "clean_sota_receipt_eligible": eligible,
            "model_specs": {"runtime": "llama_cpp"},
        },
    )


def _write_context_fixture(root: Path, *, fixture_count: int = 10) -> None:
    path = root / mod.CONTEXT_FIXTURE_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(fixture_count):
        rows.append(
            {
                "context": f"For this fixture only, item{idx} means value{idx}.",
                "exact_checker_type": "exact_alias_string",
                "expected_answer": f"value{idx}",
                "family": "symbolic_aliases",
                "fixture_id": f"ctx-{idx:03d}",
                "minimal_counterexample": {"candidate_answer": f"prior{idx}"},
                "prior_bait_answer": f"prior{idx}",
                "question": f"According to the fixture, what does item{idx} mean?",
            }
        )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _ok_nvidia() -> dict[str, Any]:
    return {
        "name": "nvidia_smi",
        "passed": True,
        "returncode": 0,
        "gpu_count": 2,
        "gpu_mem_used_mib": 4,
    }


def _ok_python_cuda() -> dict[str, Any]:
    return {
        "name": "selected_python_cuda",
        "passed": True,
        "cuda_available": True,
        "cuda_device_count": 2,
        "llama_cpp_import_ok": True,
        "llama_cpp_supports_gpu_offload": True,
    }


def _patch_single_cached_model(monkeypatch: pytest.MonkeyPatch, model_path: Path) -> None:
    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices=(0, 1): None)

    def fake_resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        return str(model_path) if hf_id == GEMMA26 else None

    monkeypatch.setattr(mod, "resolve_cached_gguf", fake_resolve)


def _model_path(root: Path, name: str = "gemma26.gguf") -> Path:
    path = root / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    return path


def _runner(decision_for_row: Any, *, gpu_mem: int = 7777) -> mod.ModelRunner:
    def run(
        rows: list[dict[str, Any]],
        model: dict[str, Any],
        random_seed: int,
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        assert random_seed == mod.DEFAULT_RANDOM_SEED
        assert policy["strict_leading_token"] is True
        out_rows = []
        for row in rows:
            decision = decision_for_row(row)
            out_rows.append(
                {
                    "row_id": row["row_id"],
                    "model_id": model["model_id"],
                    "model_path": model["model_path"],
                    "output_text": decision,
                    "decision": decision,
                    "prompt_hash": f"prompt-{row['row_id']}",
                    "transcript_hash": f"tx-{row['row_id']}",
                    "token_counts": {
                        "prompt_tokens": 10,
                        "completion_tokens": 1,
                        "total_tokens": 11,
                    },
                }
            )
        return {"rows": out_rows, "gpu_mem_used_mib": gpu_mem}

    return run


def _correct_decision(row: dict[str, Any]) -> str:
    return "ACCEPT" if row["expected_decision"] == "accept" else "REJECT"


def test_req_verify_3287_spec_anchor_declares_v15_artifact() -> None:
    """REQ-VERIFY-3287: OpenSpec declares the v15 clean rerun schema."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3287" in spec
    assert "SCENARIO-VERIFY-3287" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_verify_3287_ready_with_single_cached_mandated_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3287: zero false accepts plus coverage opens the gate."""

    _write_exp3286(tmp_path)
    _write_exp3268(tmp_path)
    _write_context_fixture(tmp_path, fixture_count=10)
    _patch_single_cached_model(monkeypatch, _model_path(tmp_path))

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=_runner(_correct_decision),
        started_s=10.0,
        now_s=14.5,
        tests_run=["SCENARIO-VERIFY-3287"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["abstention_calibrated_clean_verifier_v15_ready"] is True
    assert artifact["clean_verifier_rerun_ready"] is True
    assert artifact["repair_gate_input_clean_enough"] is True
    assert artifact["exact_checkable_row_count"] == 20
    assert artifact["n_eval"] == 20
    assert artifact["false_accept_rate"] == 0.0
    assert artifact["false_reject_rate"] == 0.0
    assert artifact["abstention_rate"] == 0.0
    assert artifact["coverage_rate"] == 1.0
    assert artifact["abstention_reason_counts"] == {}
    assert artifact["per_class_reason_counts"] == {
        "accept": {"accepted_decision": 10},
        "reject": {"rejected_decision": 10},
    }
    assert artifact["model_specs"]["cached_sota_pair_attempted"] is True
    assert artifact["model_specs"]["cached_sota_pair_available"] is False
    assert [model["model_id"] for model in artifact["models_used"]] == [GEMMA26]
    assert {model["model_id"] for model in artifact["missing_model_specs"]} == {QWEN, GEMMA31}
    assert {row["name"] for row in artifact["preconditions_checked"]} >= {
        "nvidia_smi",
        "selected_python_cuda",
        "exp3286_calibrated_rerun_plan",
        "mandated_sota_gguf_cache",
        "exact_row_fixture_availability",
    }
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3287"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=_runner(_correct_decision),
        started_s=1.0,
        now_s=2.0,
        tests_run=["writer"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert output == tmp_path / "results/out.json"
    assert saved["duration_s"] == pytest.approx(1.0)
    assert saved["tests_run"] == ["writer"]

    with pytest.raises(ValueError, match="zero false accepts"):
        mod.validate_artifact(artifact | {"false_accept_rate": 0.1})
    with pytest.raises(ValueError, match="cannot abstain on every row"):
        mod.validate_artifact(artifact | {"abstention_rate": 1.0})
    with pytest.raises(ValueError, match="non-trivial coverage"):
        mod.validate_artifact(artifact | {"coverage_rate": 0.0})
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(artifact | {"models_used": []})


def test_req_verify_3287_uses_cached_sota_pair_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3287: cached_sota_pair drives model selection when possible."""

    _write_exp3286(tmp_path)
    _write_exp3268(tmp_path)
    _write_context_fixture(tmp_path, fixture_count=2)
    qwen_path = _model_path(tmp_path, "qwen.gguf")
    gemma_path = _model_path(tmp_path, "gemma31.gguf")
    monkeypatch.setattr(
        mod,
        "cached_sota_pair",
        lambda gpu_indices=(0, 1): [
            {"name": "Qwen", "hf_id": QWEN, "gpu": gpu_indices[0], "model_path": str(qwen_path)},
            {
                "name": "Gemma31",
                "hf_id": GEMMA31,
                "gpu": gpu_indices[1],
                "model_path": str(gemma_path),
            },
        ],
    )
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args, **_kwargs: None)
    calls: list[str] = []

    def runner(
        rows: list[dict[str, Any]],
        model: dict[str, Any],
        random_seed: int,
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(model["model_id"])
        return _runner(_correct_decision)(rows, model, random_seed, policy)

    artifact = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=runner,
    )

    assert calls == [QWEN, GEMMA31]
    assert artifact["model_specs"]["cached_sota_pair_available"] is True
    assert [model["model_id"] for model in artifact["models_used"]] == [QWEN, GEMMA31]
    assert {model["model_id"] for model in artifact["missing_model_specs"]} == {GEMMA26}
    assert artifact["exact_checkable_row_count"] == 4
    assert artifact["n_eval"] == 8
    assert artifact["repair_gate_input_clean_enough"] is True


def test_req_verify_3287_false_accept_and_abstain_all_block_repair_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3287: false accepts or abstain-all outputs never open repair."""

    _write_exp3286(tmp_path)
    _write_exp3268(tmp_path)
    _write_context_fixture(tmp_path, fixture_count=10)
    _patch_single_cached_model(monkeypatch, _model_path(tmp_path))

    false_accepting = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=_runner(lambda _row: "ACCEPT"),
    )
    assert false_accepting["repair_gate_input_clean_enough"] is False
    assert false_accepting["false_accept_rate"] == 1.0
    assert "false_accept_count_nonzero" in false_accepting["gate_reasons"]

    abstain_all = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=_runner(lambda _row: "not a leading verifier token"),
    )
    assert abstain_all["repair_gate_input_clean_enough"] is False
    assert abstain_all["abstention_rate"] == 1.0
    assert abstain_all["coverage_rate"] == 0.0
    assert abstain_all["abstention_reason_counts"] == {"model_output_unparseable": 20}
    assert "abstention_rate_is_1.0" in abstain_all["gate_reasons"]


def test_req_verify_3287_preconditions_helpers_and_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3287: preconditions and validators fail closed."""

    _write_exp3286(tmp_path, identified=False)
    _write_exp3268(tmp_path)
    _write_context_fixture(tmp_path, fixture_count=1)
    _patch_single_cached_model(monkeypatch, _model_path(tmp_path))
    calls = 0

    def counting_runner(
        rows: list[dict[str, Any]],
        model: dict[str, Any],
        random_seed: int,
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _runner(_correct_decision)(rows, model, random_seed, policy)

    blocked = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=counting_runner,
    )
    assert blocked["n_eval"] == 0
    assert blocked["repair_gate_input_clean_enough"] is False
    assert "exp3286_calibrated_rerun_plan_not_ready" in blocked["gate_reasons"]
    assert calls == 0

    _write_exp3286(tmp_path, identified=True)
    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices=(0, 1): None)
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args, **_kwargs: None)
    no_model = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=counting_runner,
    )
    assert "mandated_sota_gguf_unavailable" in no_model["gate_reasons"]
    assert no_model["models_used"] == []

    _write_exp3268(tmp_path, eligible=False)
    blocked_preconditions = mod.build_artifact(
        tmp_path,
        nvidia_probe=lambda: {"name": "nvidia_smi", "passed": False},
        python_cuda_probe=lambda: {"name": "selected_python_cuda", "passed": False},
        model_runner=counting_runner,
    )
    assert "exp3268.clean_sota_receipt_eligible=false" in blocked_preconditions["gate_reasons"]
    assert "nvidia_smi_unavailable" in blocked_preconditions["gate_reasons"]
    assert "selected_python_cuda_unavailable" in blocked_preconditions["gate_reasons"]

    _write_exp3268(tmp_path, eligible=True)
    (tmp_path / mod.CONTEXT_FIXTURE_REL_PATH).unlink()
    no_fixture = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=counting_runner,
    )
    assert "exact_row_fixture_unavailable" in no_fixture["gate_reasons"]
    _write_context_fixture(tmp_path, fixture_count=1)

    _patch_single_cached_model(monkeypatch, _model_path(tmp_path, "second.gguf"))
    runner_failed = mod.build_artifact(
        tmp_path,
        nvidia_probe=_ok_nvidia,
        python_cuda_probe=_ok_python_cuda,
        model_runner=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert any(reason.startswith("model_runner_failed") for reason in runner_failed["gate_reasons"])
    assert runner_failed["abstention_rate"] == 1.0

    assert mod.normalize_decision("ACCEPT.") == "accept"
    assert mod.normalize_decision("reject") == "reject"
    assert mod.normalize_decision("ABSTAIN") == "abstain"
    assert mod.normalize_decision("") == "abstain"
    assert mod.abstention_reason({"decision": "accept", "output_text": "ACCEPT"}) == "not_abstained"
    assert (
        mod.abstention_reason({"decision": "abstain", "output_text": ""})
        == "missing_model_output"
    )
    assert (
        mod.abstention_reason({"decision": "abstain", "output_text": "ABSTAIN"})
        == "reported_abstain"
    )
    assert mod.rate(1, 0) == 0.0
    assert mod.duration(5.0, 1.0) == 0.0
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad\n", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text('\n{bad}\n{"ok": true}\n[]\n', encoding="utf-8")
    assert mod.read_jsonl_objects(mixed) == [{"ok": True}]
    assert mod.read_jsonl_objects(tmp_path / "missing.jsonl") == []
    assert mod.sha256_file(tmp_path / "none") is None
    assert len(mod.stable_hash({"b": 1})) == 64
    assert mod.per_class_reason_counts([{"expected_decision": "maybe", "decision": "other"}]) == {
        "maybe": {"unknown_decision": 1}
    }
    assert mod.max_gpu_memory([[{"memory_used_mib": "5"}], [{"memory_used_mib": 9}]]) == 9
    assert mod.max_gpu_memory([]) == 0
    assert mod.safe_int("bad") == 0
    assert mod.bounded_rate("bad", default=0.25) == 0.25
    assert mod.bounded_rate(2.0, default=0.25) == 0.25

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(runner_failed | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="rate field"):
        mod.validate_artifact(runner_failed | {"coverage_rate": 2.0})
    with pytest.raises(ValueError, match="n_eval"):
        mod.validate_artifact(runner_failed | {"n_eval": -1})
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(runner_failed | {"model_specs": []})
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(runner_failed | {"models_used": "bad"})
    with pytest.raises(ValueError, match="missing_model_specs"):
        mod.validate_artifact(runner_failed | {"missing_model_specs": "bad"})
    with pytest.raises(ValueError, match="preconditions_checked"):
        mod.validate_artifact(runner_failed | {"preconditions_checked": "bad"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(runner_failed | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="ready artifact"):
        mod.validate_artifact(
            runner_failed
            | {
                "abstention_calibrated_clean_verifier_v15_ready": True,
                "n_eval": 0,
                "abstention_rate": 0.0,
                "coverage_rate": 1.0,
            }
        )
    with pytest.raises(ValueError, match="repair gate"):
        mod.validate_artifact(
            runner_failed
            | {
                "abstention_calibrated_clean_verifier_v15_ready": False,
                "clean_verifier_rerun_ready": False,
                "repair_gate_input_clean_enough": True,
            }
        )
