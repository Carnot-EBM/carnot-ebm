"""Tests for Exp 3275 clean local SOTA verifier rerun v14.

Spec refs: REQ-VERIFY-3275, SCENARIO-VERIFY-3275.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.verify import clean_local_sota_verifier_rerun_v14 as mod


GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"

REQUIRED_FIELDS = {
    "clean_verifier_rerun_ready",
    "clean_rerun_allowed",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "n_eval",
    "exact_row_fixture_hash",
    "model_specs",
    "models_used",
    "preconditions_checked",
    "gpu_mem_used_mib",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_context_fixture(root: Path) -> None:
    rows = [
        {
            "context": "For this fixture only, mercury means banana.",
            "exact_checker_type": "exact_alias_string",
            "expected_answer": "banana",
            "family": "symbolic_aliases",
            "fixture_id": "ctx-001",
            "minimal_counterexample": {
                "candidate_answer": "planet",
                "expected_answer": "banana",
                "failure_mode": "parametric_prior_shortcut",
            },
            "prior_bait_answer": "planet",
            "question": "What does mercury mean?",
        },
        {
            "context": "For this fixture only, python means blue screwdriver.",
            "exact_checker_type": "exact_alias_string",
            "expected_answer": "blue screwdriver",
            "family": "symbolic_aliases",
            "fixture_id": "ctx-002",
            "minimal_counterexample": {
                "candidate_answer": "snake",
                "expected_answer": "blue screwdriver",
                "failure_mode": "parametric_prior_shortcut",
            },
            "prior_bait_answer": "snake",
            "question": "What does python mean?",
        },
    ]
    path = root / mod.CONTEXT_FIXTURE_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    _write_json(
        root,
        mod.EXP3223_REL_PATH,
        {
            "experiment_id": "exp3223",
            "uncertainty_sidecar_ready": True,
            "exact_row_count": 4,
            "exact_verifier_authority_preserved": True,
            "honest_verdict": "complete: exact rows ready",
        },
    )


def _write_exp3268(
    root: Path,
    *,
    eligible: bool,
    model_path: Path | None = None,
) -> None:
    model_specs = {
        "mandated_model_ids": list(mod.MANDATED_MODEL_IDS),
        "mandated_models": {
            GEMMA26: {
                "cached": model_path is not None,
                "expected_quantization": "Q4_K_M",
                "model_path": str(model_path) if model_path else None,
                "name": "Gemma4-26B-A4B-it",
                "role": "moe",
                "size_bytes": 12,
            }
        },
        "runtime": "llama_cpp",
    }
    _write_json(
        root,
        mod.EXP3268_REL_PATH,
        {
            "experiment_id": "exp3268",
            "clean_sota_receipt_eligible": eligible,
            "model_specs": model_specs,
            "models_used": [
                {
                    "model_id": GEMMA26,
                    "model_path": str(model_path) if model_path else "",
                    "cached": model_path is not None,
                    "clean_row": eligible,
                    "attempted_live_receipt": eligible,
                }
            ],
            "preconditions_checked": [{"name": "exp3268_fixture", "passed": eligible}],
            "gpu_mem_used_mib": 9000 if eligible else 0,
            "random_seed": 3268,
            "reproducibility_checksum": "receipt-checksum",
            "honest_verdict": "complete: exp3268",
        },
    )


def _cuda_probe() -> dict[str, Any]:
    return {
        "name": "cuda_runtime",
        "passed": True,
        "cuda_available": True,
        "gpu_count": 2,
        "gpu_mem_used_mib": 4,
        "returncode": 0,
    }


def _runner(decisions: dict[str, str]) -> mod.ModelRunner:
    def run(
        rows: list[dict[str, Any]],
        model: dict[str, Any],
        random_seed: int,
    ) -> dict[str, Any]:
        assert model["model_id"] == GEMMA26
        assert random_seed == mod.DEFAULT_RANDOM_SEED
        return {
            "rows": [
                {
                    "row_id": row["row_id"],
                    "model_id": model["model_id"],
                    "model_path": model["model_path"],
                    "output_text": decisions[row["row_id"]],
                    "decision": decisions[row["row_id"]],
                    "prompt_hash": f"prompt-{row['row_id']}",
                    "transcript_hash": f"tx-{row['row_id']}",
                    "token_counts": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
                }
                for row in rows
            ],
            "gpu_mem_used_mib": 7777,
        }

    return run


def test_req_verify_3275_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3275: OpenSpec declares the v14 clean local rerun contract."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3275" in spec
    assert "SCENARIO-VERIFY-3275" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "clean_sota_receipt_eligible" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3275_ineligible_receipt_writes_complete_gated_skip(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3275: ineligible Exp 3268 blocks local verifier calls."""

    _write_exp3268(tmp_path, eligible=False)
    calls = 0

    def runner(
        rows: list[dict[str, Any]],
        model: dict[str, Any],
        random_seed: int,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"rows": [], "gpu_mem_used_mib": 0}

    artifact = mod.build_artifact(
        tmp_path,
        cuda_probe=_cuda_probe,
        model_runner=runner,
        started_s=1.0,
        now_s=3.5,
        tests_run=["SCENARIO-VERIFY-3275 gated"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["clean_verifier_rerun_ready"] is False
    assert artifact["clean_rerun_allowed"] is False
    assert artifact["gated_skip"] is True
    assert artifact["n_eval"] == 0
    assert artifact["models_used"] == []
    assert artifact["gpu_mem_used_mib"] == 0
    assert artifact["false_accept_rate"] == 0.0
    assert artifact["false_reject_rate"] == 0.0
    assert artifact["abstention_rate"] == 0.0
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3275 gated"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert "exp3268.clean_sota_receipt_eligible=false" in artifact["gate_reasons"]
    assert calls == 0


def test_scenario_verify_3275_scores_exact_rows_with_local_model_stub(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3275: local decisions are scored only by exact authority."""

    model_path = tmp_path / "models" / "gemma.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"GGUF")
    _write_exp3268(tmp_path, eligible=True, model_path=model_path)
    _write_context_fixture(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        cuda_probe=_cuda_probe,
        model_runner=_runner(
            {
                "ctx-001:expected": "ACCEPT",
                "ctx-001:counterexample": "REJECT",
                "ctx-002:expected": "ACCEPT",
                "ctx-002:counterexample": "ABSTAIN",
            }
        ),
        max_eval_rows=4,
        started_s=10.0,
        now_s=14.0,
    )

    assert artifact["gated_skip"] is False
    assert artifact["clean_verifier_rerun_ready"] is True
    assert artifact["clean_rerun_allowed"] is True
    assert artifact["n_eval"] == 4
    assert artifact["false_accept_rate"] == 0.0
    assert artifact["false_reject_rate"] == 0.0
    assert artifact["abstention_rate"] == pytest.approx(0.25)
    assert len(artifact["exact_row_fixture_hash"]) == 64
    assert artifact["models_used"] == [
        {
            "model_id": GEMMA26,
            "model_path": str(model_path),
            "source": "exp3268_clean_receipt",
            "legacy_small_model": False,
        }
    ]
    assert artifact["gpu_mem_used_mib"] == 7777
    assert artifact["repair_gate_input_clean_enough"] is True
    assert {row["exact_authority"] for row in artifact["per_row_results"]} == {
        "context_exact_checker"
    }
    assert {row["source_candidate_kind"] for row in artifact["per_row_results"]} == {
        "fixture_expected_answer",
        "fixture_minimal_counterexample",
    }
    assert all(row["synthetic_shortcut_row"] is False for row in artifact["per_row_results"])
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3275_metrics_false_accepts_and_rejects_block_readiness(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3275: unsafe accepts and overblocking are visible metrics."""

    model_path = tmp_path / "models" / "gemma.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"GGUF")
    _write_exp3268(tmp_path, eligible=True, model_path=model_path)
    _write_context_fixture(tmp_path)

    artifact = mod.build_artifact(
        tmp_path,
        cuda_probe=_cuda_probe,
        model_runner=_runner(
            {
                "ctx-001:expected": "REJECT",
                "ctx-001:counterexample": "ACCEPT",
                "ctx-002:expected": "ACCEPT",
                "ctx-002:counterexample": "REJECT",
            }
        ),
        max_eval_rows=4,
    )

    assert artifact["false_accept_rate"] == pytest.approx(0.5)
    assert artifact["false_reject_rate"] == pytest.approx(0.5)
    assert artifact["abstention_rate"] == 0.0
    assert artifact["clean_verifier_rerun_ready"] is False
    assert artifact["clean_rerun_allowed"] is False
    assert artifact["repair_gate_input_clean_enough"] is False
    assert "false_accept_rate_above_threshold" in artifact["gate_reasons"]


def test_req_verify_3275_precondition_and_runner_failure_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3275: CUDA, cache, fixtures, and runner failures fail closed."""

    model_path = tmp_path / "models" / "gemma.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"GGUF")
    _write_exp3268(tmp_path, eligible=True, model_path=model_path)
    _write_context_fixture(tmp_path)

    cuda_blocked = mod.build_artifact(
        tmp_path,
        cuda_probe=lambda: {"name": "cuda_runtime", "passed": False},
        model_runner=_runner({}),
    )
    assert "cuda_unavailable" in cuda_blocked["gate_reasons"]
    assert cuda_blocked["n_eval"] == 0

    _write_exp3268(tmp_path, eligible=True, model_path=None)
    no_model = mod.build_artifact(
        tmp_path,
        cuda_probe=_cuda_probe,
        model_runner=_runner({}),
    )
    assert "mandated_sota_gguf_unavailable" in no_model["gate_reasons"]

    _write_exp3268(tmp_path, eligible=True, model_path=model_path)
    (tmp_path / mod.CONTEXT_FIXTURE_REL_PATH).unlink()
    no_fixture = mod.build_artifact(
        tmp_path,
        cuda_probe=_cuda_probe,
        model_runner=_runner({}),
    )
    assert "exact_row_fixture_unavailable" in no_fixture["gate_reasons"]

    _write_context_fixture(tmp_path)
    runner_failed = mod.build_artifact(
        tmp_path,
        cuda_probe=_cuda_probe,
        model_runner=lambda rows, model, random_seed: {
            "rows": [],
            "gpu_mem_used_mib": 4,
            "runner_error": "boom",
        },
        max_eval_rows=4,
    )
    assert any(reason.startswith("model_runner_failed") for reason in runner_failed["gate_reasons"])
    assert "gpu_mem_used_below_cuda_offload_floor" in runner_failed["gate_reasons"]
    assert "abstention_rate_above_threshold" in runner_failed["gate_reasons"]
    assert runner_failed["abstention_rate"] == 1.0


def test_req_verify_3275_writer_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3275: writer, parsers, checksum, and validators fail closed."""

    model_path = tmp_path / "models" / "gemma.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"GGUF")
    _write_exp3268(tmp_path, eligible=True, model_path=model_path)
    _write_context_fixture(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        cuda_probe=_cuda_probe,
        model_runner=_runner(
            {
                "ctx-001:expected": "ACCEPT",
                "ctx-001:counterexample": "REJECT",
                "ctx-002:expected": "ACCEPT",
                "ctx-002:counterexample": "REJECT",
            }
        ),
        max_eval_rows=4,
        started_s=5.0,
        now_s=7.0,
        tests_run=["writer"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert saved["duration_s"] == pytest.approx(2.0)
    assert saved["tests_run"] == ["writer"]
    assert len(saved["reproducibility_checksum"]) == 64
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad\n", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    assert mod.read_jsonl_objects(tmp_path / "missing.jsonl") == []
    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text('\n{bad}\n{"ok": true}\n[]\n', encoding="utf-8")
    assert mod.read_jsonl_objects(mixed_jsonl) == [{"ok": True}]
    assert mod.normalize_decision("ACCEPT") == "accept"
    assert mod.normalize_decision("reject.") == "reject"
    assert mod.normalize_decision("") == "abstain"
    assert mod.normalize_decision("I would accept") == "abstain"
    assert mod.rate(1, 0) == 0.0
    assert mod.duration(4.0, 2.0) == 0.0
    assert mod.sha256_file(tmp_path / "none") is None
    assert mod.resolve_models({"model_specs": {}}, tmp_path) == []
    assert mod.model_specs_from_receipt({}) == {
        "mandated_model_ids": list(mod.MANDATED_MODEL_IDS),
        "mandated_models": {},
    }
    assert mod.max_gpu_memory([[{"memory_used_mib": "5"}], [{"memory_used_mib": 9}]]) == 9
    assert mod.max_gpu_memory([]) == 0
    assert mod.safe_int("x") == 0
    assert mod.mapping_list("bad") == []

    rows = mod.build_exact_eval_rows(tmp_path, max_rows=3)
    assert [row["row_id"] for row in rows] == [
        "ctx-001:expected",
        "ctx-001:counterexample",
        "ctx-002:expected",
    ]

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="rate field"):
        mod.validate_artifact(saved | {"false_accept_rate": 2.0})
    with pytest.raises(ValueError, match="n_eval"):
        mod.validate_artifact(saved | {"n_eval": -1})
    with pytest.raises(ValueError, match="preconditions_checked"):
        mod.validate_artifact(saved | {"preconditions_checked": "bad"})
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(saved | {"models_used": "bad"})
    with pytest.raises(ValueError, match="exact_row_fixture_hash"):
        mod.validate_artifact(saved | {"exact_row_fixture_hash": "bad"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(saved | {"reproducibility_checksum": "bad"})
    with pytest.raises(ValueError, match="ready artifact"):
        mod.validate_artifact(saved | {"clean_verifier_rerun_ready": True, "n_eval": 0})
    with pytest.raises(ValueError, match="clean_rerun_allowed"):
        mod.validate_artifact(saved | {"clean_verifier_rerun_ready": True, "clean_rerun_allowed": False})
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(saved | {"models_used": [], "clean_verifier_rerun_ready": True})
    with pytest.raises(ValueError, match="gpu_mem_used_mib"):
        mod.validate_artifact(saved | {"gpu_mem_used_mib": 0, "clean_verifier_rerun_ready": True})
    with pytest.raises(ValueError, match="clean_rerun_allowed cannot be true"):
        mod.validate_artifact(saved | {"clean_verifier_rerun_ready": False, "clean_rerun_allowed": True})


def test_req_verify_3275_cuda_probe_parses_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3275: CUDA precondition rows expose nvidia-smi evidence."""

    def fake_success(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="0, NVIDIA RTX, 24576, 123, 0, 595.71.05\nbad,row\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_success)
    success = mod.default_cuda_probe()
    assert success["passed"] is True
    assert success["gpu_count"] == 1
    assert success["gpu_mem_used_mib"] == 123
    assert mod.parse_nvidia_smi_rows("1, GPU, 1, 2, 3, driver\nbad\n") == [
        {
            "index": 1,
            "name": "GPU",
            "memory_total_mib": 1,
            "memory_used_mib": 2,
            "utilization_gpu_pct": 3,
            "driver_version": "driver",
        }
    ]

    def fake_failure(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, stdout="", stderr="no gpu")

    monkeypatch.setattr(mod.subprocess, "run", fake_failure)
    failed = mod.default_cuda_probe()
    assert failed["passed"] is False
    assert failed["gpu_count"] == 0

    def fake_exception(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("missing nvidia-smi")

    monkeypatch.setattr(mod.subprocess, "run", fake_exception)
    errored = mod.default_cuda_probe()
    assert errored["passed"] is False
    assert errored["returncode"] is None
