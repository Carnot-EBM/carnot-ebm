"""Tests for Exp 3028 clean-methodology SOTA repair evidence.

Spec: REQ-CODE-3028, SCENARIO-CODE-3028.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval import sota_repair_clean_methodology_rerun_3028 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _controller_rule() -> dict[str, bool]:
    return {
        "require_schema_valid": True,
        "require_syntax_success": True,
        "require_entry_point_present": True,
        "require_false_accept_probe_clean": True,
        "require_no_intent_drift": True,
        "require_original_passed": True,
        "require_metamorphic_passed_all": True,
        "require_tautology_probe_clean": True,
    }


def _write_ready_sources(
    tmp_path: Path,
    *,
    n_tasks: int = 2,
    missing_transcript: bool = False,
    smoke_model: bool = False,
    intent_drift: bool = False,
) -> None:
    hard.write_artifact(
        hard.ExperimentConfig(
            repo_root=tmp_path,
            manifest_items=hard.default_items(),
            started_at=1.0,
            clock=lambda: 2.0,
            tests_run=("SCENARIO-CODE-3028-fixture",),
        )
    )
    metamorphic.write_artifact(
        metamorphic.ExperimentConfig(
            repo_root=tmp_path,
            started_at=3.0,
            clock=lambda: 4.0,
            tests_run=("SCENARIO-CODE-3028-fixture",),
        )
    )
    controller_path = tmp_path / exp.CONTROLLER_CONFIG_REL_PATH
    _write_json(
        controller_path,
        {
            "policy_type": "transparent_grid_rule",
            "selected_rule": _controller_rule(),
            "llm_judge_used": False,
        },
    )
    _write_json(
        tmp_path / exp.EXP3015_REL_PATH,
        {
            "acceptance_controller_ready": True,
            "controller_config_path": str(exp.CONTROLLER_CONFIG_REL_PATH),
            "llm_judge_used": False,
            "honest_verdict": "complete: offline repair acceptance controller ready",
        },
    )
    _write_json(
        tmp_path / exp.EXP3013_REL_PATH,
        {
            "sota_headline_ready": True,
            "sota_logprob_ready": True,
            "model_checksums": {
                HEADLINE_MODEL: {
                    "status": "available",
                    "path": "/models/gemma.gguf",
                    "bounded_sha256": "checksum",
                    "size_bytes": 123,
                }
            },
            "cache_paths": {"headline_models": {HEADLINE_MODEL: "/models/gemma.gguf"}},
            "precondition_evidence": {
                "gpu_inventory": {
                    "available": True,
                    "free_vram_mib_total": 24000,
                    "gpus": [{"index": 0, "memory_free_mib": 24000}],
                },
                "python_environment": {"selected_python": ".venv/bin/python"},
                "repo_commit": {"commit": "abc123"},
                "checksum_feasibility": {"feasible": True, "available_model_count": 1},
                "llama_cpp": {"llama_cpp_import_ok": True, "llama_cpp_supports_gpu_offload": True},
                "torch_cuda": {"cuda_available": True, "cuda_device_count": 1},
            },
        },
    )
    model_id = "Qwen/Qwen3.5-0.8B" if smoke_model else HEADLINE_MODEL
    rows: list[dict[str, Any]] = []
    transcript_paths: list[str] = []
    verifier_paths: list[str] = []
    patch_paths: list[str] = []
    for index, item in enumerate(hard.default_items()[:n_tasks]):
        item_id = str(item["item_id"])
        patch = str(item["reference_solution"])
        patch_rel = (
            exp.RAW_REL_DIR
            / "patches"
            / f"{item_id}_{model_id.replace('/', '_')}_{index}.py"
        ).as_posix()
        transcript_rel = (
            exp.RAW_REL_DIR
            / "transcripts"
            / f"{item_id}_{model_id.replace('/', '_')}_{index}.json"
        ).as_posix()
        verifier_rel = (
            Path("results/verifier_transcripts/experiment_3028_fixture")
            / f"{item_id}_{index}.json"
        ).as_posix()
        patch_path = tmp_path / patch_rel
        transcript_path = tmp_path / transcript_rel
        verifier_path = tmp_path / verifier_rel
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        verifier_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch, encoding="utf-8")
        draft_intent = (
            "Return unrelated output sorted alphabetically."
            if intent_drift and index == 0
            else str(item["expected_behavior"])
        )
        transcript = {
            "item_id": item_id,
            "model_hf_id": model_id,
            "prompt": f"repair {item_id}",
            "raw_response": json.dumps({"draft_intent": draft_intent, "final_patch": patch}),
            "draft_intent": draft_intent,
            "failing_trace": {"failing_test_ids": [f"SCENARIO-CODE-3028-{item_id}"]},
            "final_patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            "generation_duration_s": 0.25,
        }
        transcript_path.write_text(json.dumps(transcript, sort_keys=True) + "\n", encoding="utf-8")
        verifier_path.write_text(
            json.dumps({"deterministic_tests_executed": True}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if missing_transcript and index == 0:
            transcript_path.unlink()
        transcript_hash = _sha256(transcript_path) if transcript_path.is_file() else None
        rows.append(
            {
                "item_id": item_id,
                "model_hf_id": model_id,
                "seed": 302800 + index,
                "draft_intent": draft_intent,
                "schema_valid": True,
                "syntax_success": True,
                "controller_accepted": True,
                "candidate_patch_path": patch_rel,
                "live_transcript_path": transcript_rel,
                "transcript_sha256": transcript_hash,
                "verifier_log_path": verifier_rel,
                "generation_duration_s": 0.25,
                "tokens_generated": 32,
            }
        )
        transcript_paths.append(transcript_rel)
        verifier_paths.append(verifier_rel)
        patch_paths.append(patch_rel)
    _write_json(
        tmp_path / exp.EXP3016_REL_PATH,
        {
            "repair_controller_clean": True,
            "headline_result": not smoke_model,
            "n_tasks": n_tasks,
            "model_specs": {
                "headline_models": [HEADLINE_MODEL],
                "runnable_headline_models": [
                    {
                        "name": "Gemma4-26B-A4B-it",
                        "hf_id": HEADLINE_MODEL,
                        "gpu": 0,
                        "model_path": "/models/gemma.gguf",
                    }
                ],
                "smoke_only_models": ["Qwen/Qwen3.5-0.8B"],
            },
            "headline_models_used": ([] if smoke_model else [HEADLINE_MODEL]),
            "model_checksums": {
                HEADLINE_MODEL: {"status": "available", "bounded_sha256": "checksum"}
            },
            "candidate_evaluations": rows,
            "live_transcript_paths": transcript_paths,
            "verifier_log_paths": verifier_paths,
            "candidate_patch_paths": patch_paths,
            "reproducibility_checksum": "exp3016-checksum",
        },
    )
    _write_json(
        tmp_path / exp.EXP3027_REL_PATH,
        {
            "methodology_corrigendum_ready": True,
            "sota_headline_ready": True,
            "repair_rerun_required": True,
            "repair_rerun_decision": {
                "decision": "live_rerun_required",
                "reason": "top-level random_seed and transcript_hashes absent",
            },
            "honest_verdict": "complete: methodology corrigendum ready",
        },
    )


def test_req_code_3028_spec_anchor_and_script_exist() -> None:
    """REQ-CODE-3028: the clean rerun is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    script = REPO_ROOT / "scripts/experiment_3028_sota_repair_clean_methodology_rerun_v2.py"

    assert "REQ-CODE-3028" in spec
    assert "SCENARIO-CODE-3028" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) >= {
        "clean_repair_rerun_ready",
        "repair_controller_clean",
        "clean_repair_claim_promotable_candidate",
        "n_tasks",
        "n_live_transcripts",
        "model_specs",
        "legacy_smoke_only_used",
        "pass_at_1_delta",
        "pass_at_k_delta",
        "syntax_failure_rate_delta",
        "schema_failure_rate_delta",
        "false_accept_delta",
        "tautology_gate_clean",
        "intent_drift_count",
        "reproducibility_checksum",
        "inference_substrate",
        "honest_verdict",
    }
    assert script.is_file()


def test_scenario_code_3028_reconstructs_clean_live_evidence(tmp_path: Path) -> None:
    """SCENARIO-CODE-3028: nested transcript evidence repairs Exp 3016 metadata gaps."""

    _write_ready_sources(tmp_path, n_tasks=2)

    artifact = exp.write_artifact(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=10.0, clock=lambda: 14.0)
    )
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["clean_repair_rerun_ready"] is True
    assert artifact["repair_controller_clean"] is True
    assert artifact["clean_repair_claim_promotable_candidate"] is True
    assert artifact["n_tasks"] == 2
    assert artifact["n_live_transcripts"] == 2
    assert artifact["model_specs"] == [
        {
            "hf_id": HEADLINE_MODEL,
            "model_path": "/models/gemma.gguf",
            "checksum": "checksum",
        }
    ]
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["pass_at_1_delta"] == pytest.approx(1.0)
    assert artifact["pass_at_k_delta"] == pytest.approx(1.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["false_accept_delta"] == pytest.approx(0.0)
    assert artifact["tautology_gate_clean"] is True
    assert artifact["intent_drift_count"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["exp3027_repair_rerun_required"] is True
    assert artifact["inference_substrate"]["reconstruction_mode"] == "exp3016_nested_live_transcripts"
    assert artifact["candidate_evidence"][0]["random_seed"] == 302800
    assert artifact["candidate_evidence"][0]["final_patch"]
    assert artifact["candidate_evidence"][0]["validator_output"]["deterministic_tests_executed"] is True


def test_req_code_3028_blocks_without_reconstruction_or_headline(tmp_path: Path) -> None:
    """REQ-CODE-3028: missing transcript evidence and no live runner fail closed."""

    _write_ready_sources(tmp_path, n_tasks=2, missing_transcript=True)

    artifact = exp.build_artifact(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=1.0, clock=lambda: 1.5)
    )

    assert artifact["clean_repair_rerun_ready"] is False
    assert artifact["repair_controller_clean"] is False
    assert artifact["n_live_transcripts"] == 1
    assert artifact["honest_verdict"].startswith("blocked_sota_headline_model_unavailable")
    assert any(check["resource"] == "complete_transcript_reconstruction" for check in artifact["precondition_checks"])


def test_req_code_3028_rejects_smoke_only_and_intent_drift(tmp_path: Path) -> None:
    """REQ-CODE-3028: smoke models and intent drift cannot become headline evidence."""

    _write_ready_sources(tmp_path, n_tasks=2, smoke_model=True, intent_drift=True)

    artifact = exp.build_artifact(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=2.0, clock=lambda: 5.0)
    )

    assert artifact["clean_repair_rerun_ready"] is False
    assert artifact["clean_repair_claim_promotable_candidate"] is False
    assert artifact["legacy_smoke_only_used"] is True
    assert artifact["intent_drift_count"] == 0
    assert artifact["candidate_intent_drift_count"] == 1
    assert artifact["candidate_evidence"][0]["controller_accepted"] is False
    assert "legacy_smoke_only_model" in artifact["candidate_evidence"][0]["rejection_reasons"]
    assert "intent_drift" in artifact["candidate_evidence"][0]["rejection_reasons"]
    assert artifact["honest_verdict"].startswith("complete_flagged:")


def test_req_code_3028_helper_edges_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-3028: helper branches fail closed with deterministic outputs."""

    explicit_controller = tmp_path / "controller.json"
    explicit_meta = tmp_path / "meta.jsonl"
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        controller_config_path=explicit_controller,
        metamorphic_manifest_path=explicit_meta,
    )

    assert config.resolved_controller_config_path({}) == explicit_controller
    assert config.resolved_metamorphic_manifest_path({}) == explicit_meta
    assert exp._candidate_evidence_rows(
        config=config,
        exp3016={"candidate_evaluations": ["not-a-row"]},
        controller_rule={},
        hard_items=[],
        variants=[],
        tautology_gate_clean=False,
    ) == []
    assert exp._methodology_rejection_reasons(
        {"model_hf_id": "not-headline", "checker_evidence_complete": False}
    ) == ["non_headline_model", "missing_checker_evidence"]
    assert exp._model_specs_list(
        {"cache_paths": {"headline_models": {HEADLINE_MODEL: "/cache/model.gguf"}}},
        {"model_specs": {"runnable_headline_models": "not-list"}, "model_checksums": {}},
        [HEADLINE_MODEL],
    ) == [{"hf_id": HEADLINE_MODEL, "model_path": "/cache/model.gguf", "checksum": None}]
    assert exp._load_hard_items(exp.ExperimentConfig(repo_root=tmp_path / "missing")) == []
    bad_hard = tmp_path / "bad-hard.jsonl"
    bad_hard.write_text("{bad\n", encoding="utf-8")
    assert exp._load_hard_items(exp.ExperimentConfig(repo_root=tmp_path, hard_manifest_path=bad_hard)) == []
    assert exp._load_metamorphic_variants(
        exp.ExperimentConfig(repo_root=tmp_path, metamorphic_manifest_path=tmp_path / "missing.jsonl"),
        {},
    ) == []
    bad_meta = tmp_path / "bad-meta.jsonl"
    bad_meta.write_text("{bad\n", encoding="utf-8")
    assert exp._load_metamorphic_variants(
        exp.ExperimentConfig(repo_root=tmp_path, metamorphic_manifest_path=bad_meta),
        {},
    ) == []
    assert exp._run_original_check({}, "x = 1").passed is False
    assert exp._intent_preserved("", "expected behavior") is False
    assert exp._intent_preserved("only stop words", "and the") is True
    assert exp._content_tokens("finaltoken") == ["finaltoken"]
    assert exp._entry_point_present("def broken(:\n", "broken") is False
    assert exp._read_json_if_present(bad_meta) == {}
    assert exp._read_text_if_present(tmp_path / "absent.txt") == ""
    assert exp._path_string(tmp_path, tmp_path.parent / "outside.txt").endswith("outside.txt")
    sha_file = tmp_path / "sha.txt"
    sha_file.write_text("abc", encoding="utf-8")
    assert exp._sha256_file(sha_file) == hashlib.sha256(b"abc").hexdigest()

    class Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def raise_oserror(*_args: Any, **_kwargs: Any) -> Result:
        raise OSError("boom")

    monkeypatch.setattr(exp.subprocess, "run", raise_oserror)
    assert exp._git_commit(tmp_path) is None
    assert exp._nvidia_smi_inventory(tmp_path) == {}

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(1, stderr="driver unavailable"),
    )
    assert exp._nvidia_smi_inventory(tmp_path) == {
        "available": False,
        "stderr_summary": "driver unavailable",
    }
    assert exp._free_vram({"gpu_inventory": {"free_vram_mib_total": 42}}, {}) == 42
    assert exp._free_vram({}, {"free_vram_mib_total": 7}) == 7
    parsed = exp._parse_args(["--output", "out.json", "--test-run", "focused"])
    assert parsed.output == Path("out.json")
    assert parsed.test_run == ["focused"]
