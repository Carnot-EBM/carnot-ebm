"""Tests for Exp 3003 SOTA repair metamorphic false-accept rerun.

Spec: REQ-CODE-3003, SCENARIO-CODE-3003.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval import gated_sota_repair_metamorphic_false_accept_rerun as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
REQUIRED_FIELDS = {
    "repair_rerun_clean",
    "headline_result",
    "preconditions_checked",
    "n_tasks",
    "n_metamorphic_variants",
    "model_specs",
    "headline_models_used",
    "model_checksums",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "false_accept_delta",
    "tautology_gate_clean",
    "syntax_failure_rate_delta",
    "live_transcript_paths",
    "verifier_log_paths",
    "honest_verdict",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ready_sources(tmp_path: Path, *, n_items: int = 24) -> None:
    hard.write_artifact(
        hard.ExperimentConfig(
            repo_root=tmp_path,
            manifest_items=hard.default_items()[:n_items],
            started_at=10.0,
            clock=lambda: 11.0,
            tests_run=("focused-exp2990",),
        )
    )
    metamorphic.write_artifact(
        metamorphic.ExperimentConfig(
            repo_root=tmp_path,
            started_at=12.0,
            clock=lambda: 13.0,
            tests_run=("focused-exp3002",),
        )
    )
    _write_json(
        tmp_path / "results" / exp.EXP3001_FILENAME,
        {
            "artifact": "experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1",
            "sota_headline_ready": True,
            "preconditions_checked": True,
            "model_specs": {
                "headline_models": list(exp.HEADLINE_MODEL_IDS),
                "smoke_only_models": list(exp.SMOKE_ONLY_MODEL_IDS),
            },
            "model_checksums": {
                HEADLINE_MODEL: {
                    "status": "available",
                    "path": "/models/gemma.gguf",
                    "bounded_sha256": "checksum",
                }
            },
            "cache_paths": {"headline_models": {HEADLINE_MODEL: "/models/gemma.gguf"}},
            "precondition_evidence": {
                "gpu_inventory": {"available": True},
                "torch_cuda": {"cuda_available": True},
                "llama_cpp": {"llama_cpp_supports_gpu_offload": True},
            },
            "live_transcript_paths": ["/tmp/exp3001-live.json"],
        },
    )


def _write_exp2991_candidates(
    tmp_path: Path,
    *,
    n_items: int = 24,
    code_by_item: dict[str, str] | None = None,
    schema_valid: bool = True,
    syntax_success: bool = True,
) -> None:
    code_by_item = code_by_item or {}
    patch_dir = tmp_path / "results/raw/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1/patches"
    transcript_dir = (
        tmp_path / "results/raw/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1/transcripts"
    )
    verifier_dir = tmp_path / "results/verifier_transcripts/experiment_2991"
    rows: list[dict[str, Any]] = []
    patch_paths: list[str] = []
    transcript_paths: list[str] = []
    verifier_paths: list[str] = []
    for index, item in enumerate(hard.default_items()[:n_items]):
        code = code_by_item.get(item["item_id"], str(item["reference_solution"]))
        token = f"{item['item_id']}_{HEADLINE_MODEL.replace('/', '_')}_{index}"
        patch_path = patch_dir / f"{token}.py"
        transcript_path = transcript_dir / f"{token}.json"
        verifier_path = verifier_dir / f"{token}.json"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        verifier_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(code, encoding="utf-8")
        transcript_path.write_text(
            json.dumps(
                {
                    "item_id": item["item_id"],
                    "model_hf_id": HEADLINE_MODEL,
                    "prompt": f"repair {item['item_id']}",
                    "draft_intent": item["expected_behavior"],
                    "raw_response": json.dumps(
                        {
                            "draft_intent": item["expected_behavior"],
                            "final_patch": code,
                        }
                    ),
                    "generation_duration_s": 1.5,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        verifier_path.write_text("{}\n", encoding="utf-8")
        patch_rel = str(patch_path.relative_to(tmp_path))
        transcript_rel = str(transcript_path.relative_to(tmp_path))
        verifier_rel = str(verifier_path.relative_to(tmp_path))
        patch_paths.append(patch_rel)
        transcript_paths.append(transcript_rel)
        verifier_paths.append(verifier_rel)
        rows.append(
            {
                "item_id": item["item_id"],
                "model_hf_id": HEADLINE_MODEL,
                "model_path": "/models/gemma.gguf",
                "candidate_patch_path": patch_rel,
                "transcript_path": transcript_rel,
                "verifier_log_path": verifier_rel,
                "draft_intent": item["expected_behavior"],
                "generation_duration_s": 1.5,
                "generation_backend": "llama_cpp",
                "schema_valid": schema_valid,
                "syntax_success": syntax_success,
                "passed": True,
            }
        )
    _write_json(
        tmp_path / "results" / exp.EXP2991_FILENAME,
        {
            "artifact": "experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1",
            "headline_result": True,
            "n_tasks": n_items,
            "headline_models_used": [HEADLINE_MODEL],
            "candidate_evaluations": rows,
            "candidate_patch_paths": patch_paths,
            "transcript_paths": transcript_paths,
            "verifier_log_paths": verifier_paths,
            "pass_at_1_delta": 1.0,
            "pass_at_k_delta": 1.0,
            "schema_failure_rate_delta": 0.0,
            "syntax_failure_rate_delta": 0.0,
            "verifier_false_accept_delta": 0.0,
        },
    )


def _config(tmp_path: Path, *, n_tasks: int = 20) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        n_tasks=n_tasks,
        started_at=20.0,
        clock=lambda: 25.0,
        tests_run=("focused-exp3003",),
    )


def test_req_code_3003_spec_anchor_exists() -> None:
    """REQ-CODE-3003: the metamorphic false-accept rerun is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-3003" in spec
    assert "SCENARIO-CODE-3003" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "tautology_gate_clean" in spec


def test_scenario_code_3003_clean_headline_artifact_has_required_evidence(tmp_path: Path) -> None:
    """SCENARIO-CODE-3003: clean promotion requires headline and metamorphic evidence."""

    _write_ready_sources(tmp_path)
    _write_exp2991_candidates(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path, n_tasks=20))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_rerun_clean"] is True
    assert artifact["headline_result"] is True
    assert artifact["preconditions_checked"] is True
    assert artifact["n_tasks"] == 20
    assert artifact["n_metamorphic_variants"] > artifact["n_tasks"]
    assert artifact["headline_models_used"] == [HEADLINE_MODEL]
    assert artifact["model_checksums"][HEADLINE_MODEL]["bounded_sha256"] == "checksum"
    assert artifact["pass_at_1_delta"] == pytest.approx(1.0)
    assert artifact["pass_at_k_delta"] == pytest.approx(1.0)
    assert artifact["false_accept_delta"] == pytest.approx(0.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["tautology_gate_clean"] is True
    assert artifact["honest_verdict"] == "clean: metamorphic repair rerun gates passed"
    assert len(artifact["live_transcript_paths"]) == 20
    assert len(artifact["verifier_log_paths"]) == 20
    assert all((tmp_path / path).is_file() for path in artifact["candidate_patch_paths"])
    assert all((tmp_path / path).is_file() for path in artifact["verifier_log_paths"])
    assert all(row["metamorphic_variant_count"] > 0 for row in artifact["candidate_evaluations"])


def test_scenario_code_3003_false_accept_delta_flags_visible_test_overfit(tmp_path: Path) -> None:
    """SCENARIO-CODE-3003: passing original tests while failing variants is a false accept."""

    _write_ready_sources(tmp_path)
    first = hard.default_items()[0]
    overfit = (
        "def clamp_score(x, lo, hi):\n"
        "    if (x, lo, hi) == (12, 0, 10):\n"
        "        return 10\n"
        "    if (x, lo, hi) == (-3, 0, 10):\n"
        "        return 0\n"
        "    if (x, lo, hi) == (5, 0, 10):\n"
        "        return 5\n"
        "    return None\n"
    )
    _write_exp2991_candidates(tmp_path, code_by_item={first["item_id"]: overfit})

    artifact = exp.build_artifact(_config(tmp_path, n_tasks=20))
    flagged = next(row for row in artifact["candidate_evaluations"] if row["item_id"] == first["item_id"])

    assert artifact["headline_result"] is True
    assert artifact["repair_rerun_clean"] is False
    assert artifact["honest_verdict"] == "flagged: metamorphic repair rerun did not clear gates"
    assert artifact["false_accept_delta"] > 0.0
    assert flagged["original_passed"] is True
    assert flagged["metamorphic_passed_all"] is False
    assert flagged["false_accept"] is True


def test_req_code_3003_schema_or_syntax_regression_blocks_clean_gate(tmp_path: Path) -> None:
    """REQ-CODE-3003: schema and syntax regressions keep the row flagged."""

    _write_ready_sources(tmp_path)
    _write_exp2991_candidates(tmp_path, schema_valid=False, syntax_success=False)

    artifact = exp.build_artifact(_config(tmp_path, n_tasks=20))

    assert artifact["headline_result"] is True
    assert artifact["repair_rerun_clean"] is False
    assert artifact["schema_failure_rate_delta"] == pytest.approx(1.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(1.0)


def test_req_code_3003_blocks_when_preconditions_are_missing(tmp_path: Path) -> None:
    """REQ-CODE-3003: missing exp3001/exp3002 gates emit a terminal blocked artifact."""

    _write_ready_sources(tmp_path)
    _write_exp2991_candidates(tmp_path)
    _write_json(
        tmp_path / "results" / exp.EXP3001_FILENAME,
        {
            "sota_headline_ready": False,
            "preconditions_checked": True,
            "model_checksums": {},
            "precondition_evidence": {"gpu_inventory": {"available": True}},
        },
    )

    artifact = exp.build_artifact(_config(tmp_path, n_tasks=20))

    assert artifact["preconditions_checked"] is True
    assert artifact["headline_result"] is False
    assert artifact["repair_rerun_clean"] is False
    assert artifact["honest_verdict"] == "blocked: exp3003 preconditions not met"
    assert artifact["n_tasks"] == 0
    assert artifact["n_metamorphic_variants"] == 0
    assert any(
        row["resource"] == "exp3001_sota_cache_carry_forward" and row["available"] is False
        for row in artifact["precondition_checks"]
    )


def test_req_code_3003_blocks_when_sample_or_candidate_evidence_is_missing(tmp_path: Path) -> None:
    """REQ-CODE-3003: undersized samples and missing headline candidates block."""

    _write_ready_sources(tmp_path)
    _write_exp2991_candidates(tmp_path)

    too_small = exp.build_artifact(_config(tmp_path, n_tasks=3))
    missing_candidates = exp.build_artifact(_config(tmp_path / "empty", n_tasks=20))

    assert too_small["honest_verdict"] == "blocked: exp3003 preconditions not met"
    assert any(row["resource"] == "exp3003_sample_size" for row in too_small["precondition_checks"])
    assert missing_candidates["honest_verdict"] == "blocked: exp3003 preconditions not met"
    assert any(
        row["resource"] == "hard_set_integrity" and row["available"] is False
        for row in missing_candidates["precondition_checks"]
    )


def test_req_code_3003_missing_metamorphic_and_candidate_provenance_branches(tmp_path: Path) -> None:
    """REQ-CODE-3003: manifest and candidate provenance diagnostics are explicit."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path / "results" / exp.EXP2991_FILENAME,
        {
            "candidate_evaluations": [
                {
                    "item_id": hard.default_items()[0]["item_id"],
                    "model_hf_id": "Qwen/Qwen3.5-0.8B",
                    "candidate_patch_path": "missing.py",
                    "transcript_path": "missing.json",
                },
                {
                    "item_id": hard.default_items()[0]["item_id"],
                    "model_hf_id": "not-a-headline-model",
                    "candidate_patch_path": "missing.py",
                    "transcript_path": "missing.json",
                },
            ]
        },
    )

    missing_meta = exp.build_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            metamorphic_manifest_path=tmp_path / "missing-metamorphic.jsonl",
        )
    )
    no_headline = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path, n_tasks=20))
    candidate_report = exp._load_exp2991_candidates(
        exp.ExperimentConfig(repo_root=tmp_path),
        hard.default_items()[:1],
    )

    assert any(
        row["resource"] == "metamorphic_manifest_integrity" and row["available"] is False
        for row in missing_meta["precondition_checks"]
    )
    assert candidate_report["headline_candidates"] == []
    assert candidate_report["smoke_only_candidate_count"] == 1
    assert candidate_report["check"]["available"] is False
    assert any(
        row["resource"] == "exp2991_headline_candidate_provenance" and row["available"] is False
        for row in no_headline["precondition_checks"]
    )
    assert exp._metamorphic_manifest_path(
        exp.ExperimentConfig(repo_root=tmp_path, metamorphic_manifest_path=tmp_path / "explicit.jsonl"),
        {},
    ) == tmp_path / "explicit.jsonl"


def test_req_code_3003_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-3003: diagnostic helpers cover fallback and parser branches."""

    class Completed:
        returncode = 0
        stdout = "0\n"
        stderr = ""

    monkeypatch.setattr(exp.subprocess, "run", lambda *_args, **_kwargs: Completed())
    assert exp._cuda_status({"precondition_evidence": {"torch_cuda": {"cuda_available": False}}}) == {
        "available": False,
        "source": "exp3001_torch_cuda",
    }
    assert exp._cuda_status({})["source"] == "nvidia-smi"

    monkeypatch.setattr(exp, "cached_sota_pair", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert exp._call_cached_sota_pair() is None
    assert exp._candidate_syntax_success({}, "def ok():\n    return 1\n") is True
    assert exp._candidate_syntax_success({}, "def bad(:\n") is False
    assert exp._relative_or_absolute(tmp_path, tmp_path.parent / "outside.txt").is_absolute()
    parsed = exp._parse_args(["--n-tasks", "21", "--test-run", "focused"])
    assert parsed.n_tasks == 21
    assert parsed.test_run == ["focused"]
