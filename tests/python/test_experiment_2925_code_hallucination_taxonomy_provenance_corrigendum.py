"""Tests for Exp 2925 taxonomy provenance corrigendum.

Spec: REQ-CODE-2925, SCENARIO-CODE-2925.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.eval import code_hallucination_taxonomy_provenance_corrigendum as exp2925


def _load_adversarial_verify() -> Any:
    spec = importlib.util.spec_from_file_location(
        "adversarial_verify_under_test",
        Path("scripts/adversarial_verify.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clock(*values: float):
    ticks = iter(values)
    return lambda: next(ticks)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _candidate(stable_id: str, candidate_index: int, *, passed: bool = False) -> dict[str, Any]:
    return {
        "corpus": "MBPP",
        "stable_id": stable_id,
        "candidate_index": candidate_index,
        "random_seed": 2925 + candidate_index,
        "passed": passed,
        "raw_response_path": f"results/raw/exp2910/{stable_id}_{candidate_index}.txt",
    }


def _label(
    stable_id: str,
    candidate_index: int,
    labels: list[str],
    *,
    passed: bool = False,
) -> dict[str, Any]:
    return {
        "corpus": "MBPP",
        "stable_id": stable_id,
        "candidate_index": candidate_index,
        "task_key": f"MBPP:{stable_id}",
        "random_seed": 2925 + candidate_index,
        "passed": passed,
        "labels": labels,
        "raw_response_path": f"results/raw/exp2910/{stable_id}_{candidate_index}.txt",
    }


def _write_ready_sources(root: Path) -> None:
    """SCENARIO-CODE-2925: fixtures include matching Exp 2910/2911 inventories."""

    raw_manifest = {"raw_response_count": 3, "source": "fixture"}
    _write_json(root, "results/raw/exp2911_raw_manifest.json", raw_manifest)
    candidate_results = [
        _candidate("task-a", 0),
        _candidate("task-a", 1),
        _candidate("task-b", 0, passed=True),
    ]
    _write_json(
        root,
        exp2925.EXP2910_ARTIFACT,
        {
            "artifact": "experiment_2910_sota_code_generation_corrigendum_v2",
            "codegen_corrigendum_ready": True,
            "candidate_count": 3,
            "candidate_results": candidate_results,
            "per_task_results": [
                {"stable_id": "task-a", "candidate_count": 2, "pass_vector": [False, False]},
                {"stable_id": "task-b", "pass_vector": [True]},
            ],
            "model_specs": [
                {
                    "name": "FixtureModel",
                    "hf_id": "fixture/code-model",
                    "model_path": "/models/fixture-code-model.gguf",
                }
            ],
            "models_used": ["fixture/code-model"],
            "random_seed": 2910,
            "reproducibility_checksum": "upstream2910",
        },
    )
    _write_json(
        root,
        exp2925.EXP2911_ARTIFACT,
        {
            "artifact": "experiment_2911_code_hallucination_taxonomy_verifier_v1",
            "code_hallucination_verifier_ready": True,
            "upstream_candidate_count": 3,
            "upstream_per_task_result_count": 2,
            "raw_response_manifest_path": "results/raw/exp2911_raw_manifest.json",
            "per_candidate_labels": [
                _label("task-a", 0, ["syntax_error"]),
                _label("task-a", 1, ["undefined_name", "runtime_error", "true_test_failure"]),
                _label("task-b", 0, ["passed"], passed=True),
            ],
            "taxonomy_categories": list(exp2925.TAXONOMY_CATEGORIES),
            "syntax_error_rate": 1 / 3,
            "undefined_name_rate": 1 / 3,
            "runtime_error_rate": 1 / 3,
            "true_test_failure_rate": 1 / 3,
            "invented_import_rate": 0.0,
            "invented_attribute_or_method_rate": 0.0,
            "invalid_argument_rate": 0.0,
            "inference_substrate": "deterministic_verifier",
            "duration_s": 0.01,
        },
    )


def test_req_code_2925_spec_is_declared() -> None:
    """REQ-CODE-2925: OpenSpec declares the provenance corrigendum contract first."""

    spec = Path("openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2925" in spec
    assert "SCENARIO-CODE-2925" in spec
    assert "deterministic_verifier_no_new_llm_call" in spec


def test_req_code_2925_audit_accepts_deterministic_verifier_substrate() -> None:
    """REQ-CODE-2925: local audit recognizes deterministic verifier provenance."""

    audit = _load_adversarial_verify()
    flags: list[Any] = []
    payload = {
        "inference_substrate": "deterministic_verifier",
        "duration_s": 0.001,
        "upstream_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "model_specs": [{"name": "deterministic verifier", "llm_invoked": False}],
        "random_seed": 2925,
        "reproducibility_checksum": "abc",
    }

    audit.check_duration_vs_claim(payload, flags)
    audit.check_methodology_present(payload, flags)

    assert flags == []


def test_scenario_code_2925_reemits_taxonomy_provenance(tmp_path: Path) -> None:
    """SCENARIO-CODE-2925: matching sources produce a complete provenance artifact."""

    _write_ready_sources(tmp_path)
    audit = {
        "audit_available": True,
        "audit_tool": "fake-adversarial-verify",
        "returncode": 0,
        "flagged": False,
        "findings": [],
        "stderr": "",
    }

    artifact = exp2925.build_artifact(
        tmp_path,
        audit_result=audit,
        started_s=10.0,
        now_s=12.25,
    )

    assert exp2925.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["taxonomy_corrigendum_clean"] is True
    assert artifact["code_hallucination_verifier_ready"] is True
    assert artifact["deterministic_verifier_no_new_llm_call"] is True
    assert artifact["no_new_llm_call"] is True
    assert artifact["random_seed"] == 2925
    assert artifact["candidate_count"] == 3
    assert artifact["inference_substrate"] == "deterministic_verifier"
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["run_date"] == "20260523"
    assert artifact["upstream_model_specs"][0]["hf_id"] == "fixture/code-model"
    assert artifact["upstream_models_used"] == ["fixture/code-model"]
    assert artifact["model_specs"][0]["llm_invoked"] is False
    assert artifact["adversarial_audit_rerun"] == audit

    assert artifact["taxonomy_counts"] == {
        "invented_import": 0,
        "undefined_name": 1,
        "invented_attribute_or_method": 0,
        "invalid_argument": 0,
        "syntax_error": 1,
        "runtime_error": 1,
        "true_test_failure": 1,
    }
    assert artifact["taxonomy_rates"]["undefined_name"] == pytest.approx(1 / 3)
    assert artifact["syntax_error_rate"] == pytest.approx(1 / 3)
    assert artifact["undefined_name_rate"] == pytest.approx(1 / 3)
    assert artifact["true_test_failure_rate"] == pytest.approx(1 / 3)

    checksums = artifact["source_artifact_checksums"]
    assert checksums[str(exp2925.EXP2910_ARTIFACT)] == _sha256(
        tmp_path / exp2925.EXP2910_ARTIFACT
    )
    assert checksums[str(exp2925.EXP2911_ARTIFACT)] == _sha256(
        tmp_path / exp2925.EXP2911_ARTIFACT
    )
    assert checksums["results/raw/exp2911_raw_manifest.json"] == _sha256(
        tmp_path / "results/raw/exp2911_raw_manifest.json"
    )

    validation = artifact["candidate_inventory_validation"]
    assert validation["valid"] is True
    assert validation["exp2910_candidate_results_count"] == 3
    assert validation["exp2910_per_task_candidate_total"] == 3
    assert validation["exp2911_per_candidate_label_count"] == 3
    assert validation["taxonomy_rate_denominator"] == 3

    rebuilt = exp2925.build_artifact(
        tmp_path,
        audit_result={**audit, "returncode": 99},
        started_s=0.0,
        now_s=0.5,
    )
    assert rebuilt["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_code_2925_missing_upstream_blocks_before_audit(tmp_path: Path) -> None:
    """REQ-CODE-2925: absent Exp 2910 or Exp 2911 writes the mandated blocked artifact."""

    artifact = exp2925.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert artifact["taxonomy_corrigendum_clean"] is False
    assert artifact["code_hallucination_verifier_ready"] is False
    assert artifact["candidate_count"] == 0
    assert artifact["adversarial_audit_rerun"]["not_run_reason"] == "upstream_missing"
    assert artifact["source_artifact_checksums"][str(exp2925.EXP2910_ARTIFACT)] is None
    assert artifact["source_artifact_checksums"][str(exp2925.EXP2911_ARTIFACT)] is None
    assert exp2925.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()

    def fail_audit(root: Path, artifact_path: Path) -> dict[str, Any]:
        raise AssertionError("audit must not run when upstream artifacts are absent")

    out = exp2925.write_artifact(tmp_path, audit_runner=fail_audit, clock=_clock(2.0, 2.2))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "blocked_upstream_artifact_missing"


def test_req_code_2925_inventory_mismatch_is_not_clean(tmp_path: Path) -> None:
    """REQ-CODE-2925: candidate-count mismatches are explicit, not silently repaired."""

    _write_ready_sources(tmp_path)
    exp2911_path = tmp_path / exp2925.EXP2911_ARTIFACT
    exp2911 = json.loads(exp2911_path.read_text(encoding="utf-8"))
    exp2911["per_candidate_labels"] = exp2911["per_candidate_labels"][:-1]
    exp2911_path.write_text(json.dumps(exp2911, indent=2, sort_keys=True), encoding="utf-8")

    artifact = exp2925.build_artifact(
        tmp_path,
        audit_result={"audit_available": True, "flagged": False, "findings": []},
        started_s=0.0,
        now_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_candidate_inventory_mismatch"
    assert artifact["taxonomy_corrigendum_clean"] is False
    assert artifact["candidate_count"] == 3
    assert artifact["candidate_inventory_validation"]["valid"] is False
    assert "exp2911_per_candidate_label_count" in artifact["candidate_inventory_validation"][
        "mismatched_fields"
    ]


def test_req_code_2925_write_artifact_persists_final_audited_json(tmp_path: Path) -> None:
    """REQ-CODE-2925: write_artifact records the exact stable audit outcome."""

    _write_ready_sources(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def fake_audit(root: Path, artifact_path: Path) -> dict[str, Any]:
        calls.append((root, artifact_path))
        assert artifact_path.exists()
        return {
            "audit_available": True,
            "audit_tool": "fake-adversarial-verify",
            "returncode": 0,
            "flagged": False,
            "findings": [],
            "stderr": "",
        }

    out = exp2925.write_artifact(
        tmp_path,
        audit_runner=fake_audit,
        clock=_clock(4.0, 4.1, 4.75),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / exp2925.DEFAULT_OUTPUT_PATH
    assert calls == [(tmp_path, out), (tmp_path, out)]
    assert payload["taxonomy_corrigendum_clean"] is True
    assert payload["duration_s"] == pytest.approx(0.75)
    assert payload["adversarial_audit_rerun"]["audit_tool"] == "fake-adversarial-verify"


def test_req_code_2925_write_artifact_records_unstable_audit_fallback(tmp_path: Path) -> None:
    """REQ-CODE-2925: an unstable audit still leaves the latest exact finding recorded."""

    _write_ready_sources(tmp_path)
    calls = 0

    def unstable_audit(root: Path, artifact_path: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "audit_available": True,
            "audit_tool": "unstable-adversarial-verify",
            "returncode": calls,
            "flagged": True,
            "findings": [{"kind": f"KIND_{calls}", "severity": "critical", "detail": "x"}],
            "stderr": "",
        }

    out = exp2925.write_artifact(
        tmp_path,
        audit_runner=unstable_audit,
        clock=_clock(20.0, 20.1, 20.2, 20.3, 20.4),
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert calls == 3
    assert payload["taxonomy_corrigendum_clean"] is False
    assert payload["duration_s"] == pytest.approx(0.4)
    assert payload["adversarial_audit_rerun"]["returncode"] == 3
    assert payload["adversarial_audit_rerun"]["findings"][0]["kind"] == "KIND_3"


def test_req_code_2925_audit_runner_records_available_and_unavailable_tools(
    tmp_path: Path,
) -> None:
    """REQ-CODE-2925: local audit runner preserves exact command output."""

    unavailable = exp2925.run_adversarial_audit(tmp_path, tmp_path / "artifact.json")
    assert unavailable["audit_available"] is False
    assert unavailable["not_run_reason"] == "audit_tool_unavailable"

    script = tmp_path / "scripts" / "adversarial_artifact_audit.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder\n", encoding="utf-8")
    completed = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {
                "reports": [
                    {
                        "flags": [
                            {
                                "kind": "DURATION_TOO_SHORT",
                                "severity": "critical",
                                "detail": "exact",
                            }
                        ]
                    }
                ],
                "flagged_count": 1,
            }
        ),
        stderr="stderr text",
    )
    seen: dict[str, Any] = {}

    def fake_runner(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return completed

    parsed = exp2925.run_adversarial_audit(
        tmp_path,
        tmp_path / "artifact.json",
        runner=fake_runner,
        python_executable="pythonX",
    )

    assert parsed["audit_available"] is True
    assert parsed["audit_tool"] == "scripts/adversarial_artifact_audit.py"
    assert parsed["returncode"] == 1
    assert parsed["flagged"] is True
    assert parsed["findings"] == [
        {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "exact"}
    ]
    assert parsed["stderr"] == "stderr text"
    assert seen["cmd"] == ["pythonX", str(script), str(tmp_path / "artifact.json"), "--json"]
    assert seen["kwargs"]["cwd"] == str(tmp_path)


def test_req_code_2925_defensive_helper_branches(tmp_path: Path) -> None:
    """REQ-CODE-2925: malformed inputs fail closed and helper edges stay covered."""

    _write_ready_sources(tmp_path)
    no_audit = exp2925.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert no_audit["taxonomy_corrigendum_clean"] is False
    assert no_audit["adversarial_audit_rerun"]["not_run_reason"] == "audit_not_supplied"

    assert exp2925.read_json_mapping(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2925.read_json_mapping(bad) == {}
    bad.write_text("{", encoding="utf-8")
    assert exp2925.read_json_mapping(bad) == {}

    assert exp2925._source_checksums(tmp_path, [Path("missing.json")]) == {
        "missing.json": None
    }
    assert exp2925._raw_manifest_paths({"raw_response_manifest_paths": ["a.json", 3]}) == [
        Path("a.json")
    ]
    assert exp2925._raw_manifest_paths({"raw_response_manifest": "one.json"}) == [
        Path("one.json")
    ]
    assert exp2925._as_findings([{"kind": "K", "severity": "warn", "detail": "D"}, "bad"]) == [
        {"kind": "K", "severity": "warn", "detail": "D"}
    ]
    assert exp2925._as_findings(None) == []
    assert exp2925._optional_int(True) is None
    assert exp2925._taxonomy_counts({"taxonomy_counts": {"syntax_error": "2"}})[
        "syntax_error"
    ] == 2
    rates = exp2925._taxonomy_rates(
        {"taxonomy_rates": {"syntax_error": "0.5"}},
        {"syntax_error": 1, **{category: 0 for category in exp2925.TAXONOMY_CATEGORIES}},
        2,
    )
    assert rates["syntax_error"] == pytest.approx(0.5)
    assert rates["undefined_name"] == pytest.approx(0.0)
    assert exp2925._upstream_models_used({}, [{"hf_id": "fallback/model"}]) == [
        "fallback/model"
    ]
    assert exp2925._audit_equivalent({"flagged": False}, {"flagged": True}) is False
    assert exp2925._audit_equivalent(
        {"audit_available": True, "audit_tool": "a", "flagged": False, "returncode": 0},
        {"audit_available": True, "audit_tool": "a", "flagged": False, "returncode": 0},
    )
