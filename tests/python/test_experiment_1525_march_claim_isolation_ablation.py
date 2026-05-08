"""Tests for Exp 1525 MARCH claim-isolation verifier ablation.

Spec: REQ-VERIFY-1525, SCENARIO-VERIFY-1525.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import march_claim_isolation_ablation as mod


def test_req_verify_1525_extracts_stable_atomic_claim_schema() -> None:
    """REQ-VERIFY-1525: extracted claims carry stable IDs and source provenance."""

    row = _promotion_row(
        "case-reject",
        promoted_output={
            "contract_case_id": "case-reject",
            "final_deterministic_decision": "reject",
            "rationale": "The parser failed. The monitor observed a violation.",
        },
    )
    runtime_case = _contract_case("case-reject", expected_label=False, final_accept=False)

    claims = mod.extract_atomic_claims(row, runtime_case=runtime_case)

    assert [claim["claim_id"] for claim in claims] == [
        "case-reject:promoted:001",
        "case-reject:promoted:002",
    ]
    assert claims[0] == {
        "claim_id": "case-reject:promoted:001",
        "contract_case_id": "case-reject",
        "source_mode": "promoted",
        "source_family": "structural_contract",
        "claim_text": "The parser failed",
        "deterministic_expected_label": False,
        "deterministic_final_accept": False,
        "deterministic_final_decision": "reject",
    }


def test_req_verify_1525_routes_checker_acceptance_through_runtime_contracts() -> None:
    """REQ-VERIFY-1525: checker agreement is auxiliary; the deterministic ledger wins."""

    claim = {
        "claim_id": "case-reject:promoted:001",
        "contract_case_id": "case-reject",
        "claim_text": "The contract should accept.",
        "source_mode": "promoted",
        "source_family": "structural_contract",
        "deterministic_expected_label": False,
        "deterministic_final_accept": False,
        "deterministic_final_decision": "reject",
    }
    runtime_case = _contract_case("case-reject", expected_label=False, final_accept=False)

    routed = mod.route_checker_verdict(
        claim,
        runtime_case=runtime_case,
        checker_accept=True,
        mode="claim_isolated",
        model_spec=_model_spec(),
    )

    assert routed["checker_accept"] is True
    assert routed["deterministic_final_accept"] is True
    assert routed["false_accept"] is True
    assert routed["auxiliary_disagreement"] is True
    assert routed["original_answer_visible_to_checker"] is False


def test_scenario_verify_1525_runner_writes_ready_manifest_with_injected_sota(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1525: both modes run and the artifact reports budgets."""

    paths = _write_sources(tmp_path)

    def checker(prompt: str, model: dict[str, Any], mode: str, claim: dict[str, Any]) -> str:
        assert model["hf_id"] == mod.MANDATED_MODEL_SPECS[0]["hf_id"]
        if mode == "claim_isolated":
            assert "Original answer:" not in prompt
        else:
            assert "Original answer:" in prompt
        decision = "reject" if claim["deterministic_expected_label"] is False else "accept"
        return json.dumps(
            {
                "claim_id": claim["claim_id"],
                "checker_decision": decision,
            }
        )

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        manifest_path=paths["manifest"],
        promotion_artifact_path=paths["promotion_artifact"],
        promotion_manifest_path=paths["promotion_manifest"],
        runtime_contract_artifact_path=paths["runtime_artifact"],
        runtime_contract_manifest_path=paths["runtime_manifest"],
        cached_pair_fn=lambda **_: [_model_spec()],
        checker_fn=checker,
        max_models=1,
    )
    rows = _read_jsonl(paths["manifest"])

    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["claim_isolation_ablation_ready"] is True
    assert artifact["cases_loaded"] == 1
    assert artifact["claims_extracted"] == 2
    assert artifact["full_context_accept_rate"] == pytest.approx(0.0)
    assert artifact["claim_isolated_accept_rate"] == pytest.approx(0.0)
    assert artifact["claim_isolation_delta"] == pytest.approx(0.0)
    assert artifact["verifier_calls_full_context"] == 1
    assert artifact["verifier_calls_claim_isolated"] == 2
    assert artifact["budget_delta"] == 1
    assert artifact["false_accept_count"] == 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["models_used"] == [mod.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "summary"
    assert rows[0]["row_type"] == "claim_isolation_evaluation"
    assert rows[0]["claim_isolated"]["original_answer_visible_to_checker"] is False
    mod.validate_artifact(artifact, manifest_path=paths["manifest"])


def test_scenario_verify_1525_blocks_without_mandated_sota_runtime(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1525: legacy tiny models are not used for headline rows."""

    paths = _write_sources(tmp_path)
    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        output_path=paths["output"],
        manifest_path=paths["manifest"],
        promotion_artifact_path=paths["promotion_artifact"],
        promotion_manifest_path=paths["promotion_manifest"],
        runtime_contract_artifact_path=paths["runtime_artifact"],
        runtime_contract_manifest_path=paths["runtime_manifest"],
        cached_pair_fn=lambda **_: [
            {"hf_id": "legacy/tiny", "model_path": "/models/tiny.gguf", "name": "tiny"}
        ],
        resolver_fn=lambda _hf_id: None,
        checker_fn=lambda *_args, **_kwargs: "must not be called",
        max_models=1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["claim_isolation_ablation_ready"] is False
    assert artifact["models_used"] == []
    assert "no_mandated_sota_gguf_runtime" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1525_defensive_fallback_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-1525: malformed auxiliary inputs stay deterministic."""

    runtime_case = _contract_case("fallback", expected_label=None, final_accept=True)
    no_text_row = _promotion_row(
        "fallback",
        promoted_output={"contract_case_id": "fallback", "final_deterministic_decision": "accept"},
    )
    no_text_row["runtime_contract_validation"]["promoted"]["raw_output_excerpt"] = ""
    assert (
        "final_deterministic_decision"
        in mod.extract_atomic_claims(
            no_text_row,
            runtime_case=runtime_case,
        )[0]["claim_text"]
    )

    empty_row = {"contract_case_id": "fallback", "runtime_contract_validation": {"promoted": {}}}
    assert mod.extract_atomic_claims(empty_row, runtime_case=runtime_case)[0]["claim_text"] == (
        "candidate output"
    )
    empty_runtime = dict(runtime_case, proposed_output="")
    assert mod.extract_atomic_claims(empty_row, runtime_case=empty_runtime)[0]["claim_text"] == (
        "runtime contract fallback is checked"
    )

    raw_row = {
        "contract_case_id": "fallback",
        "runtime_contract_validation": {
            "promoted": {"raw_output_excerpt": "Raw checker sentence. Another sentence."}
        },
    }
    assert mod.extract_atomic_claims(raw_row, runtime_case=runtime_case)[0]["claim_text"] == (
        "Raw checker sentence"
    )

    decision_row = {
        "contract_case_id": "fallback",
        "runtime_contract_validation": {
            "promoted": {
                "parsed_contract_output": {"final_deterministic_decision": "accept"},
                "raw_output_excerpt": "",
            }
        },
    }
    assert mod._fallback_claim_text(decision_row, runtime_case=runtime_case) == (
        "final deterministic decision is accept"
    )

    assert mod.parse_checker_output("plain text") == (False, "no_json_object")
    assert mod.parse_checker_output('bad {"checker_accept": true}') == (True, "ok")
    assert mod.parse_checker_output('{"checker_decision": "maybe"}') == (
        False,
        "missing_checker_decision",
    )
    assert mod.parse_checker_output("{not-json}") == (False, "no_json_object")
    assert mod._promoted_validation({"runtime_contract_validation": []}) == {}
    assert (
        mod._original_answer(
            {
                "promotion_row": {
                    "runtime_contract_validation": {
                        "promoted": {"parsed_contract_output": {"answer": "from parsed"}}
                    }
                },
                "runtime_case": runtime_case,
            }
        )
        == '{"answer": "from parsed"}'
    )
    assert mod._original_answer({"promotion_row": {}, "runtime_case": runtime_case}) == (
        "candidate output"
    )

    joined = mod._select_joined_cases(
        [
            {"row_type": "summary"},
            {"row_type": "policy_promotion_evaluation", "contract_case_id": "missing"},
            _promotion_row("fallback"),
            _promotion_row("extra"),
        ],
        [runtime_case, _contract_case("extra")],
        limit=1,
    )
    assert [row["contract_case_id"] for row in joined] == ["fallback"]

    resolved = mod._resolve_runtime_models(
        lambda **_: None,
        lambda hf_id: (
            "/models/qwen.gguf" if hf_id == mod.MANDATED_MODEL_SPECS[0]["hf_id"] else None
        ),
        max_models=1,
    )
    assert resolved[0]["hf_id"] == mod.MANDATED_MODEL_SPECS[0]["hf_id"]

    paths = _write_sources(tmp_path)
    paths["promotion_artifact"].unlink()
    _promotion_rows, _runtime_rows, blockers = mod._load_required_sources(
        {
            "promotion_artifact": paths["promotion_artifact"],
            "promotion_manifest": paths["promotion_manifest"],
            "runtime_artifact": paths["runtime_artifact"],
            "runtime_manifest": paths["runtime_manifest"],
        }
    )
    assert any(blocker.startswith("missing_artifact:") for blocker in blockers)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    malformed_blockers: list[str] = []
    assert mod._load_json_or_blocker(malformed, malformed_blockers) is None
    assert any(blocker.startswith("malformed_artifact:") for blocker in malformed_blockers)

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{}\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == [{}]
    assert mod._display_path(Path("/outside/repo/file.json"), project_root=tmp_path) == (
        "/outside/repo/file.json"
    )

    no_case_dir = tmp_path / "no_case_src"
    no_case_paths = _write_sources(no_case_dir)
    _write_jsonl(no_case_paths["promotion_manifest"], [_promotion_row("not-in-runtime")])
    no_case = mod.run_experiment(
        project_root=no_case_dir,
        run_date="20260508",
        output_path=no_case_paths["output"],
        manifest_path=no_case_paths["manifest"],
        promotion_artifact_path=no_case_paths["promotion_artifact"],
        promotion_manifest_path=no_case_paths["promotion_manifest"],
        runtime_contract_artifact_path=no_case_paths["runtime_artifact"],
        runtime_contract_manifest_path=no_case_paths["runtime_manifest"],
        cached_pair_fn=lambda **_: None,
        resolver_fn=lambda _hf_id: None,
    )
    assert "no_promoted_runtime_contract_cases" in no_case["blockers"]
    assert "no_atomic_claims_extracted" in no_case["blockers"]


def _write_sources(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "output": tmp_path / "experiment_1525_march_claim_isolation_verifier_ablation.json",
        "manifest": tmp_path / "march_claim_isolation_1525.jsonl",
        "promotion_artifact": tmp_path / "experiment_1524.json",
        "promotion_manifest": tmp_path / "promotion.jsonl",
        "runtime_artifact": tmp_path / "experiment_1520.json",
        "runtime_manifest": tmp_path / "runtime.jsonl",
    }
    _write_json(
        paths["promotion_artifact"],
        {"status": "complete", "live_policy_promotion_ready": True},
    )
    _write_jsonl(paths["promotion_manifest"], [_promotion_row("case-reject")])
    _write_json(
        paths["runtime_artifact"],
        {"status": "complete", "runtime_contract_e2e_ready": True},
    )
    _write_jsonl(paths["runtime_manifest"], [_contract_case("case-reject")])
    return paths


def _promotion_row(
    contract_case_id: str,
    *,
    promoted_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    promoted_output = promoted_output or {
        "contract_case_id": contract_case_id,
        "final_deterministic_decision": "reject",
        "rationale": "The certificate is invalid. The contract evidence fails.",
    }
    return {
        "row_type": "policy_promotion_evaluation",
        "model_hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"],
        "policy_update_id": "daily_eval:eligible",
        "contract_case_id": contract_case_id,
        "prompt_or_case_id": contract_case_id,
        "source_family": "structural_contract",
        "baseline_task_success": False,
        "promoted_task_success": True,
        "runtime_contract_validation": {
            "promoted": {
                "raw_output_excerpt": json.dumps(promoted_output),
                "parsed_contract_output": promoted_output,
            }
        },
    }


def _contract_case(
    contract_case_id: str,
    *,
    expected_label: bool | None = False,
    final_accept: bool = False,
) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_schema_version": "runtime-contract-e2e/v1",
        "contract_case_id": contract_case_id,
        "prompt_or_case_id": contract_case_id,
        "proposed_output": "candidate output",
        "certificate_parse_result": {"linked": False},
        "safe_dsl_verifier_result": {"linked": False},
        "monitor_event_result": {"linked": False},
        "structural_contract_result": {"linked": True, "contract_family": "unit"},
        "expected_label": expected_label,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
        "source_family": "structural_contract",
        "source_path": "source.jsonl",
        "source_line": 1,
    }


def _model_spec() -> dict[str, Any]:
    return {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"],
        "gpu": 0,
        "model_path": "/models/qwen.gguf",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
