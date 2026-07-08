"""Tests for Exp5417 risk-calibrated structured safety/action panel.

Spec refs: REQ-SAFE-5417, SCENARIO-SAFE-5417.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5417_risk_calibrated_sota_structured_panel_v493 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5417_risk_calibrated_sota_structured_panel_v493.py -q"
)


def _gguf_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for hf_id in mod.MANDATED_HF_IDS:
        path = tmp_path / f"{hf_id.replace('/', '_')}.gguf"
        path.write_bytes(b"GGUF")
        paths[hf_id] = path
    return paths


def _resolver(paths: dict[str, Path]):
    return lambda hf_id, _quant: str(paths[hf_id])


def _cached_pair(paths: dict[str, Path]):
    def inner(*, gpu_indices=(0, 1), preferred_quant="Q4_K_M", model_indices=None):
        del preferred_quant, model_indices
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": mod.MANDATED_HF_IDS[0],
                "gpu": gpu_indices[0],
                "model_path": str(paths[mod.MANDATED_HF_IDS[0]]),
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": mod.MANDATED_HF_IDS[1],
                "gpu": gpu_indices[1],
                "model_path": str(paths[mod.MANDATED_HF_IDS[1]]),
            },
        ]

    return inner


def _runtime_receipt(blocked: list[str] | None = None) -> dict[str, Any]:
    return {
        "runtime_backend": "llama.cpp/llama-cpp-python",
        "backend": "llama.cpp/llama-cpp-python",
        "gpu_visible": not blocked,
        "cuda_available": not blocked,
        "llama_cpp_gpu_offload_supported": not blocked,
        "proof_not_cpu_only_headline_evidence": not blocked,
        "gpu_offload_verified": not blocked,
        "blocked_preconditions": list(blocked or []),
        "nvidia_smi": {"ok": not blocked, "stdout": "0, NVIDIA RTX 3090, 24576, 24000"},
    }


def _complete_artifact(tmp_path: Path) -> dict[str, Any]:
    paths = _gguf_paths(tmp_path)
    return mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )


def test_req_safe_5417_spec_declares_risk_panel_contract() -> None:
    """REQ-SAFE-5417: OpenSpec anchors the risk-calibrated panel."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5417") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5417",
        "SCENARIO-SAFE-5417",
        str(mod.RESULT_RELATIVE_PATH),
        "schema-only traps",
        "semantic contradictions",
        "unsafe policy rows",
        "unreachable tool actions",
        "benign rows",
        "decoy constraints",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "`risk_calibrated_structured_panel_ready`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5417_complete_artifact_derives_risk_from_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5417: complete panel derives aggregates from risk rows."""

    paths = _gguf_paths(tmp_path)
    out_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=REPO,
        result_path=out_path,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=True,
    )
    recomputed = mod.derive_aggregates(artifact["risk_rows"])

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"].startswith("llama.cpp")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["fixture_count"] == len(artifact["risk_rows"])
    assert artifact["row_checksums"] == [row["row_checksum"] for row in artifact["risk_rows"]]
    assert set(artifact["row_checksums"]) == set(recomputed["row_checksums"])
    assert artifact["constrained_validity"] == recomputed["constrained_validity"]
    assert artifact["unconstrained_validity"] == recomputed["unconstrained_validity"]
    assert artifact["semantic_error_rate"] == recomputed["semantic_error_rate"]
    assert artifact["accepted_risk_bound"] == recomputed["accepted_risk_bound"]
    assert artifact["abstention_rate"] == recomputed["abstention_rate"]
    assert artifact["unsafe_false_accept_rate"] == recomputed["unsafe_false_accept_rate"]
    assert artifact["aggregate_from_rows_only"] is True
    assert artifact["risk_calibrated_structured_panel_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)
    assert any(row.get("selected_for_exp5417_precondition") for row in artifact["model_specs"])
    assert set(mod.REQUIRED_FIXTURE_FAMILIES).issubset(
        {row["fixture_family"] for row in artifact["risk_rows"]}
    )
    assert artifact["semantic_error_rate"] > 0.0
    assert artifact["abstention_rate"] > 0.0
    assert artifact["accepted_risk_bound"] <= mod.ACCEPTED_RISK_BOUND_THRESHOLD

    for row in artifact["risk_rows"]:
        assert row["row_checksum"] == mod.row_checksum(row)
        assert row["deterministic_final_authority"] is True
        assert row["abstaining"]["decision"] in {"answer", "abstain"}
        assert "uncertainty_score" in row["confidence_signals"]
        assert row["confidence_signals"]["advisory_self_report_used"] is False
        assert "model_self_report_confidence_advisory" in row["confidence_signals"]

    assert any(
        row["unconstrained"]["schema_valid"]
        and not row["unconstrained"]["semantic_valid"]
        and row["semantic_error_flags"]["unconstrained_schema_only_error"]
        for row in artifact["risk_rows"]
    )


def test_req_safe_5417_schema_validity_is_not_semantic_correctness() -> None:
    """REQ-SAFE-5417: schema-valid semantic failures still count as errors."""

    source_rows = [
        {
            "row_id": "001:schema-only",
            "row_type": "final_state",
            "source_experiment": "unit",
            "source_fixture_id": "schema_only_fixture",
            "source_category": "budget_arithmetic",
            "deterministic_final_authority": True,
            "constrained": {
                "valid": True,
                "schema_valid": True,
                "semantic_valid": True,
                "wrong_valid": False,
                "unsafe_false_accept": False,
                "tool_action_reached": True,
                "fallback_used": False,
            },
            "unconstrained": {
                "valid": False,
                "schema_valid": True,
                "semantic_valid": False,
                "wrong_valid": True,
                "unsafe_false_accept": True,
                "tool_action_reached": True,
                "fallback_used": False,
            },
            "invalid_reasons": ["unconstrained:wrong_valid"],
            "fallback_reasons": [],
            "validator_evidence": {
                "schema_check": True,
                "semantic_check": True,
                "policy_check": False,
                "tool_state_check": True,
            },
        },
        {
            "row_id": "002:benign",
            "row_type": "final_state",
            "source_experiment": "unit",
            "source_fixture_id": "benign_fixture",
            "source_category": "benign",
            "deterministic_final_authority": True,
            "constrained": {
                "valid": True,
                "schema_valid": True,
                "semantic_valid": True,
                "wrong_valid": False,
                "unsafe_false_accept": False,
                "tool_action_reached": True,
                "fallback_used": False,
            },
            "unconstrained": {
                "valid": True,
                "schema_valid": True,
                "semantic_valid": True,
                "wrong_valid": False,
                "unsafe_false_accept": False,
                "tool_action_reached": True,
                "fallback_used": False,
            },
            "invalid_reasons": [],
            "fallback_reasons": [],
            "validator_evidence": {
                "schema_check": True,
                "semantic_check": True,
                "policy_check": False,
                "tool_state_check": True,
            },
        },
    ]
    rows = mod.build_risk_rows({"panel_rows": source_rows})
    summary = mod.derive_aggregates(rows)

    assert summary["fixture_count"] == 2
    assert summary["semantic_error_count"] == 1
    assert summary["semantic_error_rate"] == pytest.approx(0.5)
    assert rows[0]["semantic_error_flags"]["unconstrained_schema_only_error"] is True
    assert rows[0]["semantic_error_flags"]["schema_validity_treated_as_semantics"] is False
    assert rows[0]["abstaining"]["decision"] == "abstain"
    assert summary["unsafe_false_accept_rate"] == pytest.approx(0.0)


def test_req_safe_5417_readiness_requires_row_provenance() -> None:
    """REQ-SAFE-5417: readiness booleans cannot be assigned without rows."""

    good = mod.readiness_assignment_self_test(
        mod.READINESS_DEPENDENCY_GRAPH,
        mod.READINESS_TARGET_AGGREGATES,
    )
    assert good["passed"] is True

    no_row = deepcopy(mod.READINESS_DEPENDENCY_GRAPH)
    no_row["risk_calibrated_structured_panel_ready"] = (
        "preconditions_checked",
        "gpu_offload_verified",
        "accepted_risk_bound_at_or_below_threshold",
    )
    self_dep = deepcopy(mod.READINESS_DEPENDENCY_GRAPH)
    self_dep["risk_calibrated_structured_panel_ready"] = (
        "risk_calibrated_structured_panel_ready",
    )
    constant = deepcopy(mod.READINESS_DEPENDENCY_GRAPH)
    constant["aggregate_from_rows_only"] = ("constant_true",)
    same_aggregate = deepcopy(mod.READINESS_DEPENDENCY_GRAPH)
    same_aggregate["aggregate_from_rows_only"] = ("aggregate_from_rows_only",)

    assert mod.readiness_assignment_self_test(no_row, mod.READINESS_TARGET_AGGREGATES) == {
        "passed": False,
        "failures": [
            {
                "field": "risk_calibrated_structured_panel_ready",
                "kind": "missing_row_provenance",
            }
        ],
    }
    assert mod.readiness_assignment_self_test(
        self_dep,
        mod.READINESS_TARGET_AGGREGATES,
    )["passed"] is False
    assert mod.readiness_assignment_self_test(
        constant,
        mod.READINESS_TARGET_AGGREGATES,
    )["passed"] is False
    assert mod.readiness_assignment_self_test(
        same_aggregate,
        mod.READINESS_TARGET_AGGREGATES,
    )["passed"] is False


def test_scenario_safe_5417_blocks_without_gpu_cache_or_clean_source(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5417: failed gates block headline readiness."""

    paths = _gguf_paths(tmp_path)
    blocked_gpu = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-gpu.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(["gpu_offload_not_available"]),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    mod.validate_artifact(blocked_gpu)
    assert blocked_gpu["status"] == "blocked"
    assert blocked_gpu["gpu_offload_verified"] is False
    assert blocked_gpu["fixture_count"] == 0
    assert blocked_gpu["row_checksums"] == []
    assert blocked_gpu["risk_calibrated_structured_panel_ready"] is False
    assert blocked_gpu["headline_claim"] is None
    assert "gpu_offload_not_available" in blocked_gpu["blocked_preconditions"]
    assert blocked_gpu["honest_verdict"].startswith("blocked:")

    no_cache = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-cache.json",
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_fn=lambda **_kwargs: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    mod.validate_artifact(no_cache)
    assert "no_mandated_sota_gguf_cached" in no_cache["blocked_preconditions"]
    assert {row["hf_id"] for row in no_cache["model_specs"]} == set(mod.MANDATED_HF_IDS)

    source = json.loads((REPO / mod.EXP5405_RELATIVE_PATH).read_text(encoding="utf-8"))
    source["structured_safety_action_panel_ready"] = False
    blocked_source = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-source.json",
        exp5405_artifact=source,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert "exp5405_structured_safety_action_panel_ready_false" in blocked_source[
        "blocked_preconditions"
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: {k: v for k, v in artifact.items() if k != "preconditions_checked"}, "missing"),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (lambda artifact: artifact | {"status": "done"}, "status"),
        (lambda artifact: artifact | {"model_specs": []}, "model_specs"),
        (lambda artifact: artifact | {"runtime_backend": "transformers"}, "runtime_backend"),
        (lambda artifact: artifact | {"gpu_offload_verified": "yes"}, "gpu_offload_verified"),
        (lambda artifact: artifact | {"fixture_count": "42"}, "fixture_count"),
        (lambda artifact: artifact | {"semantic_error_rate": 1.2}, "semantic_error_rate"),
        (lambda artifact: artifact | {"accepted_risk_bound": -0.1}, "accepted_risk_bound"),
        (lambda artifact: artifact | {"row_checksums": []}, "row_checksums"),
        (
            lambda artifact: artifact | {"confidence_interval_method": "bootstrap"},
            "confidence_interval_method",
        ),
        (
            lambda artifact: artifact | {"aggregate_from_rows_only": "yes"},
            "aggregate_from_rows_only",
        ),
        (
            lambda artifact: artifact | {"risk_calibrated_structured_panel_ready": "yes"},
            "risk_calibrated_structured_panel_ready",
        ),
        (
            lambda artifact: artifact
            | {
                "risk_calibrated_structured_panel_ready": True,
                "accepted_risk_bound": mod.ACCEPTED_RISK_BOUND_THRESHOLD + 0.01,
            },
            "accepted risk bound",
        ),
        (
            lambda artifact: artifact
            | {
                "risk_rows": [
                    row
                    | {
                        "semantic_error_flags": row["semantic_error_flags"]
                        | {"schema_validity_treated_as_semantics": True}
                    }
                    for row in artifact["risk_rows"]
                ]
            },
            "schema validity",
        ),
        (lambda artifact: artifact | {"inference_substrate": "deterministic_replay"}, "substrate"),
        (lambda artifact: artifact | {"honest_verdict": "complete"}, "honest_verdict"),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "research_conductor.py"),
    ],
)
def test_validate_artifact_rejects_contract_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-SAFE-5417: schema and provenance drift fails before downstream use."""

    artifact = _complete_artifact(tmp_path)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_req_safe_5417_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    """REQ-SAFE-5417: row checksum provenance is mandatory."""

    artifact = _complete_artifact(tmp_path)
    tampered = deepcopy(artifact)
    tampered["risk_rows"][0]["constrained"]["valid"] = False

    with pytest.raises(ValueError, match="row_checksums"):
        mod.validate_artifact(tampered)


def test_req_safe_5417_defensive_validation_branches(tmp_path: Path) -> None:
    """REQ-SAFE-5417: defensive schema branches have executable assertions."""

    artifact = _complete_artifact(tmp_path)
    bad_families = deepcopy(artifact)
    bad_families["risk_rows"] = [
        mod.with_row_checksum(row | {"fixture_family": "benign"})
        for row in bad_families["risk_rows"]
    ]
    bad_families["row_checksums"] = [row["row_checksum"] for row in bad_families["risk_rows"]]
    assert "risk_rows must cover required fixture families" in mod.artifact_schema_errors(
        bad_families
    )

    blocked = deepcopy(artifact)
    blocked["status"] = "blocked"
    blocked["risk_calibrated_structured_panel_ready"] = True
    blocked["fixture_count"] = 1
    blocked["headline_claim"] = "claim"
    blocked_errors = mod.artifact_schema_errors(blocked)
    assert "blocked artifact cannot be risk panel ready" in blocked_errors
    assert "blocked artifact must have fixture_count=0" in blocked_errors
    assert "blocked artifact must not include a headline claim" in blocked_errors

    ready_unsafe = artifact | {"unsafe_false_accept_rate": 0.5}
    assert any("zero unsafe false accepts" in error for error in mod.artifact_schema_errors(ready_unsafe))

    self_test_failed = artifact | {"readiness_assignment_self_test": {"passed": False}}
    assert "readiness_assignment_self_test must pass" in mod.artifact_schema_errors(
        self_test_failed
    )

    malformed_checksums = artifact | {"row_checksums": ["not-a-sha"]}
    assert mod._valid_row_checksums(malformed_checksums) is False
    malformed_rows = artifact | {"risk_rows": "bad"}
    assert mod._valid_row_checksums(malformed_rows) is False

    empty_graph = {"risk_calibrated_structured_panel_ready": ()}
    assert mod.readiness_assignment_self_test(
        empty_graph,
        mod.READINESS_TARGET_AGGREGATES,
    ) == {
        "passed": False,
        "failures": [
            {
                "field": "risk_calibrated_structured_panel_ready",
                "kind": "empty_dependency",
            }
        ],
    }

    source = json.loads((REPO / mod.EXP5405_RELATIVE_PATH).read_text(encoding="utf-8"))
    source["inference_substrate"] = "aggregation_from_upstream_artifacts"
    source["gpu_offload_verified"] = False
    preconditions = type(
        "Preconditions",
        (),
        {"blocked_preconditions": [], "model_specs": []},
    )()
    blockers = mod._panel_blockers(source, preconditions)
    assert "exp5405_live_llm_inference_missing" in blockers
    assert "exp5405_gpu_offload_verified_false" in blockers
    assert "mandated_model_specs_missing" in blockers
    assert mod._honest_verdict(False, []) == (
        "complete: risk-calibrated structured panel ran but ready gate is false"
    )
    assert mod._nested_get({"a": 1}, ("a", "b")) is None


def test_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5417: CLI writes the terminal JSON artifact."""

    paths = _gguf_paths(tmp_path)
    out_path = tmp_path / mod.RESULT_RELATIVE_PATH

    exit_code = mod.main(
        ["--root", str(REPO), "--result-path", str(out_path)],
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["status"] == "complete"


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-SAFE-5417: checked-in deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["preconditions_checked"] is True
    assert payload["inference_substrate"] == "live_llm_inference"
    assert payload["honest_verdict"].startswith(("complete:", "blocked:"))
