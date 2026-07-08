"""Tests for Exp5431 structured constraint taxonomy replication.

Spec refs: REQ-SAFE-5431, SCENARIO-SAFE-5431.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5392_formal_encoding_safety_fixture_v491 as exp5392
from carnot import experiment_5431_structured_constraint_taxonomy_replication_v494 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5431_structured_constraint_taxonomy_replication_v494.py -q"
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
                "name": "gemma-4-31B-it",
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


def _preconditions(paths: dict[str, Path], blocked: list[str] | None = None):
    return exp5392.PreconditionResult(
        blocked_preconditions=list(blocked or []),
        model_specs=[
            {
                "hf_id": hf_id,
                "model_path": str(paths[hf_id]),
                "status": "local_gguf_resolved",
            }
            for hf_id in mod.MANDATED_HF_IDS
        ],
        selected_model_spec={
            "hf_id": mod.MANDATED_HF_IDS[0],
            "model_path": str(paths[mod.MANDATED_HF_IDS[0]]),
        },
        gpu_offload_receipt=_runtime_receipt(blocked),
    )


def _complete_artifact(tmp_path: Path) -> dict[str, Any]:
    paths = _gguf_paths(tmp_path)
    return mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}],
        write=False,
    )


def test_req_safe_5431_spec_declares_taxonomy_replication_contract() -> None:
    """REQ-SAFE-5431: OpenSpec anchors the taxonomy replication."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5431") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5431",
        "SCENARIO-SAFE-5431",
        str(mod.RESULT_RELATIVE_PATH),
        "structured_corrigendum_clean=true",
        "schema-only traps",
        "semantic contradictions",
        "policy violations",
        "unreachable tool actions",
        "ontology/triple updates",
        "API-like tool calls",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "`inference_substrate` MUST be `live_llm_inference`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5431_complete_artifact_derives_metrics_from_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5431: complete artifact uses row-derived taxonomy metrics."""

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
    recomputed = mod.derive_metrics(artifact["taxonomy_rows"])

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["preconditions_checked"] is True
    assert artifact["gated_upstream_clean"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"].startswith("llama.cpp")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["fixture_count"] == len(artifact["taxonomy_rows"])
    assert artifact["row_checksums"] == [row["row_checksum"] for row in artifact["taxonomy_rows"]]
    assert artifact["constraint_family_counts"] == recomputed["constraint_family_counts"]
    assert artifact["semantic_false_accept_rate"] == recomputed["semantic_false_accept_rate"]
    assert artifact["unsafe_false_accept_rate"] == recomputed["unsafe_false_accept_rate"]
    assert (
        artifact["unreachable_action_false_accept_rate"]
        == recomputed["unreachable_action_false_accept_rate"]
    )
    assert artifact["abstention_rate"] == recomputed["abstention_rate"]
    assert artifact["accepted_risk_bound"] == recomputed["accepted_risk_bound"]
    assert artifact["metric_independence_checks_passed"] is True
    assert artifact["structured_taxonomy_replication_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)

    families = {row["constraint_family"] for row in artifact["taxonomy_rows"]}
    assert set(mod.REQUIRED_CONSTRAINT_FAMILIES).issubset(families)
    assert artifact["constraint_family_counts"]["ontology_triple_update"] > 0
    assert artifact["constraint_family_counts"]["api_like_tool_call"] > 0
    assert any(row["source_model_hf_id"] in mod.MANDATED_HF_IDS for row in artifact["taxonomy_rows"])
    for row in artifact["taxonomy_rows"]:
        assert row["row_checksum"] == mod.row_checksum(row)
        assert row["deterministic_authority"]["final_authority"] is True
        assert row["model_self_report_advisory_only"] is True


def test_req_safe_5431_dirty_upstream_skips_runtime_and_rows(tmp_path: Path) -> None:
    """REQ-SAFE-5431: dirty Exp5430 blocks before runtime or generation work."""

    source = json.loads((REPO / mod.EXP5430_RELATIVE_PATH).read_text(encoding="utf-8"))
    source["structured_corrigendum_clean"] = False
    source["honest_verdict"] = "blocked: forced dirty upstream for test"

    def fail_runtime(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("runtime preflight must not run when Exp5430 is dirty")

    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        exp5430_artifact=source,
        runtime_probe=fail_runtime,
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text()) == artifact
    assert artifact["preconditions_checked"] is True
    assert artifact["gated_upstream_clean"] is False
    assert artifact["gpu_offload_verified"] is False
    assert artifact["fixture_count"] == 0
    assert artifact["taxonomy_rows"] == []
    assert artifact["structured_taxonomy_replication_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_safe_5431_metric_independence_rejects_drift() -> None:
    """REQ-SAFE-5431: copied aggregate values fail metric independence validation."""

    rows = [
        mod.taxonomy_row_with_checksum(
            {
                "row_id": "001:semantic",
                "source_row_id": "risk-001",
                "source_row_checksum": "a" * 64,
                "source_fixture_id": "semantic_case",
                "source_model_hf_id": mod.MANDATED_HF_IDS[0],
                "constraint_family": "semantic_contradiction",
                "deterministic_authority": {
                    "schema_valid": True,
                    "semantic_valid": False,
                    "policy_safe": True,
                    "risk_accepted": True,
                    "abstained": False,
                    "action_reachable": True,
                    "finite_domain_valid": True,
                    "final_authority": True,
                },
                "accepted": True,
                "model_self_report_advisory_only": True,
            }
        ),
        mod.taxonomy_row_with_checksum(
            {
                "row_id": "002:abstain",
                "source_row_id": "risk-002",
                "source_row_checksum": "b" * 64,
                "source_fixture_id": "policy_case",
                "source_model_hf_id": mod.MANDATED_HF_IDS[0],
                "constraint_family": "policy_violation",
                "deterministic_authority": {
                    "schema_valid": True,
                    "semantic_valid": True,
                    "policy_safe": False,
                    "risk_accepted": False,
                    "abstained": True,
                    "action_reachable": True,
                    "finite_domain_valid": True,
                    "final_authority": True,
                },
                "accepted": False,
                "model_self_report_advisory_only": True,
            }
        ),
        mod.taxonomy_row_with_checksum(
            {
                "row_id": "003:clean",
                "source_row_id": "risk-003",
                "source_row_checksum": "c" * 64,
                "source_fixture_id": "clean_case",
                "source_model_hf_id": mod.MANDATED_HF_IDS[0],
                "constraint_family": "benign",
                "deterministic_authority": {
                    "schema_valid": True,
                    "semantic_valid": True,
                    "policy_safe": True,
                    "risk_accepted": True,
                    "abstained": False,
                    "action_reachable": True,
                    "finite_domain_valid": True,
                    "final_authority": True,
                },
                "accepted": True,
                "model_self_report_advisory_only": True,
            }
        ),
    ]
    metrics = mod.derive_metrics(rows)

    assert metrics["semantic_false_accept_rate"] == metrics["abstention_rate"]
    assert metrics["metric_independence_checks_passed"] is True
    assert metrics["predicate_support"]["semantic_false_accept_row_ids"] != metrics[
        "predicate_support"
    ]["abstention_row_ids"]

    artifact = mod.build_artifact(
        exp5430_artifact={"structured_corrigendum_clean": True, "model_specs": []},
        preconditions=_preconditions(
            {hf_id: Path(f"/tmp/{hf_id.replace('/', '_')}.gguf") for hf_id in mod.MANDATED_HF_IDS}
        ),
        rows=rows,
        blocked_preconditions=[],
        tests_run=[TEST_COMMAND],
    )
    drifted = deepcopy(artifact)
    drifted["semantic_false_accept_rate"] = 0.0
    with pytest.raises(ValueError, match="semantic_false_accept_rate"):
        mod.validate_artifact(drifted)


def test_req_safe_5431_validation_reports_guard_failures(tmp_path: Path) -> None:
    """REQ-SAFE-5431: schema guards fail closed on malformed artifacts."""

    base = _complete_artifact(tmp_path)
    cases: list[tuple[str, Any, str]] = [
        ("field_principles", {}, "field_principles"),
        ("preconditions_checked", False, "preconditions_checked"),
        ("gated_upstream_clean", "yes", "gated_upstream_clean"),
        ("model_specs", [], "model_specs"),
        ("runtime_backend", "transformers", "runtime_backend"),
        ("gpu_offload_verified", "true", "gpu_offload_verified"),
        ("fixture_count", -1, "fixture_count"),
        ("constraint_family_counts", [], "constraint_family_counts"),
        ("row_checksums", ["bad"], "row_checksums"),
        ("semantic_false_accept_rate", 2.0, "semantic_false_accept_rate"),
        ("metric_independence_checks_passed", "yes", "metric_independence_checks_passed"),
        ("inference_substrate", "cached_text", "inference_substrate"),
        ("honest_verdict", "done", "honest_verdict"),
        ("taxonomy_rows", "bad", "taxonomy_rows"),
        ("research_conductor_modified", True, "research_conductor.py"),
    ]
    for field, value, expected in cases:
        bad = deepcopy(base)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(base)
    del missing["model_specs"]
    assert "missing required fields" in "; ".join(mod.artifact_schema_errors(missing))

    ready_cases: list[tuple[str, Any, str]] = [
        ("gated_upstream_clean", False, "Exp5430 clean"),
        ("gpu_offload_verified", False, "GPU offload"),
        ("metric_independence_checks_passed", False, "independence"),
        ("semantic_false_accept_rate", 0.1, "zero semantic_false_accept_rate"),
        ("accepted_risk_bound", 1.0, "accepted risk bound"),
    ]
    for field, value, expected in ready_cases:
        bad = deepcopy(base)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing_family = deepcopy(base)
    missing_family["constraint_family_counts"] = {"benign": 1}
    with pytest.raises(ValueError, match="all families"):
        mod.validate_artifact(missing_family)

    blocked_bad = deepcopy(base)
    blocked_bad["status"] = "blocked"
    blocked_bad["blocked_preconditions"] = ["runtime_failed"]
    blocked_bad["structured_taxonomy_replication_ready"] = True
    with pytest.raises(ValueError, match="blocked artifact"):
        mod.validate_artifact(blocked_bad)

    blocked_count_bad = deepcopy(base)
    blocked_count_bad["status"] = "blocked"
    blocked_count_bad["blocked_preconditions"] = ["runtime_failed"]
    blocked_count_bad["structured_taxonomy_replication_ready"] = False
    with pytest.raises(ValueError, match="fixture_count=0"):
        mod.validate_artifact(blocked_count_bad)


def test_req_safe_5431_helper_edges_and_cli(tmp_path: Path) -> None:
    """REQ-SAFE-5431: helper edge cases and CLI entry point stay deterministic."""

    assert mod._unique(["same", "same", "other"]) == ["same", "other"]
    assert mod._model_spec_fingerprint("bad") == []
    assert mod._runtime_backend({"backend": "llama.cpp/direct"}, {}) == "llama.cpp/direct"
    assert mod._runtime_backend({"gguf_loader_family": "llama.cpp"}, {}) == "llama.cpp"
    assert mod._runtime_backend({}, {"runtime_backend": "llama.cpp/source"}) == "llama.cpp/source"
    assert mod._normalise_test_run({"command": "pytest", "outcome": "passed"}) == {
        "command": "pytest",
        "outcome": "passed",
    }
    assert (
        mod._constraint_family({"fixture_family": "unknown", "source_category": "contradictory"})
        == "semantic_contradiction"
    )
    assert (
        mod._constraint_family({"fixture_family": "custom", "source_category": "other"})
        == "custom"
    )
    assert mod._source_model_hf_id({"model_specs": [None]}, {"model_specs": []}) == mod.MANDATED_HF_IDS[0]
    assert mod._precondition_blockers(
        {"structured_corrigendum_clean": False},
        {"risk_calibrated_structured_panel_ready": False},
        exp5392.PreconditionResult(
            blocked_preconditions=[],
            model_specs=[],
            selected_model_spec=None,
            gpu_offload_receipt=_runtime_receipt(["blocked"]),
        ),
    ) == [
        "exp5430_structured_corrigendum_clean_false",
        "exp5417_risk_calibrated_structured_panel_ready_false",
        "mandated_model_specs_missing",
    ]

    sparse = mod.taxonomy_row_with_checksum(
        {
            "row_id": "missing-authority",
            "constraint_family": "benign",
            "accepted": True,
            "model_self_report_advisory_only": True,
        }
    )
    assert mod.derive_metrics([sparse])["metric_independence_checks_passed"] is True

    paths = _gguf_paths(tmp_path)
    rc = mod.main(
        ["--root", str(REPO), "--result-path", str(tmp_path / "cli_result.json")],
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
    )
    assert rc == 0
    assert (tmp_path / "cli_result.json").exists()
