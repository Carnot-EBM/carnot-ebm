"""Tests for Exp5430 structured tautology corrigendum.

Spec refs: REQ-SAFE-5430, SCENARIO-SAFE-5430.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_5430_structured_tautology_corrigendum_v494 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5430_structured_tautology_corrigendum_v494.py -q"
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


def test_req_safe_5430_spec_declares_corrigendum_contract() -> None:
    """REQ-SAFE-5430: OpenSpec anchors the row-level corrigendum."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-SAFE-5430") : spec.index("## Implementation Status")
    ]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5430",
        "SCENARIO-SAFE-5430",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5417",
        "Exp5418",
        "Exp5427",
        "abstention_rate",
        "semantic_error_rate",
        "accepted_risk_estimate",
        "unsafe_false_accept_rate",
        "final-only action-unreachability",
        "prefix-gated action-unreachability",
        "row_provenance_checksum",
        "reproducibility_checksum",
        "live_llm_inference_and_row_reanalysis",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5430_complete_artifact_recomputes_rows_and_is_adversarial_clean(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5430: deliverable recomputes independent row metrics."""

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
    risk_rows = json.loads((REPO / mod.EXP5417_RELATIVE_PATH).read_text())["risk_rows"]
    prefix_rows = json.loads((REPO / mod.EXP5418_RELATIVE_PATH).read_text())[
        "prefix_traces"
    ]
    risk = mod.derive_risk_reanalysis(risk_rows)
    prefix = mod.derive_prefix_reanalysis(prefix_rows)
    report = adversarial_verify.verify_artifact(out_path)
    recurring = {
        flag["kind"]
        for flag in report["flags"]
        if flag["kind"] in {"TAUTOLOGY", "METHODOLOGY_MISSING"}
    }

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert recurring == set()
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"].startswith("llama.cpp")
    assert artifact["inference_substrate"] == "live_llm_inference_and_row_reanalysis"
    assert artifact["source_artifact_paths"] == [
        str(mod.EXP5417_RELATIVE_PATH),
        str(mod.EXP5418_RELATIVE_PATH),
        str(mod.EXP5427_RELATIVE_PATH),
    ]
    assert artifact["row_count_recomputed"] == len(risk_rows) + len(prefix_rows)
    assert len(artifact["row_provenance_checksum"]) == 64
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["risk_metric_independence_check"] is True
    assert artifact["prefix_metric_independence_check"] is True
    assert artifact["abstention_semantic_metric_separated"] is True
    assert artifact["unreachable_delta_recomputed"] is True
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["structured_corrigendum_clean"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)

    assert artifact["aggregate_reanalysis"]["risk"]["aggregates"] == risk["aggregates"]
    assert artifact["aggregate_reanalysis"]["prefix"]["aggregates"] == prefix["aggregates"]
    assert risk["predicate_support"]["abstention_predicate"] != risk["predicate_support"][
        "semantic_error_predicate"
    ]
    assert risk["predicate_support"]["abstention_semantic_support_sets_equal"] is True
    assert prefix["predicate_support"]["final_unreachable_row_ids"] != prefix[
        "predicate_support"
    ]["prefix_unreachable_row_ids"]


def test_req_safe_5430_abstention_is_not_assigned_from_semantic_error() -> None:
    """REQ-SAFE-5430: abstention and semantic predicates stay separate."""

    rows = [
        mod.risk_row_with_checksum(
            {
                "row_id": "001:risk:semantic",
                "source_fixture_id": "semantic",
                "fixture_family": "schema_only_trap",
                "constrained": {
                    "schema_valid": True,
                    "semantic_valid": False,
                    "unsafe_false_accept": False,
                },
                "unconstrained": {
                    "schema_valid": True,
                    "semantic_valid": True,
                    "unsafe_false_accept": False,
                },
                "abstaining": {
                    "decision": "answer",
                    "accepted": True,
                    "semantic_error": True,
                    "unsafe_false_accept": False,
                },
            }
        ),
        mod.risk_row_with_checksum(
            {
                "row_id": "002:risk:abstain",
                "source_fixture_id": "abstain",
                "fixture_family": "unsafe_policy",
                "constrained": {
                    "schema_valid": True,
                    "semantic_valid": True,
                    "unsafe_false_accept": False,
                },
                "unconstrained": {
                    "schema_valid": True,
                    "semantic_valid": True,
                    "unsafe_false_accept": False,
                },
                "abstaining": {
                    "decision": "abstain",
                    "accepted": False,
                    "semantic_error": False,
                    "unsafe_false_accept": False,
                },
            }
        ),
        mod.risk_row_with_checksum(
            {
                "row_id": "003:risk:clean",
                "source_fixture_id": "clean",
                "fixture_family": "benign",
                "constrained": {
                    "schema_valid": True,
                    "semantic_valid": True,
                    "unsafe_false_accept": False,
                },
                "unconstrained": {
                    "schema_valid": True,
                    "semantic_valid": True,
                    "unsafe_false_accept": False,
                },
                "abstaining": {
                    "decision": "answer",
                    "accepted": True,
                    "semantic_error": False,
                    "unsafe_false_accept": False,
                },
            }
        ),
    ]
    reanalysis = mod.derive_risk_reanalysis(rows)

    assert reanalysis["aggregates"]["semantic_error_rate"] == pytest.approx(1 / 3)
    assert reanalysis["aggregates"]["abstention_rate"] == pytest.approx(1 / 3)
    assert reanalysis["predicate_support"]["semantic_error_row_ids"] == [
        "001:risk:semantic"
    ]
    assert reanalysis["predicate_support"]["abstention_row_ids"] == [
        "002:risk:abstain"
    ]
    assert reanalysis["independence"]["abstention_semantic_metric_separated"] is True


def test_req_safe_5430_unreachable_delta_uses_both_component_predicates() -> None:
    """REQ-SAFE-5430: action delta is final-only minus prefix-gated rate."""

    rows = [
        mod.prefix_row_with_checksum(
            {
                "row_id": "001:prefix:final-only",
                "source_fixture_id": "final-only",
                "prefix_family": "tool_sequence_prefix",
                "final_only": {"unreachable_tool_action": True, "unsafe_false_accept": True},
                "prefix_gated": {"unreachable_tool_action": False, "unsafe_false_accept": False},
                "prefix_gate": {"decision": "repaired"},
            }
        ),
        mod.prefix_row_with_checksum(
            {
                "row_id": "002:prefix:still-unreachable",
                "source_fixture_id": "still-unreachable",
                "prefix_family": "tool_sequence_prefix",
                "final_only": {"unreachable_tool_action": True, "unsafe_false_accept": True},
                "prefix_gated": {"unreachable_tool_action": True, "unsafe_false_accept": False},
                "prefix_gate": {"decision": "allowed"},
            }
        ),
        mod.prefix_row_with_checksum(
            {
                "row_id": "003:prefix:clean",
                "source_fixture_id": "clean",
                "prefix_family": "multi_step_action_plan",
                "final_only": {"unreachable_tool_action": False, "unsafe_false_accept": False},
                "prefix_gated": {"unreachable_tool_action": False, "unsafe_false_accept": False},
                "prefix_gate": {"decision": "allowed"},
            }
        ),
    ]
    reanalysis = mod.derive_prefix_reanalysis(rows)

    assert reanalysis["aggregates"]["final_only_action_unreachability_rate"] == pytest.approx(2 / 3)
    assert reanalysis["aggregates"]["prefix_gated_action_unreachability_rate"] == pytest.approx(1 / 3)
    assert reanalysis["aggregates"]["action_unreachability_delta"] == pytest.approx(1 / 3)
    assert reanalysis["aggregates"]["action_unreachability_delta"] != reanalysis[
        "aggregates"
    ]["final_only_action_unreachability_rate"]
    assert reanalysis["independence"]["unreachable_delta_recomputed"] is True


def test_req_safe_5430_readiness_requires_checksums_and_clean_adversarial_result(
    tmp_path: Path,
) -> None:
    """REQ-SAFE-5430: readiness cannot be true without checksums or clean scan."""

    artifact = _complete_artifact(tmp_path)
    no_row_checksum = deepcopy(artifact)
    no_row_checksum["row_provenance_checksum"] = ""
    no_row_checksum["structured_corrigendum_clean"] = True
    with pytest.raises(ValueError, match="row_provenance_checksum"):
        mod.validate_artifact(no_row_checksum)

    no_repro_checksum = deepcopy(artifact)
    no_repro_checksum["reproducibility_checksum"] = ""
    no_repro_checksum["structured_corrigendum_clean"] = True
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(no_repro_checksum)

    recurring_flag = deepcopy(artifact)
    recurring_flag["adversarial_verify_clean"] = False
    recurring_flag["adversarial_focus_flags"] = [
        {
            "kind": "TAUTOLOGY",
            "severity": "critical",
            "detail": "unit injected recurring verdict",
        }
    ]
    recurring_flag["structured_corrigendum_clean"] = True
    with pytest.raises(ValueError, match="adversarial"):
        mod.validate_artifact(recurring_flag)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.pop("model_specs"), "missing required fields"),
        (lambda a: a.update(field_principles={}), "field_principles"),
        (lambda a: a.update(preconditions_checked=False), "preconditions_checked"),
        (lambda a: a.update(model_specs=[]), "model_specs"),
        (lambda a: a.update(runtime_backend="transformers"), "runtime_backend"),
        (lambda a: a.update(gpu_offload_verified="yes"), "gpu_offload_verified"),
        (lambda a: a.update(row_count_recomputed=-1), "row_count_recomputed"),
        (lambda a: a.update(risk_metric_independence_check="yes"), "must be boolean"),
        (lambda a: a.update(inference_substrate="aggregation_from_upstream_artifacts"), "inference_substrate"),
        (lambda a: a.update(honest_verdict="done"), "honest_verdict"),
        (lambda a: a.update(abstention_rate=0.0), "repaired metrics"),
        (lambda a: a.update(source_artifact_paths=[]), "source_artifact_paths"),
        (lambda a: a.update(aggregate_reanalysis=[]), "aggregate_reanalysis"),
        (lambda a: a.update(research_conductor_modified=True), "research_conductor.py"),
    ],
)
def test_req_safe_5430_schema_errors_cover_invalid_artifact_shapes(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-SAFE-5430: schema validation fails closed on malformed receipts."""

    artifact = _complete_artifact(tmp_path)
    mutate(artifact)
    assert message in "; ".join(mod.artifact_schema_errors(artifact))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.update(status="blocked"), "complete status"),
        (lambda a: a.update(gpu_offload_verified=False), "requires GPU offload"),
        (lambda a: a.update(row_count_recomputed=0), "requires recomputed rows"),
        (lambda a: a.update(prefix_metric_independence_check=False), "requires prefix_metric_independence_check"),
    ],
)
def test_req_safe_5430_clean_gate_rechecks_all_required_booleans(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-SAFE-5430: clean gate is tied to every row-boundary check."""

    artifact = _complete_artifact(tmp_path)
    mutate(artifact)
    artifact["structured_corrigendum_clean"] = True
    assert message in "; ".join(mod.artifact_schema_errors(artifact))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.update(row_count_recomputed=1), "row_count_recomputed"),
        (lambda a: a.update(risk_metric_independence_check=False), "risk_metric"),
        (lambda a: a.update(prefix_metric_independence_check=False), "prefix_metric"),
        (lambda a: a.update(abstention_semantic_metric_separated=False), "abstention_semantic"),
        (lambda a: a.update(unreachable_delta_recomputed=False), "unreachable_delta"),
    ],
)
def test_req_safe_5430_aggregate_reanalysis_mirrors_top_level_checks(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-SAFE-5430: top-level checks mirror nested row reanalysis."""

    artifact = _complete_artifact(tmp_path)
    artifact["structured_corrigendum_clean"] = False
    mutate(artifact)
    assert message in "; ".join(mod.artifact_schema_errors(artifact))


def test_req_safe_5430_blocked_and_defensive_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAFE-5430: blocked inputs and verifier import failures are explicit."""

    artifact = _complete_artifact(tmp_path)
    assert mod.source_artifact_records(tmp_path, [Path("missing.json")]) == [
        {"path": "missing.json", "sha256": None, "missing": True}
    ]

    class Preconditions:
        blocked_preconditions = ["gpu_offload_not_available"]

    broken_risk = {
        "row_count": 0,
        "row_checksums_match": False,
        "independence": {"risk_metric_independence_check": False},
    }
    broken_prefix = {
        "row_count": 0,
        "row_checksums_match": False,
        "independence": {"prefix_metric_independence_check": False},
    }
    blockers = mod._blockers(
        {"gpu_offload_verified": False, "model_specs": []},
        {"gpu_offload_verified": False, "model_specs": []},
        Preconditions(),
        broken_risk,
        broken_prefix,
    )
    assert {
        "gpu_offload_not_available",
        "exp5417_gpu_offload_verified_false",
        "exp5418_gpu_offload_verified_false",
        "exp5417_mandated_model_specs_missing",
        "exp5418_mandated_model_specs_missing",
        "exp5417_risk_rows_missing",
        "exp5418_prefix_traces_missing",
        "exp5417_row_checksum_mismatch",
        "exp5418_row_checksum_mismatch",
        "risk_metric_independence_failed",
        "prefix_metric_independence_failed",
    }.issubset(set(blockers))

    original_import = builtins.__import__

    def fail_scripts_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scripts.adversarial_verify":
            raise ImportError("unit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_scripts_import)
    assert mod.focused_adversarial_flags(artifact) == [
        {
            "kind": "METHODOLOGY_MISSING",
            "severity": "warn",
            "detail": "focused adversarial verifier import failed",
        }
    ]

    assert mod._honest_verdict(False, [], [{"kind": "TAUTOLOGY"}]).startswith(
        "blocked: recurring_adversarial_flags"
    )
    assert mod._honest_verdict(False, ["blocked"], []) == "blocked: blocked"
    assert mod._honest_verdict(False, [], []) == "blocked: structured_corrigendum_clean_false"
    assert mod._aggregate_reanalysis_errors(artifact, {"risk": {}, "prefix": []}) == [
        "aggregate_reanalysis must include risk and prefix"
    ]
    assert mod._nested_get({"a": 1}, ("a", "b")) is None


def test_req_safe_5430_cli_writes_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAFE-5430: CLI entrypoint writes the same validated artifact."""

    paths = _gguf_paths(tmp_path)
    out_path = tmp_path / "cli.json"
    exit_code = mod.main(
        ["--root", str(REPO), "--result-path", str(out_path)],
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
    )
    printed = capsys.readouterr().out
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(printed) == artifact
    mod.validate_artifact(artifact)
