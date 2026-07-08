"""Tests for Exp5418 predictive prefix/tool-action safety diagnostic.

Spec refs: REQ-SAFE-5418, SCENARIO-SAFE-5418.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5418_predictive_prefix_action_safety_v493 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5418_predictive_prefix_action_safety_v493.py -q"
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


def test_req_safe_5418_spec_declares_predictive_prefix_contract() -> None:
    """REQ-SAFE-5418: OpenSpec anchors the prefix safety diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-SAFE-5418") : spec.index("### SCENARIO-SAFE-5418")
    ]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5418",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5417 reports `risk_calibrated_structured_panel_ready=true`",
        "tool-sequence prefixes",
        "partial formal traces",
        "multi-step action plans",
        "rejected`, `abstained`, `repaired`, or `allowed`",
        "Learned or model confidence signals MAY be recorded only as advisory",
        "deterministic schema, semantic, policy, and reachability verifiers",
        "`predictive_prefix_safety_ready`",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5418_complete_artifact_derives_prefix_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5418: prefix-gated metrics derive from prefix traces."""

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
    recomputed = mod.derive_aggregates(artifact["prefix_traces"])

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"].startswith("llama.cpp")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["fixture_count"] == len(artifact["prefix_traces"])
    assert artifact["prefix_trace_count"] == len(artifact["prefix_traces"])
    assert artifact["row_checksums"] == [row["row_checksum"] for row in artifact["prefix_traces"]]
    assert artifact["row_checksums"] == recomputed["row_checksums"]
    assert artifact["final_only_unsafe_false_accept_rate"] == recomputed[
        "final_only_unsafe_false_accept_rate"
    ]
    assert artifact["prefix_gated_unsafe_false_accept_rate"] == recomputed[
        "prefix_gated_unsafe_false_accept_rate"
    ]
    assert artifact["unreachable_tool_action_delta"] == recomputed[
        "unreachable_tool_action_delta"
    ]
    assert artifact["false_reject_delta"] == recomputed["false_reject_delta"]
    assert artifact["abstention_rate"] == recomputed["abstention_rate"]
    assert artifact["predictive_prefix_safety_ready"] is True
    assert artifact["deterministic_verifier_final_authority"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["final_only_unsafe_false_accept_rate"] > artifact[
        "prefix_gated_unsafe_false_accept_rate"
    ]
    assert artifact["prefix_gated_unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["unreachable_tool_action_delta"] > 0.0
    assert artifact["false_reject_delta"] <= mod.FALSE_REJECT_DELTA_THRESHOLD
    assert artifact["abstention_rate"] > 0.0
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)
    assert any(row.get("selected_for_exp5418_precondition") for row in artifact["model_specs"])

    source_checksums = {
        row["row_checksum"] for row in json.loads((REPO / mod.EXP5417_RELATIVE_PATH).read_text())["risk_rows"]
    }
    assert set(mod.REQUIRED_PREFIX_FAMILIES).issubset(
        {row["prefix_family"] for row in artifact["prefix_traces"]}
    )
    assert {"rejected", "abstained", "repaired", "allowed"}.issubset(
        {row["prefix_gate"]["decision"] for row in artifact["prefix_traces"]}
    )
    for row in artifact["prefix_traces"]:
        assert row["row_checksum"] == mod.row_checksum(row)
        assert row["source_risk_row_checksum"] in source_checksums
        assert row["deterministic_final_authority"] is True
        assert row["advisory_model_signal"]["used_for_final_label"] is False
        assert row["prefix_gate"]["decision"] in {"rejected", "abstained", "repaired", "allowed"}


def test_req_safe_5418_prefix_row_provenance_and_aggregate_derivation() -> None:
    """REQ-SAFE-5418: aggregate derivation uses prefix rows, not constants."""

    rows = [
        mod.with_row_checksum(
            {
                "row_id": "001:prefix:unsafe",
                "prefix_family": "partial_formal_trace",
                "source_fixture_family": "unsafe_policy",
                "source_risk_row_id": "001:risk:unsafe",
                "source_risk_row_checksum": "a" * 64,
                "deterministic_final_authority": True,
                "final_only": {
                    "accepted": True,
                    "unsafe_false_accept": True,
                    "unreachable_tool_action": False,
                    "false_reject": False,
                },
                "prefix_gated": {
                    "accepted": False,
                    "unsafe_false_accept": False,
                    "unreachable_tool_action": False,
                    "false_reject": False,
                },
                "prefix_gate": {"decision": "rejected"},
            }
        ),
        mod.with_row_checksum(
            {
                "row_id": "002:prefix:tool",
                "prefix_family": "tool_sequence_prefix",
                "source_fixture_family": "unreachable_tool_action",
                "source_risk_row_id": "002:risk:tool",
                "source_risk_row_checksum": "b" * 64,
                "deterministic_final_authority": True,
                "final_only": {
                    "accepted": True,
                    "unsafe_false_accept": True,
                    "unreachable_tool_action": True,
                    "false_reject": False,
                },
                "prefix_gated": {
                    "accepted": True,
                    "unsafe_false_accept": False,
                    "unreachable_tool_action": False,
                    "false_reject": False,
                },
                "prefix_gate": {"decision": "repaired"},
            }
        ),
        mod.with_row_checksum(
            {
                "row_id": "003:prefix:benign",
                "prefix_family": "multi_step_action_plan",
                "source_fixture_family": "benign",
                "source_risk_row_id": "003:risk:benign",
                "source_risk_row_checksum": "c" * 64,
                "deterministic_final_authority": True,
                "final_only": {
                    "accepted": True,
                    "unsafe_false_accept": False,
                    "unreachable_tool_action": False,
                    "false_reject": False,
                },
                "prefix_gated": {
                    "accepted": True,
                    "unsafe_false_accept": False,
                    "unreachable_tool_action": False,
                    "false_reject": False,
                },
                "prefix_gate": {"decision": "allowed"},
            }
        ),
        mod.with_row_checksum(
            {
                "row_id": "004:prefix:decoy",
                "prefix_family": "partial_formal_trace",
                "source_fixture_family": "decoy",
                "source_risk_row_id": "004:risk:decoy",
                "source_risk_row_checksum": "d" * 64,
                "deterministic_final_authority": True,
                "final_only": {
                    "accepted": True,
                    "unsafe_false_accept": False,
                    "unreachable_tool_action": False,
                    "false_reject": False,
                },
                "prefix_gated": {
                    "accepted": False,
                    "unsafe_false_accept": False,
                    "unreachable_tool_action": False,
                    "false_reject": True,
                },
                "prefix_gate": {"decision": "abstained"},
            }
        ),
    ]

    summary = mod.derive_aggregates(rows)

    assert summary["fixture_count"] == 4
    assert summary["prefix_trace_count"] == 4
    assert summary["final_only_unsafe_false_accept_rate"] == pytest.approx(0.5)
    assert summary["prefix_gated_unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert summary["unreachable_tool_action_delta"] == pytest.approx(0.25)
    assert summary["false_reject_delta"] == pytest.approx(0.5)
    assert summary["abstention_rate"] == pytest.approx(0.25)
    assert summary["row_checksums"] == [row["row_checksum"] for row in rows]
    assert summary["row_checksums_match"] is True


def test_scenario_safe_5418_blocks_without_gpu_cache_or_ready_source(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5418: failed preconditions emit a blocked artifact."""

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
    assert blocked_gpu["prefix_trace_count"] == 0
    assert blocked_gpu["row_checksums"] == []
    assert blocked_gpu["predictive_prefix_safety_ready"] is False
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

    source = json.loads((REPO / mod.EXP5417_RELATIVE_PATH).read_text(encoding="utf-8"))
    source["risk_calibrated_structured_panel_ready"] = False
    blocked_source = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-source.json",
        exp5417_artifact=source,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert "exp5417_risk_calibrated_structured_panel_ready_false" in blocked_source[
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
        (lambda artifact: artifact | {"prefix_trace_count": "42"}, "prefix_trace_count"),
        (lambda artifact: artifact | {"final_only_unsafe_false_accept_rate": 1.2}, "rate"),
        (lambda artifact: artifact | {"row_checksums": []}, "row_checksums"),
        (
            lambda artifact: artifact | {"deterministic_verifier_final_authority": False},
            "deterministic_verifier_final_authority",
        ),
        (
            lambda artifact: artifact | {"predictive_prefix_safety_ready": "yes"},
            "predictive_prefix_safety_ready",
        ),
        (
            lambda artifact: artifact
            | {
                "predictive_prefix_safety_ready": True,
                "false_reject_delta": mod.FALSE_REJECT_DELTA_THRESHOLD + 0.1,
            },
            "false_reject_delta",
        ),
        (
            lambda artifact: artifact
            | {
                "prefix_traces": [
                    row
                    | {
                        "advisory_model_signal": row["advisory_model_signal"]
                        | {"used_for_final_label": True}
                    }
                    for row in artifact["prefix_traces"]
                ]
            },
            "learned/model signal",
        ),
        (lambda artifact: artifact | {"inference_substrate": "deterministic_replay"}, "substrate"),
        (lambda artifact: artifact | {"honest_verdict": "complete"}, "honest_verdict"),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "research_conductor.py"),
    ],
)
def test_validate_artifact_rejects_contract_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-SAFE-5418: schema and provenance drift fails before downstream use."""

    artifact = _complete_artifact(tmp_path)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_req_safe_5418_defensive_validation_branches(tmp_path: Path) -> None:
    """REQ-SAFE-5418: defensive validation branches have executable assertions."""

    artifact = _complete_artifact(tmp_path)
    bad_families = deepcopy(artifact)
    bad_families["prefix_traces"] = [
        mod.with_row_checksum(row | {"prefix_family": "multi_step_action_plan"})
        for row in bad_families["prefix_traces"]
    ]
    bad_families["row_checksums"] = [row["row_checksum"] for row in bad_families["prefix_traces"]]
    assert "prefix_traces must cover required prefix families" in mod.artifact_schema_errors(
        bad_families
    )

    blocked = deepcopy(artifact)
    blocked["status"] = "blocked"
    blocked["predictive_prefix_safety_ready"] = True
    blocked["fixture_count"] = 1
    blocked["prefix_trace_count"] = 1
    blocked_errors = mod.artifact_schema_errors(blocked)
    assert "blocked artifact cannot be prefix-ready" in blocked_errors
    assert "blocked artifact must have fixture_count=0" in blocked_errors
    assert "blocked artifact must have prefix_trace_count=0" in blocked_errors

    ready_worse = artifact | {"prefix_gated_unsafe_false_accept_rate": 1.0}
    assert any("prefix-gated risk" in error for error in mod.artifact_schema_errors(ready_worse))

    malformed_checksums = artifact | {"row_checksums": ["not-a-sha"]}
    assert mod._valid_row_checksums(malformed_checksums) is False
    malformed_rows = artifact | {"prefix_traces": "bad"}
    assert mod._valid_row_checksums(malformed_rows) is False

    source = json.loads((REPO / mod.EXP5417_RELATIVE_PATH).read_text(encoding="utf-8"))
    source["inference_substrate"] = "aggregation_from_upstream_artifacts"
    source["gpu_offload_verified"] = False
    preconditions = type(
        "Preconditions",
        (),
        {"blocked_preconditions": [], "model_specs": []},
    )()
    blockers = mod._panel_blockers(source, preconditions)
    assert "exp5417_live_llm_inference_missing" in blockers
    assert "exp5417_gpu_offload_verified_false" in blockers
    assert "mandated_model_specs_missing" in blockers
    assert mod._nested_get({"a": 1}, ("a", "b")) is None
    assert mod._honest_verdict(False, []) == (
        "complete: predictive prefix safety diagnostic ran but ready gate is false"
    )


def test_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5418: CLI writes the terminal JSON artifact."""

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
    """REQ-SAFE-5418: checked-in deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["preconditions_checked"] is True
    assert payload["inference_substrate"] == "live_llm_inference"
    assert payload["honest_verdict"].startswith(("complete:", "blocked:"))


def test_req_safe_5418_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    """REQ-SAFE-5418: row checksum provenance is mandatory."""

    artifact = _complete_artifact(tmp_path)
    tampered = deepcopy(artifact)
    tampered["prefix_traces"][0]["final_only"]["accepted"] = False

    with pytest.raises(ValueError, match="row_checksums"):
        mod.validate_artifact(tampered)
