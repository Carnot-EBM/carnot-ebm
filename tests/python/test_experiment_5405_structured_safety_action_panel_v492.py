"""Tests for Exp5405 structured safety/action panel.

Spec refs: REQ-SAFE-5405, SCENARIO-SAFE-5405.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5405_structured_safety_action_panel_v492 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5405_structured_safety_action_panel_v492.py -q"
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
        "blocked_preconditions": list(blocked or []),
        "nvidia_smi": {"ok": not blocked, "stdout": "0, NVIDIA RTX 3090, 24576, 24000"},
    }


def _load_json(relative: Path) -> dict[str, Any]:
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def test_req_safe_5405_spec_declares_combined_panel_contract() -> None:
    """REQ-SAFE-5405: OpenSpec anchors the combined safety/action panel."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5405") : spec.index("### SCENARIO-SAFE-5405")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5405",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5404 reports `formal_encoding_corrigendum_clean=true`",
        "Exp5391 structured action/state rows",
        "Exp5404 row-level formal-encoding safety rows",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "deterministic schema, semantic, policy, and tool-state checks",
        "`structured_safety_action_panel_ready`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5405_complete_panel_derives_aggregates_from_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5405: clean source artifacts produce a row-derived ready panel."""

    paths = _gguf_paths(tmp_path)
    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    recomputed = mod.derive_aggregates(artifact["panel_rows"])

    assert artifact["status"] == "complete"
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"].startswith("llama.cpp")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["fixture_count"] == 42
    assert artifact["structured_safety_action_panel_ready"] is True
    assert artifact["constrained_validity"] == recomputed["constrained_validity"]
    assert artifact["unconstrained_validity"] == recomputed["unconstrained_validity"]
    assert artifact["wrong_valid_delta"] == recomputed["wrong_valid_delta"]
    assert artifact["unsafe_false_accept_rate"] == recomputed["unsafe_false_accept_rate"]
    assert artifact["fallback_rate"] == recomputed["fallback_rate"]
    assert artifact["tool_action_reachability"] == pytest.approx(1.0)
    assert artifact["constrained_validity"] > artifact["unconstrained_validity"]
    assert artifact["wrong_valid_delta"] > 0
    assert artifact["unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert len(set(artifact["row_checksums"])) == artifact["fixture_count"]
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)
    assert any(row.get("selected_for_exp5405_precondition") for row in artifact["model_specs"])

    row_types = {row["row_type"] for row in artifact["panel_rows"]}
    assert {
        "final_state",
        "tool_action_reachability",
        "formal_encoding_safety",
        "contradictory_constraints",
        "decoy_constraints",
    }.issubset(row_types)
    for row in artifact["panel_rows"]:
        assert row["row_checksum"] == mod.row_checksum(row)
        assert row["deterministic_final_authority"] is True
        assert isinstance(row["invalid_reasons"], list)
        assert isinstance(row["fallback_reasons"], list)


def test_req_safe_5405_row_to_aggregate_derivation_is_authoritative() -> None:
    """REQ-SAFE-5405: headline rates are computed only from row records."""

    rows = [
        mod.with_row_checksum(
            {
                "row_id": "001:valid-tool",
                "row_type": "tool_action_reachability",
                "source_experiment": "unit",
                "deterministic_final_authority": True,
                "constrained": {
                    "valid": True,
                    "wrong_valid": False,
                    "unsafe_false_accept": False,
                    "tool_action_reached": True,
                    "fallback_used": False,
                },
                "unconstrained": {
                    "valid": False,
                    "wrong_valid": True,
                    "unsafe_false_accept": True,
                    "tool_action_reached": False,
                    "fallback_used": False,
                },
                "invalid_reasons": ["unconstrained:wrong_valid"],
                "fallback_reasons": [],
            }
        ),
        mod.with_row_checksum(
            {
                "row_id": "002:safety-override",
                "row_type": "formal_encoding_safety",
                "source_experiment": "unit",
                "deterministic_final_authority": True,
                "constrained": {
                    "valid": True,
                    "wrong_valid": False,
                    "unsafe_false_accept": False,
                    "tool_action_reached": None,
                    "fallback_used": True,
                },
                "unconstrained": {
                    "valid": False,
                    "wrong_valid": True,
                    "unsafe_false_accept": True,
                    "tool_action_reached": None,
                    "fallback_used": False,
                },
                "invalid_reasons": ["unconstrained:unsafe_false_accept"],
                "fallback_reasons": ["deterministic_policy_override"],
            }
        ),
    ]

    summary = mod.derive_aggregates(rows)

    assert summary["fixture_count"] == 2
    assert summary["constrained_validity"] == pytest.approx(1.0)
    assert summary["unconstrained_validity"] == pytest.approx(0.0)
    assert summary["wrong_valid_count_constrained"] == 0
    assert summary["wrong_valid_count_unconstrained"] == 2
    assert summary["wrong_valid_delta"] == 2
    assert summary["unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert summary["tool_action_reachability"] == pytest.approx(1.0)
    assert summary["fallback_rate"] == pytest.approx(0.5)
    assert summary["row_checksums"] == [row["row_checksum"] for row in rows]


def test_scenario_safe_5405_blocks_without_corrigendum_or_gpu_or_cache(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5405: failed gates block headline readiness."""

    paths = _gguf_paths(tmp_path)
    exp5404 = _load_json(mod.EXP5404_RELATIVE_PATH)
    exp5404["formal_encoding_corrigendum_clean"] = False
    blocked_gate = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-gate.json",
        exp5404_artifact=exp5404,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    mod.validate_artifact(blocked_gate)
    assert blocked_gate["status"] == "blocked"
    assert blocked_gate["fixture_count"] == 0
    assert blocked_gate["structured_safety_action_panel_ready"] is False
    assert "exp5404_formal_encoding_corrigendum_clean_false" in blocked_gate["blocked_preconditions"]
    assert blocked_gate["honest_verdict"].startswith("blocked:")

    blocked_gpu = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-gpu.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(["gpu_offload_not_available"]),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert blocked_gpu["gpu_offload_verified"] is False
    assert "gpu_offload_not_available" in blocked_gpu["blocked_preconditions"]

    no_cache = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-cache.json",
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_fn=lambda **_kwargs: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert no_cache["status"] == "blocked"
    assert "no_mandated_sota_gguf_cached" in no_cache["blocked_preconditions"]
    assert {row["hf_id"] for row in no_cache["model_specs"]} == set(mod.MANDATED_HF_IDS)


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
        (lambda artifact: artifact | {"constrained_validity": 1.2}, "rate"),
        (lambda artifact: artifact | {"wrong_valid_delta": -1}, "wrong_valid_delta"),
        (lambda artifact: artifact | {"row_checksums": []}, "row_checksums"),
        (lambda artifact: artifact | {"panel_rows": "bad"}, "panel_rows"),
        (
            lambda artifact: artifact | {"structured_safety_action_panel_ready": "yes"},
            "structured_safety_action_panel_ready",
        ),
        (
            lambda artifact: artifact
            | {
                "structured_safety_action_panel_ready": True,
                "unsafe_false_accept_rate": 0.5,
            },
            "unsafe false accepts",
        ),
        (
            lambda artifact: artifact
            | {
                "structured_safety_action_panel_ready": True,
                "constrained_validity": 0.0,
                "unconstrained_validity": 1.0,
            },
            "improve constrained validity",
        ),
        (lambda artifact: artifact | {"inference_substrate": "deterministic_replay"}, "substrate"),
        (lambda artifact: artifact | {"honest_verdict": "complete"}, "honest_verdict"),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "research_conductor.py"),
    ],
)
def test_validate_artifact_rejects_contract_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-SAFE-5405: schema drift fails before downstream use."""

    paths = _gguf_paths(tmp_path)
    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / "good.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5405: CLI writes the terminal JSON artifact."""

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
    """REQ-SAFE-5405: checked-in deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["preconditions_checked"] is True
    assert payload["inference_substrate"] == "live_llm_inference"
    assert payload["honest_verdict"].startswith(("complete:", "blocked:"))


def test_req_safe_5405_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    """REQ-SAFE-5405: row checksum provenance is mandatory."""

    paths = _gguf_paths(tmp_path)
    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / "good.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    tampered = deepcopy(artifact)
    tampered["panel_rows"][0]["constrained"]["valid"] = False

    with pytest.raises(ValueError, match="row_checksums"):
        mod.validate_artifact(tampered)
