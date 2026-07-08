"""Tests for Exp 5391 constraint-tax scale-up fixtures.

Spec refs: REQ-VERIFY-5391, SCENARIO-VERIFY-5391.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5379_live_structured_clean_gate_rerun_v490 as exp5379
from carnot import experiment_5380_constraint_tax_tool_action_panel_v3_v490 as exp5380
from carnot import experiment_5391_constraint_tax_scaleup_fixtures_v491 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5391_constraint_tax_scaleup_fixtures_v491.py -q"
)


def _ready_exp5379() -> dict[str, Any]:
    return json.loads((REPO / exp5379.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))


def _ready_exp5380() -> dict[str, Any]:
    return json.loads((REPO / exp5380.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))


def _gguf_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for hf_id in mod.MANDATED_HF_IDS:
        path = tmp_path / f"{hf_id.replace('/', '_')}.gguf"
        path.write_bytes(b"GGUF")
        paths[hf_id] = path
    return paths


def _runtime_receipt(blocked: list[str] | None = None) -> dict[str, Any]:
    return {
        "command": [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        "backend": "llama.cpp/llama-cpp-python",
        "offload_layers": -1,
        "gpu_visible": not blocked,
        "cuda_available": not blocked,
        "llama_cpp_gpu_offload_supported": not blocked,
        "proof_not_cpu_only_headline_evidence": not blocked,
        "blocked_preconditions": list(blocked or []),
        "nvidia_smi": {"ok": not blocked, "stdout": "0, NVIDIA RTX 3090, 24576, 24000"},
    }


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


def test_req_verify_5391_spec_declares_scaleup_contract() -> None:
    """REQ-VERIFY-5391: OpenSpec anchors the scaled local SOTA fixture panel."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5391") : spec.index("### REQ-VERIFY-5380")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5391",
        "SCENARIO-VERIFY-5391",
        str(mod.RESULT_RELATIVE_PATH),
        "at least 24 fixtures",
        "schema validity",
        "budget arithmetic",
        "temporal ordering",
        "tool/action reachability",
        "contradiction repair",
        "wrong-valid traps",
        "cached_sota_pair()",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "generated text SHALL NOT be accepted as a verifier",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5391_complete_scaleup_records_deterministic_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5391: clean gates and GPU/offload preconditions run the scale-up."""

    paths = _gguf_paths(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        exp5379_artifact=_ready_exp5379(),
        exp5380_artifact=_ready_exp5380(),
        model_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.07.491"
    assert artifact["fixture_count"] >= 24
    assert artifact["constraint_tax_scaleup_ready"] is True
    assert artifact["constrained_semantic_validity_rate"] == pytest.approx(1.0)
    assert artifact["unconstrained_semantic_validity_rate"] < 1.0
    assert artifact["wrong_valid_count_constrained"] == 0
    assert artifact["wrong_valid_count_unconstrained"] > 0
    assert artifact["unsafe_false_accept_count"] == 0
    assert artifact["tool_action_reachability_rate"] == pytest.approx(1.0)
    assert artifact["latency_ratio_constrained_vs_unconstrained"] > 1.0
    assert artifact["token_ratio_constrained_vs_unconstrained"] > 1.0
    assert artifact["constraint_tax_deltas"]["semantic_validity_delta"] > 0
    assert artifact["constraint_tax_deltas"]["wrong_valid_reduction"] > 0
    assert artifact["gpu_offload_receipt"]["proof_not_cpu_only_headline_evidence"] is True
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)
    assert any(row["selected_for_exp5391_precondition"] for row in artifact["model_specs"])
    assert artifact["model_specs"][1]["ran_in_upstream_live_structured_receipt"] is True

    categories = {row["category"] for row in artifact["fixture_results"]}
    assert categories == {
        "schema_validity",
        "budget_arithmetic",
        "temporal_ordering",
        "tool_action_reachability",
        "contradiction_repair",
        "wrong_valid_trap",
    }
    for row in artifact["fixture_results"]:
        assert row["initial_state"]
        assert row["expected_final_state"]
        assert row["required_action_sequence"]
        assert row["validator_evidence"]["deterministic_validator"] is True
        for arm in ("constrained", "unconstrained"):
            result = row[arm]
            assert {
                "raw_output",
                "parse_valid",
                "schema_valid",
                "semantic_valid",
                "final_state_valid",
                "wrong_valid",
                "latency_s",
                "token_count",
            }.issubset(result)


def test_scenario_verify_5391_blocks_without_upstream_or_gpu_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5391: failed clean gates or GPU/offload checks block completion."""

    paths = _gguf_paths(tmp_path)
    upstream = _ready_exp5379()
    upstream["structured_protocol_clean"] = False

    blocked_upstream = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-upstream.json",
        exp5379_artifact=upstream,
        exp5380_artifact=_ready_exp5380(),
        model_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert blocked_upstream["status"] == "blocked"
    assert blocked_upstream["constraint_tax_scaleup_ready"] is False
    assert "exp5379_structured_protocol_clean_false" in blocked_upstream["blocked_preconditions"]
    assert blocked_upstream["fixture_count"] == 0
    assert blocked_upstream["honest_verdict"].startswith("blocked:")

    blocked_gpu = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-gpu.json",
        exp5379_artifact=_ready_exp5379(),
        exp5380_artifact=_ready_exp5380(),
        model_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(["llama_cpp_cpu_only"]),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert blocked_gpu["status"] == "blocked"
    assert "llama_cpp_cpu_only" in blocked_gpu["blocked_preconditions"]
    assert blocked_gpu["gpu_offload_receipt"]["proof_not_cpu_only_headline_evidence"] is False

    no_cache = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-cache.json",
        exp5379_artifact=_ready_exp5379(),
        exp5380_artifact=_ready_exp5380(),
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_fn=lambda **_kwargs: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert no_cache["status"] == "blocked"
    assert "no_mandated_sota_gguf_cached" in no_cache["blocked_preconditions"]

    exp5380_block = _ready_exp5380()
    exp5380_block["constraint_tax_panel_ready"] = False
    blocked_panel = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-exp5380.json",
        exp5379_artifact=_ready_exp5379(),
        exp5380_artifact=exp5380_block,
        model_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert "exp5380_constraint_tax_panel_ready_false" in blocked_panel["blocked_preconditions"]


def test_req_verify_5391_validators_do_not_trust_generated_text() -> None:
    """REQ-VERIFY-5391: final authority is action/state replay, not response prose."""

    fixture = copy.deepcopy(mod.DEFAULT_SCALEUP_FIXTURES[0])
    payload = json.loads(fixture["constrained_output"])
    payload["actions"] = []
    payload["answer"] = "I completed the move successfully."
    fixture["constrained_output"] = json.dumps(payload)

    summary = mod.evaluate_panel([fixture])
    row = summary["fixture_results"][0]
    assert row["constrained"]["parse_valid"] is True
    assert row["constrained"]["schema_valid"] is True
    assert row["constrained"]["semantic_valid"] is False
    assert row["constrained"]["wrong_valid"] is True
    assert row["constrained"]["final_state_valid"] is False
    assert row["constrained"]["response_text_accepted_as_verifier"] is False


def test_req_verify_5391_precondition_receipt_handles_single_cached_model(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5391: one cached mandated GGUF plus GPU/offload proof is sufficient."""

    paths = _gguf_paths(tmp_path)

    def no_arg_cached_pair():
        return None

    one_cached = mod.collect_preconditions(
        exp5379_artifact={},
        model_resolver=lambda hf_id, _quant: str(paths[hf_id])
        if hf_id == mod.MANDATED_HF_IDS[0]
        else None,
        cached_pair_fn=no_arg_cached_pair,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
    )

    assert one_cached.blocked_preconditions == []
    selected = [row for row in one_cached.model_specs if row["selected_for_exp5391_precondition"]]
    assert [row["hf_id"] for row in selected] == [mod.MANDATED_HF_IDS[0]]

    not_live_upstream = mod.collect_preconditions(
        exp5379_artifact={
            "selected_model_spec": {"hf_id": mod.MANDATED_HF_IDS[1]},
            "inference_substrate": {"live_local_sota_inference_ran": False},
        },
        model_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
    )
    assert not any(row["ran_in_upstream_live_structured_receipt"] for row in not_live_upstream.model_specs)


def test_req_verify_5391_complete_but_not_ready_verdict(tmp_path: Path) -> None:
    """REQ-VERIFY-5391: a run can complete without claiming scale-up readiness."""

    paths = _gguf_paths(tmp_path)
    preconditions = mod.collect_preconditions(
        exp5379_artifact=_ready_exp5379(),
        model_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
    )
    panel = mod.evaluate_panel(mod.DEFAULT_SCALEUP_FIXTURES)
    panel["constrained_semantic_validity_rate"] = 0.0
    panel["unconstrained_semantic_validity_rate"] = 1.0

    artifact = mod.build_artifact(preconditions=preconditions, panel=panel, tests_run=[TEST_COMMAND])

    assert artifact["status"] == "complete"
    assert artifact["constraint_tax_scaleup_ready"] is False
    assert artifact["honest_verdict"] == "complete: scale-up ran but constraint_tax_scaleup_ready=false."


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: {k: v for k, v in artifact.items() if k != "status"}, "missing"),
        (lambda artifact: artifact | {"milestone": "2026.07.490"}, "milestone"),
        (lambda artifact: artifact | {"status": "done"}, "status"),
        (lambda artifact: artifact | {"model_specs": []}, "model_specs"),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (
            lambda artifact: artifact
            | {"gpu_offload_receipt": artifact["gpu_offload_receipt"] | {"backend": "cpu"}},
            "gpu_offload_receipt",
        ),
        (lambda artifact: artifact | {"fixture_count": "24"}, "fixture_count"),
        (lambda artifact: artifact | {"fixture_count": 3}, "fixture_count"),
        (
            lambda artifact: artifact | {"constrained_semantic_validity_rate": 1.2},
            "rate fields",
        ),
        (
            lambda artifact: artifact | {"wrong_valid_count_constrained": -1},
            "count fields",
        ),
        (
            lambda artifact: artifact | {"constraint_tax_scaleup_ready": True, "unsafe_false_accept_count": 1},
            "unsafe_false_accept_count",
        ),
        (
            lambda artifact: artifact
            | {
                "constraint_tax_scaleup_ready": True,
                "constrained_semantic_validity_rate": 0.0,
                "unconstrained_semantic_validity_rate": 1.0,
            },
            "improve semantic validity",
        ),
        (
            lambda artifact: artifact
            | {"status": "blocked", "constraint_tax_scaleup_ready": True},
            "blocked artifact",
        ),
        (
            lambda artifact: artifact | {"latency_ratio_constrained_vs_unconstrained": 0.0},
            "ratio fields",
        ),
        (
            lambda artifact: artifact | {"deterministic_validator_final_authority": False},
            "deterministic validators",
        ),
        (
            lambda artifact: artifact | {"generated_text_accepted_as_verifier": True},
            "generated text",
        ),
        (
            lambda artifact: artifact | {"research_conductor_modified": True},
            "research_conductor",
        ),
        (lambda artifact: artifact | {"honest_verdict": "complete"}, "honest_verdict"),
    ],
)
def test_validate_artifact_rejects_contract_drift(mutate, message: str) -> None:
    """REQ-VERIFY-5391: schema drift fails before the result is trusted downstream."""

    artifact = mod.build_artifact(
        preconditions=mod.PreconditionResult(
            blocked_preconditions=[],
            model_specs=[
                {
                    "hf_id": hf_id,
                    "name": hf_id.rsplit("/", 1)[-1],
                    "status": "local_gguf_resolved",
                    "selected_for_exp5391_precondition": index < 2,
                    "ran_in_exp5391": False,
                    "ran_in_upstream_live_structured_receipt": hf_id
                    == "unsloth/gemma-4-31B-it-GGUF",
                }
                for index, hf_id in enumerate(mod.MANDATED_HF_IDS)
            ],
            gpu_offload_receipt=_runtime_receipt(),
        ),
        panel=mod.evaluate_panel(mod.DEFAULT_SCALEUP_FIXTURES),
        tests_run=[TEST_COMMAND],
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5391: CLI writes the terminal JSON artifact."""

    paths = _gguf_paths(tmp_path)
    exp5379_path = tmp_path / exp5379.RESULT_RELATIVE_PATH
    exp5380_path = tmp_path / exp5380.RESULT_RELATIVE_PATH
    exp5379_path.parent.mkdir(parents=True, exist_ok=True)
    exp5380_path.parent.mkdir(parents=True, exist_ok=True)
    exp5379_path.write_text(json.dumps(_ready_exp5379()), encoding="utf-8")
    exp5380_path.write_text(json.dumps(_ready_exp5380()), encoding="utf-8")
    out_path = tmp_path / mod.RESULT_RELATIVE_PATH

    exit_code = mod.main(
        [
            "--root",
            str(tmp_path),
            "--artifact-path",
            str(out_path),
            "--exp5379-path",
            str(exp5379_path),
            "--exp5380-path",
            str(exp5380_path),
        ],
        model_resolver=lambda hf_id, _quant: str(paths[hf_id]),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["status"] == "complete"


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-VERIFY-5391: checked-in deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["milestone"] == "2026.07.491"
    assert payload["fixture_count"] >= 24 or payload["status"] == "blocked"
