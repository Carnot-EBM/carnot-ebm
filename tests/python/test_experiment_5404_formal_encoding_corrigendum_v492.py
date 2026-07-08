"""Tests for Exp5404 formal-encoding tautology corrigendum.

Spec refs: REQ-SAFE-5404, SCENARIO-SAFE-5404.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_5404_formal_encoding_corrigendum_v492 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5404_formal_encoding_corrigendum_v492.py -q --no-cov"
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
                "name": "Gemma4-26B-A4B-it",
                "hf_id": mod.MANDATED_HF_IDS[2],
                "gpu": gpu_indices[1],
                "model_path": str(paths[mod.MANDATED_HF_IDS[2]]),
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
        "gpu_offload_verified": not blocked,
        "blocked_preconditions": list(blocked or []),
        "nvidia_smi": {"ok": not blocked, "stdout": "0, NVIDIA RTX 3090, 24576, 24000"},
    }


def _model_runner(*, fixture: dict[str, Any], model_spec: dict[str, Any]) -> dict[str, Any]:
    proposed = (
        mod.INTENT_BENIGN
        if fixture["fixture_family"] == "decoy"
        else fixture["expected_intent_label"]
    )
    return {
        "raw_output": json.dumps(
            {
                "intent_label": proposed,
                "policy_label": "reject" if proposed == mod.INTENT_DISALLOWED else "allow",
                "rationale": "classification only; no operational details",
            },
            sort_keys=True,
        ),
        "parsed_output": {
            "intent_label": proposed,
            "policy_label": "reject" if proposed == mod.INTENT_DISALLOWED else "allow",
            "rationale": "classification only; no operational details",
        },
        "schema_valid": True,
        "model_hf_id": model_spec["hf_id"],
        "wall_clock_s": 12.5,
    }


def test_req_safe_5404_spec_declares_corrigendum_contract() -> None:
    """REQ-SAFE-5404: OpenSpec anchors row-level formal-encoding repair."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5404") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5404",
        "SCENARIO-SAFE-5404",
        str(mod.RESULT_RELATIVE_PATH),
        "benign",
        "harmful",
        "disguised-formal",
        "contradictory",
        "decoy",
        "row checksum",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "`formal_encoding_corrigendum_clean`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_safe_5404_identifies_exp5392_tautology_and_self_confirming_fields() -> None:
    """REQ-SAFE-5404: the source review names the quarantined aggregate pair."""

    source = mod.load_source_artifact(REPO)
    review = mod.identify_source_failure(source)

    assert review["source_flagged_artifact"] == str(mod.SOURCE_ARTIFACT_RELATIVE_PATH)
    assert review["tautological_fields"] == [
        {
            "left": "constrained_intent_consistency_rate",
            "right": "constrained_model_false_negative_rate",
            "left_value": 2 / 3,
            "right_value": 2 / 3,
            "why_suspect": (
                "intent consistency and model false-negative rate are distinct "
                "aggregate concepts; exact equality was the Exp5392 CRITICAL "
                "TAUTOLOGY."
            ),
        }
    ]
    assert review["self_confirming_fields"] == [
        "encoded_intent_false_negative_rate",
        "benign_false_positive_rate",
        "formal_encoding_fixture_ready",
    ]


def test_scenario_safe_5404_complete_artifact_uses_row_checksums(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5404: aggregates derive from row records."""

    paths = _gguf_paths(tmp_path)
    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=_model_runner,
        tests_run=[TEST_COMMAND],
        duration_s_override=63.0,
        write=True,
    )
    report = adversarial_verify.verify_artifact(tmp_path / mod.RESULT_RELATIVE_PATH)
    critical = [flag for flag in report["flags"] if flag["severity"] == "critical"]

    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert critical == []
    assert artifact["status"] == "complete"
    assert artifact["preconditions_checked"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["runtime_backend"].startswith("llama.cpp")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["fixture_count"] == len(mod.DEFAULT_FIXTURES)
    assert set(row["fixture_family"] for row in artifact["fixture_rows"]) == set(
        mod.REQUIRED_FIXTURE_FAMILIES
    )
    assert artifact["row_checksums"] == [row["row_checksum"] for row in artifact["fixture_rows"]]
    assert len(set(artifact["row_checksums"])) == artifact["fixture_count"]
    assert artifact["deterministic_policy_authority"] is True
    assert artifact["false_positive_rate"] == pytest.approx(0.0)
    assert artifact["false_negative_rate"] == pytest.approx(0.0)
    assert artifact["forbidden_leak_rate"] == pytest.approx(0.0)
    assert artifact["aggregate_from_rows_only"] is True
    assert artifact["formal_encoding_corrigendum_clean"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)
    assert any(row["ran_in_exp5404"] for row in artifact["model_specs"])

    for row in artifact["fixture_rows"]:
        assert row["expected_policy_label"] in {"allow", "reject"}
        assert row["expected_intent_label"] in {mod.INTENT_BENIGN, mod.INTENT_DISALLOWED}
        assert row["deterministic_verifier_output"]["solver_authoritative"] is True
        assert row["model_output"]["response_accepted_as_authority"] is False
        assert row["final_decision"] in {"accept", "reject"}
        assert len(row["row_checksum"]) == 64


def test_scenario_safe_5404_blocks_without_cache_or_gpu_offload(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5404: failed preconditions emit no headline rows."""

    paths = _gguf_paths(tmp_path)
    blocked_gpu = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-gpu.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(["gpu_offload_not_available"]),
        generation_runner=_model_runner,
        tests_run=[TEST_COMMAND],
        write=False,
    )
    mod.validate_artifact(blocked_gpu)

    assert blocked_gpu["status"] == "blocked"
    assert blocked_gpu["preconditions_checked"] is True
    assert blocked_gpu["gpu_offload_verified"] is False
    assert blocked_gpu["fixture_count"] == 0
    assert blocked_gpu["row_checksums"] == []
    assert blocked_gpu["formal_encoding_corrigendum_clean"] is False
    assert blocked_gpu["honest_verdict"].startswith("blocked:")
    assert blocked_gpu["headline_claim"] is None

    no_cache = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked-cache.json",
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_fn=lambda **_kwargs: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=_model_runner,
        tests_run=[TEST_COMMAND],
        write=False,
    )
    mod.validate_artifact(no_cache)
    assert no_cache["fixture_count"] == 0
    assert "no_mandated_sota_gguf_cached" in no_cache["blocked_preconditions"]
    assert {row["hf_id"] for row in no_cache["model_specs"]} == set(mod.MANDATED_HF_IDS)


def test_req_safe_5404_readiness_self_tests_catch_circular_assignments() -> None:
    """REQ-SAFE-5404: readiness booleans cannot prove themselves."""

    passed = mod.readiness_assignment_self_test(
        mod.READINESS_DEPENDENCY_GRAPH,
        mod.READINESS_TARGET_AGGREGATES,
    )
    assert passed["passed"] is True

    self_dep = deepcopy(mod.READINESS_DEPENDENCY_GRAPH)
    self_dep["formal_encoding_corrigendum_clean"] = (
        "formal_encoding_corrigendum_clean",
    )
    assert mod.readiness_assignment_self_test(
        self_dep,
        mod.READINESS_TARGET_AGGREGATES,
    )["passed"] is False

    constant_dep = deepcopy(mod.READINESS_DEPENDENCY_GRAPH)
    constant_dep["aggregate_from_rows_only"] = ("constant_true",)
    assert mod.readiness_assignment_self_test(
        constant_dep,
        mod.READINESS_TARGET_AGGREGATES,
    )["passed"] is False

    same_aggregate = deepcopy(mod.READINESS_DEPENDENCY_GRAPH)
    same_aggregate["aggregate_from_rows_only"] = ("row_metric_bundle",)
    assert mod.readiness_assignment_self_test(
        same_aggregate,
        {"aggregate_from_rows_only": "row_metric_bundle"},
    )["passed"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: {k: v for k, v in artifact.items() if k != "preconditions_checked"}, "missing"),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (lambda artifact: artifact | {"status": "done"}, "status"),
        (lambda artifact: artifact | {"model_specs": []}, "model_specs"),
        (lambda artifact: artifact | {"runtime_backend": "transformers"}, "runtime_backend"),
        (lambda artifact: artifact | {"gpu_offload_verified": "yes"}, "gpu_offload_verified"),
        (lambda artifact: artifact | {"fixture_count": "5"}, "fixture_count"),
        (lambda artifact: artifact | {"row_checksums": []}, "row_checksums"),
        (lambda artifact: artifact | {"row_checksums": "bad"}, "row_checksums"),
        (lambda artifact: artifact | {"fixture_rows": "bad"}, "row_checksums"),
        (
            lambda artifact: artifact | {"deterministic_policy_authority": False},
            "deterministic_policy_authority",
        ),
        (lambda artifact: artifact | {"false_positive_rate": 1.5}, "rate"),
        (
            lambda artifact: artifact | {"aggregate_from_rows_only": "yes"},
            "aggregate_from_rows_only",
        ),
        (
            lambda artifact: artifact | {"aggregate_from_rows_only": False},
            "aggregate_from_rows_only",
        ),
        (
            lambda artifact: artifact | {"formal_encoding_corrigendum_clean": "yes"},
            "formal_encoding_corrigendum_clean",
        ),
        (
            lambda artifact: artifact | {"formal_encoding_corrigendum_clean": True, "status": "blocked"},
            "formal_encoding_corrigendum_clean",
        ),
        (
            lambda artifact: artifact
            | {"formal_encoding_corrigendum_clean": True, "false_negative_rate": 0.2},
            "formal_encoding_corrigendum_clean",
        ),
        (
            lambda artifact: artifact
            | {"formal_encoding_corrigendum_clean": True, "forbidden_leak_rate": 0.2},
            "formal_encoding_corrigendum_clean",
        ),
        (lambda artifact: artifact | {"inference_substrate": "deterministic_verifier"}, "substrate"),
        (lambda artifact: artifact | {"honest_verdict": "complete"}, "honest_verdict"),
        (
            lambda artifact: artifact
            | {"fixture_rows": [artifact["fixture_rows"][0]] * artifact["fixture_count"]},
            "fixture_rows",
        ),
        (
            lambda artifact: artifact | {"research_conductor_modified": True},
            "research_conductor.py",
        ),
    ],
)
def test_validate_artifact_rejects_contract_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-SAFE-5404: schema drift fails before downstream use."""

    paths = _gguf_paths(tmp_path)
    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / "good.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=_model_runner,
        tests_run=[TEST_COMMAND],
        duration_s_override=63.0,
        write=False,
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_req_safe_5404_leak_or_tampered_checksum_blocks_cleanliness(tmp_path: Path) -> None:
    """REQ-SAFE-5404: leak and checksum drift block the corrigendum gate."""

    paths = _gguf_paths(tmp_path)

    def leaky_runner(*, fixture: dict[str, Any], model_spec: dict[str, Any]) -> dict[str, Any]:
        output = _model_runner(fixture=fixture, model_spec=model_spec)
        if fixture["expected_policy_label"] == "reject":
            output["raw_output"] = output["raw_output"] + " payload"
        return output

    leaky = mod.run(
        root=REPO,
        result_path=tmp_path / "leaky.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=leaky_runner,
        tests_run=[TEST_COMMAND],
        duration_s_override=63.0,
        write=False,
    )
    mod.validate_artifact(leaky)
    assert leaky["forbidden_leak_rate"] > 0.0
    assert leaky["formal_encoding_corrigendum_clean"] is False

    tampered = deepcopy(leaky)
    tampered["row_checksums"][0] = "0" * 64
    with pytest.raises(ValueError, match="row_checksums"):
        mod.validate_artifact(tampered)


def test_req_safe_5404_helper_branches_are_covered(tmp_path: Path) -> None:
    """REQ-SAFE-5404: parser and blocked-validator branches stay exercised."""

    paths = _gguf_paths(tmp_path)
    blocked = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(["gpu_offload_not_available"]),
        generation_runner=_model_runner,
        tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}],
        write=False,
    )

    for mutated, message in (
        (blocked | {"formal_encoding_corrigendum_clean": True}, "blocked artifact"),
        (blocked | {"fixture_count": 1}, "headline rows"),
        (blocked | {"headline_claim": "unsafe headline"}, "headline claim"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(mutated)

    row = mod.evaluate_fixture_row(
        99,
        mod.DEFAULT_FIXTURES[0],
        '{"intent_label":"benign_control","policy_label":"allow","rationale":"ok"}',
        {"hf_id": mod.MANDATED_HF_IDS[0]},
    )
    assert row["model_output"]["schema_valid"] is True
    assert mod._parse_model_json("no-json") == {}
    assert mod._parse_model_json("{bad") == {}
    assert mod._parse_model_json("{bad}") == {}
    assert mod._parse_model_json("[1]") == {}
    assert mod._normalise_test_run({"command": "unit", "outcome": "passed"}) == {
        "command": "unit",
        "outcome": "passed",
    }
    assert mod.readiness_assignment_self_test(
        {"aggregate_from_rows_only": ()},
        mod.READINESS_TARGET_AGGREGATES,
    )["passed"] is False


def test_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5404: CLI writes the terminal JSON artifact."""

    paths = _gguf_paths(tmp_path)
    out_path = tmp_path / mod.RESULT_RELATIVE_PATH

    exit_code = mod.main(
        ["--root", str(REPO), "--result-path", str(out_path)],
        model_resolver=_resolver(paths),
        cached_pair_fn=_cached_pair(paths),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_runner=_model_runner,
        duration_s_override=63.0,
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["status"] == "complete"


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-SAFE-5404: checked-in deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["preconditions_checked"] is True
    assert payload["inference_substrate"] == "live_llm_inference"
    assert payload["status"] in {"complete", "blocked"}
