"""Tests for Exp5474 SOTA GGUF CSL scale-up.

Spec refs: REQ-LEARN-5474,
SCENARIO-LEARN-5474-LIVE-OR-BLOCKED,
SCENARIO-LEARN-5474-SAME-ROWS,
SCENARIO-LEARN-5474-FROZEN-WEIGHTS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5474_sota_csl_scale_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5474_sota_csl_scale_v497.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5474_sota_csl_scale_v497.py "
    "-m pytest tests/python/test_experiment_5474_sota_csl_scale_v497.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5474_sota_csl_scale_v497.py "
    "--fail-under=100"
)


def _fake_model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"fake-gguf:{hf_id}".encode("utf-8"))
        paths[hf_id] = path
    return paths


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, _preferred_quant: str = "Q4_K_M") -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolve


def _runtime_probe(*, ready: bool) -> dict[str, Any]:
    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "runtime_ready": ready,
        "cuda_visible": ready,
        "cuda_device_count": 1 if ready else 0,
        "llama_cpp_gpu_offload": ready,
        "blocked_reasons": [] if ready else ["cuda_or_offload_unavailable"],
    }


def _complete_artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        write=True,
        cache_resolver=_resolver(_fake_model_paths(tmp_path)),
        runtime_probe=lambda: _runtime_probe(ready=True),
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
    )


def test_req_learn_5474_spec_declares_sota_scale_contract() -> None:
    """REQ-LEARN-5474: OpenSpec anchors the V497 SOTA CSL scale-up lane."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5474") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5474",
        "SCENARIO-LEARN-5474-LIVE-OR-BLOCKED",
        "SCENARIO-LEARN-5474-SAME-ROWS",
        "SCENARIO-LEARN-5474-FROZEN-WEIGHTS",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "AutoTokenizer.from_pretrained",
        "repeated tasks",
        "support-removal tasks",
        "conflicting-memory tasks",
        "downstream action-use tasks",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_HF_IDS:
        assert hf_id in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5474_live_or_blocked_writes_honest_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5474-LIVE-OR-BLOCKED: no GPU offload blocks headlines."""

    artifact = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        write=True,
        cache_resolver=_resolver(_fake_model_paths(tmp_path)),
        runtime_probe=lambda: _runtime_probe(ready=False),
        tests_run=[TEST_COMMAND],
    )

    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["csl_scale_ready"] is False
    assert artifact["headline_models_run"] == []
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert set(artifact["model_file_checksums"]) == set(mod.MANDATED_HF_IDS)
    assert all(
        receipt["offload_verified"] is False
        for receipt in artifact["gpu_offload_receipts"]
    )


def test_scenario_learn_5474_same_rows_and_scores_are_kan_assured(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5474-SAME-ROWS: all conditions share row IDs."""

    artifact = _complete_artifact(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["csl_scale_ready"] is True
    assert artifact["headline_models_run"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["n_samples"] == 4
    assert artifact["no_memory_score"] == pytest.approx(0.25)
    assert artifact["naive_icl_score"] == pytest.approx(0.5)
    assert artifact["kan_assured_csl_score"] == pytest.approx(1.0)
    assert artifact["delta_vs_no_memory"] == pytest.approx(0.75)
    assert artifact["delta_vs_naive_icl"] == pytest.approx(0.5)
    assert artifact["negative_transfer_deflection_rate"] == pytest.approx(1.0)
    assert artifact["rollback_trigger_count"] == 1
    assert artifact["exact_validator_pass_rate"] == pytest.approx(1.0)
    assert artifact["context_token_cost_delta"] > 0.0
    assert artifact["model_weight_mutation"] is False
    assert artifact["threshold_offset_summary"]["max"] > 0.0
    assert artifact["threshold_offset_summary"]["count"] == 4

    row_ids_by_condition = artifact["row_ids_by_condition"]
    assert set(row_ids_by_condition) == set(mod.CONDITION_NAMES)
    assert len({tuple(ids) for ids in row_ids_by_condition.values()}) == 1
    tags = {tag for row in artifact["panel_rows"] for tag in row["panel_task_tags"]}
    assert tags == {
        "repeated_task",
        "support_removal",
        "conflicting_memory",
        "downstream_action_use",
    }
    assert all(row["exact_validator_authority"] == "exact_task_verifier" for row in artifact["panel_rows"])
    assert all(row["final_authority_bypassed"] is False for row in artifact["panel_rows"])


def test_scenario_learn_5474_frozen_weights_validation_rejects_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5474-FROZEN-WEIGHTS: checksum and mutation drift fail."""

    artifact = _complete_artifact(tmp_path)

    cases = [
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("research_conductor_modified", True, "scripts/research_conductor.py"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("honest_verdict", "done", "honest_verdict"),
        ("csl_scale_ready", "yes", "csl_scale_ready"),
        ("model_specs", artifact["model_specs"][:2], "model_specs"),
    ]
    for field, value, expected in cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    bad_offload = deepcopy(artifact)
    bad_offload["gpu_offload_receipts"][0]["offload_verified"] = False
    with pytest.raises(ValueError, match="verified GPU offload"):
        mod.validate_artifact(bad_offload)

    bad_checksum = deepcopy(artifact)
    model_id = mod.MANDATED_HF_IDS[0]
    bad_checksum["model_file_checksums"][model_id]["unchanged"] = False
    with pytest.raises(ValueError, match="model file receipts"):
        mod.validate_artifact(bad_checksum)

    bad_rows = deepcopy(artifact)
    bad_rows["panel_rows"][0]["condition"] = "bad"
    with pytest.raises(ValueError, match="same row IDs"):
        mod.validate_artifact(bad_rows)

    bad_authority = deepcopy(artifact)
    bad_authority["panel_rows"][0]["exact_validator_authority"] = "model_self_verdict"
    with pytest.raises(ValueError, match="exact validator"):
        mod.validate_artifact(bad_authority)

    bad_bypass = deepcopy(artifact)
    bad_bypass["panel_rows"][0]["final_authority_bypassed"] = True
    with pytest.raises(ValueError, match="final authority"):
        mod.validate_artifact(bad_bypass)

    schema_mutations = []
    missing = deepcopy(artifact)
    missing.pop("model_specs")
    schema_mutations.append((missing, "missing required fields"))

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    schema_mutations.append((bad_principles, "field_principles mismatch"))

    bad_seed = deepcopy(artifact)
    bad_seed["random_seed"] = 1
    schema_mutations.append((bad_seed, "random_seed mismatch"))

    bad_panel_type = deepcopy(artifact)
    bad_panel_type["panel_rows"] = "bad"
    schema_mutations.append((bad_panel_type, "panel_rows must be a list"))

    bad_gpu_type = deepcopy(artifact)
    bad_gpu_type["gpu_offload_receipts"] = {}
    schema_mutations.append((bad_gpu_type, "gpu_offload_receipts must be a list"))

    bad_spec_type = deepcopy(artifact)
    bad_spec_type["model_specs"] = "bad"
    schema_mutations.append((bad_spec_type, "model_specs must be a list"))

    bad_legacy = deepcopy(artifact)
    bad_legacy["model_specs"][0]["legacy_smoke_only"] = True
    schema_mutations.append((bad_legacy, "legacy smoke"))

    bad_checksum_type = deepcopy(artifact)
    bad_checksum_type["model_file_checksums"] = []
    schema_mutations.append((bad_checksum_type, "model_file_checksums must be a dict"))

    missing_checksum = deepcopy(artifact)
    missing_checksum["model_file_checksums"].pop(mod.MANDATED_HF_IDS[0])
    schema_mutations.append((missing_checksum, "model_file_checksums must include"))

    no_headline_ready = deepcopy(artifact)
    no_headline_ready["headline_models_run"] = []
    schema_mutations.append((no_headline_ready, "ready requires headline_models_run"))

    invalid_headline = deepcopy(artifact)
    invalid_headline["headline_models_run"] = ["legacy/tiny"]
    schema_mutations.append((invalid_headline, "headline_models_run must contain only"))

    for payload, expected in schema_mutations:
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(payload)


def test_req_learn_5474_defensive_helpers_and_repository_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5474: helpers and committed deliverable stay schema-valid."""

    paths = _fake_model_paths(tmp_path)
    specs = mod.model_specs_from_cache(cache_resolver=_resolver(paths))
    checksums = mod.model_file_checksums(specs)
    assert set(checksums) == set(mod.MANDATED_HF_IDS)
    assert all(row["sha256_before"] == row["sha256_after"] for row in checksums.values())
    assert mod.model_file_checksums([{"hf_id": "missing", "model_path": None}]) == {}
    failed_preconditions = mod.evaluate_preconditions(
        routing_artifact={},
        assurance_artifact={},
        model_specs=[
            {
                "hf_id": mod.MANDATED_HF_IDS[0],
                "local_model_present": False,
                "model_path": None,
            }
        ],
        runtime_precondition={"blocked_reasons": ["runtime_down"]},
    )
    assert set(failed_preconditions["blocked_preconditions"]) >= {
        "exp5461_live_routing_not_ready",
        "exp5473_kan_assurance_not_ready",
        "mandated_model_specs_missing",
        "non_empty_mandated_model_paths_missing",
        "local_sota_gguf_gpu_offload_unavailable",
        "no_upstream_headline_model_run",
        "runtime_down",
    }
    assert mod._normalise_tests_run([]) == [{"command": "not_recorded", "outcome": "not_recorded"}]
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._read_json_value(tmp_path / "missing.json") == {}
    scalar = tmp_path / "scalar.json"
    scalar.write_text("7", encoding="utf-8")
    assert mod._read_json(scalar) == {}
    assert mod._rate(0, 0) == 0.0

    minimal_blocked = mod.build_blocked_artifact(
        model_specs=specs,
        model_file_checksums=checksums,
        runtime_precondition={"blocked_reasons": ["test"]},
        tests_run=[],
        duration_s=0.01,
    )
    mod.validate_artifact(minimal_blocked)
    assert minimal_blocked["csl_scale_ready"] is False

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(result)
    assert result["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(result)
