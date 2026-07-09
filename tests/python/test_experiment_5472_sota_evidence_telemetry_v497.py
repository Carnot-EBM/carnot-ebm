"""Tests for Exp5472 local SOTA GGUF evidence telemetry.

Spec refs: REQ-SAFE-5472, SCENARIO-SAFE-5472.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5472_sota_evidence_telemetry_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5472_sota_evidence_telemetry_v497.py -q"
)


def _fake_model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        model_dir = tmp_path / hf_id.replace("/", "--")
        model_dir.mkdir(parents=True)
        model_path = model_dir / f"fixture-{index}-Q4_K_M.gguf"
        model_path.write_bytes(f"fake-gguf:{hf_id}".encode("utf-8"))
        paths[hf_id] = model_path
    return paths


def _resolver(paths: dict[str, Path]):
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        return str(paths[hf_id])

    return resolve


def _gpu_probe(*, gpu_offload: bool) -> dict[str, Any]:
    return {
        "cuda_available": True,
        "cuda_device_count": 1,
        "llama_cpp_python_available": True,
        "llama_cpp_gpu_offload": gpu_offload,
        "native_llama_cli_available": False,
        "runtime_ready": gpu_offload,
        "blocked_reasons": [] if gpu_offload else ["llama_cpp_gpu_offload_false"],
    }


class _FakeHeadlineRuntime:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.prompts: list[str] = []

    def preflight_gpu_offload(self) -> dict[str, Any]:
        return {
            "model_hf_id": self.spec["hf_id"],
            "runtime_backend": "fake_llama_cpp_python",
            "gpu_offload_supported": True,
            "offload_verified": True,
            "n_gpu_layers": -1,
            "vram_before_mb": 1000,
            "vram_after_load_mb": 4096,
            "vram_delta_mb": 3096,
            "pre_generation": True,
        }

    def generate(self, prompt_text: str) -> dict[str, Any]:
        self.prompts.append(prompt_text)
        if "5470-valid-fact-paraphrase" in prompt_text:
            output = "ACCEPT\nThe rewrite preserves the licensed fact."
        elif "5470-hidden-premise" in prompt_text:
            output = "REJECT\nThe rewrite adds an unsupported premise."
        elif "5470-json-semantic-invalid" in prompt_text:
            output = "ACCEPT\nThe JSON looks locally valid."
        elif "5470-factual-distortion" in prompt_text:
            output = "ABSTAIN\nThe fact anchor is ambiguous."
        else:  # pragma: no cover - the selected fixture set is fixed.
            output = "ABSTAIN"
        return {
            "output_text": output,
            "duration_s": 0.125,
            "runtime_backend": "fake_llama_cpp_python",
            "token_logprobs": [-0.1, -0.2],
            "top_logprobs": [{" ACCEPT": -0.1, " REJECT": -2.0}],
        }

    def close(self) -> None:
        return None


class _PreflightFalseRuntime:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def preflight_gpu_offload(self) -> dict[str, Any]:
        return {
            "model_hf_id": self.spec["hf_id"],
            "runtime_backend": "fake_llama_cpp_python",
            "offload_verified": False,
            "pre_generation": True,
        }

    def generate(self, prompt_text: str) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("generation must not run without verified offload")

    def close(self) -> None:
        return None


class _RaisingRuntime:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def preflight_gpu_offload(self) -> dict[str, Any]:
        raise RuntimeError("boom")

    def close(self) -> None:
        return None


def test_req_safe_5472_spec_declares_sota_telemetry_contract() -> None:
    """REQ-SAFE-5472: OpenSpec anchors the local SOTA telemetry contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5472") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5472",
        "SCENARIO-SAFE-5472",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5471 rows",
        "valid row",
        "hidden-premise row",
        "semantic-invalid row",
        "factual-distortion temptation",
        "AutoTokenizer.from_pretrained",
        "non-empty local model file path",
        "GPU-offload receipt",
        "CPU-only headline results",
        mod.INFERENCE_SUBSTRATE,
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for hf_id in mod.MANDATED_HF_IDS:
        assert hf_id in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_safe_5472_balanced_fixture_subset_reports_exact_labels() -> None:
    """REQ-SAFE-5472: selected rows preserve exact Exp5471 validator labels."""

    panel_rows = mod.build_fixture_panel_rows()

    assert [row["candidate_id"] for row in panel_rows] == list(mod.SELECTED_FIXTURE_IDS)
    assert {row["fixture_bucket"] for row in panel_rows} == {
        "valid",
        "hidden_premise",
        "semantic_invalid",
        "factual_distortion",
    }

    by_id = {row["candidate_id"]: row for row in panel_rows}
    assert by_id["5470-valid-fact-paraphrase"]["exact_validator_label"]["accepted"] is True
    assert by_id["5470-hidden-premise"]["exact_validator_label"]["accepted"] is False
    assert "hidden_premise" in by_id["5470-hidden-premise"]["exact_validator_label"][
        "violation_kinds"
    ]
    assert by_id["5470-json-semantic-invalid"]["exact_validator_label"]["accepted"] is False
    assert "semantic_graph_guard" in by_id["5470-json-semantic-invalid"][
        "exact_validator_label"
    ]["caught_by_guards"]
    assert by_id["5470-factual-distortion"]["exact_validator_label"]["accepted"] is False
    assert "distortion_guard" in by_id["5470-factual-distortion"]["exact_validator_label"][
        "caught_by_guards"
    ]
    for row in panel_rows:
        assert row["prompt_text"].startswith("Decide whether this candidate rewrite")
        assert row["exact_validator_label"]["final_authority"] == mod.EXACT_FINAL_AUTHORITY


def test_scenario_safe_5472_blocks_without_gpu_offload_and_no_cpu_headline(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5472: no GPU offload means blocked, not CPU headline fallback."""

    paths = _fake_model_paths(tmp_path)

    def fail_factory(spec: dict[str, Any]) -> _FakeHeadlineRuntime:  # pragma: no cover
        raise AssertionError(f"runtime factory should not be called for {spec['hf_id']}")

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=result_path,
        write=True,
        cache_resolver=_resolver(paths),
        runtime_probe=lambda: _gpu_probe(gpu_offload=False),
        runtime_factory=fail_factory,
        tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["headline_models_run"] == []
    assert artifact["sota_evidence_telemetry_ready"] is False
    assert artifact["guided_decoding_used"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert {spec["hf_id"] for spec in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)
    assert all(spec["local_model_present"] is True for spec in artifact["model_specs"])
    assert all(
        receipt["offload_verified"] is False for receipt in artifact["gpu_offload_receipts"]
    )
    assert set(artifact["model_file_checksums"]) == set(mod.MANDATED_HF_IDS)


def test_scenario_safe_5472_runs_one_headline_model_and_derives_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5472: mocked GPU-offloaded headline run drives row metrics."""

    paths = _fake_model_paths(tmp_path)
    runtimes: list[_FakeHeadlineRuntime] = []

    def factory(spec: dict[str, Any]) -> _FakeHeadlineRuntime:
        runtime = _FakeHeadlineRuntime(spec)
        runtimes.append(runtime)
        return runtime

    artifact = mod.run(
        result_path=tmp_path / "artifact.json",
        write=False,
        cache_resolver=_resolver(paths),
        runtime_probe=lambda: _gpu_probe(gpu_offload=True),
        runtime_factory=factory,
        max_headline_models=1,
        tests_run=[TEST_COMMAND],
    )

    mod.validate_artifact(artifact)
    assert artifact["headline_models_run"] == [mod.MANDATED_HF_IDS[0]]
    assert artifact["n_samples"] == 4
    assert artifact["exact_validator_accuracy"] == pytest.approx(0.5)
    assert artifact["semantic_false_accept_rate"] == pytest.approx(1.0)
    assert artifact["factual_distortion_rate"] == pytest.approx(0.0)
    assert artifact["abstention_rate"] == pytest.approx(0.25)
    assert artifact["logprob_telemetry_available"] is True
    assert artifact["sota_evidence_telemetry_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(runtimes) == 1
    assert len(runtimes[0].prompts) == artifact["n_samples"]
    assert all(
        row["model_outputs"][0]["model_hf_id"] == mod.MANDATED_HF_IDS[0]
        for row in artifact["fixture_rows"]
    )
    assert all(
        row["model_outputs"][0]["logprob_telemetry"]["available"] is True
        for row in artifact["fixture_rows"]
    )


def test_req_safe_5472_validation_fails_closed_on_schema_or_readiness_drift(
    tmp_path: Path,
) -> None:
    """REQ-SAFE-5472: required fields, authority, and readiness drift are rejected."""

    paths = _fake_model_paths(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / "artifact.json",
        write=False,
        cache_resolver=_resolver(paths),
        runtime_probe=lambda: _gpu_probe(gpu_offload=True),
        runtime_factory=_FakeHeadlineRuntime,
        max_headline_models=1,
        tests_run=[TEST_COMMAND],
    )

    missing = deepcopy(artifact)
    missing.pop("model_specs")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_guided = deepcopy(artifact)
    bad_guided["guided_decoding_used"] = True
    with pytest.raises(ValueError, match="guided_decoding_used"):
        mod.validate_artifact(bad_guided)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_specs = deepcopy(artifact)
    bad_specs["model_specs"] = bad_specs["model_specs"][:2]
    with pytest.raises(ValueError, match="model_specs must include"):
        mod.validate_artifact(bad_specs)

    bad_ready = deepcopy(artifact)
    bad_ready["headline_models_run"] = []
    with pytest.raises(ValueError, match="ready requires headline_models_run"):
        mod.validate_artifact(bad_ready)

    bad_offload = deepcopy(artifact)
    bad_offload["gpu_offload_receipts"][0]["offload_verified"] = False
    with pytest.raises(ValueError, match="ready requires verified GPU offload"):
        mod.validate_artifact(bad_offload)

    bad_authority = deepcopy(artifact)
    bad_authority["fixture_rows"][0]["exact_validator_label"]["final_authority"] = "model"
    with pytest.raises(ValueError, match="exact validator authority"):
        mod.validate_artifact(bad_authority)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)


def test_req_safe_5472_defensive_branches_and_runtime_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAFE-5472: defensive helpers expose deterministic blocked states."""

    paths = _fake_model_paths(tmp_path)
    assert mod.parse_decision("") == "abstain"
    assert mod.parse_decision("No clear call.") == "abstain"
    assert mod.parse_decision("Unsure, but probably valid.") == "abstain"

    missing_path = tmp_path / "missing_exp5471.json"
    assert len(mod.load_exp5471_rows(missing_path)) >= len(mod.SELECTED_FIXTURE_IDS)
    bad_rows_path = tmp_path / "bad_exp5471.json"
    bad_rows_path.write_text(json.dumps({"row_results": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="row_results"):
        mod.load_exp5471_rows(bad_rows_path)
    with pytest.raises(ValueError, match="missing selected"):
        mod.select_fixture_subset([])

    absent_specs = mod.model_specs_from_cache(cache_resolver=lambda _hf_id, _quant: None)
    assert mod.model_file_checksums(absent_specs) == {}
    assert mod._is_nonempty_model_file(None) is False
    no_exist = tmp_path / ".no_exist" / "model.gguf"
    no_exist.parent.mkdir()
    no_exist.write_bytes(b"not-real")
    assert mod._is_nonempty_model_file(str(no_exist)) is False
    assert "no_nonempty_mandated_local_model_path" in mod._blocked_reason({}, [])

    monkeypatch.setattr(mod, "_detect_cuda", lambda: (False, 0))
    monkeypatch.setattr(mod, "_llama_cpp_python_status", lambda: (False, False, "missing"))
    monkeypatch.setattr(mod, "_find_native_llama_cli", lambda: None)
    probe = mod.default_runtime_probe()
    assert probe["runtime_ready"] is False
    assert probe["blocked_reasons"] == [
        "cuda_unavailable",
        "llama_cpp_gpu_offload_unavailable",
    ]

    preflight_false = mod.run(
        result_path=tmp_path / "preflight_false.json",
        write=False,
        cache_resolver=_resolver(paths),
        runtime_probe=lambda: _gpu_probe(gpu_offload=True),
        runtime_factory=_PreflightFalseRuntime,
        max_headline_models=1,
    )
    assert preflight_false["headline_models_run"] == []
    assert preflight_false["gpu_offload_receipts"][0]["offload_verified"] is False

    raised = mod.run(
        result_path=tmp_path / "raised.json",
        write=False,
        cache_resolver=_resolver(paths),
        runtime_probe=lambda: _gpu_probe(gpu_offload=True),
        runtime_factory=_RaisingRuntime,
        max_headline_models=1,
    )
    assert "runtime_error:RuntimeError:boom" in raised["gpu_offload_receipts"][0][
        "blocked_reason"
    ]

    artifact = mod.run(
        result_path=tmp_path / "artifact.json",
        write=False,
        cache_resolver=_resolver(paths),
        runtime_probe=lambda: _gpu_probe(gpu_offload=True),
        runtime_factory=_FakeHeadlineRuntime,
        max_headline_models=1,
    )

    mutations = []
    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    mutations.append((bad_principles, "field_principles mismatch"))

    bad_seed = deepcopy(artifact)
    bad_seed["random_seed"] = 1
    mutations.append((bad_seed, "random_seed mismatch"))

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    mutations.append((bad_conductor, "research_conductor.py"))

    bad_headline_type = deepcopy(artifact)
    bad_headline_type["headline_models_run"] = "bad"
    mutations.append((bad_headline_type, "headline_models_run must be a list"))

    bad_headline_id = deepcopy(artifact)
    bad_headline_id["headline_models_run"] = ["legacy/tiny"]
    mutations.append((bad_headline_id, "headline_models_run must contain only mandated"))

    bad_rows_type = deepcopy(artifact)
    bad_rows_type["fixture_rows"] = "bad"
    mutations.append((bad_rows_type, "fixture_rows must be a list"))

    bad_metric = deepcopy(artifact)
    bad_metric["n_samples"] = 99
    mutations.append((bad_metric, "n_samples must match row recomputation"))

    bad_gpu_type = deepcopy(artifact)
    bad_gpu_type["gpu_offload_receipts"] = {}
    mutations.append((bad_gpu_type, "gpu_offload_receipts must be a list"))

    bad_checksum_type = deepcopy(artifact)
    bad_checksum_type["model_file_checksums"] = []
    mutations.append((bad_checksum_type, "model_file_checksums must be a dict"))

    bad_ready_type = deepcopy(artifact)
    bad_ready_type["sota_evidence_telemetry_ready"] = "yes"
    mutations.append((bad_ready_type, "sota_evidence_telemetry_ready must be boolean"))

    bad_order = deepcopy(artifact)
    bad_order["fixture_rows"] = list(reversed(bad_order["fixture_rows"]))
    mutations.append((bad_order, "fixture_rows must preserve selected Exp5471 order"))

    bad_row_type = deepcopy(artifact)
    bad_row_type["fixture_rows"][0] = "bad"
    mutations.append((bad_row_type, "fixture row must be a mapping"))

    bad_score = deepcopy(artifact)
    bad_score["fixture_rows"][0]["exact_validator_label"][
        "computed_from_repair_score"
    ] = True
    mutations.append((bad_score, "exact validator label must not use repair proposal score"))

    bad_prompt = deepcopy(artifact)
    bad_prompt["fixture_rows"][0]["prompt_text"] = ""
    mutations.append((bad_prompt, "fixture row prompt_text must be non-empty"))

    for payload, expected_error in mutations:
        assert expected_error in "; ".join(mod.artifact_schema_errors(payload))


def test_deliverable_file_validates_for_scenario_safe_5472() -> None:
    """SCENARIO-SAFE-5472: checked-in deliverable satisfies the V497 contract."""

    path = REPO / mod.RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["guided_decoding_used"] is False
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert {spec["hf_id"] for spec in artifact["model_specs"]} == set(mod.MANDATED_HF_IDS)
