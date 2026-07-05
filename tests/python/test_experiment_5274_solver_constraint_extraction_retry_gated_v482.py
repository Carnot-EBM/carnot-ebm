"""Tests for Exp 5274 solver constraint extraction retry.

Spec refs: REQ-VERIFY-5274, SCENARIO-VERIFY-5274.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5273_solver_fixture_rebuild_v482 as fixture_mod
from carnot import experiment_5274_solver_constraint_extraction_retry_gated_v482 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _fake_model_specs(tmp_path: Path, *, dense_ready: bool = True) -> dict[str, Any]:
    paths: dict[str, Path | None] = {}
    for slot in ("flagship_moe", "flagship_dense", "middle_moe"):
        model_path = tmp_path / f"{slot}-Q4_K_M.gguf"
        model_path.write_bytes((slot + "\n").encode("utf-8"))
        paths[slot] = model_path
    if not dense_ready:
        paths["flagship_dense"] = None
    return mod.build_model_specs(model_paths=paths)


def _ready_fixture_artifact() -> dict[str, Any]:
    return fixture_mod.run(write=False, tests_run=[{"command": "fixture unit", "outcome": "passed"}])


def test_req_verify_5274_spec_declares_retry_contract() -> None:
    """REQ-VERIFY-5274: OpenSpec anchors the gated SOTA retry contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5274") : spec.index("### REQ-VERIFY-5263")
    ]

    for marker in (
        "REQ-VERIFY-5274",
        "SCENARIO-VERIFY-5274",
        str(mod.RESULT_RELATIVE_PATH),
        "solver_fixture_ready=true",
        "live_llm_inference_local_gguf_sota",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "malformed model outputs",
        "external text scorers",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5274_schema_errors_do_not_reach_solver() -> None:
    """REQ-VERIFY-5274: malformed outputs are counted before solver scoring."""

    fixture = fixture_mod.fixture_set()[0]
    prompt = mod.render_prompt(fixture)
    raw_output = """
    Notes before payload.
    ```json
    {"schema_version":"solver_constraint_ir_v1","variables":{"x":{"type":"int"}},"constraints":["x >= 0"]}
    ```
    """

    class ExplodingZ3:
        sat = "sat"
        unsat = "unsat"

        @staticmethod
        def Solver() -> Any:
            raise AssertionError("schema-invalid payload must not reach solver")

    row = mod.evaluate_model_output(
        fixture,
        model_slot="flagship_moe",
        prompt=prompt,
        raw_output=raw_output,
        z3_module=ExplodingZ3,
    )

    assert row["json_parseable"] is True
    assert row["schema_valid"] is False
    assert row["malformed"] is True
    assert row["solver_status"] == "schema_error"
    assert row["prompt_sha256"] == mod.sha256(prompt)
    assert row["output_sha256"] == mod.sha256(raw_output)
    assert row["score"]["counterexample"]["schema_errors"]

    decode_error = mod.evaluate_model_output(
        fixture,
        model_slot="flagship_moe",
        prompt=prompt,
        raw_output="{bad}",
    )
    assert decode_error["json_parseable"] is False
    assert decode_error["parse_error"].startswith("json_decode_error:")

    embedded_payload = dict(fixture.reference_encoding)
    embedded_payload["metadata"] = 'brace } and quote " stays inside string'
    embedded = mod.evaluate_model_output(
        fixture,
        model_slot="flagship_moe",
        prompt=prompt,
        raw_output=f"prefix {json.dumps(embedded_payload)} suffix",
    )
    assert embedded["json_parseable"] is True
    assert embedded["json_extracted"] is True
    assert embedded["schema_valid"] is True

    unclosed = mod.evaluate_model_output(
        fixture,
        model_slot="flagship_moe",
        prompt=prompt,
        raw_output="prefix {",
    )
    assert unclosed["parse_error"] == "no_json_object"


def test_scenario_verify_5274_runs_injected_sota_retry_and_improves(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5274: SOTA extraction rows are solver-scored vs baselines."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    fixture_artifact = _ready_fixture_artifact()
    model_specs = _fake_model_specs(tmp_path)
    seen_prompts: list[tuple[str, str, str]] = []

    def fake_proposer(model_spec: dict[str, Any], fixture: fixture_mod.SolverFixture, prompt: str) -> str:
        seen_prompts.append((model_spec["slot"], fixture.fixture_id, prompt))
        return json.dumps(fixture.reference_encoding, sort_keys=True)

    artifact = mod.run(
        result_path=result_path,
        fixture_artifact=fixture_artifact,
        model_specs=model_specs,
        proposal_fn=fake_proposer,
        commands_run=[{"command": "unit retry", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("complete: retry improved")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["solver_extraction_improved"]["value"] is True
    assert artifact["validity_rate"]["value"] == 1.0
    assert artifact["baseline_validity"]["value"] == 0.5
    assert artifact["malformed_rate"]["value"] == 0.0
    assert artifact["unsafe_false_accepts"]["value"] == 0
    assert artifact["preconditions_checked"]["value"]["exp5273_solver_fixture_ready"] is True
    assert artifact["preconditions_checked"]["value"]["deterministic_solver_available"] is True
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["headline_included"] is True
    assert artifact["MODEL_SPECS"]["value"]["flagship_dense"]["headline_included"] is True
    assert artifact["MODEL_SPECS"]["value"]["middle_moe"]["headline_included"] is True
    assert artifact["external_text_scorer_used"] is False
    assert len(artifact["extraction_results"]) == 3 * len(fixture_mod.fixture_set())
    assert len(seen_prompts) == 3 * len(fixture_mod.fixture_set())
    assert artifact["fixture_checksums"]["value"]["source_fixture_set_sha256"]
    assert artifact["fixture_checksums"]["value"]["prompt_output_checksums"]["flagship_moe"]


def test_req_verify_5274_aggregate_separates_malformed_from_solver_invalid() -> None:
    """REQ-VERIFY-5274: malformed, solver-invalid, and false accepts are distinct."""

    fixtures = {fixture.fixture_id: fixture for fixture in fixture_mod.fixture_set()}
    rows = [
        mod.evaluate_model_output(
            fixtures["single_even_high"],
            model_slot="flagship_moe",
            prompt="p1",
            raw_output="not json",
        ),
        mod.evaluate_model_output(
            fixtures["small_pair_sum"],
            model_slot="flagship_moe",
            prompt="p2",
            raw_output=json.dumps(fixtures["small_pair_sum"].reference_encoding),
        ),
        mod.evaluate_model_output(
            fixtures["fixed_schedule_window"],
            model_slot="flagship_dense",
            prompt="p3",
            raw_output=json.dumps(fixtures["even_and_odd"].reference_encoding),
        ),
        mod.evaluate_model_output(
            fixtures["even_and_odd"],
            model_slot="flagship_dense",
            prompt="p4",
            raw_output=json.dumps(mod.empty_encoding()),
        ),
    ]

    aggregate = mod.aggregate_rows(rows, baseline_validity=0.5, prior_v481_validity=0.25)

    assert aggregate["validity_rate"] == 0.25
    assert aggregate["malformed_rate"] == 0.25
    assert aggregate["malformed_outputs"] == 1
    assert aggregate["solver_invalid_constraints"] == 2
    assert aggregate["unsafe_false_accepts"] == 1
    assert aggregate["satisfiable_label_accuracy"] == 0.5
    assert aggregate["counterexample_agreement_rate"] == 1.0
    assert aggregate["retry_outcome"] == "nulled"
    assert aggregate["improved"] is False

    regressed = mod.aggregate_rows([rows[0]], baseline_validity=0.5, prior_v481_validity=0.25)
    assert regressed["retry_outcome"] == "regressed"


def test_req_verify_5274_preconditions_and_artifact_schema_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5274: missing fixture/model preconditions block honestly."""

    blocked_fixture = dict(_ready_fixture_artifact())
    blocked_fixture["solver_fixture_ready"] = False
    proposer_called = False

    def forbidden_proposer(
        model_spec: dict[str, Any],
        fixture: fixture_mod.SolverFixture,
        prompt: str,
    ) -> str:
        nonlocal proposer_called
        proposer_called = True
        return "{}"

    blocked = mod.run(
        result_path=tmp_path / "blocked_fixture.json",
        fixture_artifact=blocked_fixture,
        model_specs=_fake_model_specs(tmp_path),
        proposal_fn=forbidden_proposer,
        commands_run=[{"command": "unit blocked fixture", "outcome": "passed"}],
    )

    mod.validate_artifact(blocked)
    assert blocked["honest_verdict"]["value"].startswith("blocked_preconditions:")
    assert "retry was unmeasured" in blocked["honest_verdict"]["value"]
    assert blocked["solver_extraction_improved"]["value"] is False
    assert blocked["preconditions_checked"]["value"]["exp5273_solver_fixture_ready"] is False
    assert proposer_called is False

    blocked_model = mod.run(
        result_path=tmp_path / "blocked_model.json",
        fixture_artifact=_ready_fixture_artifact(),
        model_specs=_fake_model_specs(tmp_path, dense_ready=False),
        proposal_fn=forbidden_proposer,
        commands_run=[],
    )
    assert blocked_model["honest_verdict"]["value"].startswith("blocked_preconditions:")
    assert "required_headline_models_unavailable" in blocked_model["blockers"]
    assert proposer_called is False

    blocked_solver = mod.run(
        result_path=tmp_path / "blocked_solver.json",
        fixture_artifact=_ready_fixture_artifact(),
        model_specs=_fake_model_specs(tmp_path),
        proposal_fn=forbidden_proposer,
        z3_module=None,
        commands_run=[],
    )
    assert "deterministic_solver_unavailable" in blocked_solver["blockers"]
    assert blocked_solver["baseline_validity"]["value"] == 0.0
    assert proposer_called is False

    monkeypatch.setattr(
        mod,
        "_live_runtime_precondition",
        lambda proposal_fn: {
            "required": proposal_fn is None,
            "llama_cpp_import_ok": True,
            "llama_cpp_gpu_offload_supported": proposal_fn is not None,
        },
    )
    blocked_offload = mod.run(
        result_path=tmp_path / "blocked_offload.json",
        fixture_artifact=_ready_fixture_artifact(),
        model_specs=_fake_model_specs(tmp_path),
        proposal_fn=None,
        commands_run=[],
    )
    assert "llama_cpp_gpu_offload_unavailable" in blocked_offload["blockers"]
    assert proposer_called is False

    unwrapped_fixture = dict(_ready_fixture_artifact())
    unwrapped_fixture["fixture_checksums"] = unwrapped_fixture["fixture_checksums"]["value"]

    def one_model_reference(
        model_spec: dict[str, Any],
        fixture: fixture_mod.SolverFixture,
        prompt: str,
    ) -> str:
        del model_spec, prompt
        return json.dumps(fixture.reference_encoding, sort_keys=True)

    unwrapped = mod.run(
        result_path=tmp_path / "unwrapped_fixture_checksums.json",
        fixture_artifact=unwrapped_fixture,
        model_specs=_fake_model_specs(tmp_path),
        proposal_fn=one_model_reference,
        commands_run=[],
        write=False,
    )
    assert unwrapped["fixture_checksums"]["value"]["source_fixture_set_sha256"]

    broken = dict(blocked)
    broken.pop("validity_rate")
    with pytest.raises(AssertionError, match="missing required field validity_rate"):
        mod.validate_artifact(broken)

    broken = dict(blocked)
    broken["honest_verdict"] = {
        "value": "complete but vague",
        "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
    }
    with pytest.raises(AssertionError, match="complete: or blocked_"):
        mod.validate_artifact(broken)

    broken = dict(blocked)
    broken["inference_substrate"] = {
        "value": "live_llm_inference",
        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
    }
    with pytest.raises(AssertionError, match=mod.INFERENCE_SUBSTRATE):
        mod.validate_artifact(broken)
