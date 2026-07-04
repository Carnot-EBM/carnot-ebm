"""Tests for Exp 5238 VerIbmc methodology-correct rerun.

Spec refs: REQ-VERIFY-5238, SCENARIO-VERIFY-5238.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as av
from carnot import experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479 as mod


JsonDict = dict[str, Any]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
FAKE_MODEL_SPECS = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "gpu": 0,
        "model_path": "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    }
]


def _value(artifact: JsonDict, field: str) -> Any:
    return artifact[field]["value"]


def _null_proposer(prompt: Any) -> str:
    if prompt.example.example_id == "paired_decrement":
        return '{"invariant": "x >= 0 and y >= 0"}'
    return "no formal proposal"


def _artifact_report(tmp_path: Path, artifact: JsonDict) -> JsonDict:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return av.verify_artifact(path)


def _flag_kinds(report: JsonDict) -> set[str]:
    return {str(flag["kind"]) for flag in report["flags"]}


def test_req_verify_5238_spec_declares_methodology_correct_contract() -> None:
    """REQ-VERIFY-5238: OpenSpec anchors the rerun and receipt schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5238") :]

    for marker in (
        "REQ-VERIFY-5238",
        "SCENARIO-VERIFY-5238",
        str(mod.RESULT_RELATIVE_PATH),
        "scripts.experiment_template.cached_sota_pair",
        "local_sota_gguf_plus_deterministic_solver_feedback",
        "retire_current_veribmc_path=true",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5238_model_specs_use_experiment_template_cached_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5238: MODEL_SPECS come from cached_sota_pair()."""

    calls: list[tuple[int, int]] = []

    def fake_cached_pair(gpu_indices: tuple[int, int]) -> list[JsonDict]:
        calls.append(gpu_indices)
        return [dict(FAKE_MODEL_SPECS[0])]

    monkeypatch.setattr(mod, "cached_sota_pair", fake_cached_pair)

    specs = mod.resolve_model_specs_for_rerun()

    assert calls == [(0, 1)]
    assert specs == FAKE_MODEL_SPECS
    assert mod.select_target_model(specs) == "unsloth/Qwen3.6-35B-A3B-GGUF"


def test_scenario_verify_5238_clean_null_writes_retired_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5238: clean null uplift retires the current path."""

    result_path = tmp_path / "experiment_5238.json"

    artifact = mod.run_experiment(
        result_path=result_path,
        proposal_fn=_null_proposer,
        model_specs_provider=lambda: [dict(FAKE_MODEL_SPECS[0])],
        duration_s=61.0,
        validation_commands_run=["unit fixture: PASS"],
        enforce_duration_floor=False,
    )

    mod.validate_artifact(artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert _value(artifact, "preconditions_checked") is True
    assert _value(artifact, "methodology_receipts_complete") is True
    assert _value(artifact, "retire_current_veribmc_path") is True
    assert _value(artifact, "n_examples") == 3
    assert _value(artifact, "solver_only_solved") == 1
    assert _value(artifact, "llm_only_solved") == 1
    assert _value(artifact, "llm_solver_feedback_solved") == 1
    assert _value(artifact, "solver_feedback_uplift") == 0.0
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "target_model") == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert _value(artifact, "models_used") == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert len(_value(artifact, "reproducibility_checksum")) >= 8
    assert len(_value(artifact, "prompt_template_hash")) == 16
    assert len(_value(artifact, "verifier_pass_fail_log")) == 9
    assert all("verifier_passed" in row for row in _value(artifact, "verifier_pass_fail_log"))
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "retired" in _value(artifact, "honest_verdict")

    report = _artifact_report(tmp_path, artifact)
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_scenario_verify_5238_blocked_preconditions_do_not_claim_headline(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5238: missing GGUFs emit a precondition-only block."""

    artifact = mod.run_experiment(
        result_path=tmp_path / "blocked.json",
        proposal_fn=_null_proposer,
        model_specs_provider=list,
        duration_s=0.01,
        validation_commands_run=["blocked preflight: PASS"],
        enforce_duration_floor=False,
    )

    mod.validate_artifact(artifact)
    assert _value(artifact, "preconditions_checked") is True
    assert _value(artifact, "methodology_receipts_complete") is False
    assert _value(artifact, "target_model") == "blocked"
    assert _value(artifact, "models_used") == []
    assert _value(artifact, "inference_substrate") == mod.PRECONDITION_SUBSTRATE
    assert _value(artifact, "honest_verdict").startswith("blocked_")
    assert _value(artifact, "retire_current_veribmc_path") is False

    report = _artifact_report(tmp_path, artifact)
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_verify_5238_validation_fails_closed_on_bad_receipts(tmp_path: Path) -> None:
    """REQ-VERIFY-5238: clean artifacts require every methodology receipt."""

    artifact = mod.run_experiment(
        result_path=tmp_path / "valid.json",
        proposal_fn=_null_proposer,
        model_specs_provider=lambda: [dict(FAKE_MODEL_SPECS[0])],
        duration_s=61.0,
        validation_commands_run=["unit fixture: PASS"],
        enforce_duration_floor=False,
    )

    too_short = json.loads(json.dumps(artifact))
    too_short["duration_s"] = 59.0
    with pytest.raises(ValueError, match="duration_floor"):
        mod.validate_artifact(too_short)

    missing_checksum = json.loads(json.dumps(artifact))
    missing_checksum["reproducibility_checksum"]["value"] = ""
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(missing_checksum)

    bad_substrate = json.loads(json.dumps(artifact))
    bad_substrate["inference_substrate"]["value"] = mod.PRECONDITION_SUBSTRATE
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    missing_log = json.loads(json.dumps(artifact))
    missing_log["verifier_pass_fail_log"]["value"] = []
    with pytest.raises(ValueError, match="verifier_pass_fail_log"):
        mod.validate_artifact(missing_log)


def test_req_verify_5238_validation_rejects_each_contract_break(tmp_path: Path) -> None:
    """REQ-VERIFY-5238: validator covers malformed wrappers and receipt fields."""

    artifact = mod.run_experiment(
        result_path=tmp_path / "valid.json",
        proposal_fn=_null_proposer,
        model_specs_provider=lambda: [dict(FAKE_MODEL_SPECS[0])],
        duration_s=61.0,
        validation_commands_run=["unit fixture: PASS"],
        enforce_duration_floor=False,
    )
    blocked = mod.run_experiment(
        result_path=tmp_path / "blocked.json",
        proposal_fn=_null_proposer,
        model_specs_provider=list,
        duration_s=0.01,
        validation_commands_run=["blocked preflight: PASS"],
        enforce_duration_floor=False,
    )

    def expect_broken(payload: JsonDict, edit: Any, pattern: str) -> None:
        broken = deepcopy(payload)
        edit(broken)
        with pytest.raises(ValueError, match=pattern):
            mod.validate_artifact(broken)

    expect_broken(artifact, lambda d: d.pop("model_specs"), "missing required fields")
    expect_broken(artifact, lambda d: d.__setitem__("model_specs", []), "principle-wrapped")
    expect_broken(artifact, lambda d: d.__setitem__("field_principles", {}), "field_principles")
    expect_broken(artifact, lambda d: d["honest_verdict"].__setitem__("value", "not terminal"), "honest_verdict")
    expect_broken(artifact, lambda d: d["preconditions_checked"].__setitem__("value", False), "preconditions_checked")
    expect_broken(artifact, lambda d: d["random_seed"].__setitem__("value", "seed"), "random_seed")
    expect_broken(artifact, lambda d: d["prompt_template_hash"].__setitem__("value", "short"), "prompt_template_hash")
    expect_broken(artifact, lambda d: d["verifier_command"].__setitem__("value", ""), "verifier_command")
    expect_broken(artifact, lambda d: d["validation_commands_run"].__setitem__("value", [1]), "validation_commands_run")
    expect_broken(artifact, lambda d: d["n_examples"].__setitem__("value", -1), "n_examples")
    expect_broken(artifact, lambda d: d["solver_only_solved"].__setitem__("value", 99), "solver_only_solved")
    expect_broken(artifact, lambda d: d["solver_feedback_uplift"].__setitem__("value", "0"), "solver_feedback_uplift")
    expect_broken(
        artifact,
        lambda d: d["methodology_receipts_complete"].__setitem__("value", "yes"),
        "methodology_receipts_complete",
    )
    expect_broken(
        artifact,
        lambda d: d["retire_current_veribmc_path"].__setitem__("value", "yes"),
        "retire_current_veribmc_path",
    )
    expect_broken(artifact, lambda d: d["target_model"].__setitem__("value", "legacy"), "target_model")
    expect_broken(
        artifact,
        lambda d: d["model_specs"].__setitem__("value", [{"hf_id": "legacy", "model_path": "/m.gguf"}]),
        "model_specs",
    )
    expect_broken(artifact, lambda d: d["models_used"].__setitem__("value", []), "models_used")
    expect_broken(
        artifact,
        lambda d: d["retire_current_veribmc_path"].__setitem__("value", False),
        "retire_current_veribmc_path inconsistent",
    )
    expect_broken(
        blocked,
        lambda d: d["inference_substrate"].__setitem__("value", "bad"),
        "inference_substrate invalid",
    )
    expect_broken(
        blocked,
        lambda d: d["verifier_pass_fail_log"].__setitem__("value", {}),
        "verifier_pass_fail_log must be list",
    )
    expect_broken(
        blocked,
        lambda d: d["verifier_pass_fail_log"].__setitem__("value", [{"verifier_passed": "yes"}]),
        "verifier_pass_fail_log rows",
    )


def test_req_verify_5238_prompt_hash_and_verifier_receipts_are_stable() -> None:
    """REQ-VERIFY-5238: prompt and verifier receipts are deterministic."""

    examples = mod.fixture_examples()
    first = mod.prompt_template_hash(examples)
    second = mod.prompt_template_hash(examples)

    assert first == second
    assert first != mod.prompt_template_hash(examples[:1])

    solver = [mod.run_solver_only_baseline(example) for example in examples]
    log = mod.verifier_pass_fail_log(solver, [], [])
    assert [row["example_id"] for row in log] == [example.example_id for example in examples]
    assert {row["arm"] for row in log} == {"solver_only"}

    assert mod._uplift(0, 0, 0, 0) == 0.0
    assert mod._duration(None, time.perf_counter()) >= 0.0
    assert mod.honest_verdict(
        complete=True,
        methodology_complete=False,
        uplift=0.0,
        retire=False,
        blocked_reason="",
    ).startswith("blocked_methodology_receipts_incomplete")
    assert "improved" in mod.honest_verdict(
        complete=True,
        methodology_complete=True,
        uplift=0.1,
        retire=False,
        blocked_reason="",
    )
    assert mod.honest_verdict(
        complete=True,
        methodology_complete=True,
        uplift=0.0,
        retire=False,
        blocked_reason="",
    ).endswith("receipts")
