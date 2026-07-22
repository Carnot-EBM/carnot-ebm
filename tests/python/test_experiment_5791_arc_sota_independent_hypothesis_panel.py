"""Tests for Exp5791 ARC SOTA independent hypothesis panel.

Spec refs: REQ-ARC-WMTE-5791,
SCENARIO-ARC-WMTE-5791-PRECONDITION-BLOCKS-TINY-OR-CPU-EVIDENCE,
SCENARIO-ARC-WMTE-5791-INDEPENDENT-IMMUTABLE-HASHES-NO-FEEDBACK,
SCENARIO-ARC-WMTE-5791-ADMISSION-PANEL-NO-SOLVE-CREDIT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5791_arc_sota_independent_hypothesis_panel as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "-m pytest tests/python/test_experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5791_arc_sota_independent_hypothesis_panel.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _sha(label: str) -> str:
    return "sha256:" + (label.encode("utf-8").hex() * 8)[:64].ljust(64, "0")


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, split in enumerate(
        ("seen", "seen", "heldout", "heldout", "unseen_action", "heldout")
    ):
        row = {
            "row_id": f"r{index}",
            "anonymous_trace_id": f"trace-{index % 2}",
            "observation_hash": _sha(f"obs-{index}"),
            "action": "PAINT" if split == "unseen_action" else "MOVE",
            "successor_hash": _sha(f"succ-{index}"),
            "split": split,
            "seed": 57_910 + (index % 3),
            "agent_owned": True,
            "provenance": "live_agent_observation_receipt",
            "step_index": index,
            "terminal_before": False,
            "terminal_after": index == 2,
            "object_effect_count": 5,
            "policy_votes": ["left", "left"],
            "counterfactual_successor_hashes": [_sha(f"succ-{index}")],
            "play_cost": 1.0,
        }
        rows.append(row)
    rows[0]["reversal_observed"] = True
    rows[3]["object_effect_count"] = 1
    rows[4]["policy_votes"] = ["left", "right"]
    rows[5]["counterfactual_successor_hashes"] = [_sha("succ-5"), _sha("alt-5")]
    return rows


def _predictions(rows: list[dict[str, Any]], wrong: set[str] | None = None) -> dict[str, str]:
    wrong = wrong or set()
    return {
        row["row_id"]: (_sha(f"wrong-{row['row_id']}") if row["row_id"] in wrong else row["successor_hash"])
        for row in rows
    }


def _hypothesis(
    family: str,
    sample_index: int,
    rows: list[dict[str, Any]],
    *,
    wrong: set[str] | None = None,
    compile_ok: bool = True,
) -> dict[str, Any]:
    source = (
        f"def engine_{family}_{sample_index}(state, action): return {sample_index}"
        if compile_ok
        else f"def broken_{family}_{sample_index}(:\n    pass"
    )
    return {
        "hypothesis_id": f"{family}-unit0-s{sample_index}",
        "family": family,
        "capacity": {
            "qwen35b": "35B total / 3B active MoE",
            "gemma31b": "31B dense",
            "gemma26b": "26B total / 4B active MoE",
        }[family],
        "induction_unit": "unit0",
        "sample_index": sample_index,
        "model_id": f"{family}-fresh-{sample_index}",
        "immutable": True,
        "syntax_compile_passed": compile_ok,
        "sandbox_passed": compile_ok,
        "executed_through_live_e3": False,
        "edited_after_freeze": False,
        "closed_loop_proxy_utility": 0.25,
        "source": source,
        "metadata": {"temperature": 0.7, "seed": 57_910 + sample_index, "family": family},
        "predictions": _predictions(rows, wrong),
    }


def _fresh_hypotheses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for family in ("qwen35b", "gemma31b", "gemma26b"):
        hypotheses.append(_hypothesis(family, 0, rows))
        hypotheses.append(_hypothesis(family, 1, rows, wrong={"r3"}))
        hypotheses.append(_hypothesis(family, 2, rows, wrong={"r2", "r3"}, compile_ok=family != "gemma26b"))
    return hypotheses


def _preconditions_fixture() -> dict[str, Any]:
    specs = [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "family": "qwen35b",
            "role": "flagship_moe",
            "model_path": "/models/qwen.gguf",
            "gguf_sha256": _sha("qwen"),
            "quantization": "Q4_K_M",
            "chat_template": "embedded",
            "cuda_layers": 999,
            "runtime": "llama.cpp CUDA test",
            "prompt_id": "arc_world_model_single_shot_v1",
            "sampling": {"temperature": 0.7, "top_p": 0.95},
            "stop_policy": ["```", "<|eot_id|>"],
            "seeds": [57_910, 57_911, 57_912],
            "gpu": 0,
            "real_sota": True,
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "family": "gemma31b",
            "role": "flagship_dense",
            "model_path": "/models/gemma31.gguf",
            "gguf_sha256": _sha("gemma31"),
            "quantization": "Q4_K_M",
            "chat_template": "embedded",
            "cuda_layers": 999,
            "runtime": "llama.cpp CUDA test",
            "prompt_id": "arc_world_model_single_shot_v1",
            "sampling": {"temperature": 0.7, "top_p": 0.95},
            "stop_policy": ["```", "<end_of_turn>"],
            "seeds": [57_910, 57_911, 57_912],
            "gpu": 1,
            "real_sota": True,
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "family": "gemma26b",
            "role": "middle_moe",
            "model_path": "/models/gemma26.gguf",
            "gguf_sha256": _sha("gemma26"),
            "quantization": "Q4_K_M",
            "chat_template": "embedded",
            "cuda_layers": 999,
            "runtime": "llama.cpp CUDA test",
            "prompt_id": "arc_world_model_single_shot_v1",
            "sampling": {"temperature": 0.7, "top_p": 0.95},
            "stop_policy": ["```", "<end_of_turn>"],
            "seeds": [57_910, 57_911, 57_912],
            "gpu": 0,
            "real_sota": True,
        },
    ]
    registry = {
        "source": "ops/arc_solve_registry.yaml",
        "registry_hash": _sha("registry"),
        "checked_before_scoring": True,
        "public_game_count": 25,
        "registry_level_count": 183,
        "full_game_clear_count": 25,
        "all_public_games_complete": True,
        "no_public_level_can_be_credited_as_new": True,
        "ok": True,
    }
    return {
        "ok": True,
        "failures": [],
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": specs[:2],
        "registry_precheck": registry,
        "MODEL_SPECS": specs,
        "intended_models": [spec["hf_id"] for spec in specs],
        "models_used": [],
        "model_runtime_receipts": {
            "llama_cpp_python_cuda": True,
            "llama_cpp_runtime": "llama.cpp CUDA test",
            "runtime_build_hash": _sha("runtime"),
        },
        "gpu_offload_receipts": [
            {"gpu": 0, "name": "NVIDIA GeForce RTX 3090", "offload_ok": True, "vram_delta_mib": 18000},
            {"gpu": 1, "name": "NVIDIA GeForce RTX 3090", "offload_ok": True, "vram_delta_mib": 21000},
        ],
        "agent_owned_trace_hashes": {"trace": _sha("trace")},
        "checkpoint_paths": {
            "output": str(mod.RESULT_RELATIVE_PATH),
            "checkpoint": "results/checkpoints/experiment_5791/checkpoint.json",
            "fresh_cells_present": True,
            "resume_checkpoint_file_present": True,
        },
        "disk_ram": {"ok": True, "disk_free_mb": 9999, "ram_free_mb": 9999},
    }


def test_req_arc_wmte_5791_spec_declares_panel_contract() -> None:
    """REQ-ARC-WMTE-5791: OpenSpec lists every required field and scenario."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5791") :]
    normalized = " ".join(section.split())

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    for marker in (
        "SCENARIO-ARC-WMTE-5791-PRECONDITION-BLOCKS-TINY-OR-CPU-EVIDENCE",
        "SCENARIO-ARC-WMTE-5791-INDEPENDENT-IMMUTABLE-HASHES-NO-FEEDBACK",
        "SCENARIO-ARC-WMTE-5791-ADMISSION-PANEL-NO-SOLVE-CREDIT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in section or marker in normalized


def test_scenario_arc_wmte_5791_precondition_blocks_tiny_or_cpu_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5791-PRECONDITION-BLOCKS-TINY-OR-CPU-EVIDENCE."""

    blocked = _preconditions_fixture()
    blocked["ok"] = False
    blocked["failures"] = ["headline_gpu_offload_missing"]
    blocked["gpu_offload_receipts"][0]["offload_ok"] = False
    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: blocked)

    artifact = mod.build_artifact(
        root=tmp_path,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )

    assert tuple(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["status"] == "blocked"
    assert artifact["solve_claimed"] is False
    assert artifact["registry_credit"] is False
    assert artifact["models_used"] == []
    assert artifact["model_runtime_receipts"]["matched_inference_executed"] is False
    assert artifact["panel_ready_score"] == pytest.approx(0.0)
    assert artifact["admissible_hypothesis_count"] == 0
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_scenario_arc_wmte_5791_independent_hashes_and_admission_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5791-INDEPENDENT-IMMUTABLE-HASHES-NO-FEEDBACK.

    SCENARIO-ARC-WMTE-5791-ADMISSION-PANEL-NO-SOLVE-CREDIT.
    """

    rows = _rows()
    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: _preconditions_fixture())
    monkeypatch.setattr(mod, "load_agent_owned_transition_rows", lambda *_args, **_kw: rows)
    monkeypatch.setattr(mod, "load_fresh_matched_hypotheses", lambda *_args, **_kw: _fresh_hypotheses(rows))
    monkeypatch.setattr(
        mod,
        "development_anchor_import_receipt",
        lambda *_args, **_kw: {
            "source": str(mod.DEVELOPMENT_ANCHOR_RELATIVE_PATH),
            "artifact_sha256": _sha("anchor"),
            "imported_as_development_proxy_only": True,
            "excluded_from_matched_comparison": True,
            "protocol_mismatch": ["old_split_and_single_family_only"],
            "pooled_heldout_accuracy_delta": 0.190883,
        },
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["status"] == "complete"
    assert saved["solve_claimed"] is False
    assert saved["registry_credit"] is False
    assert saved["models_used"] == list(mod.MANDATED_HF_IDS)
    assert saved["model_runtime_receipts"]["matched_inference_executed"] is True
    assert saved["development_anchor_import_receipt"]["excluded_from_matched_comparison"] is True
    assert saved["fresh_matched_cells"]["families_complete"] is True
    assert saved["prompt_and_sampling_receipts"]["feedback_used_for_generation"] is False
    assert saved["independence_receipts"]["all_samples_independent_single_shot"] is True
    assert saved["no_refinement_receipts"]["repaired_rejected_hypothesis_count"] == 0
    assert len(saved["hypothesis_hashes"]) == 9
    assert all(row["source_sha256"] for row in saved["hypothesis_hashes"])
    assert all(row["freeze_stage"] == "pre_compile" for row in saved["hypothesis_hashes"])
    assert saved["compile_sandbox_receipts"]["failed_hypotheses_preserved_in_denominator"] is True
    assert saved["compile_sandbox_receipts"]["payload_compile_flags_trusted"] is False
    assert saved["compile_sandbox_receipts"]["compile_pass_count"] == 8
    assert saved["compile_sandbox_receipts"]["sandbox_pass_count"] == 8
    assert saved["admission_rung_scores"]["hypothesis_count"] == 9
    assert saved["ordinary_transition_metrics"]["denominator_includes_compile_failures"] is True
    assert saved["source_game_identity_leaks"] == []
    assert saved["admissible_hypothesis_count"] >= 2
    assert saved["real_sota_model_count"] == 3
    assert saved["panel_ready_score"] == pytest.approx(1.0)
    assert saved["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_scenario_arc_wmte_5791_compile_sandbox_is_source_derived() -> None:
    """SCENARIO-ARC-WMTE-5791-INDEPENDENT-IMMUTABLE-HASHES-NO-FEEDBACK."""

    rows = _rows()
    syntax_bad = _hypothesis("qwen35b", 0, rows)
    syntax_bad["source"] = "def broken(:\n    pass"
    syntax_bad["syntax_compile_passed"] = True
    syntax_bad["sandbox_passed"] = True
    prepared = mod._prepare_hypothesis_for_scoring(syntax_bad)
    assert prepared["syntax_compile_passed"] is False
    assert prepared["sandbox_passed"] is False
    assert prepared["compile_sandbox_receipt"]["payload_compile_flags_trusted"] is False
    assert prepared["compile_sandbox_receipt"]["error"].startswith("syntax_error:")

    missing = _hypothesis("qwen35b", 1, rows)
    missing["source"] = ""
    assert mod._prepare_hypothesis_for_scoring(missing)["compile_sandbox_receipt"]["error"] == "missing_source"

    no_entrypoint = _hypothesis("qwen35b", 2, rows)
    no_entrypoint["source"] = "PREDICTION_TABLE = {'r0': 'not executable'}"
    assert (
        mod._prepare_hypothesis_for_scoring(no_entrypoint)["compile_sandbox_receipt"]["error"]
        == "missing_executable_entrypoint"
    )

    forbidden = _hypothesis("gemma31b", 0, rows)
    forbidden["source"] = "def engine(state, action):\n    return eval('1')"
    forbidden_receipt = mod._prepare_hypothesis_for_scoring(forbidden)["compile_sandbox_receipt"]
    assert forbidden_receipt["syntax_compile_passed"] is True
    assert forbidden_receipt["sandbox_passed"] is False
    assert forbidden_receipt["forbidden_sandbox_hits"] == ["eval"]

    import_forbidden = _hypothesis("gemma31b", 1, rows)
    import_forbidden["source"] = "import os\n\ndef engine(state, action):\n    return 1"
    assert mod._prepare_hypothesis_for_scoring(import_forbidden)["compile_sandbox_receipt"][
        "forbidden_sandbox_hits"
    ] == ["import"]

    no_hash = {"hypothesis_id": "raw", "family": "qwen35b", "source": "def engine(): return 1"}
    assert mod._compile_sandbox_receipt_for_frozen(no_hash)["error"] == "missing_precompile_freeze_hash"

    fallback_receipts = mod._compile_receipts([_hypothesis("gemma26b", 1, rows)])
    assert fallback_receipts["hypothesis_count"] == 1
    assert fallback_receipts["compile_pass_count"] == 1
    assert fallback_receipts["per_hypothesis"][0]["hash_computed_before_compile"] is True

    scored, _pivotal = mod._score_panel(rows, [_hypothesis("gemma31b", 1, rows)])
    assert scored[0]["hypothesis"]["compile_sandbox_receipt"]["sandbox_passed"] is True


def test_req_arc_wmte_5791_validation_rejects_manual_overclaims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-5791: schema validation fails closed on unsafe manual edits."""

    rows = _rows()
    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: _preconditions_fixture())
    monkeypatch.setattr(mod, "load_agent_owned_transition_rows", lambda *_args, **_kw: rows)
    monkeypatch.setattr(mod, "load_fresh_matched_hypotheses", lambda *_args, **_kw: _fresh_hypotheses(rows))
    artifact = mod.build_artifact(root=tmp_path)

    mutators = [
        ("required field order", lambda data: data.__setitem__("status", data.pop("status"))),
        ("solve_claimed", lambda data: data.__setitem__("solve_claimed", True)),
        ("registry_credit", lambda data: data.__setitem__("registry_credit", True)),
        ("MODEL_SPECS", lambda data: data.__setitem__("MODEL_SPECS", "bad")),
        ("MODEL_SPECS", lambda data: data.__setitem__("MODEL_SPECS", data["MODEL_SPECS"][:2])),
        ("models_used", lambda data: data["models_used"].append("mock/tiny")),
        ("source_game_identity_leaks", lambda data: data["source_game_identity_leaks"].append({"leak": True})),
        ("producer_gate_fields", lambda data: data.__setitem__("producer_gate_fields", [])),
        ("producer_gate_fields", lambda data: data.__setitem__("panel_ready_score", {})),
        ("real_sota_model_count", lambda data: data.__setitem__("real_sota_model_count", 2)),
        ("admissible_hypothesis_count", lambda data: data.__setitem__("admissible_hypothesis_count", 1)),
        ("panel_ready_score", lambda data: data.__setitem__("panel_ready_score", 0.5)),
        ("family_comparison", lambda data: data["family_comparison"].__setitem__("fresh_only", False)),
        ("hypothesis_hashes", lambda data: data["hypothesis_hashes"][0].__setitem__("source_sha256", "")),
        (
            "compile_sandbox_receipts",
            lambda data: data["compile_sandbox_receipts"].__setitem__("payload_compile_flags_trusted", True),
        ),
        (
            "compile_sandbox_receipts",
            lambda data: data["compile_sandbox_receipts"].__setitem__("hypothesis_count", 0),
        ),
        (
            "compile_sandbox_receipts",
            lambda data: data["compile_sandbox_receipts"]["per_hypothesis"][0].__setitem__(
                "hash_computed_before_compile", False
            ),
        ),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "live_llm_inference")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "ok")),
        ("reproducibility_checksum", lambda data: data["family_comparison"].__setitem__("clustered_by", "bad")),
    ]

    for message, mutate in mutators:
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad)

    blocked = deepcopy(artifact)
    blocked["status"] = "blocked"
    blocked["models_used"] = []
    blocked["panel_ready_score"] = 1.0
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="panel_ready_score"):
        mod.validate_artifact(blocked)

    blocked_models = deepcopy(artifact)
    blocked_models["status"] = "blocked"
    blocked_models["panel_ready_score"] = 0.0
    blocked_models["reproducibility_checksum"] = mod.payload_checksum(blocked_models)
    with pytest.raises(ValueError, match="models_used"):
        mod.validate_artifact(blocked_models)


def test_req_arc_wmte_5791_degeneracy_and_leak_taxonomy_edges() -> None:
    """REQ-ARC-WMTE-5791: degeneracy and leak taxonomy covers every named edge."""

    base_score = {"leak_receipt": {"leak_classes": []}, "decision": {"failed_rung": None}}
    clean = {"syntax_compile_passed": True, "sandbox_passed": True}

    assert mod._degeneracy_for(clean, {"leak_receipt": {"leak_classes": ["source"]}, "decision": {}}) == "source_or_identity_leak"
    assert mod._degeneracy_for({"syntax_compile_passed": True, "sandbox_passed": False}, base_score) == "sandbox_failed"
    assert mod._degeneracy_for(clean, {"leak_receipt": {"leak_classes": []}, "decision": {"failed_rung": "L1"}}) == "seen_replay_failed"
    assert mod._degeneracy_for(clean, {"leak_receipt": {"leak_classes": []}, "decision": {"failed_rung": "L3"}}) == "rollout_calibration_failed"
    assert mod._degeneracy_for(clean, {"leak_receipt": {"leak_classes": []}, "decision": {"failed_rung": "L4"}}) == "pivotal_coverage_failed"

    leaks = mod._leaks(
        [
            {
                "hypothesis": {"hypothesis_id": "h0", "family": "qwen35b"},
                "score": {
                    "leak_receipt": {
                        "leak_classes": ["source"],
                        "forbidden_keys": {"source": ["source_file"]},
                    }
                },
            }
        ]
    )
    assert leaks == [
        {
            "hypothesis_id": "h0",
            "family": "qwen35b",
            "leak_classes": ["source"],
            "forbidden_keys": {"source": ["source_file"]},
        }
    ]


def test_req_arc_wmte_5791_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5791: checked-in JSON is the stable terminal panel receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["solve_claimed"] is False
    assert artifact["registry_credit"] is False
    assert artifact["honest_verdict"].startswith(("complete:", "blocked:"))
    if artifact["status"] == "complete":
        assert artifact["panel_ready_score"] == pytest.approx(1.0)
        assert artifact["real_sota_model_count"] == 3
        assert artifact["source_game_identity_leaks"] == []
    else:
        assert artifact["panel_ready_score"] == pytest.approx(0.0)
