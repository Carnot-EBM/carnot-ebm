"""Tests for Exp 5339 gated SOTA claim/rewrite panel.

Spec refs: REQ-VERIFY-5339, SCENARIO-VERIFY-5339.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5339_gated_sota_claim_rewrite_panel_v487 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_model_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "flagship_moe": tmp_path / "qwen.gguf",
        "flagship_dense": tmp_path / "gemma-31b.gguf",
        "middle_moe": tmp_path / "gemma-26b.gguf",
    }
    for path in paths.values():
        path.write_text("GGUF", encoding="utf-8")
    return paths


def _model_specs(model_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        role: {
            "role": role,
            "hf_id": mod.EXPECTED_HF_BY_ROLE[role],
            "model_path": str(model_paths[role]),
            "status": "local_gguf_resolved",
            "cached": True,
            "autotokenizer_used": False,
        }
        for role in mod.EXPECTED_ROLES
    }


def _write_runtime(path: Path, binary: Path, model_paths: dict[str, Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "-m",
        str(model_paths["flagship_dense"]),
        "-p",
        "old prompt",
        "-n",
        "8",
        "-c",
        "512",
        "--seed",
        "5337",
    ]
    payload = {
        "status": {"value": "complete", "principle": "runtime status"},
        "inference_substrate": {"value": "live_llm_inference", "principle": "runtime substrate"},
        "sota_runtime_clean_receipt_ready": True,
        "runtime_unblocked_min_one_mandated": True,
        "selected_backend_command": {
            "value": {
                "backend_kind": "llama-cli",
                "backend_variant": "llama-cli-single-turn-batch512",
                "command": command,
                "model_path": str(model_paths["flagship_dense"]),
                "model_role": "flagship_dense",
                "timeout_s": 240.0,
            },
            "principle": "selected command",
        },
        "MODEL_SPECS": {"value": _model_specs(model_paths), "principle": "model specs"},
        "preconditions_checked": {
            "value": {
                "gpu_visible": True,
                "model_file_presence": {
                    "flagship_dense": True,
                    "flagship_moe": True,
                    "middle_moe": True,
                },
                "blocked_preconditions": [],
            },
            "principle": "runtime preconditions",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_protocol(path: Path, model_paths: dict[str, Path], *, ready: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    variant = {
        "variant_id": "final_sentinel_post_think_json_v1",
        "n_predict": 640,
        "sentinel": "FINAL_JSON:",
        "end_sentinel": "END_FINAL_JSON",
        "increased_token_budget": True,
        "explicit_final_only_sentinel": True,
        "post_think_json_extraction": True,
        "forbids_analysis_in_final": True,
        "parser_side_strips_llama_cpp_banners": True,
        "stop_sequences_requested": ["END_FINAL_JSON"],
        "stop_sequences_supported": False,
        "ready": ready,
    }
    payload = {
        "status": {"value": "complete" if ready else "blocked", "principle": "protocol status"},
        "honest_verdict": {
            "value": "complete: structured_output_protocol_ready=final_sentinel_post_think_json_v1"
            if ready
            else "blocked_structured_output_protocol_ready_false",
            "principle": "protocol verdict",
        },
        "inference_substrate": {
            "value": "live_llm_inference",
            "principle": "protocol substrate",
        },
        "MODEL_SPECS": {"value": _model_specs(model_paths), "principle": "model specs"},
        "selected_model_spec": {
            "value": _model_specs(model_paths)["flagship_dense"],
            "principle": "selected model",
        },
        "preconditions_checked": {
            "value": {
                "gpu_visible": True,
                "rewrite_state_fixture_ready": True,
                "blocked_preconditions": [],
            },
            "principle": "protocol preconditions",
        },
        "protocol_variants": {"value": [variant], "principle": "variants"},
        "selected_variant_id": variant["variant_id"],
        "structured_output_protocol_ready": ready,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_kwargs(tmp_path: Path) -> dict[str, Any]:
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    model_paths = _write_model_files(tmp_path)
    protocol_path = _write_protocol(tmp_path / mod.exp5338.RESULT_RELATIVE_PATH, model_paths)
    runtime_path = _write_runtime(tmp_path / mod.exp5338.exp5337.RESULT_RELATIVE_PATH, binary, model_paths)
    return {
        "root": tmp_path,
        "artifact_path": tmp_path / mod.RESULT_RELATIVE_PATH,
        "protocol_artifact_path": protocol_path,
        "runtime_artifact_path": runtime_path,
        "paraphrase_groups": mod.exp5310.load_fixture(),
        "rewrite_cases": mod.exp5325.load_fixture(),
        "tests_run": [{"command": "unit exp5339", "outcome": "passed"}],
    }


def _final_json(variant: dict[str, Any], payload: dict[str, Any]) -> str:
    return (
        "[Start thinking]\nfixture-local reasoning outside the final object\n"
        f"{variant['sentinel']} {json.dumps(payload, separators=(',', ':'))}\n"
        f"{variant['end_sentinel']}\n"
    )


def _successful_generation(**kwargs: Any) -> dict[str, Any]:
    prompt_id = kwargs["prompt_spec"]["prompt_id"]
    outputs = {
        "paraphrase_supported": {
            "id": prompt_id,
            "accepted": True,
            "text": "Under amber load, the Aster-9 battery lasted 47 minutes.",
            "premise_valid": True,
            "facts": {
                "duration_minutes": "47",
                "subject": "aster-9 battery",
                "test": "amber-load",
            },
            "attributes": {},
            "citations": [],
        },
        "paraphrase_contradictory": {
            "id": prompt_id,
            "accepted": False,
            "text": "Cedar lab opened in 2023 with eleven benches.",
            "premise_valid": True,
            "facts": {"bench_count": "11", "opened_year": "2023", "subject": "cedar lab"},
            "attributes": {},
            "citations": [],
        },
        "paraphrase_premise_invalid": {
            "id": prompt_id,
            "accepted": False,
            "text": "Route 6 skips Pear Gate today because it never served Pear Gate.",
            "premise_valid": False,
            "facts": {
                "current_stop": "pear gate",
                "service_status": "skips",
                "subject": "route 6",
            },
            "attributes": {},
            "citations": [],
        },
        "paraphrase_surface_only": {
            "id": prompt_id,
            "accepted": True,
            "text": "Noma audit recorded checksum 8f12.",
            "premise_valid": True,
            "facts": {"checksum": "8f12", "subject": "noma audit"},
            "attributes": {},
            "citations": [],
        },
        "rewrite_safe_paraphrase": {
            "id": prompt_id,
            "accepted": True,
            "text": "Under the amber-load test, the Aster-9 battery lasted 47 minutes.",
            "premise_valid": True,
            "facts": {
                "duration_minutes": "47",
                "subject": "aster-9 battery",
                "test": "amber-load",
            },
            "attributes": {"wording": "lasted"},
            "citations": ["battery-log-47"],
        },
        "rewrite_numeric_contradiction": {
            "id": prompt_id,
            "accepted": False,
            "text": "Under the amber-load test, the Aster-9 battery lasted 74 minutes.",
            "premise_valid": True,
            "facts": {
                "duration_minutes": "74",
                "subject": "aster-9 battery",
                "test": "amber-load",
            },
            "attributes": {"wording": "lasted"},
            "citations": ["battery-log-47"],
        },
        "rewrite_missing_required_change": {
            "id": prompt_id,
            "accepted": False,
            "text": "The delta audit recorded a minor temperature drift.",
            "premise_valid": True,
            "facts": {
                "finding": "temperature drift",
                "severity": "minor",
                "subject": "delta audit",
            },
            "attributes": {"review_state": "draft"},
            "citations": ["delta-audit-note"],
        },
        "rewrite_fabricated_premise_citation": {
            "id": prompt_id,
            "accepted": False,
            "text": "The Hollow sensor calibration passed.",
            "premise_valid": True,
            "facts": {"calibration": "passed", "subject": "hollow sensor"},
            "attributes": {"wording": "summarized"},
            "citations": ["sensor-log-12"],
        },
        "rewrite_invalid_premise_preserved": {
            "id": prompt_id,
            "accepted": False,
            "text": "Since Pear Gate was supposedly never on Route 6, it still skips Pear Gate.",
            "premise_valid": False,
            "facts": {
                "current_stop": "pear gate",
                "service_status": "skips",
                "subject": "route-6",
            },
            "attributes": {"repair_mode": "required"},
            "citations": ["route-table-6"],
        },
        "rewrite_overbroad_rewrite": {
            "id": prompt_id,
            "accepted": False,
            "text": "For maintainers, Relay-2 requires a manual reset across all modules.",
            "premise_valid": True,
            "facts": {
                "scope": "all-modules",
                "status": "requires manual reset",
                "subject": "relay-2",
            },
            "attributes": {"audience": "maintainer"},
            "citations": ["relay-note-2"],
        },
    }
    return {
        "completed": True,
        "timed_out": False,
        "returncode": 0,
        "stdout": _final_json(kwargs["variant"], outputs[prompt_id]),
        "stderr": "",
        "wall_clock_s": 0.2,
    }


def test_req_verify_5339_spec_declares_claim_rewrite_panel_contract() -> None:
    """REQ-VERIFY-5339: OpenSpec anchors the gated panel contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5339") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5339",
        "SCENARIO-VERIFY-5339",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "live_llm_inference",
        "prompt_count",
        "parse_success_rate",
        "paraphrase_label_preservation_rate",
        "rewrite_acceptability_rate",
        "citation_preservation_rate",
        "unsafe_false_accepts",
        "headline_quality_claim=false",
        "sota_claim_rewrite_panel_ready",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(mod.FIELD_PRINCIPLES[field].split()) in normalized_section


def test_scenario_verify_5339_calibrated_protocol_scores_fixed_panel(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5339: calibrated protocol outputs are fixture-scored."""

    calls: list[tuple[str, list[str]]] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["prompt_spec"]["prompt_id"], kwargs["command"]))
        return _successful_generation(**kwargs)

    artifact = mod.run(**_base_kwargs(tmp_path), generation_probe=probe, write=True)

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert [prompt_id for prompt_id, _command in calls] == [
        prompt["prompt_id"] for prompt in mod.DEFAULT_PANEL_PROMPTS
    ]
    assert all("-p" in command and "-n" in command and "--seed" in command for _, command in calls)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == "live_llm_inference"
    assert artifact["prompt_count"] == 10
    assert artifact["parse_success_rate"] == pytest.approx(1.0)
    assert artifact["paraphrase_label_preservation_rate"] == pytest.approx(1.0)
    assert artifact["rewrite_acceptability_rate"] == pytest.approx(1.0)
    assert artifact["citation_preservation_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["headline_quality_claim"] is False
    assert artifact["sota_claim_rewrite_panel_ready"] is True
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert {row["hf_id"] for row in artifact["MODEL_SPECS"]["value"].values()} == set(
        mod.EXPECTED_MODEL_IDS
    )
    assert artifact["parse_failures"] == []
    assert artifact["semantic_failures"] == []


def test_req_verify_5339_blocks_before_generation_when_protocol_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5339: missing Exp5338 protocol readiness blocks generation."""

    kwargs = _base_kwargs(tmp_path)
    protocol = json.loads(kwargs["protocol_artifact_path"].read_text(encoding="utf-8"))
    protocol["structured_output_protocol_ready"] = False
    protocol["status"]["value"] = "blocked"
    kwargs["protocol_artifact_path"].write_text(json.dumps(protocol), encoding="utf-8")
    calls: list[str] = []

    artifact = mod.run(
        **kwargs,
        generation_probe=lambda **kw: calls.append(kw["prompt_spec"]["prompt_id"]) or {},
        write=False,
    )

    mod.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["prompt_count"] == 0
    assert artifact["parse_success_rate"] == pytest.approx(0.0)
    assert artifact["sota_claim_rewrite_panel_ready"] is False
    assert artifact["headline_quality_claim"] is False
    assert "structured_output_protocol_not_ready" in artifact["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]


def test_scenario_verify_5339_records_parse_and_semantic_failures_separately(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5339: parse failures are distinct from semantic failures."""

    def mixed_probe(**kwargs: Any) -> dict[str, Any]:
        if kwargs["prompt_spec"]["prompt_id"] == "paraphrase_supported":
            return {
                "completed": True,
                "timed_out": False,
                "returncode": 0,
                "stdout": "not-json",
                "stderr": "",
                "wall_clock_s": 0.2,
            }
        raw = _successful_generation(**kwargs)
        if kwargs["prompt_spec"]["prompt_id"] == "rewrite_safe_paraphrase":
            payload = {
                "id": "rewrite_safe_paraphrase",
                "accepted": False,
                "text": "Under the amber-load test, the Aster-9 battery lasted 47 minutes.",
                "premise_valid": True,
                "facts": {
                    "duration_minutes": "47",
                    "subject": "aster-9 battery",
                    "test": "amber-load",
                },
                "attributes": {"wording": "lasted"},
                "citations": ["phantom-log"],
            }
            raw["stdout"] = _final_json(kwargs["variant"], payload)
        return raw

    artifact = mod.run(**_base_kwargs(tmp_path), generation_probe=mixed_probe, write=False)

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["prompt_count"] == 10
    assert artifact["parse_success_rate"] == pytest.approx(0.9)
    assert artifact["sota_claim_rewrite_panel_ready"] is False
    assert [row["prompt_id"] for row in artifact["parse_failures"]] == ["paraphrase_supported"]
    assert {row["prompt_id"] for row in artifact["semantic_failures"]} == {
        "rewrite_safe_paraphrase"
    }
    assert artifact["rewrite_acceptability_rate"] < 1.0
    assert artifact["citation_preservation_rate"] < 1.0


def test_req_verify_5339_schema_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5339: schema validation catches malformed panel artifacts."""

    artifact = mod.run(
        **_base_kwargs(tmp_path),
        generation_probe=_successful_generation,
        write=False,
    )

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(artifact))

    malformed_cases = [
        (lambda a: (a.pop("MODEL_SPECS"), a)[1], "missing required fields"),
        (lambda a: (a.__setitem__("experiment_id", mod.EXPERIMENT_ID), a)[1], "principle-wrapped"),
        (lambda a: (a["honest_verdict"].__setitem__("value", "done"), a)[1], "honest_verdict"),
        (lambda a: (a["milestone"].__setitem__("value", "wrong"), a)[1], "milestone mismatch"),
        (lambda a: (a["status"].__setitem__("value", "running"), a)[1], "status must be complete or blocked"),
        (
            lambda a: (a["inference_substrate"].__setitem__("value", "cached"), a)[1],
            "inference_substrate mismatch",
        ),
        (lambda a: (a.__setitem__("prompt_count", "10"), a)[1], "prompt_count must be a bare integer"),
        (
            lambda a: (a.__setitem__("parse_success_rate", 1.5), a)[1],
            "parse_success_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("citation_preservation_rate", -0.1), a)[1],
            "citation_preservation_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("unsafe_false_accepts", "0"), a)[1],
            "unsafe_false_accepts must be a bare integer",
        ),
        (
            lambda a: (a.__setitem__("headline_quality_claim", True), a)[1],
            "headline_quality_claim must be bare false",
        ),
        (
            lambda a: (a.__setitem__("sota_claim_rewrite_panel_ready", "yes"), a)[1],
            "sota_claim_rewrite_panel_ready must be a bare boolean",
        ),
        (
            lambda a: (a["MODEL_SPECS"]["value"].pop("middle_moe"), a)[1],
            "MODEL_SPECS roles mismatch",
        ),
        (
            lambda a: (a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"), a)[1],
            "hf_id mismatch",
        ),
        (
            lambda a: (a["tests_run"].__setitem__("value", "bad"), a)[1],
            "tests_run must be a list",
        ),
        (
            lambda a: (a["tests_run"].__setitem__("principle", "wrong"), a)[1],
            "tests_run must be principle-wrapped",
        ),
        (
            lambda a: (a["selected_model_spec"].__setitem__("value", "bad"), a)[1],
            "selected_model_spec must be an object or null",
        ),
        (
            lambda a: (a["status"].__setitem__("value", "blocked"), a)[1],
            "ready artifact must have complete status",
        ),
        (
            lambda a: (a.__setitem__("sota_claim_rewrite_panel_ready", False), a)[1],
            "blocked artifact must have blocked status",
        ),
    ]

    for mutate, expected in malformed_cases:
        joined = "; ".join(mod.artifact_schema_errors(mutate(clone())))
        assert expected in joined

    with pytest.raises(AssertionError, match="headline_quality_claim"):
        bad = clone()
        bad["headline_quality_claim"] = True
        mod.validate_artifact(bad)


def test_req_verify_5339_defensive_helpers_cover_precondition_and_parser_branches(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5339: defensive helpers classify malformed inputs deterministically."""

    kwargs = _base_kwargs(tmp_path)
    protocol = json.loads(kwargs["protocol_artifact_path"].read_text(encoding="utf-8"))
    runtime = json.loads(kwargs["runtime_artifact_path"].read_text(encoding="utf-8"))
    variant = protocol["protocol_variants"]["value"][0]
    prompt = mod.DEFAULT_PANEL_PROMPTS[0]

    assert mod._raw_or_wrapped_value({"plain": "value"}, "plain") == "value"
    assert mod._string_map(None) == {}
    assert mod._string_tuple("not-array") == ()
    assert mod._rate([], "ok") == 0.0
    assert set(mod._default_model_specs()) == set(mod.EXPECTED_ROLES)

    assert mod._selected_protocol_variant({"protocol_variants": {"value": "bad"}}) is None
    fallback_protocol = json.loads(json.dumps(protocol))
    fallback_protocol.pop("selected_variant_id")
    assert (
        mod._selected_protocol_variant(fallback_protocol)["variant_id"]
        == "final_sentinel_post_think_json_v1"
    )

    bad_specs = {"flagship_dense": {"hf_id": "wrong", "model_path": "/missing"}}
    blockers = mod._model_specs_blockers(bad_specs)
    assert "model_specs_missing_or_drift" in blockers

    bad_protocol = json.loads(json.dumps(protocol))
    bad_protocol["protocol_variants"]["value"] = []
    bad_protocol.pop("MODEL_SPECS")
    bad_protocol.pop("selected_model_spec")
    bad_protocol["preconditions_checked"]["value"]["gpu_visible"] = False
    bad_runtime = json.loads(json.dumps(runtime))
    bad_runtime["status"]["value"] = "blocked"
    bad_runtime["preconditions_checked"]["value"]["gpu_visible"] = False
    bad_runtime.pop("selected_backend_command")
    _command, _specs, _model, _variant, blockers = mod._selected_context(
        bad_protocol, bad_runtime
    )
    assert "selected_protocol_variant_missing" in blockers
    assert "model_specs_missing_or_drift" in blockers
    assert "selected_model_spec_missing" in blockers
    assert "gpu_not_visible" in blockers
    assert "runtime_receipt_not_clean" in blockers
    assert "selected_backend_command_missing" in blockers

    missing_file_protocol = json.loads(json.dumps(protocol))
    missing_file_protocol["selected_model_spec"]["value"]["model_path"] = str(
        tmp_path / "missing-selected.gguf"
    )
    missing_file_protocol["selected_model_spec"]["value"]["hf_id"] = "wrong/model"
    missing_file_runtime = json.loads(json.dumps(runtime))
    missing_file_runtime["selected_backend_command"]["value"]["command"][0] = str(
        tmp_path / "missing-binary"
    )
    missing_file_runtime["selected_backend_command"]["value"]["model_path"] = str(
        tmp_path / "missing-command-model.gguf"
    )
    missing_file_runtime["selected_backend_command"]["value"]["model_role"] = "middle_moe"
    _command, _specs, _model, _variant, blockers = mod._selected_context(
        missing_file_protocol, missing_file_runtime
    )
    assert "selected_model_not_mandated" in blockers
    assert "selected_binary_missing" in blockers
    assert "selected_command_model_file_missing" in blockers
    assert "selected_model_role_mismatch" in blockers

    missing_selected_file_protocol = json.loads(json.dumps(protocol))
    missing_selected_file_protocol["selected_model_spec"]["value"]["model_path"] = str(
        tmp_path / "missing-selected-only.gguf"
    )
    _command, _specs, _model, _variant, blockers = mod._selected_context(
        missing_selected_file_protocol, runtime
    )
    assert "selected_model_file_missing" in blockers

    appended = mod.command_for_panel(
        ["llama-cli"],
        "prompt",
        n_predict=32,
        seed=7,
        variant={**variant, "stop_sequences_supported": True},
    )
    assert appended == [
        "llama-cli",
        "-p",
        "prompt",
        "-n",
        "32",
        "--seed",
        "7",
        "--reverse-prompt",
        "END_FINAL_JSON",
    ]

    base_receipt = {
        "prompt_id": prompt["prompt_id"],
        "kind": prompt["kind"],
        "category": prompt["category"],
        "completed": True,
        "stdout_tail": "",
    }
    assert (
        mod._parse_receipt({**base_receipt, "completed": False}, prompt, variant)[
            "parse_failure_reason"
        ]
        == "generation_incomplete"
    )
    assert (
        mod._parse_receipt({**base_receipt, "stdout_tail": f"{variant['sentinel']} text"}, prompt, variant)[
            "parse_failure_reason"
        ]
        == "missing_final_json_object"
    )

    malformed_payloads = [
        ({"id": "x"}, "missing_required_keys"),
        (
            {
                "id": "x",
                "accepted": "yes",
                "text": "x",
                "premise_valid": True,
                "facts": {},
                "attributes": {},
                "citations": [],
            },
            "accepted_not_boolean",
        ),
        (
            {
                "id": "x",
                "accepted": True,
                "text": "x",
                "premise_valid": "true",
                "facts": {},
                "attributes": {},
                "citations": [],
            },
            "premise_valid_not_boolean",
        ),
        (
            {
                "id": "x",
                "accepted": True,
                "text": "x",
                "premise_valid": True,
                "facts": [],
                "attributes": {},
                "citations": [],
            },
            "facts_not_object",
        ),
        (
            {
                "id": "x",
                "accepted": True,
                "text": "x",
                "premise_valid": True,
                "facts": {},
                "attributes": [],
                "citations": [],
            },
            "attributes_not_object",
        ),
        (
            {
                "id": "x",
                "accepted": True,
                "text": "x",
                "premise_valid": True,
                "facts": {},
                "attributes": {},
                "citations": "none",
            },
            "citations_not_array",
        ),
    ]
    for payload, expected in malformed_payloads:
        parsed = mod._parse_receipt(
            {**base_receipt, "stdout_tail": _final_json(variant, payload)},
            prompt,
            variant,
        )
        assert str(parsed["parse_failure_reason"]).startswith(expected)

    parse_fail = {
        "prompt_id": "rewrite_safe_paraphrase",
        "kind": "rewrite",
        "category": "safe_paraphrase",
        "parse_success": False,
        "parsed_object": None,
    }
    assert (
        mod._score_rewrite(
            parse_fail,
            mod.DEFAULT_PANEL_PROMPTS[4],
            mod.exp5325.load_fixture(),
        )["scored"]
        is False
    )

    unsafe_accept = {
        "prompt_id": "rewrite_numeric_contradiction",
        "kind": "rewrite",
        "category": "numeric_contradiction",
        "parse_success": True,
        "parsed_object": {
            "id": "rewrite_numeric_contradiction",
            "accepted": True,
            "text": "Under the amber-load test, the Aster-9 battery lasted 74 minutes.",
            "premise_valid": True,
            "facts": {
                "duration_minutes": "74",
                "subject": "aster-9 battery",
                "test": "amber-load",
            },
            "attributes": {"wording": "lasted"},
            "citations": ["battery-log-47"],
        },
    }
    unsafe_row = mod._score_rewrite(
        unsafe_accept,
        mod.DEFAULT_PANEL_PROMPTS[5],
        mod.exp5325.load_fixture(),
    )
    assert unsafe_row["unsafe_false_accept"] is True
    assert "unsafe_false_accept" in unsafe_row["semantic_failures"]

    unavailable_kwargs = dict(kwargs)
    unavailable_kwargs["paraphrase_groups"] = ()
    unavailable_kwargs["rewrite_cases"] = ()
    unavailable = mod.run(
        **unavailable_kwargs,
        generation_probe=lambda **kw: pytest.fail("generation should not run"),
        write=False,
    )
    assert "paraphrase_fixture_unavailable" in unavailable["readiness_blockers"]
    assert "rewrite_state_fixture_unavailable" in unavailable["readiness_blockers"]
