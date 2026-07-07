"""Tests for Exp 5326 gated SOTA paraphrase/rewrite smoke.

Spec refs: REQ-VERIFY-5326, SCENARIO-VERIFY-5326.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import pytest

from carnot import experiment_5326_gated_sota_paraphrase_rewrite_smoke_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _resolver_from_paths(paths: dict[str, Path]):
    def resolver(hf_id: str, _quant: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolver


def _write_prior(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stable_exp5324_artifact(binary: Path, model_path: Path) -> dict[str, Any]:
    command = [
        str(binary),
        "-m",
        str(model_path),
        "-p",
        "Write eight lowercase color words separated by spaces.",
        "-n",
        "8",
        "-c",
        "512",
        "-b",
        "512",
        "-ub",
        "128",
        "-ngl",
        "all",
        "-sm",
        "layer",
        "--temp",
        "0",
        "--seed",
        "5324",
        "--no-display-prompt",
        "--simple-io",
        "-st",
        "--perf",
    ]
    return {
        "status": {"value": "complete", "principle": "prior status"},
        "honest_verdict": {"value": "complete: stable", "principle": "prior verdict"},
        "sota_runtime_unblocked_stable": True,
        "selected_backend_command": {
            "value": {
                "backend_kind": "llama-cli",
                "backend_variant": "llama-cli-single-turn-batch512",
                "command": command,
                "model_path": str(model_path),
                "model_role": "flagship_dense",
                "n_predict": 8,
                "timeout_s": 240.0,
                "gpu_memory_delta_mb": 9000,
            },
            "principle": "selected command",
        },
        "selected_model_spec": {
            "value": {
                "role": "flagship_dense",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "model_path": str(model_path),
                "status": "local_gguf_resolved",
                "cached": True,
                "autotokenizer_used": False,
            },
            "principle": "selected model",
        },
    }


def _current_preconditions(*, gpu_visible: bool = True) -> dict[str, Any]:
    return {
        "gpu_visible": gpu_visible,
        "free_vram_mb": 48240 if gpu_visible else 0,
        "cuda_backend_evidence": gpu_visible,
        "raw_nvidia_smi": {"ok": gpu_visible, "stdout": "CUDA UMD Version: 13.3"},
        "nvidia_smi": {
            "ok": gpu_visible,
            "stdout": "0, NVIDIA RTX 3090, 610.43.02, 24576, 24000, 0",
        },
        "blocked_preconditions": [],
    }


def _successful_generation(**kwargs: Any) -> dict[str, Any]:
    prompt_id = kwargs["prompt_spec"]["prompt_id"]
    outputs = {
        "paraphrase_supported": {
            "text": "Under the amber-load test, Aster-9's battery lasted 47 minutes.",
            "premise_valid": True,
            "facts": {
                "duration_minutes": "47",
                "subject": "aster-9 battery",
                "test": "amber-load",
            },
        },
        "rewrite_safe_paraphrase": {
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
        "rewrite_unsafe_contradiction": {
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
    return {
        "completed": True,
        "timed_out": False,
        "returncode": 0,
        "stdout": json.dumps(outputs[prompt_id]),
        "stderr": "load_tensors: offloaded 49/49 layers to GPU",
        "wall_clock_s": 20.0,
    }


def _base_run_kwargs(tmp_path: Path) -> dict[str, Any]:
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    prior_path = _write_prior(
        tmp_path / mod.exp5324.RESULT_RELATIVE_PATH,
        _stable_exp5324_artifact(binary, model_path),
    )
    return {
        "root": tmp_path,
        "artifact_path": tmp_path / mod.RESULT_RELATIVE_PATH,
        "prior_artifact_path": prior_path,
        "model_resolver": _resolver_from_paths({"unsloth/gemma-4-31B-it-GGUF": model_path}),
        "current_preconditions_provider": lambda: _current_preconditions(),
        "tests_run": [{"command": "unit exp5326", "outcome": "passed"}],
    }


def test_req_verify_5326_spec_declares_gated_smoke_contract() -> None:
    """REQ-VERIFY-5326: OpenSpec anchors the bounded smoke contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5326") : spec.index("### REQ-VERIFY-5325")]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5326",
        "SCENARIO-VERIFY-5326",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "local_sota_gguf_bounded_smoke",
        "prompt_count",
        "paraphrase_label_preservation_rate",
        "rewrite_acceptability_rate",
        "unsafe_false_accepts",
        "sota_quality_measured",
        "headline_quality_claim=false",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(mod.FIELD_PRINCIPLES[field].split()) in normalized_section


def test_scenario_verify_5326_stable_runtime_scores_fixture_outputs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5326: bounded outputs are scored by deterministic fixtures."""

    calls: list[tuple[str, list[str]]] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["prompt_spec"]["prompt_id"], kwargs["command"]))
        return _successful_generation(**kwargs)

    artifact = mod.run(
        **_base_run_kwargs(tmp_path),
        generation_probe=probe,
        write=True,
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert [prompt_id for prompt_id, _command in calls] == [
        "paraphrase_supported",
        "rewrite_safe_paraphrase",
        "rewrite_unsafe_contradiction",
    ]
    assert all(command[0].endswith("llama-cli") for _prompt_id, command in calls)
    assert all("-p" in command and "-n" in command for _prompt_id, command in calls)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["prompt_count"] == 3
    assert artifact["paraphrase_label_preservation_rate"] == pytest.approx(1.0)
    assert artifact["rewrite_acceptability_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["sota_quality_measured"] is True
    assert artifact["headline_quality_claim"] is False
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert {row["hf_id"] for row in artifact["MODEL_SPECS"]["value"].values()} == {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    }


def test_req_verify_5326_blocks_before_generation_when_exp5324_not_stable(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5326: missing stable Exp5324 receipt blocks before generation."""

    kwargs = _base_run_kwargs(tmp_path)
    prior_path = kwargs["prior_artifact_path"]
    _write_prior(prior_path, {"sota_runtime_unblocked_stable": False})
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
    assert artifact["sota_quality_measured"] is False
    assert artifact["headline_quality_claim"] is False
    assert "exp5324_stable_runtime_missing" in artifact["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]


def test_scenario_verify_5326_generation_or_parse_failures_close_measurement(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5326: failed bounded output keeps quality measured false."""

    def malformed_probe(**kwargs: Any) -> dict[str, Any]:
        if kwargs["prompt_spec"]["prompt_id"] == "rewrite_safe_paraphrase":
            return {
                "completed": False,
                "timed_out": True,
                "returncode": None,
                "stdout": "",
                "stderr": "timeout",
                "wall_clock_s": 240.0,
            }
        if kwargs["prompt_spec"]["prompt_id"] == "paraphrase_supported":
            return {
                "completed": True,
                "timed_out": False,
                "returncode": 0,
                "stdout": "not json",
                "stderr": "",
                "wall_clock_s": 20.0,
            }
        return _successful_generation(**kwargs)

    artifact = mod.run(
        **_base_run_kwargs(tmp_path),
        generation_probe=malformed_probe,
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["sota_quality_measured"] is False
    assert artifact["prompt_count"] == 2
    assert artifact["paraphrase_label_preservation_rate"] == pytest.approx(0.0)
    assert artifact["rewrite_acceptability_rate"] == pytest.approx(0.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert "generation failed: rewrite_safe_paraphrase" in artifact["readiness_blockers"]


def test_req_verify_5326_schema_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5326: schema validation catches malformed smoke artifacts."""

    artifact = mod.run(
        **_base_run_kwargs(tmp_path),
        generation_probe=_successful_generation,
        write=False,
    )

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(artifact))

    malformed_cases = [
        (lambda a: (a.pop("MODEL_SPECS"), a)[1], "missing required fields"),
        (lambda a: (a.__setitem__("experiment_id", mod.EXPERIMENT_ID), a)[1], "principle-wrapped"),
        (
            lambda a: (
                a["honest_verdict"].__setitem__("value", "done"),
                a,
            )[1],
            "honest_verdict",
        ),
        (
            lambda a: (
                a["milestone"].__setitem__("value", "wrong"),
                a,
            )[1],
            "milestone mismatch",
        ),
        (
            lambda a: (
                a["status"].__setitem__("value", "running"),
                a,
            )[1],
            "status must be complete or blocked",
        ),
        (
            lambda a: (
                a["inference_substrate"].__setitem__("value", "wrong"),
                a,
            )[1],
            "inference_substrate mismatch",
        ),
        (
            lambda a: (
                a["tests_run"].__setitem__("principle", "wrong"),
                a,
            )[1],
            "tests_run must be principle-wrapped",
        ),
        (
            lambda a: (
                a.__setitem__("headline_quality_claim", True),
                a,
            )[1],
            "headline_quality_claim must be bare false",
        ),
        (
            lambda a: (
                a.__setitem__("sota_quality_measured", "yes"),
                a,
            )[1],
            "sota_quality_measured must be a bare boolean",
        ),
        (
            lambda a: (
                a.__setitem__("prompt_count", "3"),
                a,
            )[1],
            "prompt_count must be a bare integer",
        ),
        (
            lambda a: (
                a.__setitem__("paraphrase_label_preservation_rate", 2.0),
                a,
            )[1],
            "paraphrase_label_preservation_rate must be in [0, 1]",
        ),
        (
            lambda a: (
                a.__setitem__("unsafe_false_accepts", "0"),
                a,
            )[1],
            "unsafe_false_accepts must be a bare integer",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"].pop("middle_moe"),
                a,
            )[1],
            "MODEL_SPECS roles mismatch",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"),
                a,
            )[1],
            "hf_id mismatch",
        ),
        (
            lambda a: (
                a["tests_run"].__setitem__("value", "bad"),
                a,
            )[1],
            "tests_run must be a list",
        ),
        (
            lambda a: (
                a["status"].__setitem__("value", "blocked"),
                a,
            )[1],
            "measured artifact must have complete status",
        ),
        (
            lambda a: (
                a["selected_model_spec"].__setitem__("value", None),
                a,
            )[1],
            "selected_model_spec must be an object when measured",
        ),
    ]

    for mutate, expected in malformed_cases:
        joined = "; ".join(mod.artifact_schema_errors(mutate(clone())))
        assert expected in joined
    with pytest.raises(AssertionError, match="headline_quality_claim"):
        bad = clone()
        bad["headline_quality_claim"] = True
        mod.validate_artifact(bad)


def test_req_verify_5326_parsers_and_command_rewrite_are_defensive(tmp_path: Path) -> None:
    """REQ-VERIFY-5326: command edits are bounded and JSON parsing is structured."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._read_json(bad_json) == {}

    command = ["llama-cli", "-m", "model.gguf", "-p", "old", "-n", "8", "--seed", "1"]
    rewritten = mod.command_for_prompt(command, "new prompt", n_predict=64, seed=99)

    assert rewritten[:4] == ["llama-cli", "-m", "model.gguf", "-p"]
    assert rewritten[rewritten.index("-p") + 1] == "new prompt"
    assert rewritten[rewritten.index("-n") + 1] == "64"
    assert rewritten[rewritten.index("--seed") + 1] == "99"

    appended = mod.command_for_prompt(["llama-cli", "-m", "model.gguf"], "prompt", 32, 7)
    assert appended[-6:] == ["-p", "prompt", "-n", "32", "--seed", "7"]
    assert mod.extract_json_object("{bad}{\"ok\": true}") == {"ok": True}
    assert mod.extract_json_object("prefix ```json\n{\"ok\": true}\n``` suffix") == {"ok": True}
    assert mod.extract_json_object("no-json") is None
    assert mod._string_map(None) == {}
    assert mod._string_tuple("not-a-list") == ()

    blockers = mod._precondition_blockers(
        selected_command={"command": ["llama-cli"]},
        prior_selected_model={"role": "flagship_dense"},
        selected_model={
            "status": "local_gguf_resolved",
            "model_path": str(tmp_path / "missing.gguf"),
            "hf_id": "wrong",
        },
        current_preconditions={"gpu_visible": False},
        fixture_status={"paraphrase_fixture_ready": False, "rewrite_state_fixture_ready": False},
    )
    assert blockers == [
        "gpu_not_visible",
        "selected_model_file_missing",
        "selected_model_not_mandated",
        "paraphrase_fixture_unavailable",
        "rewrite_state_fixture_unavailable",
    ]

    rewrite_parse_error = mod._score_rewrite_output(
        {"prompt_id": "rewrite_safe_paraphrase", "output_text": "not json"},
        {"fixture_case_id": "rsf-001-safe-paraphrase"},
        mod.exp5325.load_fixture(),
    )
    assert rewrite_parse_error["parse_ok"] is False
    assert rewrite_parse_error["acceptability_matches_expected"] is False
