"""Tests for Exp 5251 Token-Guard-inspired Carnot fragment pilot.

Spec refs: REQ-VERIFY-5251, SCENARIO-VERIFY-5251.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from carnot.verify import token_guard_carnot_pilot_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _fixture(
    fixture_id: str = "unit-fixture",
    *,
    prompt: str = (
        "Three notebooks cost $4 each. After a $2 discount, a $3 service fee is added. "
        "How much is the final charge?"
    ),
    expected: str = "$13 final charge",
) -> mod.PilotFixture:
    return mod.PilotFixture(
        fixture_id=fixture_id,
        prompt=prompt,
        expected_outcome=expected,
        source_artifact="unit",
        taxonomy_label="omitted_premises",
    )


class FakeGenerator:
    """Deterministic stand-in for REQ-VERIFY-5251 live GGUF calls."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int, seed: int, tag: str) -> mod.GenerationReceipt:
        del max_tokens
        self.prompts.append(prompt)
        text = self.outputs.pop(0)
        return mod.GenerationReceipt(
            tag=tag,
            prompt=prompt,
            text=text,
            seed=seed,
            command=("fake-gguf", tag),
            duration_s=0.01,
            returncode=0,
            stderr_tail="",
            stdout_tail=text[-80:],
        )


def _preconditions(runtime_command: tuple[str, ...] = ("fake-gguf",)) -> mod.PreconditionReport:
    return mod.PreconditionReport(
        ok=True,
        checks=[
            {"resource": "cuda_gpu", "available": True},
            {"resource": "local_gguf_runtime", "available": True},
            {"resource": "mandated_sota_gguf", "available": True},
        ],
        selected_model={
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "model_path": "/tmp/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "quantization": "UD-Q4_K_M",
        },
        runtime_command=runtime_command,
    )


def _receipt(tag: str = "unit:baseline", text: str = "FINAL: 13") -> dict[str, object]:
    return mod.GenerationReceipt(
        tag=tag,
        prompt="unit prompt",
        text=text,
        seed=mod.RANDOM_SEED,
        command=("fake-gguf", tag),
        duration_s=0.01,
        returncode=0,
        stderr_tail="",
        stdout_tail=text,
    ).compact()


def _row(
    *,
    baseline_unsupported: int = 0,
    gated_unsupported: int = 0,
    baseline_violations: int = 0,
    gated_violations: int = 0,
    baseline_accuracy: bool = True,
    gated_accuracy: bool = True,
    regeneration_count: int = 0,
    false_accept: bool = False,
) -> dict[str, object]:
    return {
        "fixture_id": "unit",
        "taxonomy_label": "unit",
        "expected_outcome": "$13",
        "baseline": {
            "receipt": _receipt("unit:baseline"),
            "check": {
                "unsupported_claim_count": baseline_unsupported,
                "deterministic_violation_count": baseline_violations,
                "accuracy": baseline_accuracy,
            },
        },
        "gated": {
            "receipts": [_receipt("unit:fragment0"), _receipt("unit:gated_final")],
            "gate_decisions": [],
            "accepted_fragment_count": 1,
            "regeneration_count": regeneration_count,
            "final_check": {
                "unsupported_claim_count": gated_unsupported,
                "deterministic_violation_count": gated_violations,
                "accuracy": gated_accuracy,
            },
            "false_accept": false_accept,
        },
    }


def _complete_artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    return mod.build_complete_artifact(
        rows=rows,
        preconditions=_preconditions(),
        started_at="2026-07-05T00:00:00Z",
        finished_at="2026-07-05T00:00:01Z",
        duration_s=1.0,
    )


def test_req_verify_5251_spec_declares_contract() -> None:
    """REQ-VERIFY-5251: OpenSpec anchors Exp 5251 before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_verify_5251_fixture_selection_uses_existing_corpus() -> None:
    """SCENARIO-VERIFY-5251: selected fixtures come from Exp 214 deterministic corpus."""

    fixtures = mod.load_selected_fixtures(REPO)

    assert 8 <= len(fixtures) <= 12
    assert {fixture.source_artifact for fixture in fixtures} == {"exp214_followup"}
    assert all(fixture.expected_numbers for fixture in fixtures)
    assert all(fixture.fixture_id.startswith("exp214-followup-") for fixture in fixtures)


def test_req_verify_5251_fragment_gate_uses_provenance_and_energy() -> None:
    """REQ-VERIFY-5251: fragment gates reject unsupported claims before final scoring."""

    fixture = _fixture()
    accepted = mod.score_fragment(
        fixture,
        "The notebooks cost 3 x 4 = 12. Then 12 - 2 + 3 = 13.",
        prior_fragments=[],
    )
    rejected = mod.score_fragment(
        fixture,
        "Assume there are 99 notebooks and the service fee is 99.",
        prior_fragments=[],
    )

    assert accepted.accepted is True
    assert accepted.unsupported_claim_count == 0
    assert accepted.energy_score < rejected.energy_score
    assert rejected.accepted is False
    assert rejected.unsupported_claim_count >= 1
    assert "unsupported_numeric_claim" in rejected.reasons

    contradiction = mod.score_fragment(
        mod.PilotFixture(
            fixture_id="energy-unit",
            prompt="The total can be 2 or 3. What is the total?",
            expected_outcome="2",
            source_artifact="unit",
            taxonomy_label="unit",
        ),
        "The total is 2. The total is 3.",
        prior_fragments=[],
    )
    assert contradiction.accepted is False
    assert "semantic_consistency_energy" in contradiction.reasons


def test_scenario_verify_5251_fake_ab_run_reports_deltas_and_regeneration() -> None:
    """SCENARIO-VERIFY-5251: A/B run reports required deltas without Phase D scoring."""

    fixtures = [_fixture(f"f{index}", expected="$13") for index in range(8)]
    good_fragment = "The notebooks cost 3 x 4 = 12. Then 12 - 2 + 3 = 13."
    good_final = (
        "Three notebooks cost $4 each, so 3 x 4 = 12. After the $2 discount "
        "the subtotal is 10. Add the $3 service fee. FINAL: 13"
    )
    outputs = [
        "Assume there are 99 notebooks. Answer: 10",
        "Assume there are 99 notebooks.",
        good_fragment,
        good_final,
    ]
    for _index in range(7):
        outputs.extend(["Assume there are 99 notebooks. Answer: 10", good_fragment, good_final])
    fake = FakeGenerator(outputs)

    artifact = mod.run_pilot(
        repo_root=REPO,
        generator=fake,
        fixtures=fixtures,
        preconditions=_preconditions(("fake-gguf",)),
        write=False,
    )

    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["retired_phase_d_path_reopened"]["value"] is False
    assert artifact["fixtures_count"]["value"] == 8
    assert artifact["unsupported_claim_delta"]["value"] < 0
    assert artifact["deterministic_violation_delta"]["value"] < 0
    assert artifact["regeneration_count"]["value"] == 1
    assert artifact["false_accepts"]["value"] == 0
    assert artifact["accuracy_change"]["value"] > 0
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5251_blocked_artifact_has_no_tiny_model_headline() -> None:
    """REQ-VERIFY-5251: missing preconditions produce blocked artifact only."""

    artifact = mod.run_pilot(
        repo_root=REPO,
        generator=FakeGenerator([]),
        fixtures=[_fixture()],
        preconditions=mod.PreconditionReport(
            ok=False,
            checks=[
                {"resource": "cuda_gpu", "available": False, "detail": "torch.cuda unavailable"}
            ],
            selected_model=None,
            runtime_command=(),
            blocked_reason="blocked_precondition_cuda_gpu",
        ),
        write=False,
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["fixtures_count"]["value"] == 0
    assert artifact["retired_phase_d_path_reopened"]["value"] is False
    assert artifact["model_specs"]["value"]["headline_model"] is None
    assert "tiny" not in str(artifact["model_specs"]).lower()
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5251_llama_completion_runner_records_replayable_receipt(monkeypatch) -> None:
    """REQ-VERIFY-5251: live-runtime wrapper records command and cleans llama.cpp logs."""

    class Proc:
        stdout = (
            "124.456.789.000 I sampler params:\n"
            "\trepeat_last_n = 64\n"
            "<|channel>thought <channel|>The total is 13.<turn|>"
            "124.456.789.001 I common_perf_print: 1 tokens per second\n"
        )
        stderr = "main: warning\n"
        returncode = 0

    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return Proc()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    runner = mod.LlamaCompletionRunner(
        model_path="/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        runtime_path="/bin/llama-completion",
        n_ctx=256,
        n_gpu_layers=42,
        timeout_s=7,
    )

    receipt = runner.generate("Answer with FINAL: 13", max_tokens=12, seed=5251, tag="unit")

    assert receipt.text == "The total is 13."
    assert receipt.returncode == 0
    assert receipt.command == seen["cmd"]
    assert seen["kwargs"]["timeout"] == 7
    assert "-ngl" in receipt.command
    assert "42" in receipt.command
    assert receipt.prompt_checksum == mod.sha16("Answer with FINAL: 13")

    def timeout_run(cmd, **kwargs):  # noqa: ANN001
        del kwargs
        raise mod.subprocess.TimeoutExpired(cmd=cmd, timeout=7, output="partial", stderr="slow")

    monkeypatch.setattr(mod.subprocess, "run", timeout_run)
    timed_out = runner.generate("Answer slowly", max_tokens=12, seed=5251, tag="timeout")
    assert timed_out.returncode == -124
    assert timed_out.text == "partial slow timeout_s=7"


def test_req_verify_5251_parser_and_model_selection_edges() -> None:
    """REQ-VERIFY-5251: parsing helpers expose deterministic fallback behavior."""

    assert mod.render_gemma_turn_prompt("  hi  ").startswith("<|turn>user\nhi")
    assert (
        mod.clean_llama_completion_output(
            "124.456.789.000 I full log line\n"
            "build info\nmodel params\n[end of text]\nUseful [end of text] 0.14.503.864 I\n"
            "Clean 0.14.596.824 I\n"
        )
        == "Useful Clean"
    )
    assert mod.final_numbers("No explicit result here; candidates were 2 and then 3") == (3.0,)
    assert mod.infer_quantization("/tmp/model-UD-Q5_K_M.gguf") == "UD-Q5_K_M"
    assert mod.infer_quantization("/tmp/model.gguf") is None
    assert mod.select_headline_model(
        [
            {"hf_id": "other/model", "model_path": "/tmp/other.gguf"},
            {
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "model_path": "/tmp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            },
        ]
    )["quantization"] == "UD-Q4_K_M"
    assert mod.select_headline_model([{"hf_id": "other/model", "model_path": "/tmp/other.gguf"}]) is None


def test_scenario_verify_5251_complete_artifact_reports_null_and_harmful() -> None:
    """SCENARIO-VERIFY-5251: consumer recommendation follows measured deltas."""

    null_artifact = _complete_artifact([_row() for _ in range(8)])
    harmful_artifact = _complete_artifact(
        [
            _row(
                baseline_unsupported=0,
                gated_unsupported=1,
                baseline_violations=0,
                gated_violations=1,
                baseline_accuracy=True,
                gated_accuracy=False,
                false_accept=True,
            )
            for _ in range(8)
        ]
    )

    assert "null" in null_artifact["honest_verdict"]["value"]
    assert null_artifact["consumer_recommendation"]["value"].startswith("retire_or_redesign")
    assert "harmful" in harmful_artifact["honest_verdict"]["value"]
    assert harmful_artifact["consumer_recommendation"]["value"].startswith("redesign_or_retire")
    assert harmful_artifact["false_accepts"]["value"] == 8


def test_req_verify_5251_schema_errors_cover_required_failures() -> None:
    """REQ-VERIFY-5251: artifact schema guard catches field, model, and count errors."""

    good = _complete_artifact([_row() for _ in range(8)])

    missing_object = copy.deepcopy(good)
    missing_object["false_accepts"] = 0
    assert "missing_object_field:false_accepts" in mod.artifact_schema_errors(missing_object)

    missing_value = copy.deepcopy(good)
    missing_value["honest_verdict"]["value"] = "blocked_schema_test"
    missing_value["fixtures_count"].pop("value")
    missing_value["consumer_recommendation"]["principle"] = "wrong"
    errors = mod.artifact_schema_errors(missing_value)
    assert "missing_value:fixtures_count" in errors
    assert "missing_principle:consumer_recommendation" in errors

    invalid = copy.deepcopy(good)
    invalid["honest_verdict"]["value"] = "pending"
    invalid["retired_phase_d_path_reopened"]["value"] = True
    invalid["model_specs"]["value"]["headline_model"] = "tiny/model"
    errors = mod.artifact_schema_errors(invalid)
    assert "honest_verdict_prefix" in errors
    assert "retired_phase_d_reopened" in errors
    assert "headline_model_not_mandated_sota" in errors

    bad_count = copy.deepcopy(good)
    bad_count["fixtures_count"]["value"] = 2
    assert "complete_fixture_count_out_of_bounds" in mod.artifact_schema_errors(bad_count)


def test_req_verify_5251_check_preconditions_detects_cuda_runtime_and_model(
    monkeypatch, tmp_path
) -> None:
    """REQ-VERIFY-5251: precondition check requires CUDA, CUDA-linked runtime, and SOTA GGUF."""

    runtime = tmp_path / "llama-completion"
    runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime.chmod(0o755)
    model = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model.write_bytes(b"gguf")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    )

    class LddProc:
        stdout = "libggml-cuda.so.0 => /tmp/libggml-cuda.so\nlibcuda.so.1 => /tmp/libcuda.so\n"
        stderr = ""
        returncode = 0

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CARNOT_LLAMA_COMPLETION", str(runtime))
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: LddProc())
    monkeypatch.setattr(
        mod,
        "cached_sota_pair",
        lambda gpu_indices=(0, 1): [
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "model_path": str(model),
            }
        ],
    )

    report = mod.check_preconditions(tmp_path)

    assert report.ok is True
    assert report.blocked_reason == ""
    assert report.selected_model["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert str(runtime) == report.runtime_command[0]
    assert "-m" in report.runtime_command
    assert str(model) in report.runtime_command


def test_scenario_verify_5251_run_pilot_constructs_live_runner_and_writes(
    monkeypatch, tmp_path
) -> None:
    """SCENARIO-VERIFY-5251: successful pilot path writes the required artifact."""

    fixtures = [_fixture(f"f{index}", expected="$13") for index in range(8)]
    good_fragment = "The notebooks cost 3 x 4 = 12. Then 12 - 2 + 3 = 13."
    good_final = (
        "Three notebooks cost $4 each, so 3 x 4 = 12. After the $2 discount "
        "the subtotal is 10. Add the $3 service fee. FINAL: 13"
    )

    class Runner:
        def __init__(self, *, model_path: str, runtime_path: str) -> None:
            assert model_path.endswith(".gguf")
            assert runtime_path == "fake-gguf"

        def generate(self, prompt: str, *, max_tokens: int, seed: int, tag: str) -> mod.GenerationReceipt:
            del prompt, max_tokens, seed
            text = good_fragment if tag.endswith(":fragment0") else good_final
            return mod.GenerationReceipt(
                tag=tag,
                prompt="runner prompt",
                text=text,
                seed=mod.RANDOM_SEED,
                command=("fake-gguf", tag),
                duration_s=0.01,
                returncode=0,
                stderr_tail="",
                stdout_tail=text,
            )

    monkeypatch.setattr(mod, "LlamaCompletionRunner", Runner)

    artifact = mod.run_pilot(
        repo_root=tmp_path,
        generator=None,
        fixtures=fixtures,
        preconditions=_preconditions(("fake-gguf",)),
        write=True,
    )
    written = tmp_path / mod.RESULT_RELATIVE_PATH

    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["experiment"] == mod.EXPERIMENT
    assert artifact["schema_errors"] == []


def test_req_verify_5251_main_prints_result(monkeypatch, capsys) -> None:
    """REQ-VERIFY-5251: CLI entrypoint reports the deliverable path."""

    monkeypatch.setattr(
        mod,
        "run_pilot",
        lambda write=True: {
            "honest_verdict": {"value": "blocked_precondition_unit", "principle": "unit"}
        },
    )

    mod.main()

    assert mod.RESULT_RELATIVE_PATH in capsys.readouterr().out
