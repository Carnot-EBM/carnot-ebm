"""Tests for Exp 5272 gated internal/logit hallucination probe.

Spec refs: REQ-VERIFY-5272, SCENARIO-VERIFY-5272.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5272_internal_hallucination_probe_gated_v482 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _ready_telemetry_artifact() -> dict[str, Any]:
    return {
        "telemetry_harness_ready": True,
        "telemetry_harness_ready_principle": "ready through flagship_moe and flagship_dense",
        "inference_substrate": {
            "value": "live_llm_internal_telemetry_local_gguf_sota",
            "principle": "upstream",
        },
        "MODEL_SPECS": {
            "value": {
                "flagship_moe": {
                    "role": "flagship_moe",
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "quantization": "Q4_K_M",
                    "runtime_status": "telemetry_ready",
                    "status": "local_gguf_resolved",
                    "model_path": "/models/qwen.gguf",
                    "file_receipts": {
                        "path": "/models/qwen.gguf",
                        "size_bytes": 123,
                        "checksum_head_1m_sha256": "abc",
                        "checksum_sha256": None,
                    },
                },
                "flagship_dense": {
                    "role": "flagship_dense",
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "quantization": "Q4_K_M",
                    "runtime_status": "telemetry_ready",
                    "status": "local_gguf_resolved",
                    "model_path": "/models/gemma31.gguf",
                    "file_receipts": {
                        "path": "/models/gemma31.gguf",
                        "size_bytes": 456,
                        "checksum_head_1m_sha256": "def",
                        "checksum_sha256": None,
                    },
                },
                "middle_moe": {
                    "role": "middle_moe",
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "quantization": "Q4_K_M",
                    "runtime_status": "not_attempted",
                    "status": "missing_local_gguf",
                    "model_path": None,
                    "file_receipts": None,
                },
            }
        },
        "exposed_telemetry_fields": {
            "value": {
                "flagship_moe": {
                    "logits": {"availability": "available", "steps": 5, "top_k_count": 8},
                    "token_logprobs": {"availability": "available", "top_logprobs_count": 2},
                    "hidden_states": {"availability": "capability_absent"},
                    "attention_summaries": {"availability": "capability_absent"},
                },
                "flagship_dense": {
                    "logits": {"availability": "available", "steps": 6, "top_k_count": 8},
                    "token_logprobs": {"availability": "available", "top_logprobs_count": 2},
                    "hidden_states": {"availability": "capability_absent"},
                    "attention_summaries": {"availability": "capability_absent"},
                },
                "middle_moe": {
                    "logits": {"availability": "missing_local_gguf"},
                    "token_logprobs": {"availability": "missing_local_gguf"},
                    "hidden_states": {"availability": "missing_local_gguf"},
                    "attention_summaries": {"availability": "missing_local_gguf"},
                },
            }
        },
        "duration_receipts": {
            "value": {
                "total_wall_clock_s": 90.0,
                "per_model": {
                    "flagship_moe": {"wall_clock_s": 30.0, "runtime_ready": True},
                    "flagship_dense": {"wall_clock_s": 40.0, "runtime_ready": True},
                },
            }
        },
    }


def _custom_fixtures() -> list[mod.FactualFixture]:
    return [
        mod.FactualFixture(
            fixture_id="t-supported-paraphrase",
            evidence="Orin log: the lantern test lasted forty seven minutes.",
            claim="The lantern test lasted 47 minutes.",
            case_type="supported",
            unsupported_label=False,
        ),
        mod.FactualFixture(
            fixture_id="t-unsupported-absent",
            evidence="Vela note: the cold room used a brass sensor.",
            claim="The cold room used a platinum sensor.",
            case_type="unsupported",
            unsupported_label=True,
        ),
        mod.FactualFixture(
            fixture_id="t-contradiction",
            evidence="Mira note: Trial Nacre enrolled 18 participants.",
            claim="Trial Nacre enrolled 81 participants.",
            case_type="contradiction",
            unsupported_label=True,
        ),
    ]


def test_req_verify_5272_spec_declares_gated_internal_probe_contract() -> None:
    """REQ-VERIFY-5272: OpenSpec anchors the gated internal probe contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5272") :]

    for marker in (
        "REQ-VERIFY-5272",
        "SCENARIO-VERIFY-5272",
        str(mod.RESULT_RELATIVE_PATH),
        "blocked_telemetry_harness_not_ready",
        "live_llm_internal_telemetry_local_gguf_sota",
        "retired_external_scorer_reopened.value` SHALL be false",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5272_fixture_checksums_keep_labels_out_of_prompts() -> None:
    """REQ-VERIFY-5272: fixture labels are independent from rendered prompts."""

    fixtures = mod.default_fixtures()
    checksums = mod.fixture_checksums(fixtures)

    assert len(fixtures) == 9
    assert {fixture.case_type for fixture in fixtures} == {
        "supported",
        "unsupported",
        "contradiction",
    }
    assert checksums["fixture_count"] == 9
    assert checksums["case_type_counts"] == {
        "contradiction": 3,
        "supported": 3,
        "unsupported": 3,
    }
    assert checksums == mod.fixture_checksums(mod.default_fixtures())
    for fixture in fixtures:
        prompt = mod.render_prompt(fixture)
        assert fixture.claim in prompt
        assert fixture.evidence in prompt
        assert fixture.case_type not in prompt
        assert "unsupported_label" not in prompt
        assert fixture.label_source == "curated_local_evidence_label"


def test_req_verify_5272_internal_feature_math_and_controls() -> None:
    """REQ-VERIFY-5272: logit/logprob features and controls are deterministic."""

    features = mod.compute_internal_features(
        {
            "token_logprobs": [math.log(0.8), math.log(0.5)],
            "top_logprobs": [{" SAFE": math.log(0.7), " RISK": math.log(0.3)}],
            "final_logits": [0.0, 1.0],
            "logit_receipt": {"steps": 2, "vocab_size": 2},
        },
        exposed_fields={
            "logits": {"availability": "available"},
            "token_logprobs": {"availability": "available"},
            "hidden_states": {"availability": "capability_absent"},
            "attention_summaries": {"availability": "capability_absent"},
        },
    )

    assert features["signal_available"] is True
    assert features["sequence_marginal_energy"] == pytest.approx(-math.log(0.8 * 0.5) / 2)
    assert features["sequence_spilled_energy"] == pytest.approx(0.35)
    assert features["entropy_logprob_baseline"] == pytest.approx(
        -(0.7 * math.log(0.7) + 0.3 * math.log(0.3))
    )
    assert features["full_logit_spilled_energy"] == pytest.approx(1.0 / (1.0 + math.e))
    assert features["primary_internal_score"] == pytest.approx(features["sequence_marginal_energy"])

    rows = [
        {
            "unsupported_label": False,
            "case_type": "supported",
            "scores": {"internal": 0.1, "lexical": 0.9},
        },
        {
            "unsupported_label": False,
            "case_type": "supported",
            "scores": {"internal": 0.2, "lexical": 0.8},
        },
        {
            "unsupported_label": True,
            "case_type": "unsupported",
            "scores": {"internal": 0.8, "lexical": 0.3},
        },
        {
            "unsupported_label": True,
            "case_type": "contradiction",
            "scores": {"internal": 0.9, "lexical": 0.2},
        },
    ]
    summary = mod.summarize_rows(rows)

    assert summary["sample_count"] == 4
    assert summary["internal"]["auroc"] == pytest.approx(1.0)
    assert summary["lexical"]["auroc"] == pytest.approx(0.0)
    assert summary["delta_over_lexical_baseline"] == pytest.approx(1.0)
    assert summary["false_accepts"] == 0
    assert summary["shuffled_label_control"]["sample_count"] == 4


def test_scenario_verify_5272_blocks_when_telemetry_harness_gate_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5272: a closed Exp 5271 gate writes an unmeasured artifact."""

    artifact = mod.run_probe(
        result_path=tmp_path / "blocked.json",
        telemetry_artifact={"telemetry_harness_ready": False},
        generation_runner=lambda fixture, model_spec, exposed_fields, seed: {
            "raw_response": "SUPPORTED"
        },
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("blocked_telemetry_harness_not_ready")
    assert "unmeasured" in artifact["honest_verdict"]["value"]
    assert artifact["internal_signal_available"]["value"] is False
    assert artifact["delta_over_lexical_baseline"]["value"] == 0.0
    assert artifact["auroc"]["value"] is None
    assert artifact["false_accepts"]["value"] == 0
    assert artifact["retired_external_scorer_reopened"]["value"] is False
    assert artifact["pilot_rows"] == []
    mod.validate_artifact(artifact)


def test_scenario_verify_5272_runs_injected_multimodel_logprob_probe(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5272: available logprob receipts produce controlled metrics."""

    fixtures = _custom_fixtures()

    def runner(
        fixture: mod.FactualFixture,
        model_spec: dict[str, Any],
        exposed_fields: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        assert model_spec["role"] in {"flagship_moe", "flagship_dense"}
        assert exposed_fields["logits"]["availability"] == "available"
        base = math.log(0.35 if fixture.unsupported_label else 0.95)
        return {
            "raw_response": "UNSUPPORTED" if fixture.unsupported_label else "SUPPORTED",
            "token_logprobs": [base, base],
            "top_logprobs": [
                {
                    " UNSUPPORTED": math.log(0.6 if fixture.unsupported_label else 0.1),
                    " SUPPORTED": math.log(0.4 if fixture.unsupported_label else 0.9),
                }
            ],
            "final_logits": [0.0, 1.0] if fixture.unsupported_label else [1.0, 0.0],
            "token_count": 2,
            "seed": seed,
            "logit_receipt": {"steps": 1, "vocab_size": 2},
        }

    artifact = mod.run_probe(
        result_path=tmp_path / "ready.json",
        telemetry_artifact=_ready_telemetry_artifact(),
        generation_runner=runner,
        fixtures=fixtures,
        tests_run=[{"command": "unit ready", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / "ready.json").read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "positive" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["internal_signal_available"]["value"] is True
    assert artifact["auroc"]["value"] == pytest.approx(1.0)
    assert artifact["delta_over_lexical_baseline"]["value"] > 0.0
    assert artifact["false_accepts"]["value"] == 0
    assert artifact["telemetry_receipts"]["value"]["sample_count"] == 6
    assert artifact["telemetry_receipts"]["value"]["model_roles_scored"] == [
        "flagship_moe",
        "flagship_dense",
    ]
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["headline_metric_role"] is True
    assert artifact["MODEL_SPECS"]["value"]["middle_moe"]["headline_metric_role"] is False
    assert len(artifact["pilot_rows"]) == len(fixtures) * 2
    assert all(row["prompt_checksum"] for row in artifact["pilot_rows"])
    assert artifact["retired_external_scorer_reopened"]["value"] is False


def test_req_verify_5272_schema_rejects_malformed_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5272: malformed artifacts fail closed instead of looking complete."""

    valid = mod.run_probe(
        result_path=tmp_path / "valid.json",
        telemetry_artifact={"telemetry_harness_ready": False},
        generation_runner=lambda fixture, model_spec, exposed_fields, seed: {},
        tests_run=[],
    )

    for mutation, message in (
        (
            lambda art: {key: value for key, value in art.items() if key != "honest_verdict"},
            "missing required field",
        ),
        (
            lambda art: (
                art
                | {
                    "honest_verdict": {
                        "value": "pending",
                        "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                    }
                }
            ),
            "honest_verdict",
        ),
        (
            lambda art: (
                art
                | {
                    "internal_signal_available": {
                        "value": "false",
                        "principle": mod.FIELD_PRINCIPLES["internal_signal_available"],
                    }
                }
            ),
            "internal_signal_available",
        ),
        (
            lambda art: (
                art
                | {
                    "delta_over_lexical_baseline": {
                        "value": None,
                        "principle": mod.FIELD_PRINCIPLES["delta_over_lexical_baseline"],
                    }
                }
            ),
            "delta_over_lexical_baseline",
        ),
        (
            lambda art: (
                art
                | {
                    "retired_external_scorer_reopened": {
                        "value": True,
                        "principle": mod.FIELD_PRINCIPLES["retired_external_scorer_reopened"],
                    }
                }
            ),
            "retired_external_scorer_reopened",
        ),
        (
            lambda art: (
                art
                | {
                    "inference_substrate": {
                        "value": "live_llm_inference_local_gguf_sota",
                        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                    }
                }
            ),
            "inference_substrate",
        ),
    ):
        with pytest.raises(AssertionError, match=message):
            mod.validate_artifact(mutation(valid))


def test_req_verify_5272_blocker_paths_and_optional_middle_role(tmp_path: Path) -> None:
    """REQ-VERIFY-5272: precondition blockers and optional roles are explicit."""

    missing_dense = _ready_telemetry_artifact()
    missing_dense["MODEL_SPECS"]["value"]["flagship_dense"]["model_path"] = None
    missing_dense["MODEL_SPECS"]["value"]["flagship_dense"]["file_receipts"] = None
    missing_dense["exposed_telemetry_fields"]["value"]["flagship_dense"]["logits"] = {
        "availability": "missing_local_gguf"
    }

    blocked_model = mod.run_probe(
        result_path=tmp_path / "missing-model.json",
        telemetry_artifact=missing_dense,
        generation_runner=lambda fixture, model_spec, exposed_fields, seed: {
            "token_logprobs": [-0.1]
        },
        tests_run=[],
        write=False,
    )
    assert blocked_model["honest_verdict"]["value"].startswith(
        "blocked_headline_models_unavailable"
    )

    inadequate = mod.run_probe(
        result_path=tmp_path / "bad-fixtures.json",
        telemetry_artifact=_ready_telemetry_artifact(),
        generation_runner=lambda fixture, model_spec, exposed_fields, seed: {
            "token_logprobs": [-0.1]
        },
        fixtures=_custom_fixtures()[:2],
        tests_run=[],
        write=False,
    )
    assert inadequate["honest_verdict"]["value"].startswith("blocked_fixture_labels_inadequate")

    no_receipts = mod.run_probe(
        result_path=tmp_path / "no-receipts.json",
        telemetry_artifact=_ready_telemetry_artifact(),
        generation_runner=lambda fixture, model_spec, exposed_fields, seed: {
            "raw_response": "SUPPORTED"
        },
        fixtures=_custom_fixtures(),
        tests_run=[],
        write=False,
    )
    assert no_receipts["honest_verdict"]["value"].startswith(
        "blocked_live_internal_signal_unmeasured"
    )
    assert no_receipts["pilot_rows"]
    assert no_receipts["internal_signal_available"]["value"] is False

    with_middle = _ready_telemetry_artifact()
    with_middle["MODEL_SPECS"]["value"]["middle_moe"].update(
        {
            "runtime_status": "telemetry_ready",
            "status": "local_gguf_resolved",
            "model_path": "/models/gemma26.gguf",
            "file_receipts": {
                "path": "/models/gemma26.gguf",
                "size_bytes": 789,
                "checksum_head_1m_sha256": "ghi",
                "checksum_sha256": None,
            },
        }
    )
    with_middle["exposed_telemetry_fields"]["value"]["middle_moe"] = {
        "logits": {"availability": "available"},
        "token_logprobs": {"availability": "available"},
        "hidden_states": {"availability": "capability_absent"},
        "attention_summaries": {"availability": "capability_absent"},
    }

    def middle_runner(
        fixture: mod.FactualFixture,
        model_spec: dict[str, Any],
        exposed_fields: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        del exposed_fields, seed
        assert model_spec["role"] in {"flagship_moe", "flagship_dense", "middle_moe"}
        base = math.log(0.2 if fixture.unsupported_label else 0.9)
        return {"raw_response": "OK", "token_logprobs": [base], "token_count": 1}

    complete = mod.run_probe(
        result_path=tmp_path / "middle.json",
        telemetry_artifact=with_middle,
        generation_runner=middle_runner,
        fixtures=_custom_fixtures(),
        tests_run=[],
        write=False,
    )
    assert complete["telemetry_receipts"]["value"]["model_roles_scored"] == [
        "flagship_moe",
        "flagship_dense",
        "middle_moe",
    ]
    assert complete["MODEL_SPECS"]["value"]["middle_moe"]["headline_metric_role"] is True


def test_req_verify_5272_feature_fallbacks_and_helper_edges() -> None:
    """REQ-VERIFY-5272: edge helpers fail neutral instead of inventing signal."""

    field_logits = {
        "logits": {"availability": "available"},
        "token_logprobs": {"availability": "capability_absent"},
        "hidden_states": {"availability": "capability_absent"},
        "attention_summaries": {"availability": "capability_absent"},
    }
    receipt_features = mod.compute_internal_features(
        {"logit_receipt": {"top1_probability": 0.75, "entropy_topk": 0.5}},
        exposed_fields=field_logits,
    )
    assert receipt_features["primary_internal_score_name"] == "full_logit_spilled_energy"
    assert receipt_features["primary_internal_score"] == pytest.approx(0.25)

    field_logprobs = {
        "logits": {"availability": "capability_absent"},
        "token_logprobs": {"availability": "available"},
        "hidden_states": {"availability": "capability_absent"},
        "attention_summaries": {"availability": "capability_absent"},
    }
    top_only = mod.compute_internal_features(
        {"top_logprobs": [{"A": math.log(0.6), "B": math.log(0.4)}]},
        exposed_fields=field_logprobs,
    )
    assert top_only["primary_internal_score_name"] == "final_token_spilled_energy"
    assert top_only["entropy_logprob_baseline"] == pytest.approx(
        -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
    )

    assert (
        mod.lexical_risk_score(mod.FactualFixture("empty", "anything", "", "supported", False))
        == 0.0
    )
    assert mod._rotated_labels([1]) == [1]
    assert mod._auroc([1, 1], [0.1, 0.2]) is None
    assert mod._availability({"logits": True}, "logits") == "missing_receipt"
    assert mod._first_choice({"choices": [{"text": "OK", "logprobs": {}}]})["text"] == "OK"
    assert mod._first_choice("raw")["text"] == "raw"
    assert mod._full_logit_summary([]) == {}
    assert (
        mod._summary_from_generation({"logit_receipt": {"top1_probability": 0.5}})[
            "top1_probability"
        ]
        == 0.5
    )
    assert mod._summary_from_generation({}) == {}
    assert mod._final_logits({"logits": [[0.1, 0.2]]}) == [0.1, 0.2]
    assert mod._final_logits({}) == []
    assert mod._top_logprob_rows("bad") == []
    assert mod._numeric_values("bad") == []
    assert mod._optional_float(True) is None
    assert mod._softmax_log_values([]) == []
    assert mod._entropy([]) is None
    assert mod._selected_final_probability([]) is None
    assert mod._token_count({"tokens": ["a", "b"]}, []) == 2
    assert mod._nested_value({"x": {"value": 3}}, "x") == 3
    assert mod._nested_value("bad", "x") is None
    assert mod._format_optional_float(None) == "null"
    assert (
        mod._role_ready(
            {"flagship_moe": {"model_path": "/models/qwen.gguf", "legacy_tiny_model": True}},
            {"flagship_moe": {"logits": {"availability": "available"}}},
            "flagship_moe",
        )
        is False
    )
    assert "harmful" in mod._complete_verdict(delta=-0.1, auroc=0.25, sample_count=4)
    assert "null" in mod._complete_verdict(delta=0.0, auroc=None, sample_count=4)


def test_req_verify_5272_schema_error_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5272: schema diagnostics cover malformed edge cases."""

    valid = mod.run_probe(
        result_path=tmp_path / "valid-edge.json",
        telemetry_artifact={"telemetry_harness_ready": False},
        generation_runner=lambda fixture, model_spec, exposed_fields, seed: {},
        tests_run=[],
        write=False,
    )

    cases = [
        (
            valid
            | {
                "honest_verdict": {
                    "value": "complete: finished",
                    "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                }
            },
            "must state positive",
        ),
        (
            valid
            | {
                "honest_verdict": {
                    "value": valid["honest_verdict"]["value"],
                    "principle": "wrong",
                }
            },
            "principle mismatch",
        ),
        (
            valid
            | {
                "auroc": {
                    "value": True,
                    "principle": mod.FIELD_PRINCIPLES["auroc"],
                }
            },
            "auroc.value",
        ),
        (
            valid
            | {
                "false_accepts": {
                    "value": "0",
                    "principle": mod.FIELD_PRINCIPLES["false_accepts"],
                }
            },
            "false_accepts.value",
        ),
        (valid | {"tests_run": "pytest"}, "tests_run"),
        (
            valid
            | {
                "MODEL_SPECS": {
                    "value": "bad",
                    "principle": mod.FIELD_PRINCIPLES["MODEL_SPECS"],
                }
            },
            "MODEL_SPECS.value must be an object",
        ),
        (
            valid
            | {
                "MODEL_SPECS": {
                    "value": {
                        key: value
                        for key, value in valid["MODEL_SPECS"]["value"].items()
                        if key != "middle_moe"
                    },
                    "principle": mod.FIELD_PRINCIPLES["MODEL_SPECS"],
                }
            },
            "missing role middle_moe",
        ),
        (
            valid
            | {
                "MODEL_SPECS": {
                    "value": valid["MODEL_SPECS"]["value"]
                    | {
                        "flagship_moe": valid["MODEL_SPECS"]["value"]["flagship_moe"]
                        | {"hf_id": "wrong"}
                    },
                    "principle": mod.FIELD_PRINCIPLES["MODEL_SPECS"],
                }
            },
            "hf_id mismatch",
        ),
        (
            valid
            | {
                "telemetry_receipts": {
                    "value": "bad",
                    "principle": mod.FIELD_PRINCIPLES["telemetry_receipts"],
                }
            },
            "telemetry_receipts.value must be an object",
        ),
        (
            valid
            | {
                "telemetry_receipts": {
                    "value": {"duration": {}},
                    "principle": mod.FIELD_PRINCIPLES["telemetry_receipts"],
                }
            },
            "field_availability missing",
        ),
        (
            valid
            | {
                "telemetry_receipts": {
                    "value": {"field_availability": {}},
                    "principle": mod.FIELD_PRINCIPLES["telemetry_receipts"],
                }
            },
            "duration missing",
        ),
    ]

    for artifact, expected in cases:
        assert any(expected in error for error in mod.artifact_schema_errors(artifact))
