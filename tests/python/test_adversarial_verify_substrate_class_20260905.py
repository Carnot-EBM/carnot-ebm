"""The closed substrate class: declared by the artifact, held to the gate's own floors.

Spec: REQ-SUBSTRATE-CLASS-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-SUBSTRATE-CLASS-1 (absent class with an unrecognised name: warn,
never a stamp), SCENARIO-SUBSTRATE-CLASS-2 (absent class with a recognised name:
nothing -- the stated gap), SCENARIO-SUBSTRATE-CLASS-3 (a value outside the enum, or
not a bare string: critical), SCENARIO-SUBSTRATE-CLASS-4 (blocked_no_run pairs with a
blocked verdict; a model class forbids one), SCENARIO-SUBSTRATE-CLASS-5 (typed
invocation evidence contradicts the class: critical), SCENARIO-SUBSTRATE-CLASS-6
(duration below the class floor: critical; blocked runs and None floors skip),
SCENARIO-SUBSTRATE-CLASS-7 (a dict-shaped inference_substrate is malformed: warn, and
read as missing), SCENARIO-SUBSTRATE-CLASS-8 (wired into the full verifier; a
principle-wrapped class unwraps).

Origin: docs/research-notes/substrate-vocabulary-census-and-recommendation-2026-09-05.md
(1036 distinct substrate strings over 6056 artifacts; 1153 unknown to the gate).
Nothing here reads or writes results/**.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.adversarial_verify as av  # noqa: E402

MISSING = av.SUBSTRATE_CLASS_MISSING_KIND
MISMATCH = av.SUBSTRATE_CLASS_MISMATCH_KIND
MALFORMED = av.SUBSTRATE_DECLARATION_MALFORMED_KIND


def _class_flags(d: dict) -> list:
    flags: list = []
    av.check_substrate_class(d, flags)
    return [f for f in flags if f.kind in {MISSING, MISMATCH}]


def _kinds(d: dict) -> list[tuple[str, str]]:
    return [(f.kind, f.severity) for f in _class_flags(d)]


def test_the_enum_is_the_seven_classes_keyed_to_the_floors_the_gate_applies() -> None:
    assert av.SUBSTRATE_CLASSES == frozenset(
        {
            "aggregation",
            "no_model_load",
            "model_load_no_generation",
            "model_bounded_generation",
            "model_full_generation",
            "hardware_board",
            "blocked_no_run",
        }
    )
    floors = av.SUBSTRATE_CLASS_FLOORS
    assert floors["aggregation"] == av.AGGREGATION_MIN_DURATION_S
    assert floors["no_model_load"] == av.NO_LLM_DECLARED_MIN_DURATION_S
    assert floors["model_load_no_generation"] == av.LLM_EMBEDDING_EXTRACTION_MIN_DURATION_S
    assert floors["model_bounded_generation"] == av.LOCAL_SOTA_GGUF_SMALL_N_MIN_DURATION_S
    assert floors["model_full_generation"] == av.COMPUTE_BOUND_MIN_DURATION_S == 60.0
    # Stated, not invented: the gate has no hardware floor today, and a blocked run has none.
    assert floors["hardware_board"] is None
    assert floors["blocked_no_run"] is None


def test_absent_class_with_an_unrecognised_name_warns_and_names_the_enum() -> None:
    # SCENARIO-SUBSTRATE-CLASS-1: the population where the class would have decided the floor.
    out = _class_flags(
        {
            "inference_substrate": "fresh_process_exact_candidate_selection_replay_x9",
            "duration_s": 1.0,
        }
    )
    assert [(f.kind, f.severity) for f in out] == [(MISSING, "warn")]
    assert "inference_substrate_class" in out[0].detail
    assert "model_full_generation" in out[0].detail


def test_absent_class_with_a_recognised_name_draws_nothing_which_is_the_stated_gap() -> None:
    # SCENARIO-SUBSTRATE-CLASS-2. Measured 2026-09-05: a universal warn broke an existing
    # empty-flag test on a real artifact (exp4628); 33 such assertions exist by AST parse,
    # 9 of them on a capstone's stored verify report. So a recognised name is NOT nudged
    # here; adoption for it rides on the planner prompt and CLAUDE.md.
    assert _class_flags({"inference_substrate": "aggregation_from_upstream_artifacts"}) == []
    assert _class_flags({"inference_substrate": "cached_replay_fixture_no_llm"}) == []
    assert _class_flags({"inference_substrate": "live_llm_inference", "duration_s": 3.0}) == []
    assert _class_flags({"duration_s": 5.0}) == []


def test_a_value_outside_the_enum_or_not_a_bare_string_is_critical() -> None:
    # SCENARIO-SUBSTRATE-CLASS-3
    out = _class_flags({"inference_substrate_class": "banana"})
    assert [(f.kind, f.severity) for f in out] == [(MISMATCH, "critical")]
    assert "closed enum" in out[0].detail
    assert _kinds({"inference_substrate_class": {"kind": "aggregation"}}) == [
        (MISMATCH, "critical")
    ]
    assert _kinds({"inference_substrate_class": ["aggregation"]}) == [(MISMATCH, "critical")]
    assert _kinds({"inference_substrate_class": "Aggregation"}) == [(MISMATCH, "critical")]


def test_blocked_no_run_pairs_with_a_blocked_verdict() -> None:
    # SCENARIO-SUBSTRATE-CLASS-4
    assert (
        _kinds(
            {
                "inference_substrate_class": "blocked_no_run",
                "honest_verdict": "complete: blocked_model_not_cached_gemma_4_31B",
                "duration_s": 0.0,
            }
        )
        == []
    )
    out = _class_flags(
        {"inference_substrate_class": "blocked_no_run", "honest_verdict": "complete: ok"}
    )
    assert [(f.kind, f.severity) for f in out] == [(MISMATCH, "critical")]
    assert "blocked_*" in out[0].detail


def test_a_model_class_with_a_blocked_verdict_is_critical() -> None:
    # SCENARIO-SUBSTRATE-CLASS-4, the other direction. The floor check is skipped for a
    # blocked run, so exactly one flag: the contradiction.
    out = _class_flags(
        {
            "inference_substrate_class": "model_full_generation",
            "honest_verdict": "blocked_cuda_unavailable",
            "duration_s": 0.1,
        }
    )
    assert [(f.kind, f.severity) for f in out] == [(MISMATCH, "critical")]
    assert "blocked_no_run" in out[0].detail


def test_typed_live_evidence_contradicts_a_no_model_class() -> None:
    # SCENARIO-SUBSTRATE-CLASS-5
    for cls in ("no_model_load", "aggregation"):
        out = _class_flags(
            {"inference_substrate_class": cls, "generation_invoked": True, "duration_s": 5.0}
        )
        assert [(f.kind, f.severity) for f in out] == [(MISMATCH, "critical")], cls
        assert "generation_invoked" in out[0].detail


def test_typed_negative_evidence_contradicts_a_model_class() -> None:
    # SCENARIO-SUBSTRATE-CLASS-5
    out = _class_flags(
        {
            "inference_substrate_class": "model_full_generation",
            "generation_invoked": False,
            "duration_s": 500.0,
        }
    )
    assert [(f.kind, f.severity) for f in out] == [(MISMATCH, "critical")]
    assert "did not" in out[0].detail


def test_duration_below_the_class_floor_is_critical_and_none_floors_skip() -> None:
    # SCENARIO-SUBSTRATE-CLASS-6
    assert _kinds({"inference_substrate_class": "model_full_generation", "duration_s": 2.0}) == [
        (MISMATCH, "critical")
    ]
    assert _kinds({"inference_substrate_class": "model_full_generation", "duration_s": 120.0}) == []
    assert _kinds({"inference_substrate_class": "model_bounded_generation", "duration_s": 9.0}) == [
        (MISMATCH, "critical")
    ]
    assert _kinds({"inference_substrate_class": "model_load_no_generation", "duration_s": 1.0}) == [
        (MISMATCH, "critical")
    ]
    assert _kinds({"inference_substrate_class": "aggregation", "duration_s": 0.0}) == [
        (MISMATCH, "critical")
    ]
    assert _kinds({"inference_substrate_class": "aggregation", "duration_s": 0.01}) == []
    assert _kinds({"inference_substrate_class": "hardware_board", "duration_s": 0.0}) == []
    # An absent duration draws no floor flag here; presence is a separate concern.
    assert _kinds({"inference_substrate_class": "model_full_generation"}) == []


def test_a_dict_shaped_declaration_is_malformed_and_read_as_missing() -> None:
    # SCENARIO-SUBSTRATE-CLASS-7: 169 corpus artifacts on 2026-09-05 wrote evidence into the
    # field; the gate used to judge the stringified dict like a name.
    d = {
        "inference_substrate": {"executes_models": False, "live_model_invoked": False},
        "duration_s": 0.3,
    }
    assert av._inference_substrate_text(d) == ""
    assert av._classify_inference_substrate(d)["source"] == "missing_top_level_inference_substrate"
    flags: list = []
    av.check_substrate_declaration_shape(d, flags)
    assert [(f.kind, f.severity) for f in flags] == [(MALFORMED, "warn")]
    assert "executes_models" in flags[0].detail
    # A principle wrapper still unwraps; a plain string draws no shape flag.
    assert (
        av._inference_substrate_text(
            {"inference_substrate": {"value": "x_no_llm", "principle": "p"}}
        )
        == "x_no_llm"
    )
    plain: list = []
    av.check_substrate_declaration_shape({"inference_substrate": "live_llm_inference"}, plain)
    assert plain == []


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_wired_into_the_full_verifier_and_a_wrapped_class_unwraps(tmp_path: Path) -> None:
    # SCENARIO-SUBSTRATE-CLASS-8
    base = {
        "experiment": 9990,
        "honest_verdict": "complete: ok",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.5,
    }
    bad = _write(
        tmp_path / "experiment_9990_bad_class.json", {**base, "inference_substrate_class": "banana"}
    )
    report = av.verify_artifact(bad)
    assert [(f["kind"], f["severity"]) for f in report["flags"] if f["kind"] == MISMATCH] == [
        (MISMATCH, "critical")
    ]

    wrapped = _write(
        tmp_path / "experiment_9991_wrapped_class.json",
        {
            **base,
            "experiment": 9991,
            "inference_substrate_class": {
                "value": "aggregation",
                "principle": "reads upstream JSON",
            },
        },
    )
    report = av.verify_artifact(wrapped)
    assert [f for f in report["flags"] if f["kind"] in {MISSING, MISMATCH, MALFORMED}] == []

    shaped = _write(
        tmp_path / "experiment_9992_dict_shaped.json",
        {
            **base,
            "experiment": 9992,
            "inference_substrate": {"kind": "aggregation", "executes_models": False},
        },
    )
    report = av.verify_artifact(shaped)
    assert [(f["kind"], f["severity"]) for f in report["flags"] if f["kind"] == MALFORMED] == [
        (MALFORMED, "warn")
    ]

    unknown = _write(
        tmp_path / "experiment_9993_unknown_name.json",
        {**base, "experiment": 9993, "inference_substrate": "brand_new_name_nobody_reviewed_v3"},
    )
    report = av.verify_artifact(unknown)
    assert [(f["kind"], f["severity"]) for f in report["flags"] if f["kind"] == MISSING] == [
        (MISSING, "warn")
    ]
