"""Regression tests for the no-LLM substrate name rule (the exp6593 incident).

Exp6593 declared `immutable_qwen_gemma_cfr_row_reducer_no_llm`, a value outside the
six-value taxonomy. The linter held it to the 60s live-model floor and the fabrication
gate quarantined an honest 1.16s replay. The band-aid was to append that one string to
NO_LLM_SUBSTRATE_ALIASES; 41 of that tuple's 61 entries were already the same shape, so
the list was enumerating a concept instead of stating it. These tests pin the RULE, not
the alias, so deleting the alias entry keeps exp6593 clean while a live-model
declaration still gets the live floor.

Spec refs: REQ-VERIFY-6593, SCENARIO-VERIFY-6593-NAME-RULE,
SCENARIO-VERIFY-6593-LIVE-CONTROL, SCENARIO-VERIFY-6593-FLOOR-STILL-BITES,
SCENARIO-VERIFY-6593-VISIBLE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av

# The incident input, verbatim from
# results/experiment_6593_cfr_independent_row_reducer.json.
EXP6593_SUBSTRATE = "immutable_qwen_gemma_cfr_row_reducer_no_llm"
EXP6593_DURATION_S = 1.1603620913811028


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _kinds(tmp_path: Path, payload: dict[str, Any]) -> set[str]:
    return {f["kind"] for f in av.verify_artifact(_write(tmp_path, payload))["flags"]}


def _critical_kinds(tmp_path: Path, payload: dict[str, Any]) -> set[str]:
    return {
        f["kind"]
        for f in av.verify_artifact(_write(tmp_path, payload))["flags"]
        if str(f.get("severity", "")).lower() == "critical"
    }


def _exp6593_like(**overrides: Any) -> dict[str, Any]:
    """The incident shape: a no-LLM row reducer quoting upstream GPU receipts."""
    payload: dict[str, Any] = {
        "experiment": 6593,
        "honest_verdict": (
            "complete: Qwen, Gemma, and pooled CFR effects replayed; every "
            "exact-success delta is 0.0 with zero direct headroom"
        ),
        "inference_substrate": EXP6593_SUBSTRATE,
        "duration_s": EXP6593_DURATION_S,
        "reproducibility_checksum": "sha256:" + "e" * 64,
        # Vestigial: the upstream receipt this artifact REPLAYS names a live server.
        # The marker must be a real COMPUTE_BOUND_MARKERS string (they are
        # case-sensitive), or the floor is never reached and the test proves nothing.
        "model_identity_replay_rows": [
            {
                "gpu_process_receipts": {
                    "process": {
                        "command": [
                            "llama-server",
                            "--model",
                            "unsloth/Qwen3.6-35B-A3B-GGUF/model.gguf",
                        ]
                    }
                }
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_req_verify_6593_incident_substrate_and_duration_are_not_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6593-NAME-RULE: the exact incident input stays unquarantined."""

    assert "DURATION_TOO_SHORT" not in _critical_kinds(tmp_path, _exp6593_like())


def test_req_verify_6593_rule_holds_without_the_allowlist_entry() -> None:
    """SCENARIO-VERIFY-6593-NAME-RULE: recognition comes from the name, not the alias.

    This is the point of the change, so the alias MUST be gone. exp6593's own agent added
    it to the guard in exp6593's own commit; while it stayed, every test below passed with
    the rule deleted and this file proved nothing.
    """

    assert EXP6593_SUBSTRATE not in av.NO_LLM_SUBSTRATE_ALIASES, (
        "the self-served allowlist entry must not come back: with it present the name "
        "rule is untested and the guard admits the artifact that needed admitting"
    )
    assert av._declares_no_llm_by_name(EXP6593_SUBSTRATE) is True
    classification = av._classify_inference_substrate({"inference_substrate": EXP6593_SUBSTRATE})
    assert classification["kind"] == av.SUBSTRATE_KIND_NO_LLM
    assert classification["source"] == "no_llm_name_suffix"


def test_req_verify_6593_name_rule_covers_the_concept_the_alias_list_enumerated() -> None:
    """SCENARIO-VERIFY-6593-NAME-RULE: every no-LLM-named alias matches the rule."""

    by_name = [
        alias
        for alias in av.NO_LLM_SUBSTRATE_ALIASES
        if alias.endswith(("_no_llm", "_no_new_llm", "_no_experiment_llm"))
    ]
    assert len(by_name) >= 40, "expected the alias list to be dominated by this shape"
    assert all(av._declares_no_llm_by_name(alias) for alias in by_name)


def test_req_verify_6593_trailing_principle_note_does_not_defeat_the_rule() -> None:
    """SCENARIO-VERIFY-6593-NAME-RULE: a human note after the value must not hide it."""

    for raw in (
        f"{EXP6593_SUBSTRATE} -- reads upstream rows, no model load (100us floor).",
        f"{EXP6593_SUBSTRATE}; 100us floor.",
        f"{EXP6593_SUBSTRATE}: replay only",
    ):
        assert av._declares_no_llm_by_name(raw) is True, raw


def test_req_verify_6593_principle_wrapped_substrate_is_unwrapped(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6593-NAME-RULE: the field may arrive principle-annotated."""

    payload = _exp6593_like(
        inference_substrate={
            "principle": "No model is loaded; this reducer replays frozen upstream rows.",
            "value": EXP6593_SUBSTRATE,
        }
    )
    assert "DURATION_TOO_SHORT" not in _critical_kinds(tmp_path, payload)


def test_req_verify_6593_live_llm_declaration_still_gets_the_live_floor(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6593-LIVE-CONTROL: the rule cannot pull a live claim off 60s."""

    payload = _exp6593_like(
        inference_substrate="live_llm_inference",
        model_specs={"gguf_path": "/tmp/model.gguf"},
        random_seed=1,
    )
    assert "DURATION_TOO_SHORT" in _critical_kinds(tmp_path, payload)


def test_req_verify_6593_allowlist_wins_over_the_name_rule() -> None:
    """SCENARIO-VERIFY-6593-LIVE-CONTROL: allowlists are consulted before the name rule.

    Exercised against a REAL conflicting value. No live-model alias currently ends in the
    no-LLM shape, so the live case cannot express the ordering; five AGGREGATION aliases
    do, and those must classify as aggregation rather than being captured by the fallback.
    """

    conflicting = [
        alias for alias in av.AGGREGATION_SUBSTRATE_ALIASES if av._declares_no_llm_by_name(alias)
    ]
    assert conflicting, "expected aggregation aliases that also match the name shape"
    for alias in conflicting:
        result = av._classify_inference_substrate({"inference_substrate": alias})
        assert result["kind"] == av.SUBSTRATE_KIND_AGGREGATION, alias
        assert result["source"] == "top_level_inference_substrate", alias


def test_req_verify_6593_live_substrate_classifies_as_live_model() -> None:
    """SCENARIO-VERIFY-6593-LIVE-CONTROL: the strict live value is unaffected."""

    live_first = av._classify_inference_substrate({"inference_substrate": av.LIVE_LLM_SUBSTRATE})
    assert live_first["kind"] == av.SUBSTRATE_KIND_LIVE_MODEL


def test_req_verify_6593_absent_duration_still_gets_the_visibility_warning(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6593-VISIBLE: the warning must not sit behind the duration guard.

    A no-LLM claim nobody reviewed, with NO measured duration at all, is the case most
    worth surfacing. 50 corpus artifacts sat in that blind spot when the warning was
    emitted after the early return.
    """

    payload = {
        "experiment": 6593,
        "honest_verdict": "complete: no model was loaded",
        "inference_substrate": "brand_new_unreviewed_reducer_no_llm",
    }
    assert "duration_s" not in payload
    assert "SUBSTRATE_NO_LLM_BY_NAME" in _kinds(tmp_path, payload)


def test_req_verify_6593_note_separators_agree_with_the_sibling_matcher() -> None:
    """SCENARIO-VERIFY-6593-NAME-RULE: both note strippers accept the same boundaries.

    They split the same string for the same reason, so a separator one accepts and the
    other rejects is a latent disagreement, not a design choice.
    """

    for sep in (" ", ";", ",", ":", "."):
        assert av._declares_no_llm_by_name(f"foo_no_llm{sep} 100us floor.") is True, sep
        assert av._inference_substrate_value_matches(f"simulation{sep} note", "simulation")


def test_req_verify_6593_malformed_substrate_does_not_crash_the_linter() -> None:
    """SCENARIO-VERIFY-6593-VISIBLE: a bad declaration is unrecognized, not an exception."""

    for bad in (None, 7, ["a"], {"k": "v"}, b"bytes", True):
        assert av._declares_no_llm_by_name(bad) is False
        assert av._substrate_leading_token(bad) == ""


def test_req_verify_6593_floor_still_bites_below_the_no_llm_minimum(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6593-FLOOR-STILL-BITES: recognition is not exemption.

    exp6288 declares `deterministic_partial_atom_text_adapter_no_llm` at 3.7e-07s and
    carries a vestigial GGUF marker. The name rule gives it a floor rather than no
    floor, so it MUST stay flagged. Verified against the real artifact, not only here.
    """

    payload = _exp6593_like(
        inference_substrate="deterministic_partial_atom_text_adapter_no_llm",
        duration_s=3.708992153406143e-07,
    )
    assert "DURATION_TOO_SHORT" in _critical_kinds(tmp_path, payload)


def test_req_verify_6593_marker_free_no_llm_artifact_reaches_no_floor_check(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6593-FLOOR-STILL-BITES: records a measured PRE-EXISTING gap.

    `check_duration_vs_claim` returns before applying any substrate floor when the
    artifact carries no compute-bound marker. That predates this change and affects
    allowlisted substrates identically -- `simulation` at duration 0.0 is also
    unflagged. Closing it would newly quarantine 102 historical artifacts, so it is
    recorded here and referred to the operator rather than changed silently.
    This test pins the CURRENT behaviour so the day someone closes the gap, it fails
    loudly and they must confront the 102 rather than discover them later.
    """

    marker_free = {
        "inference_substrate": "deterministic_partial_atom_text_adapter_no_llm",
        "honest_verdict": "complete: no model was loaded",
        "duration_s": 0.0,
    }
    assert av._has_compute_bound_marker(marker_free) is False
    floor = av.duration_floor_for_artifact(marker_free)
    assert floor is not None and floor["min_duration_s"] == 0.0001
    assert "DURATION_TOO_SHORT" not in _critical_kinds(tmp_path, marker_free)

    # The same hole on an allowlisted substrate: this is not caused by the name rule.
    allowlisted = dict(marker_free, inference_substrate="simulation")
    assert "DURATION_TOO_SHORT" not in _critical_kinds(tmp_path, allowlisted)


def test_req_verify_6593_unrecognized_substrate_without_the_shape_stays_unfloored(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6593-VISIBLE: the rule does not silence unknown declarations.

    `ising_energy_optimization_cpu` is a real corpus value with no no-LLM name shape.
    No marker here, so it must still reach the unrecognised-substrate warning rather
    than quietly acquiring the no-LLM floor.
    """

    payload = {
        "experiment": 6593,
        "honest_verdict": "complete: cpu optimization",
        "inference_substrate": "ising_energy_optimization_cpu",
        "duration_s": EXP6593_DURATION_S,
    }
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" in _kinds(tmp_path, payload)


def test_req_verify_6593_name_recognition_is_reported_not_silent(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6593-VISIBLE: a self-declared match stays auditable.

    Nobody reviewed a name-shape match the way somebody reviewed each alias entry, so it
    must warn rather than pass quietly.
    """

    payload = _exp6593_like(inference_substrate="brand_new_unreviewed_reducer_no_llm")
    kinds = _kinds(tmp_path, payload)
    assert "SUBSTRATE_NO_LLM_BY_NAME" in kinds
    assert "DURATION_TOO_SHORT" not in _critical_kinds(tmp_path, payload)


def test_req_verify_6593_versioned_suffix_does_not_match(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6593-VISIBLE: the rule is anchored, so `_no_llm_v2` is unknown."""

    assert av._declares_no_llm_by_name("some_reducer_no_llm_v2") is False
    assert av._declares_no_llm_by_name("no_llm_but_actually_live") is False


def test_req_verify_6593_floor_descriptor_names_the_recognition_source() -> None:
    """SCENARIO-VERIFY-6593-VISIBLE: readers can tell an alias match from a name match."""

    by_name = av.duration_floor_for_artifact(
        {"inference_substrate": "brand_new_unreviewed_reducer_no_llm", "duration_s": 1.0}
    )
    assert by_name is not None
    assert by_name["reason"] == "no_llm_declared_by_name"

    allowlisted = av.duration_floor_for_artifact(
        {"inference_substrate": "simulation", "duration_s": 1.0}
    )
    assert allowlisted is not None
    assert allowlisted["reason"] == "no_llm_declared"
