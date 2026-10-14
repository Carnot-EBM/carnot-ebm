"""REQ-REPORT-8509, REQ-VERIFY-8509: missing effects cannot become failed treatments."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_redirect_effect_8509 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_original_seals_and_mode_joins(tmp_path):
    """SCENARIO-REPORT-8509: retain all rows without inventing application operands."""
    source = e.authenticate(e.ROOT, tmp_path)
    result = e.reduce(source)
    assert len(result["rows"]) == 32
    assert len(result["redirect_effect_rows"]) == len(result["shadow_rows"]) == 16
    assert all(r["application_disposition"] is None for r in result["redirect_effect_rows"])
    assert all(
        r["effect_class"] == "application_unobserved" for r in result["redirect_effect_rows"]
    )
    assert all(r["effect_class"] == "shadow_recommendation" for r in result["shadow_rows"])
    assert all(r["snapshot"]["induced"] for r in result["rows"] if r["arm"] == "allow_reinduction")
    assert all(r["subsequent_action_trace"] for r in result["rows"])
    assert result["censoring_counts"] == {"applied": 16, "shadow": 16}
    assert not result["arm_refinement_authorized"]
    assert all(not r["support_gate_passed"] for r in result["leave_one_game_out"])
    assert all(r["resolved_count"] == 0 for r in result["effect_support_by_arm"])
    assert all(
        r["induction_tier_reentry_evidence"]["later_attempt_count"] == 2
        for r in result["redirect_effect_rows"]
        if r["arm"] == "allow_reinduction"
    )
    assert all(
        r["induction_tier_reentry_evidence"]["later_attempt_count"] == 1
        for r in result["shadow_rows"]
        if r["arm"] == "allow_reinduction"
    )
    broken = deepcopy(source)
    broken["operands"][0]["steps"]["expected_sha256"] = "bad"
    with pytest.raises(ValueError, match="primitive_pin"):
        e.reduce(broken)
    path = tmp_path / "bad_steps.jsonl"
    event = json.loads(
        Path(source["operands"][0]["steps"]["snapshot_path"]).read_text().splitlines()[0]
    )
    event["supervisor_events"] = [
        dict(
            snapshot=dict(
                level=0,
                induced=False,
                induction_attempts=0,
                goal_bias_installed=False,
                diversity_active=False,
                new_transitions_since_induction=0,
            ),
            redirect={"arm": "invented"},
        )
    ]
    path.write_text(json.dumps(event) + "\n")
    from carnot.reporting.current_work_receipt import sha256_file

    broken["operands"][0]["steps"].update(
        snapshot_path=str(path), expected_sha256=sha256_file(path)
    )
    with pytest.raises(ValueError, match="redirect_drift"):
        e.reduce(broken)


def test_missing_and_changed_producer_rejected(tmp_path):
    """REQ-VERIFY-8509: independent pins reject changed primaries before reduction."""
    with pytest.raises(OSError):
        e.authenticate(tmp_path, tmp_path / "raw")
    target = tmp_path / e.UPSTREAM
    atomic_json(target, {"experiment_id": 8496})
    with pytest.raises(ValueError, match="upstream_pin"):
        e.authenticate(tmp_path, tmp_path / "raw")


@pytest.mark.parametrize("mode", ["applied", "shadow"])
def test_real_policy_mechanisms(tmp_path, mode):
    """SCENARIO-REPORT-8509: real mutations and model-disabled tier differ from generation."""
    rows = e.fixtures(tmp_path / mode, mode)
    by_name = {r["mechanism"]: r for r in rows}
    assert all(r["fixture_is_live_evidence"] is False for r in rows)
    assert all(r["llm_calls"] == 0 for r in rows)
    if mode == "shadow":
        assert all(r["old_state_hash"] == r["new_state_hash"] for r in rows)
        assert all(r["effect_class"] == "shadow_recommendation" for r in rows)
    else:
        assert by_name["allow_true_latch"]["effect_class"] == "applied_state_change"
        assert by_name["allow_false_latch"]["effect_class"] == "selected_but_no_state_change"
        assert by_name["application_exception"]["effect_class"] == "application_exception"
        assert by_name["force_diversity"]["effect_class"] == "applied_state_change"
        assert by_name["allow_true_latch"]["induction_tier_entered"]
        assert by_name["allow_true_latch"]["induction_attempt"]["skipped"] == "disabled_by_env"


def test_effect_classifier_missing_and_exception():
    """REQ-REPORT-8509: classify only state hashes and dispositions actually present."""
    assert e.effect("shadow", {}) == "shadow_recommendation"
    assert e.effect("applied", {}) == "application_unobserved"
    assert (
        e.effect("applied", {"application_disposition": "exception:ValueError"})
        == "application_exception"
    )
    assert (
        e.effect("applied", {"old_state_hash": "a", "new_state_hash": "a"})
        == "selected_but_no_state_change"
    )
    assert (
        e.effect("applied", {"old_state_hash": "a", "new_state_hash": "b"})
        == "applied_state_change"
    )


def test_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-VERIFY-8509: a self-rehashed work manifest cannot replace upstream receipts."""
    source = e.authenticate(e.ROOT, tmp_path / "inputs")
    work = dict(
        source=source,
        fixture_rows=[],
        preconditions_checked={},
        failures=[],
        duration_s=1.0,
        phase_spans=[],
    )
    work_path = tmp_path / "work.json"
    atomic_json(work_path, work)
    value = e.build(work_path, [], tmp_path / (e.NAME + ".json"))
    assert e.replay(value)
    changed = deepcopy(value)
    changed["rows"][0]["snapshot"]["induced"] = False
    changed["reproducibility_checksum"] = canonical_hash(changed["rows"])
    assert not e.replay(changed)
    changed = deepcopy(value)
    changed["work_reference"]["path"] = str(tmp_path / "missing")
    assert not e.replay(changed)
    work["source"]["primary"]["rows"][0]["supervisor_outcome_rows"][0]["resolved_by_levelup"] = True
    atomic_json(work_path, work)
    assert not e.replay(e.build(work_path, [], tmp_path / (e.NAME + ".json")))
    work["source"] = source
    # Original source objects are copied by JSON publication; refresh the sealed
    # source to distinguish each rejection from the aggregate mutation above.
    work["source"]["primary"] = json.loads(
        Path(source["primary_reference"]["snapshot_path"]).read_bytes()
    )
    original = deepcopy(work)
    for mechanism in ["work_hash", "primitive_seal", "owned_code", "fixture", "log"]:
        work = deepcopy(original)
        receipts = []
        if mechanism == "primitive_seal":
            work["source"]["operands"][0]["terminal"]["expected_sha256"] = "bad"
        elif mechanism == "owned_code":
            work["code_config_hashes"] = {str(e.ROOT / e.CLI): "bad"}
        elif mechanism == "fixture":
            work["fixture_rows"] = [dict(mechanism="invented")]
        elif mechanism == "log":
            log = tmp_path / "log.txt"
            log.write_text("changed")
            receipts = [dict(passed=True, stdout_path=str(log), stdout_sha256="bad")]
        atomic_json(work_path, work)
        value = e.build(work_path, receipts, tmp_path / (e.NAME + ".json"))
        if mechanism == "work_hash":
            value["work_reference"]["sha256"] = "bad"
        assert not e.replay(value), mechanism


def test_authentication_rejection_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8509: qualification, original work, replay and date each fail independently."""
    original = json.loads((e.ROOT / e.UPSTREAM).read_bytes())
    target = tmp_path / e.UPSTREAM
    from carnot.reporting.current_work_receipt import sha256_file

    for mechanism in ["validation", "work", "replay", "identity"]:
        value = deepcopy(original)
        if mechanism == "work":
            value["work_reference"]["sha256"] = "bad"
        if mechanism == "identity":
            value["run_date"] = "20261014"
        atomic_json(target, value)
        with monkeypatch.context() as patcher:
            patcher.setattr(e, "UPSTREAM_PIN", sha256_file(target))
            patcher.setattr(
                e,
                "read_bound_sidecar",
                lambda *_: dict(report=dict(passed=mechanism != "validation")),
            )
            patcher.setattr(e.upstream, "replay", lambda *_: mechanism != "replay")
            with pytest.raises(
                ValueError,
                match={
                    "validation": "upstream_validation",
                    "work": "upstream_work_pin",
                    "replay": "original_sealed_replay",
                    "identity": "upstream_identity",
                }[mechanism],
            ):
                e.authenticate(tmp_path, tmp_path / "raw")


def test_fixture_model_tripwire_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8509: an attempted model seam invalidates a private mechanism fixture."""
    from contextlib import contextmanager
    from carnot.reporting import arc_redirect_fixtures_8509 as f

    @contextmanager
    def attempted():
        yield ["forbidden_model_seam"]

    monkeypatch.setattr(f, "no_models", attempted)
    with pytest.raises(ValueError, match="fixture_model_tripwire"):
        f.fixtures(tmp_path, "shadow")


def test_support_floor_and_no_shadow_treatment():
    """REQ-REPORT-8509: support counts never mix shadow controls with applied outcomes."""
    rows = [
        dict(game=g, mode=m, arm="a", resolved=True, right_censored=False)
        for g in ["g1", "g2", "g3", "g4"]
        for m in ["applied", "shadow"]
        for _ in range(5)
    ]
    support = e.support(rows)
    assert all(r["support_gate_passed"] for r in support["leave_one_game_out"])
    assert support["arm_refinement_authorized"]
    shadow_unresolved = deepcopy(rows)
    for row in shadow_unresolved:
        if row["mode"] == "shadow":
            row.update(resolved=False, right_censored=True)
    assert not e.support(shadow_unresolved)["arm_refinement_authorized"]
    disjoint = deepcopy(rows)
    for row in disjoint:
        if row["mode"] == "shadow":
            row["game"] += "_control"
    assert not e.support(disjoint)["arm_refinement_authorized"]
    for row in rows:
        if row["mode"] == "applied":
            row["resolved"] = False
            row["right_censored"] = True
    assert not e.support(rows)["arm_refinement_authorized"]
