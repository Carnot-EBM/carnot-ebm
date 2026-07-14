"""Tests for the FR-11 shadow adapter in the verify/repair path.

Spec refs: REQ-LEARN-5640,
SCENARIO-LEARN-5640-EQUIVALENCE,
SCENARIO-LEARN-5640-SHADOW,
SCENARIO-LEARN-5640-REPLAY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import fr11_shadow_adapter as shadow_mod
from carnot.pipeline.fr11_shadow_adapter import (
    ACTIONS,
    ExactVerificationReceipt,
    FR11ShadowAdapter,
    ledger_lineage_complete,
    load_ledger,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


def _receipt(
    receipt_id: str,
    *,
    exact_valid: bool | None = True,
    action_set: tuple[str, ...] = ("retain", "smooth", "abstain"),
    delayed_label: bool = False,
    poison: bool = False,
    rollback_required: bool = False,
) -> ExactVerificationReceipt:
    return ExactVerificationReceipt(
        receipt_id=receipt_id,
        input_payload={"question": "What is 2 + 2?", "response": "2 + 2 = 4."},
        checkpoint_parent="sha256:parent",
        conformal_action_set=action_set,
        exact_valid=exact_valid,
        delayed_label=delayed_label,
        poison=poison,
        rollback_required=rollback_required,
    )


def test_scenario_learn_5640_disabled_adapter_leaves_verify_result_unchanged(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5640-EQUIVALENCE: disabled adapter writes nothing."""

    baseline = VerifyRepairPipeline().verify(
        question="What is 47 + 28?",
        response="The answer is 47 + 28 = 76.",
        domain="arithmetic",
    )
    candidate = VerifyRepairPipeline(
        fr11_shadow_adapter_enabled=False,
        fr11_shadow_ledger_path=tmp_path / "shadow.jsonl",
    ).verify(
        question="What is 47 + 28?",
        response="The answer is 47 + 28 = 76.",
        domain="arithmetic",
    )

    assert candidate.verified == baseline.verified
    assert candidate.energy == baseline.energy
    assert candidate.mode == baseline.mode
    assert candidate.skipped == baseline.skipped
    assert candidate.certificate == baseline.certificate
    assert [v.constraint_type for v in candidate.violations] == [
        v.constraint_type for v in baseline.violations
    ]
    assert not (tmp_path / "shadow.jsonl").exists()


def test_scenario_learn_5640_enabled_pipeline_appends_exact_gated_shadow_row(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5640-SHADOW: exact rejection forces shadow abstention."""

    ledger_path = tmp_path / "ledger.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    result = VerifyRepairPipeline(
        fr11_shadow_adapter_enabled=True,
        fr11_shadow_ledger_path=ledger_path,
        fr11_shadow_checkpoint_path=checkpoint_path,
    ).verify(
        question="What is 47 + 28?",
        response="The answer is 47 + 28 = 76.",
        domain="arithmetic",
    )

    assert result.verified is False
    shadow = result.certificate["fr11_shadow_adapter"]
    assert shadow["recommendation"] == "abstain"
    assert shadow["exact_disposition"] == "reject"
    assert shadow["rollback_reason"] == "exact_rejection_authoritative"
    assert shadow["conformal_action_set"] == ["abstain"]

    rows = load_ledger(ledger_path)
    assert len(rows) == 1
    row = rows[0]
    for field in (
        "input_hash",
        "checkpoint_parent",
        "conformal_action_set",
        "recommendation",
        "exact_disposition",
        "rollback_reason",
        "ledger_hash",
        "previous_ledger_hash",
    ):
        assert field in row
    assert row["recommendation"] == "abstain"
    assert row["unsafe_update_accepted"] is False
    assert row["checkpoint_hash"].startswith("sha256:")
    assert checkpoint_path.exists()
    assert ledger_lineage_complete(rows) is True


def test_req_learn_5640_adapter_recommends_only_frozen_actions_and_fails_closed(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5640: unsupported, duplicate, delayed, poison, rollback states abstain."""

    adapter = FR11ShadowAdapter(
        ledger_path=tmp_path / "ledger.jsonl",
        checkpoint_path=tmp_path / "checkpoint.json",
        enabled=True,
    )

    accepted = adapter.observe(_receipt("accepted", action_set=("adapt", "abstain")))
    duplicate = adapter.observe(_receipt("accepted", action_set=("adapt", "abstain")))
    delayed = adapter.observe(_receipt("delayed", delayed_label=True))
    poison = adapter.observe(_receipt("poison", poison=True))
    rollback = adapter.observe(_receipt("rollback", rollback_required=True))
    unsupported = adapter.observe(_receipt("unsupported", exact_valid=None))
    rejected = adapter.observe(_receipt("rejected", exact_valid=False))

    assert accepted.recommendation == "adapt"
    assert accepted.recommendation in ACTIONS
    for decision, reason in (
        (duplicate, "duplicate_delivery"),
        (delayed, "delayed_label_pending"),
        (poison, "poison_rejected"),
        (rollback, "rollback_required"),
        (unsupported, "unsupported_exact_state"),
        (rejected, "exact_rejection_authoritative"),
    ):
        assert decision.recommendation == "abstain"
        assert decision.rollback_reason == reason
        assert decision.unsafe_update_accepted is False

    rows = load_ledger(tmp_path / "ledger.jsonl")
    assert len(rows) == 7
    assert ledger_lineage_complete(rows) is True
    assert {row["recommendation"] for row in rows}.issubset(set(ACTIONS))
    assert sum(int(row["unsafe_update_accepted"]) for row in rows) == 0


def test_scenario_learn_5640_restart_replay_and_corrupt_checkpoint_recover(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5640-REPLAY: restart and corrupt checkpoints fail closed."""

    ledger_path = tmp_path / "ledger.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    first = FR11ShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    first.observe(_receipt("row-1", action_set=("retain", "abstain")))

    restarted = FR11ShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    second = restarted.observe(_receipt("row-2", action_set=("smooth", "abstain")))
    assert second.checkpoint_parent.startswith("sha256:")
    assert ledger_lineage_complete(load_ledger(ledger_path)) is True

    checkpoint_path.write_text("{corrupt", encoding="utf-8")
    recovered = FR11ShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    third = recovered.observe(_receipt("row-3", action_set=("reset", "abstain")))
    rows = load_ledger(ledger_path)

    assert third.corrupted_checkpoint is True
    assert third.recommendation == "abstain"
    assert third.rollback_reason == "corrupted_checkpoint_recovered"
    assert ledger_lineage_complete(rows) is True
    assert not list(tmp_path.glob("*.tmp"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["checkpoint_hash"].startswith("sha256:")


def test_req_learn_5640_ledger_and_checkpoint_parsers_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5640: malformed ledger and checkpoint state is rejected or recovered."""

    missing = tmp_path / "missing.jsonl"
    assert load_ledger(missing) == []

    valid_adapter = FR11ShadowAdapter(
        ledger_path=tmp_path / "valid.jsonl",
        checkpoint_path=tmp_path / "valid.checkpoint.json",
        enabled=True,
    )
    valid_adapter.observe(_receipt("valid"))
    valid_row = load_ledger(tmp_path / "valid.jsonl")[0]

    blank_then_valid = tmp_path / "blank.jsonl"
    blank_then_valid.write_text("\n" + shadow_mod.canonical_json(valid_row) + "\n", encoding="utf-8")
    assert len(load_ledger(blank_then_valid)) == 1

    non_object = tmp_path / "non_object.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        load_ledger(non_object)

    bad_hash = tmp_path / "bad_hash.jsonl"
    bad_hash.write_text(
        shadow_mod.canonical_json({**valid_row, "recommendation": "adapt"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_ledger(bad_hash)

    assert shadow_mod.ledger_lineage_complete([{**valid_row, "sequence": 99}]) is False
    assert (
        shadow_mod.ledger_lineage_complete(
            [{**valid_row, "previous_ledger_hash": shadow_mod.GENESIS_HASH.replace("0", "1", 1)}]
        )
        is False
    )
    assert shadow_mod.ledger_lineage_complete([{**valid_row, "ledger_hash": "sha256:bad"}]) is False

    bad_lineage = tmp_path / "bad_lineage.jsonl"
    lineage_row = {**valid_row, "sequence": 99}
    lineage_row["ledger_hash"] = shadow_mod._row_hash(lineage_row)  # noqa: SLF001
    bad_lineage.write_text(shadow_mod.canonical_json(lineage_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger lineage incomplete"):
        FR11ShadowAdapter(
            ledger_path=bad_lineage,
            checkpoint_path=tmp_path / "unused.checkpoint.json",
            enabled=True,
        )

    for payload, message in (
        ([], "checkpoint is not an object"),
        ({"schema": "wrong", "checkpoint_hash": ""}, "checkpoint schema"),
        ({"schema": shadow_mod.CHECKPOINT_SCHEMA, "checkpoint_hash": "sha256:bad"}, "hash"),
    ):
        checkpoint_path = tmp_path / f"{message.split()[0]}.json"
        checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message.split()[0]):
            shadow_mod._validate_checkpoint(checkpoint_path)  # noqa: SLF001

    checkpoint = json.loads((tmp_path / "valid.checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["ledger_tail_hash"] = shadow_mod.GENESIS_HASH
    checkpoint["checkpoint_hash"] = shadow_mod._checkpoint_hash(checkpoint)  # noqa: SLF001
    mismatch_path = tmp_path / "mismatch.checkpoint.json"
    mismatch_path.write_text(shadow_mod.canonical_json(checkpoint), encoding="utf-8")
    recovered = FR11ShadowAdapter(
        ledger_path=tmp_path / "valid.jsonl",
        checkpoint_path=mismatch_path,
        enabled=True,
    )
    decision = recovered.observe(_receipt("after-mismatch"))
    assert decision is not None
    assert decision.rollback_reason == "corrupted_checkpoint_recovered"


def test_req_learn_5640_private_recommendation_edges(tmp_path: Path) -> None:
    """REQ-LEARN-5640: empty/abstain-only action sets remain fail closed."""

    adapter = FR11ShadowAdapter(
        ledger_path=tmp_path / "ledger.jsonl",
        checkpoint_path=tmp_path / "checkpoint.json",
        enabled=True,
    )
    abstain = adapter.observe(_receipt("abstain-only", action_set=("abstain",)))

    assert abstain is not None
    assert abstain.recommendation == "abstain"
    assert abstain.rollback_reason == "conformal_action_set_abstained"
    assert shadow_mod.FR11ShadowAdapter._choose_recommendation(("invalid",)) == "abstain"  # noqa: SLF001
