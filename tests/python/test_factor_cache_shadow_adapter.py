"""Tests for the FR-11 factor-cache shadow adapter.

Spec refs: REQ-PIPELINE-6479, SCENARIO-PIPELINE-6479-SHADOW,
REQ-LEARN-6479, SCENARIO-LEARN-6479-EXACT-ADMIT,
SCENARIO-LEARN-6479-RESTART.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot.pipeline import factor_cache_shadow_adapter as shadow_mod
from carnot.pipeline.factor_cache_shadow_adapter import (
    ADAPTER_API_VERSION,
    GENESIS_HASH,
    FactorCacheEventReceipt,
    FR11FactorCacheShadowAdapter,
    adapter_api_schema_hash,
    load_ledger,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


def _receipt(
    event_id: str,
    *,
    raw_hash: str = "sha256:" + "1" * 64,
    raw_unit_binding: str = "unit-a",
    unit_binding: str = "unit-a",
    exact_outcome: str = "pass",
    checker_outcome: str | None = None,
    checker_ran_before_write: bool = True,
    checker_authority_passed: bool = True,
    self_signed: bool = False,
    chronology_index: int = 0,
    cache_parent_hash: str = "sha256:" + "0" * 64,
) -> FactorCacheEventReceipt:
    """Build one deterministic receipt for exact-admission tests."""

    return FactorCacheEventReceipt(
        event_id=event_id,
        raw_hash=raw_hash,
        unit_binding=unit_binding,
        raw_unit_binding=raw_unit_binding,
        checker_hash="sha256:" + "2" * 64,
        exact_outcome=exact_outcome,
        checker_receipt={
            "exact_outcome": checker_outcome if checker_outcome is not None else exact_outcome,
            "checker_ran_before_write": checker_ran_before_write,
            "checker_authority_passed": checker_authority_passed,
        },
        chronology_index=chronology_index,
        factor_id="arithmetic:verified_binding",
        model_confidence=0.8,
        selected_features=("verified_binding",),
        cache_parent_hash=cache_parent_hash,
        self_signed=self_signed,
    )


def _public_result(result: object) -> dict[str, object]:
    cert = {
        key: value
        for key, value in result.certificate.items()
        if key != "fr11_factor_cache_shadow_adapter"
    }
    return {
        "verified": result.verified,
        "energy": result.energy,
        "violations": [violation.constraint_type for violation in result.violations],
        "mode": result.mode,
        "skipped": result.skipped,
        "certificate": cert,
    }


def test_req_learn_6479_versioned_interface_and_disabled_noop(tmp_path: Path) -> None:
    """REQ-LEARN-6479: adapter API is versioned and disabled writes nothing."""

    adapter = FR11FactorCacheShadowAdapter(
        ledger_path=tmp_path / "ledger.jsonl",
        checkpoint_path=tmp_path / "checkpoint.json",
        enabled=False,
    )

    for method in (
        "observe",
        "exact_admit",
        "propose_rank",
        "tombstone",
        "rollback",
        "save",
        "load",
        "close",
    ):
        assert callable(getattr(adapter, method))

    assert ADAPTER_API_VERSION == "carnot.fr11.factor_cache_shadow_adapter.v1"
    assert adapter_api_schema_hash().startswith("sha256:")
    assert adapter.observe(_receipt("disabled")) is None
    adapter.save()
    adapter.close()
    assert not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "checkpoint.json").exists()


def test_scenario_learn_6479_exact_admit_rejects_identity_and_checker_attacks(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6479-EXACT-ADMIT: exact receipts own cache writes."""

    adapter = FR11FactorCacheShadowAdapter(
        ledger_path=tmp_path / "ledger.jsonl",
        checkpoint_path=tmp_path / "checkpoint.json",
        enabled=True,
    )

    admitted = adapter.observe(_receipt("event-1", exact_outcome="fail"))
    assert admitted is not None
    assert admitted.exact_admission["admitted"] is True
    assert admitted.cache_write["write_admitted"] is True
    assert admitted.cache_write["update_sign"] == -1
    assert admitted.shadow_rank["recommendation"] == "rank"

    attacks = [
        _receipt("event-1", raw_hash="sha256:" + "3" * 64),
        _receipt("event-2", raw_hash="sha256:" + "1" * 64),
        _receipt("event-3", raw_hash="sha256:" + "4" * 64, raw_unit_binding="unit-b"),
        _receipt(
            "event-4",
            raw_hash="sha256:" + "5" * 64,
            exact_outcome="pass",
            checker_outcome="fail",
        ),
        _receipt("event-5", raw_hash="sha256:" + "6" * 64, checker_ran_before_write=False),
        _receipt("event-6", raw_hash="sha256:" + "7" * 64, checker_authority_passed=False),
        _receipt("event-7", raw_hash="sha256:" + "8" * 64, self_signed=True),
    ]
    reasons = []
    for attack in attacks:
        decision = adapter.observe(attack)
        assert decision is not None
        assert decision.exact_admission["admitted"] is False
        assert decision.cache_write["write_admitted"] is False
        reasons.append(decision.exact_admission["reject_reason"])

    assert reasons == [
        "duplicate_event_id",
        "duplicate_raw_hash",
        "wrong_unit_binding",
        "forged_exact_outcome",
        "write_before_check",
        "checker_authority_failed",
        "self_signed_receipt",
    ]
    assert adapter.state_summary()["admitted_write_count"] == 1
    assert adapter.state_summary()["quarantine_count"] == len(attacks)
    assert len(load_ledger(tmp_path / "ledger.jsonl")) == 8


def test_scenario_learn_6479_restart_tombstone_and_rollback_non_resurrection(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6479-RESTART: tombstoned events stay inactive after load."""

    ledger_path = tmp_path / "ledger.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    adapter = FR11FactorCacheShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    decision = adapter.observe(_receipt("event-1", exact_outcome="pass"))
    assert decision is not None
    admitted_hash = decision.cache_write["post_cache_hash"]

    tombstone = adapter.tombstone("event-1", reason="stale_cache")
    rollback = adapter.rollback(target_cache_hash=decision.cache_write["pre_cache_hash"], reason="stale_cache")
    adapter.close()

    restored = FR11FactorCacheShadowAdapter.load(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    summary = restored.state_summary()
    assert tombstone["tombstone_hash"] in summary["tombstone_hashes"]
    assert rollback["rollback_hash"] in summary["rollback_hashes"]
    assert admitted_hash not in summary["active_cache_hashes"]
    assert "event-1" in summary["tombstoned_event_ids"]

    replay = restored.observe(_receipt("event-1", raw_hash="sha256:" + "9" * 64, chronology_index=1))
    assert replay is not None
    assert replay.exact_admission["admitted"] is False
    assert replay.exact_admission["reject_reason"] == "tombstoned_event"


def test_req_pipeline_6479_default_off_shadow_mode_and_env_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PIPELINE-6479 and SCENARIO-PIPELINE-6479-SHADOW: pipeline output is stable."""

    kwargs = {
        "question": "What is 47 + 28?",
        "response": "The answer is 47 + 28 = 76.",
        "domain": "arithmetic",
    }
    baseline = VerifyRepairPipeline(and_compose_verifier=False).verify(**kwargs)
    disabled = VerifyRepairPipeline(
        and_compose_verifier=False,
        fr11_factor_cache_shadow_adapter_enabled=False,
        fr11_factor_cache_shadow_ledger_path=tmp_path / "disabled.jsonl",
    ).verify(**kwargs)
    assert _public_result(disabled) == _public_result(baseline)
    assert disabled.certificate == baseline.certificate
    assert not (tmp_path / "disabled.jsonl").exists()

    monkeypatch.setenv("CARNOT_FR11_FACTOR_CACHE_SHADOW_ADAPTER", "1")
    env_only = VerifyRepairPipeline(
        and_compose_verifier=False,
        fr11_factor_cache_shadow_ledger_path=tmp_path / "env.jsonl",
    ).verify(**kwargs)
    assert _public_result(env_only) == _public_result(baseline)
    assert "fr11_factor_cache_shadow_adapter" not in env_only.certificate
    assert not (tmp_path / "env.jsonl").exists()

    enabled = VerifyRepairPipeline(
        and_compose_verifier=False,
        fr11_factor_cache_shadow_adapter_enabled=True,
        fr11_factor_cache_shadow_ledger_path=tmp_path / "enabled.jsonl",
        fr11_factor_cache_shadow_checkpoint_path=tmp_path / "enabled.checkpoint.json",
    ).verify(**kwargs)
    assert _public_result(enabled) == _public_result(baseline)
    shadow = enabled.certificate["fr11_factor_cache_shadow_adapter"]
    assert shadow["mode"] == "shadow"
    assert shadow["release_authority"] == "exact_verifier"
    assert shadow["exact_admission"]["admitted"] is True
    assert shadow["cache_write"]["write_admitted"] is True
    assert Path(tmp_path / "enabled.jsonl").exists()


def test_req_learn_6479_ledger_checkpoint_and_private_edges(tmp_path: Path) -> None:
    """REQ-LEARN-6479: parsers, restore, stale, and abstain branches fail closed."""

    ledger_path = tmp_path / "ledger.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    adapter = FR11FactorCacheShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    decision = adapter.observe(_receipt("edge-1", raw_hash="sha256:" + "a" * 64))
    assert decision is not None

    blank_then_valid = tmp_path / "blank.jsonl"
    blank_then_valid.write_text("\n" + shadow_mod.canonical_json(load_ledger(ledger_path)[0]) + "\n")
    assert len(load_ledger(blank_then_valid)) == 1

    malformed_cases = [
        ("non-object", "[]\n", "not an object"),
        (
            "schema",
            shadow_mod.canonical_json({**load_ledger(ledger_path)[0], "schema": "wrong"}) + "\n",
            "schema",
        ),
        (
            "sequence",
            shadow_mod.canonical_json({**load_ledger(ledger_path)[0], "sequence": 9}) + "\n",
            "sequence",
        ),
        (
            "lineage",
            shadow_mod.canonical_json(
                {**load_ledger(ledger_path)[0], "previous_row_hash": "sha256:" + "b" * 64}
            )
            + "\n",
            "lineage",
        ),
        (
            "hash",
            shadow_mod.canonical_json(
                {**load_ledger(ledger_path)[0], "row_hash": "sha256:" + "c" * 64}
            )
            + "\n",
            "hash",
        ),
    ]
    for name, payload, message in malformed_cases:
        path = tmp_path / f"{name}.jsonl"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_ledger(path)

    assert adapter.propose_rank(
        _receipt(
            "unsupported-rank",
            raw_hash="sha256:" + "d" * 64,
            exact_outcome="unsupported",
            checker_outcome="unsupported",
        )
    )["recommendation"] == "abstain"

    missing = adapter.exact_admit(_receipt("", raw_hash="sha256:" + "e" * 64))
    bad_hash = adapter.exact_admit(_receipt("bad-hash", raw_hash="bad"))
    unsupported = adapter.exact_admit(
        _receipt(
            "unsupported",
            raw_hash="sha256:" + "f" * 64,
            exact_outcome="unsupported",
            checker_outcome="unsupported",
        )
    )
    stale = adapter.exact_admit(
        _receipt(
            "stale",
            raw_hash="sha256:" + "9" * 64,
            chronology_index=1,
            cache_parent_hash=GENESIS_HASH,
        )
    )
    assert missing["reject_reason"] == "missing_required_identity"
    assert bad_hash["reject_reason"] == "bad_hash_format"
    assert unsupported["reject_reason"] == "unsupported_exact_outcome"
    assert stale["reject_reason"] == "stale_cache_parent"

    active_rollback = adapter.rollback(target_cache_hash=decision.cache_write["post_cache_hash"], reason="edge")
    assert active_rollback["state_hash_after"] == decision.cache_write["post_cache_hash"]
    adapter.tombstone("edge-1", reason="edge")
    adapter.rollback(target_cache_hash=GENESIS_HASH, reason="edge")
    checkpoint_path.unlink()
    replayed = FR11FactorCacheShadowAdapter(
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        enabled=True,
    )
    assert replayed.state_summary()["rollback_count"] == 2

    valid_checkpoint = tmp_path / "valid.checkpoint.json"
    checkpoint = {
        "schema": shadow_mod.CHECKPOINT_SCHEMA,
        "adapter_api_version": ADAPTER_API_VERSION,
        "adapter_api_schema_hash": adapter_api_schema_hash(),
        "sequence": 0,
        "ledger_tail_hash": GENESIS_HASH,
        "state_hash": GENESIS_HASH,
        "cache": {},
        "tombstones": {},
        "quarantine": [],
        "rollbacks": [],
        "seen_event_ids": [],
        "seen_raw_hashes": [],
        "event_cache_hashes": {},
        "active_cache_hashes": [],
        "last_chronology_index": -1,
        "checkpoint_hash": "",
    }
    checkpoint["checkpoint_hash"] = shadow_mod._checkpoint_hash(checkpoint)  # noqa: SLF001
    valid_checkpoint.write_text(shadow_mod.canonical_json(checkpoint), encoding="utf-8")
    assert FR11FactorCacheShadowAdapter(
        ledger_path=tmp_path / "empty.jsonl",
        checkpoint_path=valid_checkpoint,
        enabled=True,
    ).state_hash == GENESIS_HASH

    checkpoint_errors = [
        ("not-object", [], "not an object"),
        ("schema", {**checkpoint, "schema": "wrong"}, "schema"),
        ("hash", {**checkpoint, "state_hash": "sha256:" + "1" * 64}, "hash"),
        ("tail", {**checkpoint, "ledger_tail_hash": "sha256:" + "2" * 64}, "tail"),
    ]
    for name, payload, message in checkpoint_errors:
        path = tmp_path / f"{name}.checkpoint.json"
        if isinstance(payload, dict) and name != "hash":
            payload["checkpoint_hash"] = shadow_mod._checkpoint_hash(payload)  # noqa: SLF001
        path.write_text(shadow_mod.canonical_json(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            FR11FactorCacheShadowAdapter(
                ledger_path=tmp_path / f"{name}.jsonl",
                checkpoint_path=path,
                enabled=True,
            )

    direct = FR11FactorCacheShadowAdapter(
        ledger_path=tmp_path / "direct.jsonl",
        checkpoint_path=tmp_path / "direct.checkpoint.json",
        enabled=True,
    )
    schema_path = tmp_path / "schema.checkpoint.json"
    hash_path = tmp_path / "hash.checkpoint.json"
    with pytest.raises(ValueError, match="schema"):
        direct._restore_checkpoint(schema_path)  # noqa: SLF001
    with pytest.raises(ValueError, match="hash"):
        direct._restore_checkpoint(hash_path)  # noqa: SLF001
