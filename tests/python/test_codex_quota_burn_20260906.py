"""Quota burn is read from the provider's records, and a reset never flattens the slope.

REQ-QUOTA-BURN-1. The load-bearing rule is segment detection: the used fraction FALLS
when a window rolls, and a slope fitted across that fall averages consumption with a
refill. That single error would make the tool worse than not having it, because it
understates burn exactly when burn matters.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import codex_quota_burn as burn  # noqa: E402

T0 = datetime(2026, 9, 6, 0, 0, tzinfo=UTC)


def _s(hours: float, used: float, resets_at: int | None = None) -> burn.Sample:
    return burn.Sample(
        when=T0 + timedelta(hours=hours),
        used_fraction=used,
        window_minutes=10080,
        resets_at=resets_at,
    )


def test_rising_samples_give_a_positive_rate_from_the_segment_ends() -> None:
    # SCENARIO-QUOTA-BURN-1
    rep = burn.burn_report([_s(0, 0.10), _s(5, 0.20), _s(10, 0.30)])
    assert rep["segment_samples"] == 3
    assert rep["burn_fraction_per_hour"] == (0.30 - 0.10) / 10.0


def test_a_fall_starts_a_new_segment_and_only_the_tail_is_fitted() -> None:
    # SCENARIO-QUOTA-BURN-2. This is the whole point of the tool being trustworthy.
    samples = [_s(0, 0.80), _s(1, 0.90), _s(2, 0.05), _s(6, 0.25)]
    rep = burn.burn_report(samples)
    assert rep["segment_samples"] == 2, "the pre-reset samples must be excluded"
    # 0.05 -> 0.25 over 4h. Fitting across the reset would give a NEGATIVE or tiny slope.
    assert rep["burn_fraction_per_hour"] == (0.25 - 0.05) / 4.0
    assert rep["segment_started_at"] == _s(2, 0.05).when.isoformat()


def test_fitting_across_a_reset_would_have_understated_burn() -> None:
    # SCENARIO-QUOTA-BURN-2, stated as the defect it prevents rather than as a value.
    samples = [_s(0, 0.80), _s(1, 0.90), _s(2, 0.05), _s(6, 0.25)]
    naive = (samples[-1].used_fraction - samples[0].used_fraction) / 6.0
    correct = burn.burn_report(samples)["burn_fraction_per_hour"]
    assert naive < 0 < correct


def test_a_single_sample_since_reset_yields_no_projection() -> None:
    # SCENARIO-QUOTA-BURN-3
    rep = burn.burn_report([_s(0, 0.90), _s(1, 0.10)])
    assert "burn_fraction_per_hour" not in rep
    assert "fewer than two samples" in rep["projection"]


def test_a_flat_segment_yields_no_projection() -> None:
    # SCENARIO-QUOTA-BURN-3. Never emit a number when the slope cannot support one.
    rep = burn.burn_report([_s(0, 0.40), _s(5, 0.40)])
    assert "burn_fraction_per_hour" not in rep
    assert "hours_to_full_at_this_rate" not in rep
    assert "flat or falling" in rep["projection"]


def test_outpaces_window_compares_against_the_records_own_reset_stamp() -> None:
    # SCENARIO-QUOTA-BURN-4. Both directions, so a constant-true cannot pass.
    # Fast arm: 0.10 -> 0.90 over 10h is 0.08/h; remaining 0.10 needs 1.25h, against
    # 10h to reset from the LATEST sample. Both arms are spelled out because I got the
    # arithmetic wrong twice writing them, in opposite directions.
    # `now` is pinned to the latest sample so this test measures the COMPARISON, not
    # the wall clock. Before SCENARIO-QUOTA-BURN-6 the reset was measured from the
    # sample implicitly; making it explicit is what keeps this deterministic.
    reset_soon = int((T0 + timedelta(hours=20)).timestamp())
    at_last_sample = T0 + timedelta(hours=10)
    fast = burn.burn_report([_s(0, 0.10, reset_soon), _s(10, 0.90, reset_soon)], now=at_last_sample)
    assert fast["hours_to_full_at_this_rate"] < fast["hours_to_reset"]
    assert fast["outpaces_window"] is True
    # Slow arm, arithmetic stated so a future edit cannot quietly invert it:
    # 0.10 -> 0.101 over 10h is 1e-4/h; remaining 0.899 needs 8990h, well past the
    # 490h to reset. A first draft used 0.12 here, which needs 440h and is therefore
    # ALSO outpacing -- the assertion caught my arithmetic, not the code.
    reset_late = int((T0 + timedelta(hours=500)).timestamp())
    slow = burn.burn_report(
        [_s(0, 0.10, reset_late), _s(10, 0.101, reset_late)], now=at_last_sample
    )
    assert slow["hours_to_full_at_this_rate"] > slow["hours_to_reset"]
    assert slow["outpaces_window"] is False


def test_no_records_is_reported_not_guessed() -> None:
    rep = burn.burn_report([])
    assert rep["samples"] == 0
    assert "burn_fraction_per_hour" not in rep


def test_unreadable_files_are_counted_not_dropped(tmp_path: Path) -> None:
    # SCENARIO-QUOTA-BURN-5. A partial read must not present itself as complete.
    day = tmp_path / "2026" / "09" / "06"
    day.mkdir(parents=True)
    good = day / "rollout-good.jsonl"
    good.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-06T01:00:00.000Z",
                "payload": {
                    "rate_limits": {"primary": {"used_percent": 12.5, "window_minutes": 10080}}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (day / "rollout-garbage.jsonl").write_text('{"rate_limits": NOT JSON\n', encoding="utf-8")
    samples, read, skipped = burn.collect_samples(str(tmp_path), days=3650)
    assert len(samples) == 1
    assert samples[0].used_fraction == 0.125, "percent must be converted to a fraction"
    assert read == 2 and skipped == 0, "a malformed LINE is skipped without losing the file"


def test_the_tool_writes_nothing(tmp_path: Path) -> None:
    # REQ-QUOTA-BURN-1 rule 2. Read-only is a contract, so assert it.
    day = tmp_path / "2026" / "09" / "06"
    day.mkdir(parents=True)
    (day / "rollout-a.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-09-06T01:00:00.000Z",
                "payload": {"rate_limits": {"primary": {"used_percent": 5.0}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = sorted(p.name for p in day.iterdir())
    burn.main(["--root", str(tmp_path), "--days", "3650", "--json"])
    assert sorted(p.name for p in day.iterdir()) == before


def test_hours_to_reset_is_measured_from_now_not_from_the_latest_sample() -> None:
    """SCENARIO-QUOTA-BURN-6. First real use exposed this: with a 6.8h-old sample the
    tool reported 10.06h to reset against a true 3.23h. Measuring from the sample
    overstates the remaining window by exactly the staleness -- and the samples go
    stale precisely when calls are being rejected, which is when the number matters."""
    reset_at = int((T0 + timedelta(hours=20)).timestamp())
    now = T0 + timedelta(hours=15)  # 5h after the last sample
    rep = burn.burn_report([_s(0, 0.10, reset_at), _s(10, 0.20, reset_at)], now=now)
    assert rep["hours_to_reset"] == 5.0, "20h reset minus 15h now"
    # Measuring from the latest sample (hour 10) would have said 10.0.
    assert rep["hours_to_reset"] != 10.0


def test_a_stale_sample_is_flagged_with_its_age() -> None:
    """SCENARIO-QUOTA-BURN-6. A stale reading presented as current is worse than no
    reading, because it looks authoritative."""
    now = T0 + timedelta(hours=17)
    rep = burn.burn_report([_s(0, 0.10), _s(10, 0.20)], now=now)
    assert rep["sample_age_hours"] == 7.0
    assert "7.0h old" in rep["staleness_warning"]


def test_a_fresh_sample_carries_no_staleness_warning() -> None:
    """SCENARIO-QUOTA-BURN-6. The warning must discriminate, or it is decoration."""
    now = T0 + timedelta(hours=10, minutes=6)
    rep = burn.burn_report([_s(0, 0.10), _s(10, 0.20)], now=now)
    assert rep["sample_age_hours"] < 1.0
    assert "staleness_warning" not in rep


def test_projections_closer_than_the_blind_spot_are_not_ordered() -> None:
    """SCENARIO-QUOTA-BURN-7, the historical case. On 2026-09-06 the tool reported
    opposite verdicts five minutes apart on projections 3.210h and 3.157h under a 6.9h
    stale sample. A 0.05h margin inside a 6.9h blind spot is not a measurement."""
    # 0.10 -> 0.90 over 10h is 0.08/h; remaining 0.10 needs 1.25h to full.
    now = T0 + timedelta(hours=17)  # 7h after the last sample
    reset_at = int((now + timedelta(hours=1.3)).timestamp())  # ~0.05h from hours_to_full
    rep = burn.burn_report([_s(0, 0.10, reset_at), _s(10, 0.90, reset_at)], now=now)
    assert "outpaces_window" not in rep, "an unresolvable order must not be asserted"
    assert "too close to call" in rep["verdict"]
    assert rep["verdict_resolution_hours"] == 7.0


def test_a_clear_separation_still_gets_a_verdict() -> None:
    """SCENARIO-QUOTA-BURN-7. Suppression must discriminate; a guard that always fires
    would delete the tool's answer entirely."""
    now = T0 + timedelta(hours=10, minutes=6)  # fresh: floor applies, not staleness
    reset_at = int((now + timedelta(hours=40)).timestamp())
    rep = burn.burn_report([_s(0, 0.10, reset_at), _s(10, 0.90, reset_at)], now=now)
    assert rep["outpaces_window"] is True
    assert "verdict" not in rep
    assert rep["verdict_margin_hours"] > rep["verdict_resolution_hours"]


def test_the_floor_applies_when_the_sample_is_fresh() -> None:
    """SCENARIO-QUOTA-BURN-7. With a fresh sample the staleness term goes to ~0, so the
    0.25h judgement floor is what stops the tool splitting hairs on a two-point rate."""
    now = T0 + timedelta(hours=10)  # zero staleness
    # hours_to_full is 1.25h; put the reset 0.1h away from it, inside the floor.
    reset_at = int((now + timedelta(hours=1.35)).timestamp())
    rep = burn.burn_report([_s(0, 0.10, reset_at), _s(10, 0.90, reset_at)], now=now)
    assert rep["verdict_resolution_hours"] == 0.25, "the floor, not staleness"
    assert "outpaces_window" not in rep
    assert "too close to call" in rep["verdict"]
