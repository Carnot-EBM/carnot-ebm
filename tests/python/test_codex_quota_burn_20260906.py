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
    reset_soon = int((T0 + timedelta(hours=20)).timestamp())
    fast = burn.burn_report([_s(0, 0.10, reset_soon), _s(10, 0.90, reset_soon)])
    assert fast["hours_to_full_at_this_rate"] < fast["hours_to_reset"]
    assert fast["outpaces_window"] is True
    # Slow arm, arithmetic stated so a future edit cannot quietly invert it:
    # 0.10 -> 0.101 over 10h is 1e-4/h; remaining 0.899 needs 8990h, well past the
    # 490h to reset. A first draft used 0.12 here, which needs 440h and is therefore
    # ALSO outpacing -- the assertion caught my arithmetic, not the code.
    reset_late = int((T0 + timedelta(hours=500)).timestamp())
    slow = burn.burn_report([_s(0, 0.10, reset_late), _s(10, 0.101, reset_late)])
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
