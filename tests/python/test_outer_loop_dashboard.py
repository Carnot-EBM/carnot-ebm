"""REQ-INFRA-6840: the hourly outer-loop dashboard reports measured state, never recalled state.

The hourly status report drifted from a compact scannable block into paragraphs, and the
operator asked for the block back. A script fixes that permanently -- same fields, same order,
every hour -- and removes two error classes this session hit repeatedly: elapsed time estimated
rather than computed (days-old events reported as months old), and process liveness assumed
rather than read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "outer_loop_dashboard.py"


def _module():
    spec = importlib.util.spec_from_file_location("_dash", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dash"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_age_is_computed_not_estimated() -> None:
    """The whole point: 8 days must read as 8, not as "a while back"."""
    m = _module()
    now = datetime(2026, 8, 30, tzinfo=UTC)
    assert m.days_since((now - timedelta(days=8)).isoformat(), now) == 8
    assert m.days_since((now - timedelta(days=1)).isoformat(), now) == 1


def test_an_unparseable_date_returns_none_rather_than_a_guess() -> None:
    """A wrong age is worse than an absent one -- the wrong one gets repeated as fact."""
    assert _module().days_since("not-a-date") is None


def test_liveness_is_read_from_proc_not_assumed() -> None:
    m = _module()
    assert m.pid_alive(os.getpid())
    assert not m.pid_alive(999_999_999)


def test_outcomes_are_counted_from_the_log(tmp_path, monkeypatch) -> None:
    """Counted, never recalled. A remembered outcome mix is how a bad day gets called good."""
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(
        "| 2026-08-30 01:00 UTC | a | OK | x |\n"
        "| 2026-08-30 02:00 UTC | b | FAIL | y |\n"
        "| 2026-08-30 03:00 UTC | c | OK | z |\n"
        "| 2026-08-29 01:00 UTC | d | OK | other day |\n"
    )
    assert m.outcome_mix("2026-08-30") == {"OK": 2, "FAIL": 1}


def test_a_dead_job_with_no_receipt_says_so_explicitly(tmp_path) -> None:
    """The 2026-08-29 case: a 7-hour job vanished and nothing said what killed it.

    A missing receipt is itself evidence (SIGKILL, OOM, kernel) and must be reported as such
    rather than shown as a blank.
    """
    out = _module().render([("ab", 999_999_999, tmp_path / "absent.json")])
    assert "NO receipt" in out
    assert "REQ-INFRA-6830" in out


def test_a_dead_job_with_a_receipt_names_the_signal(tmp_path) -> None:
    import json

    receipt = tmp_path / "death.json"
    receipt.write_text(
        json.dumps(
            {"signal_name": "SIGTERM", "elapsed_s": 25200.0, "progress": {"cells": 38, "of": 39}}
        )
    )
    out = _module().render([("ab", 999_999_999, receipt)])
    assert "KILLED by SIGTERM" in out
    assert "38" in out


def test_a_live_job_is_reported_alive(tmp_path) -> None:
    out = _module().render([("self", os.getpid(), None)])
    assert "alive" in out


def test_the_header_carries_the_actual_date() -> None:
    """The report must state the date it was produced -- the drift this exists to stop."""
    out = _module().render([])
    assert datetime.now(UTC).strftime("%Y-%m-%d") in out


# --- Efficiency and generalization lines (2026-08-30) -----------------------------------------
# The operator asked for an efficiency score and a submission score. The first version computed
# the competition's capped rule over banked solve artifacts and returned a uniform 1.000 -- real
# arithmetic over the wrong quantity, because the stored `solution` is a WINNING PATH REPLAYED and
# is 1.2x-5.3x shorter than a human who had to explore. A submission score was declined outright:
# all 25 public games are cleared by hand-built per-game adapters that do not transfer, so a
# public number would read as a hidden-game prediction it cannot support.


def test_efficiency_score_matches_the_competition_rule() -> None:
    """Kept because the rule itself is right; what was wrong was the data fed to it."""
    m = _module()
    assert m.efficiency_score(10, 20) == 0.25
    assert m.efficiency_score(20, 10) == 1.0  # capped at human parity
    assert m.efficiency_score(0, 10) is None
    assert m.efficiency_score(10, 0) is None


def test_a_missing_side_is_none_not_zero() -> None:
    """An unmeasured level and a maximally inefficient one are different findings.

    Averaging the second into a headline is how a coverage gap becomes a bad score.
    """
    assert _module().efficiency_score(5, 0) is None


def test_the_efficiency_line_refuses_to_claim_discovery_cost(tmp_path, monkeypatch) -> None:
    """The degeneracy guard. A banked replay must never be rendered as an efficiency score."""
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_agi3_game_characterization.json").write_text(
        json.dumps({"games": [{"game_id": "aa11-x", "baseline_actions": [100, 100]}]})
    )
    (tmp_path / "results" / "arc_loop_solve_aa11.json").write_text(
        json.dumps(
            {
                "game": "aa11-x",
                "reached_level": 2,
                "solution": ["a"] * 20,
                "solve_provenance": "development_proxy",
            }
        )
    )
    out = m.render([])
    assert "DISCOVERY COST NOT MEASURED" in out
    assert "10.0x shorter than human" in out  # 200/20, uncapped


def test_the_ratio_is_uncapped_so_degeneracy_stays_visible(tmp_path, monkeypatch) -> None:
    """Capping is what hid the bug: every game clamped to 1.0 and read as a perfect score."""
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_agi3_game_characterization.json").write_text(
        json.dumps({"games": [{"game_id": "bb22-y", "baseline_actions": [500]}]})
    )
    (tmp_path / "results" / "arc_loop_solve_bb22.json").write_text(
        json.dumps(
            {
                "game": "bb22-y",
                "reached_level": 1,
                "solution": ["a"] * 10,
                "solve_provenance": "development_proxy",
            }
        )
    )
    assert m.public_set_efficiency()["mean_ratio"] == 50.0


def test_generalization_says_not_measured_rather_than_zero(tmp_path, monkeypatch) -> None:
    """A public-set count is a DIFFERENT quantity and must not stand in for a hidden-game one.

    Zero would read as a measured result; "not measured" is the honest state while every solve
    carries development_proxy.
    """
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_loop_solve_cc33.json").write_text(
        json.dumps(
            {"game": "cc33", "reproduced_levels": 7, "solve_provenance": "development_proxy"}
        )
    )
    gen = m.generalization_levels()
    # Field checks, not dict equality: the shape gained a `source` key and an exact-match
    # assertion breaks on every future field without telling you anything useful.
    assert (gen["measured"], gen["levels"], gen["games"]) == (False, 0, 0)
    assert "not measured" in m.render([])


def test_generalization_counts_only_live_self_discovery(tmp_path, monkeypatch) -> None:
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_loop_solve_dd44.json").write_text(
        json.dumps(
            {
                "game": "dd44",
                "reproduced_levels": 3,
                "solve_provenance": "live_agent_self_discovery",
            }
        )
    )
    (tmp_path / "results" / "arc_loop_solve_ee55.json").write_text(
        json.dumps(
            {"game": "ee55", "reproduced_levels": 99, "solve_provenance": "development_proxy"}
        )
    )
    gen = m.generalization_levels()
    assert (gen["measured"], gen["levels"], gen["games"]) == (True, 3, 1)


# --- Worker walk + provenance on the generalization number (2026-08-30) ----------------------


def test_the_generalization_number_carries_policy_and_age(tmp_path, monkeypatch) -> None:
    """A 48-day-old `policy=explorer` floor was being shown as the current e3 measurement.

    A number without its provenance is how a stale floor becomes a headline.
    """
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_leaderboard_eval.json").write_text(
        json.dumps({"policy": "explorer", "live_levels": 5, "per_game": {"a": 1, "b": 2}})
    )
    gen = m.generalization_levels()
    assert gen["policy"] == "explorer"
    assert isinstance(gen["age_days"], int)
    out = m.render([])
    assert "policy=explorer" in out and "d old]" in out


def test_the_leaderboard_source_is_preferred_over_solve_artifacts(tmp_path, monkeypatch) -> None:
    """Reading only arc_loop_solve_*.json left this stuck at "not measured" forever, because the
    adapter-free eval writes a different file."""
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_leaderboard_eval.json").write_text(
        json.dumps({"policy": "e3", "live_levels": 12, "per_game": {}})
    )
    (tmp_path / "results" / "arc_loop_solve_zz99.json").write_text(
        json.dumps(
            {"game": "zz99", "reproduced_levels": 99, "solve_provenance": "development_proxy"}
        )
    )
    assert m.generalization_levels()["levels"] == 12


def test_zero_live_levels_does_not_count_as_measured(tmp_path, monkeypatch) -> None:
    m = _module()
    monkeypatch.setattr(m, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc_leaderboard_eval.json").write_text(
        json.dumps({"policy": "e3", "live_levels": 0})
    )
    assert m.generalization_levels()["measured"] is False


def test_the_worker_walk_finds_a_descendant_not_the_supervisor(monkeypatch) -> None:
    """The 2026-08-30 misreading: the dashboard showed a parent asleep in poll while the real
    work ran two levels down at 99% CPU."""
    m = _module()
    monkeypatch.setattr(m, "_run", lambda *a: "4242, 17884 MiB")
    monkeypatch.setattr(m.Path, "read_text", lambda self, **k: "4242 (x) S 999", raising=False)
    assert m.gpu_worker_for(999) == 4242


def test_a_job_with_no_gpu_worker_says_so(monkeypatch) -> None:
    m = _module()
    monkeypatch.setattr(m, "_run", lambda *a: "")
    monkeypatch.setattr(m, "pid_alive", lambda pid: True)
    assert "no GPU worker" in m.render([("j", os.getpid(), None)])


def test_the_walk_prefers_the_busiest_descendant_not_the_first(monkeypatch) -> None:
    """The launcher can itself appear in nvidia-smi's holder list.

    It does for the leaderboard eval, so "first descendant" picked the supervisor at 1.2% CPU
    over its own worker at 629% -- reproducing the exact misreading the walk was written to end.
    A pid is a descendant of itself at zero hops, which is why the naive version matched it.
    """
    m = _module()
    monkeypatch.setattr(
        m,
        "_run",
        lambda *a: (
            "500, 100 MiB\n600, 17884 MiB"
            if a[0] == "nvidia-smi"
            else ("1.2" if a[-1] == "500" else "629.0")
        ),
    )
    stats = {500: "500 (x) S 1", 600: "600 (x) R 500"}

    def fake_read(self, **kwargs):
        return stats[int(str(self).split("/")[2])]

    monkeypatch.setattr(m.Path, "read_text", fake_read, raising=False)
    # 500 is the launcher (a holder, and its own descendant); 600 is its busy worker.
    assert m.gpu_worker_for(500) == 600
