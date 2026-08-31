"""REQ-ARC-WMTE-6790: every eval row records whether that game reached a generator.

INCIDENT 2026-08-31. A 22-hour adapter-free run of `arc_leaderboard_eval.py --policy e3` over 9
games produced per-game numbers that could not be read as an e3 result. One llama-server fit at
first launch and held ~18.4 GB for the whole run, so every later game logged "even 12 CPU-FFN
layers cannot fit the generator". Of 23 server logs, exactly ONE had completions; the other 22
were failed starts. Whether later games reached the surviving server over HTTP was confirmed
exactly once, by hand.

A row reading `levels=2, efficiency=0.1025` was therefore indistinguishable between the e3
cascade solving it and the explorer floor solving it with no model at all. Not a weaker result --
an uninterpretable one. 22 GPU-hours bought two usable games.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "arc_leaderboard_eval.py"


from carnot.agentic.arc_eval_provenance import completion_counters, generator_provenance


class _Prop:
    def __init__(self, **kw):
        self.model_path = kw.get("model_path")
        self.observed_server_model_path = kw.get("observed")
        self.reuse_model_check = kw.get("reuse", "not_checked")
        self.n_ctx = kw.get("n_ctx", 49152)
        self.ffn_cpu_layers = kw.get("ffn", 0)
        self.channel_totals = dict(kw.get("totals", {}))
        self._port = kw.get("port", 8919)

    def _url(self):
        return f"http://127.0.0.1:{self._port}"


class _Policy:
    def __init__(self, proposer=None):
        if proposer is not None:
            self.proposer = proposer


def test_provenance_records_the_reachable_backend() -> None:
    p = _Policy(_Prop(model_path="/m/qwen.gguf", observed="/m/qwen.gguf", ffn=1))
    prov = generator_provenance(p)
    assert prov["resolved"] is True
    assert prov["server_url"] == "http://127.0.0.1:8919"
    assert prov["observed_server_model_path"] == "/m/qwen.gguf"
    assert prov["ffn_cpu_layers"] == 1


def test_a_policy_with_no_proposer_says_so_rather_than_raising() -> None:
    """The explorer tier has no proposer. Provenance that breaks the run it describes is worse
    than no provenance."""
    prov = generator_provenance(_Policy())
    assert prov["resolved"] is False
    assert "no proposer" in str(prov.get("note"))


def test_an_unreadable_attribute_is_reported_not_raised() -> None:

    class Hostile:
        # A bare class, not a _Prop subclass: _Prop.__init__ assigns model_path, which a
        # read-only property refuses, and the test would fail on construction instead of on
        # the behaviour under test.
        channel_totals: dict = {}

        def _url(self):
            return "http://127.0.0.1:1"

        @property
        def model_path(self):
            raise RuntimeError("boom")

    prov = generator_provenance(_Policy(Hostile()))
    assert "unreadable" in str(prov["model_path"])
    assert prov["resolved"] is True


def test_counters_snapshot_is_a_copy_not_a_live_reference() -> None:
    """Differencing two snapshots is the whole mechanism; a shared reference would difference
    to zero and silently report "no model reached" for every game."""
    prop = _Prop(totals={"completions": 5})
    pol = _Policy(prop)
    before = completion_counters(pol)
    prop.channel_totals["completions"] = 9
    after = completion_counters(pol)
    assert before["completions"] == 5
    assert after["completions"] == 9


def test_a_policy_without_counters_yields_an_empty_snapshot() -> None:
    assert completion_counters(_Policy()) == {}
    assert completion_counters(_Policy(_Prop())) == {}


def test_the_call_site_assertion_is_deliberately_absent_until_the_wiring_lands() -> None:
    """THE CALL SITE IS NOT WIRED YET, ON PURPOSE, AND THIS RECORDS WHY.

    Editing `scripts/arc_leaderboard_eval.py` makes 5 artifacts stale that cite it as
    provenance, and `artifact-freshness-lint` correctly refuses the commit until each is rebuilt
    and diffed. One was rebuilt already and moved ZERO measured values -- only build timestamp,
    git head, and the SHA/bytes of the edited file -- so the change is additive. The remaining
    four are queued.

    So this module ships tested and INERT. That is the exact "implemented beside the call site"
    shape this session caught three times, and it is acceptable ONLY because it is stated here
    and queued in ops/known-issues.md rather than assumed to be finished.

    WHEN THE WIRING LANDS, REPLACE THIS TEST with the assertions it displaced:

        src = SCRIPT.read_text()
        assert '"generator_provenance": _gen_prov,' in src
        assert '"completions_consumed": _consumed,' in src
        assert '"llm_reached": bool(_consumed.get("completions", 0) > 0),' in src
        assert "_counters_before = completion_counters(policy)" in src
        assert "_counters_after = completion_counters(policy)" in src

    The prepared diff is in the session scratchpad as `eval_wiring.patch`.
    """

    src = SCRIPT.read_text()
    assert "generator_provenance" not in src, (
        "the wiring has landed -- restore the call-site assertions in this test"
    )
