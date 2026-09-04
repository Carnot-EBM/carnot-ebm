"""Spec: REQ-ARC-WMTE-6641, SCENARIO-ARC-WMTE-6641-EVAL-ROW

The live-path eval emits the generator's induce-channel counters on every row.

INCIDENT 2026-09-04. The proposer counts what the generator actually returned -- chat_completions,
chars_final, chars_reasoning -- and the only code that read them into an artifact was
`arc_scored_path_lever_harness.py`, frozen on the RETIRED Qwen3.5-9B-MTP pin and unable to start
against the live generator. All eleven artifacts in results/arc_leaderboard_eval_runs/ carry zero
rows with these counters, including runs whose own `llm_reached` is true.

The cost was concrete: after the n_ctx=98304 fix, "did the induce tier emit anything" was not
answerable from any artifact, so a multi-hour live run would have produced another row that could
not answer it. Same shape as the trajectory-supervisor gap fixed three days earlier.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import arc_leaderboard_eval as ale  # noqa: E402


class _Proposer:
    def __init__(self, totals: Any) -> None:
        self.channel_totals = totals


class _Policy:
    def __init__(self, proposer: Any) -> None:
        self.proposer = proposer


def test_real_counters_pass_through() -> None:
    totals = {"chat_completions": 24, "chars_final": 0, "chars_reasoning": 2_400_000}
    assert ale.generator_channels_row_field(_Policy(_Proposer(totals))) == totals


def test_an_absent_proposer_is_reported_not_errored() -> None:
    """The LLM tier legitimately may never fire; that is a state, not a broken instrument."""
    assert ale.generator_channels_row_field(_Policy(None)) == {"proposer": "absent"}


def test_a_missing_attribute_is_an_error_marker() -> None:
    class _Bare:
        pass

    assert ale.generator_channels_row_field(_Policy(_Bare())) == {"error": "no_channel_totals_attr"}


def test_a_non_dict_value_is_an_error_marker() -> None:
    """A bare None would read as 'the generator returned nothing' rather than 'instrument broke'."""
    out = ale.generator_channels_row_field(_Policy(_Proposer([1, 2])))
    assert out == {"error": "non_dict_channel_totals:list"}


def test_every_policy_state_yields_a_dict() -> None:
    class _Boom:
        @property
        def proposer(self) -> Any:
            raise RuntimeError("x")

    for pol in (_Policy(_Proposer({})), _Policy(None), _Policy(object()), _Boom()):
        assert isinstance(ale.generator_channels_row_field(pol), dict)


def test_the_row_carries_the_field() -> None:
    """The call-site test. A helper nothing calls is precisely what this incident WAS."""
    src = (REPO / "scripts" / "arc_leaderboard_eval.py").read_text()
    assert '"generator_channels": generator_channels_row_field(policy),' in src
