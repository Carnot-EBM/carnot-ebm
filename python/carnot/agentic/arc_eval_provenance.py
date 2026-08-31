"""Per-game generator provenance for the ARC leaderboard eval (REQ-ARC-WMTE-6790).

Lives here rather than in `scripts/arc_leaderboard_eval.py` so it can be tested without
importing that module, which pulls torch/jax and costs ~557MB -- enough to trip the suite's
own memory watchdog at teardown and report a passing test as an ERROR. A check that cannot be
tested cheaply tends not to stay tested.
"""

from __future__ import annotations


def generator_provenance(policy) -> dict:
    """What model, if any, this policy can actually reach right now.

    WHY THIS EXISTS (2026-08-31). A 22-hour adapter-free run over 9 games produced numbers that
    could not be read as an e3 result, because nothing recorded whether each game HAD a
    generator. One llama-server fit at first launch and then held ~18.4 GB for the whole run, so
    every later game logged "even 12 CPU-FFN layers cannot fit the generator" and failed to load
    its own. Of 23 server logs from that run, exactly ONE had any completions; the other 22 were
    failed starts. Whether a later game then reached the surviving server over HTTP was confirmed
    exactly once, by hand, with `ss -tnp`.

    Without this, a row reading `levels=2, efficiency=0.1025` is indistinguishable between "the
    e3 cascade solved it" and "the explorer floor solved it with no model at all". That is not a
    weaker result; it is an uninterpretable one, and 22 GPU-hours bought two usable games.

    Everything here is best-effort and never raises: provenance that can break the run it is
    describing is worse than no provenance.
    """

    out: dict[str, object] = {"resolved": False}
    prop = getattr(policy, "proposer", None)
    if prop is None:
        out["note"] = "policy exposes no proposer (explorer-tier policy has none)"
        return out
    out["resolved"] = True
    for name, attr in (
        ("server_url", "_url"),
        ("model_path", "model_path"),
        ("observed_server_model_path", "observed_server_model_path"),
        ("reuse_model_check", "reuse_model_check"),
        ("n_ctx", "n_ctx"),
        ("ffn_cpu_layers", "ffn_cpu_layers"),
    ):
        try:
            v = getattr(prop, attr, None)
            out[name] = v() if callable(v) else v
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unreadable: {type(exc).__name__}: {exc}"
    return out


def completion_counters(policy) -> dict:
    """A snapshot of the proposer's monotone completion counters (REQ-ARC-WMTE-6710).

    Differencing two snapshots gives the completions ONE game actually consumed -- the counters
    were declared monotone for exactly this use. `completions == 0` across a game is the
    unambiguous signal that no model was reached, which is the fact the 2026-08-31 run could not
    establish for 7 of its 9 games.
    """

    prop = getattr(policy, "proposer", None)
    totals = getattr(prop, "channel_totals", None)
    return dict(totals) if isinstance(totals, dict) else {}
