#!/usr/bin/env python3
"""Roster-level summary of the prompt audit. Quantiles over GAMES, never one game."""

import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "out" / "rows.json").read_text())
ok = [r for r in rows if r.get("status") == "ok"]
gaps = [r for r in rows if r.get("status") != "ok"]


def q(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n = len(v)

    def pct(p):
        return v[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "min": v[0],
        "q1": pct(0.25),
        "median": median(v),
        "q3": pct(0.75),
        "max": v[-1],
        "n": n,
    }


def frac(pred, path):
    hits = []
    for r in ok:
        cur = r
        for p in path:
            cur = (cur or {}).get(p)
        hits.append(bool(pred(cur)))
    return {
        "n_true": sum(hits),
        "n": len(hits),
        "games_true": sorted(r["game"] for r, h in zip(ok, hits, strict=True) if h),
    }


T = lambda k: [r["tokens"].get(k) for r in ok]  # noqa: E731
X = lambda k: [r["transitions"].get(k) for r in ok]  # noqa: E731

# The real per-slot prompt budget the server runs with, from the module's own arithmetic:
# n_ctx 81920 / 4 slots (kv_unified) - max_tokens 4096.
N_CTX, SLOTS, MAX_TOK = 81920, 4, 4096
BUDGET = N_CTX // SLOTS - MAX_TOK

out = {
    "n_games_attempted": len(rows),
    "n_ok": len(ok),
    "coverage_gaps": [
        {"game": r["game"], "status": r["status"], "reason": r.get("error")} for r in gaps
    ],
    "prompt_budget_tokens": BUDGET,
    "budget_derivation": f"n_ctx {N_CTX} / {SLOTS} slots (kv_unified) - max_tokens {MAX_TOK}",
    "tokens": {
        "live_all": q(T("live_all")),
        "live_k8": q(T("live_k8")),
        "as_sent_all": q(T("as_sent_all")),
        "window_all": q(T("window_all")),
        "goal_shipped": q(T("goal_shipped")),
        "goal_with_transitions": q(T("goal_with_transitions")),
    },
    "budget_headroom": {
        "max_as_sent_tokens": max(x for x in T("as_sent_all") if x is not None),
        "max_pct_of_budget": round(
            100 * max(x for x in T("as_sent_all") if x is not None) / BUDGET, 1
        ),
        "n_games_over_budget": sum(1 for x in T("as_sent_all") if x is not None and x > BUDGET),
        "n_games_over_half_budget": sum(
            1 for x in T("as_sent_all") if x is not None and x > BUDGET / 2
        ),
    },
    "truncation": {
        "n_games_where_k8_would_change_the_prompt": sum(
            1 for r in ok if r.get("k8_changes_prompt")
        ),
        "games_where_k8_would_change_the_prompt": sorted(
            r["game"] for r in ok if r.get("k8_changes_prompt")
        ),
        "n_changed_dropped_by_k8": q(X("n_changed_dropped_k8")),
        "n_games_char_budget_bound": sum(
            1 for r in ok if r["transitions"].get("char_budget_bound_all")
        ),
        "action_coverage_lost_by_k8": sum(
            1
            for r in ok
            if r["transitions"]["n_distinct_actions_shown_k8"]
            < r["transitions"]["n_distinct_actions_live"]
        ),
    },
    "transitions": {
        "n_live_trans": q(X("n_live_trans")),
        "n_changed": q(X("n_changed")),
        "n_noop": q(X("n_noop")),
        "n_shown_all": q(X("n_shown_all")),
        "n_distinct_actions_live": q(X("n_distinct_actions_live")),
    },
    "action_space": {
        "n_observed_actions": q([r["action_space"]["n_observed"] for r in ok]),
        "coverage_fraction": q([r["action_space"]["coverage_fraction"] for r in ok]),
        "n_games_single_action_only": sum(1 for r in ok if r["action_space"]["single_action_only"]),
        "games_single_action_only": sorted(
            r["game"] for r in ok if r["action_space"]["single_action_only"]
        ),
        "n_games_observing_all_7": sum(1 for r in ok if r["action_space"]["n_observed"] == 7),
    },
    "change_sparsity": {
        "changed_cell_fraction_median": q(
            [r["change_sparsity"]["changed_cell_fraction_median"] for r in ok]
        ),
        "changed_cell_fraction_max": q(
            [r["change_sparsity"]["changed_cell_fraction_max"] for r in ok]
        ),
        "identity_cellwise_accuracy_median": q(
            [r["change_sparsity"]["identity_cellwise_accuracy_median"] for r in ok]
        ),
        "cells_total": q([r["change_sparsity"]["cells_total"] for r in ok]),
    },
    "asks": {
        "engine_returns_full_grid": frac(bool, ["asks", "engine_returns_full_grid"]),
        "evidence_is_delta_encoded": frac(bool, ["asks", "evidence_is_delta_encoded"]),
        "mentions_delta_output_format": frac(bool, ["asks", "mentions_delta_output_format"]),
        "says_prefer_simple_general": frac(bool, ["asks", "says_prefer_simple_general"]),
    },
    "identity_surface": {
        k: frac(bool, ["identity_surface", k])
        for k in (
            "says_all_other_cells_unchanged",
            "noop_rendered_as_no_change",
            "codeonly_directive_present",
            "codeonly_forbids_grid_analysis",
            "no_think_prefix",
            "forbids_identity",
            "mentions_word_identity",
        )
    },
    "n_no_change_examples": q([r["identity_surface"]["n_no_change_examples"] for r in ok]),
    "first_rendered_transition_is_noop": frac(bool, ["first_rendered_transition_is_noop"]),
    "blocks": {
        k: frac(bool, ["blocks", k])
        for k in (
            "win_transition_block",
            "opening_board_block",
            "object_structure_block",
            "initial_grid_block",
            "playbook_exemplars",
        )
    },
    "goal_prompt": {
        "shipped_is_evidence_free": frac(bool, ["goal_prompt", "shipped_is_evidence_free"]),
        "shipped_carries_transitions": frac(bool, ["goal_prompt", "shipped_carries_transitions"]),
        "shipped_carries_any_grid": frac(bool, ["goal_prompt", "shipped_carries_any_grid"]),
        "flag_on_carries_transitions": frac(bool, ["goal_prompt", "flag_on_carries_transitions"]),
        "receives_win_transition": frac(bool, ["goal_prompt", "receives_win_transition"]),
    },
    "per_game": [
        {
            "game": r["game"],
            "grid": r["transitions"]["grid_shape"],
            "n_live": r["transitions"]["n_live_trans"],
            "n_changed": r["transitions"]["n_changed"],
            "n_noop": r["transitions"]["n_noop"],
            "n_shown_all": r["transitions"]["n_shown_all"],
            "n_shown_k8": r["transitions"]["n_shown_k8"],
            "tok_live_all": r["tokens"]["live_all"],
            "tok_as_sent": r["tokens"]["as_sent_all"],
            "pct_budget": round(100 * (r["tokens"]["as_sent_all"] or 0) / BUDGET, 1),
            "k8_changes_prompt": r.get("k8_changes_prompt"),
            "n_no_change_examples": r["identity_surface"]["n_no_change_examples"],
            "tok_goal_shipped": r["tokens"]["goal_shipped"],
            "n_observed_actions": r["action_space"]["n_observed"],
            "observed_actions": r["action_space"]["observed_actions"],
            "changed_cell_frac_median": r["change_sparsity"]["changed_cell_fraction_median"],
        }
        for r in ok
    ],
}
(HERE / "out" / "analysis.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "per_game"}, indent=1))
