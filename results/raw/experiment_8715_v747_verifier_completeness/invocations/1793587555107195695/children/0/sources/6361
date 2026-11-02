#!/usr/bin/env python3
"""Phase 2a follow-up analysis: the CORRECT pre-registered primary metric for the expanded
gemma think-mode A/B (`results/experiment_6221_gemma_think_mode_ab_expanded_roster.json`).

WHY THIS EXISTS SEPARATELY FROM THE RUN SCRIPT. exp6199's own `_verdict()` function (reused
unmodified by the expanded-roster driver) still decides its headline on `levelup_positive_recall`
-- the metric Finding 3 of the improvement plan named as the WEAKER, noisier read. Finding 3's
own re-analysis of the original 12-game run found the metric that actually matters is HELD-OUT
EXACT ACCURACY reaching the live admission gate's own bar (`heldout_accuracy >= 1.0`, the
`min_heldout_accuracy=1.0` threshold `arc_competition_agent.py` uses to decide whether an induced
engine is trusted at all) -- and that under THAT metric, think won 4-0-6 against no_think on the
original run, not the mixed/negative-looking headline the run's own verdict string reported.

THE PRE-REGISTERED TEST (Phase 2a's own gate, stated before this analysis exists): per-game sign
test on which arm reaches the admission bar, treating each game as one discordant/concordant
observation (both arms admit, both fail, or one admits and the other does not). Gate: >= 6
discordant games favoring think at p < 0.05 (exact two-sided binomial sign test) -> think stays
on and Phase 2b unlocks; fewer -> record honestly, think's default reverts to the exp6199
evidence base alone.

CPU-only; reads an already-written artifact, invokes no model.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "results/experiment_6221_gemma_think_mode_ab_expanded_roster.json"
OUT_PATH = REPO_ROOT / "results/analysis_exp6221_admission_rate_20260809.json"
ADMISSION_BAR = 1.0  # arc_competition_agent.py's min_heldout_accuracy=1.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign_test_two_sided_exact(n_favor_a: int, n_favor_b: int) -> float:
    """Exact two-sided binomial sign test, p=0.5 null, no external dependency (scipy not assumed
    available in every environment this repeats in). Standard sum-of-tails construction."""
    n = n_favor_a + n_favor_b
    if n == 0:
        return 1.0
    from math import comb

    k = min(n_favor_a, n_favor_b)

    def _pmf(i: int) -> float:
        return comb(n, i) * (0.5**n)

    # Sum every outcome at least as extreme (as far from n/2) as the observed minority count.
    total = 0.0
    for i in range(0, n + 1):
        if abs(i - n / 2) >= abs(k - n / 2) - 1e-9:
            total += _pmf(i)
    return min(1.0, total)


def build_artifact() -> dict:
    t0 = time.time()
    if not INPUT_PATH.exists():
        return {
            "experiment": "analysis_exp6221_admission_rate",
            "honest_verdict": "blocked_input_artifact_not_yet_written",
            "input_path": str(INPUT_PATH.relative_to(REPO_ROOT)),
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_s": round(time.time() - t0, 3),
        }
    raw = json.loads(INPUT_PATH.read_text())
    rows = raw.get("per_game_results", [])
    by_game: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("arm") not in ("no_think", "think"):
            continue
        by_game.setdefault(r["game"], {})[r["arm"]] = r

    per_game = []
    n_favor_think = 0
    n_favor_nothink = 0
    n_tie = 0
    n_incomplete = 0
    for game, arms in sorted(by_game.items()):
        nt = arms.get("no_think")
        th = arms.get("think")
        if nt is None or th is None:
            n_incomplete += 1
            per_game.append({"game": game, "status": "incomplete", "have_arms": list(arms)})
            continue
        nt_admit = bool(
            nt.get("induction_ok") and (nt.get("heldout_accuracy") or 0.0) >= ADMISSION_BAR
        )
        th_admit = bool(
            th.get("induction_ok") and (th.get("heldout_accuracy") or 0.0) >= ADMISSION_BAR
        )
        if th_admit and not nt_admit:
            n_favor_think += 1
            outcome = "think_favored"
        elif nt_admit and not th_admit:
            n_favor_nothink += 1
            outcome = "no_think_favored"
        else:
            n_tie += 1
            outcome = "tie_both_admit" if th_admit else "tie_neither_admits"
        per_game.append(
            {
                "game": game,
                "status": "complete",
                "no_think_admitted": nt_admit,
                "think_admitted": th_admit,
                "no_think_heldout_accuracy": nt.get("heldout_accuracy"),
                "think_heldout_accuracy": th.get("heldout_accuracy"),
                "no_think_induction_ok": nt.get("induction_ok"),
                "think_induction_ok": th.get("induction_ok"),
                "outcome": outcome,
            }
        )

    n_discordant = n_favor_think + n_favor_nothink
    p_value = _sign_test_two_sided_exact(n_favor_think, n_favor_nothink)
    gate_met = n_favor_think >= 6 and p_value < 0.05

    art: dict = {
        "experiment": "analysis_exp6221_admission_rate",
        "title": (
            "Phase 2a follow-up: the pre-registered admission-rate primary metric for the "
            "expanded gemma think-mode A/B"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "On the held-out exact-accuracy admission bar (>=1.0, the live agent's own trust "
            "threshold), does think beat no_think on >=6 discordant games at p<0.05 (exact "
            "two-sided sign test)?"
        ),
        "admission_bar": ADMISSION_BAR,
        "per_game": per_game,
        "headline": {
            "n_games_total": len(by_game),
            "n_games_complete": len(by_game) - n_incomplete,
            "n_games_incomplete": n_incomplete,
            "n_favor_think": n_favor_think,
            "n_favor_no_think": n_favor_nothink,
            "n_tie": n_tie,
            "n_discordant": n_discordant,
            "p_value_two_sided_sign_test": round(p_value, 6),
            "gate_condition": ">=6 discordant games favoring think AND p<0.05",
            "gate_met": gate_met,
            "reading": (
                f"{n_favor_think} games favor think, {n_favor_nothink} favor no_think, {n_tie} "
                f"tie, out of {len(by_game) - n_incomplete} complete games "
                f"({n_incomplete} incomplete at analysis time). p={p_value:.4f}. "
                f"Gate {'MET' if gate_met else 'NOT MET'} at analysis time."
            ),
        },
        "caveats": [
            "This reads whatever per_game_results rows exist in the input artifact AT THE TIME "
            "THIS SCRIPT RUNS. If the underlying A/B was still in progress or was cut short by "
            "the 1-hour external timeout wrapper, n_games_incomplete reflects that honestly -- "
            "this is not a claim that the full 16-game roster completed.",
            "Admission is defined here as induction_ok AND heldout_accuracy>=1.0 together, "
            "matching arc_competition_agent.py's actual live gate (an engine that fails to "
            "parse/run at all cannot be admitted regardless of what heldout_accuracy would "
            "otherwise read).",
        ],
        "verifier_is_oracle": True,
        "verifier_is_oracle_principle": (
            "heldout_accuracy>=1.0 IS the live agent's own admission gate; this measures "
            "whether the treatment moves that exact gate, not an oracle-distinct claim."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_principle": (
            "aggregation over an already-run offline dev-twin A/B; no new solve is claimed here."
        ),
        "arc_solve_claim": False,
        "random_seed": 6221,
    }

    art["honest_verdict"] = (
        f"complete_admission_rate_analysis_"
        f"{n_favor_think}_favor_think_{n_favor_nothink}_favor_no_think_{n_tie}_tie_"
        f"p_{p_value:.4f}_gate_{'met' if gate_met else 'not_met'}_"
        f"{n_incomplete}_of_{len(by_game)}_games_incomplete_at_analysis_time"
    )
    art["honest_verdict_principle"] = (
        "terminal `complete_` prefix; states the full discordant breakdown, the p-value, the "
        "gate outcome, and the completeness caveat all in one string so none can be read in "
        "isolation."
    )

    try:
        code = [
            {"path": "scripts/analyze_exp6221_admission_rate.py", "sha256": _sha(Path(__file__))}
        ]
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
        art["git_head"] = head
        art["provenance"] = {
            "git_head": head,
            "code": code,
            "rows_sources": {
                "cited_artifacts": [
                    {
                        "path": "results/experiment_6221_gemma_think_mode_ab_expanded_roster.json",
                        "sha256": _sha(INPUT_PATH),
                    }
                ]
            },
        }
    except Exception as exc:  # noqa: BLE001
        art["provenance"] = {"error": f"{type(exc).__name__}:{exc}"}

    art["duration_s"] = round(time.time() - t0, 3)
    art["inference_substrate"] = "aggregation_from_upstream_artifacts"
    art["inference_substrate_principle"] = (
        "reads already-persisted rows from the exp6221 artifact and recomputes a different "
        "primary metric; invokes no model and reruns no induction."
    )

    payload = json.dumps(
        {k: art[k] for k in art if k not in ("run_date", "duration_s")},
        sort_keys=True,
        default=str,
    ).encode()
    art["reproducibility_checksum"] = hashlib.sha256(payload).hexdigest()
    return art


def main() -> int:
    art = build_artifact()
    # Carry hand-authored keys through the rebuild (REQ-OPS-REBUILD-PRESERVE-1).
    import sys as _sys
    from pathlib import Path as _P

    if str(_P(__file__).resolve().parent) not in _sys.path:
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from artifact_merge_preserve import merge_preserve_with_file

    art = merge_preserve_with_file(OUT_PATH, art)
    OUT_PATH.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(art.get("headline", {"honest_verdict": art.get("honest_verdict")}), indent=2))
    print("verdict:", art["honest_verdict"])
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
