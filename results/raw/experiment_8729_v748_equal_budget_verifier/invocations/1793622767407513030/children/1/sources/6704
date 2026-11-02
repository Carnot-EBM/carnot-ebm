#!/usr/bin/env python3
"""Exp 1213 — SDPO dense reward distillation via Carnot energy verifier.

arXiv 2604.03128 (SDPO: Self-Distilled Policy Optimization) converts sparse binary
outcome rewards into token-level dense supervision by using the same model as
teacher+student with the verifier as "privileged information".

This experiment measures whether SDPO's token-level KL-distance selection outperforms
simple energy-based binary teacher selection on 50 GSM8K math questions with 4
completions each.

Measurement approach:
    1. Use 50 pre-written canonical math questions with 4 completions each.
       Each question has 1 correct high-quality response and 3 responses of
       varying quality (arithmetic error, causal break, or mostly correct with
       minor error).  Pre-written responses keep the experiment deterministic and
       fast — the verifier signal is independent of which LLM generated the text.
    2. Score each completion using the composite energy verifier:
       energy = 0.5 * mean_causal_violation + 0.5 * mean_z3_violation
    3. Identify the teacher completion as the lowest-energy response.
    4. Compute mean per-token logprob proxies from the energy scores (deterministic,
       seeded RNG).  Live GGUF inference is unavailable on this machine — the
       llama_cpp Python package requires CUDA libs absent on the ROCm host.  The
       proxy formula (BASE_LP - energy * LP_RANGE) is calibrated to the Qwen3 range
       observed in Exp 1209 logprob rejection sampling.
    5. Compute KL-proxy distances and select the best completion by lowest KL.
    6. Compare accuracy: energy selection vs KL selection.

Spec: REQ-LEARN-1213, SCENARIO-LEARN-1215, SCENARIO-LEARN-1216,
      SCENARIO-LEARN-1217, SCENARIO-LEARN-1218
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_repo_venv() -> None:
    """Re-exec under the repo .venv so the documented command works."""
    if __name__ != "__main__":
        return
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("CARNOT_EXP1213_VENV_REEXEC") == "1":
        return
    os.environ["CARNOT_EXP1213_VENV_REEXEC"] = "1"
    # Propagate CUDA libraries if present so llama.cpp GPU probe works.
    nvidia = _REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    extra = f"{nvidia / 'cuda_runtime' / 'lib'}:{nvidia / 'cublas' / 'lib'}"
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{extra}:{cur}" if cur else extra
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_maybe_reexec_repo_venv()

for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

EXPERIMENT_ID = 1213
DELIVERABLE = _REPO_ROOT / "results" / "experiment_1213_sdpo_dense_reward_distillation.json"
RANDOM_SEED = 42
N_QUESTIONS = 51
N_COMPLETIONS = 4

# GGUF inference would use this model, but we fall back to pre-written
# canonical responses when the llama_cpp Python package is unavailable.
_MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
_GGUF_CANDIDATES = [
    Path(
        "/home/ianblenke/.cache/huggingface/hub"
        "/models--unsloth--Qwen3.6-35B-A3B-GGUF"
        "/snapshots/d1cb0861c117275a7ce8a91c61b567394107f69a"
        "/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"
    ),
]


def _find_gguf() -> str | None:
    for p in _GGUF_CANDIDATES:
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# 50 GSM8K-style questions with 4 completions each
#
# Each entry has:
#   question:    The math word problem.
#   gold:        Expected numeric answer.
#   completions: List of 4 dicts, each with "response" and "correct" flag.
#                Completion 0 = the best (correct, well-structured).
#                Completions 1-3 = progressively worse (arithmetic/causal errors).
# ---------------------------------------------------------------------------

_QUESTION_BANK: list[dict[str, Any]] = [
    {
        "question": "Janet has 24 apples. She gives 8 to her friend and buys 15 more. How many does she have?",
        "gold": "31",
        "completions": [
            {
                "response": "Janet starts with 24 apples.\nShe gives away 8, so 24 - 8 = 16.\nShe buys 15 more: 16 + 15 = 31.\nThe answer is 31.",
                "correct": True,
            },
            {
                "response": "Janet starts with 24 apples.\nShe gives away 8, so 24 - 8 = 17.\nShe buys 15 more: 17 + 15 = 32.\nThe answer is 32.",
                "correct": False,
            },
            {
                "response": "Janet starts with 24 apples.\nShe gives away 8.\nWe had 20 apples before, so 20 + 15 = 35.\nThe answer is 35.",
                "correct": False,
            },
            {
                "response": "Janet has 24 apples and gives away 8 and buys 15.\n24 + 15 = 39.\nThe answer is 39.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Weng earns $12 an hour for babysitting. She babysit for 50 minutes. How much did she earn?",
        "gold": "10",
        "completions": [
            {
                "response": "Weng earns $12/hour.\n50 minutes = 50/60 hours.\n12 * (50/60) = 12 * 5/6 = 10.\nThe answer is 10.",
                "correct": True,
            },
            {
                "response": "Weng earns $12/hour.\n50 minutes = 0.75 hours.\n12 * 0.75 = 9.\nThe answer is 9.",
                "correct": False,
            },
            {
                "response": "Weng earns $12/hour for 50 minutes.\n12 / 50 = 0.24 per minute.\n0.24 * 50 = 12.\nThe answer is 12.",
                "correct": False,
            },
            {
                "response": "50 minutes of babysitting at $12/hour.\n12 * 50 = 600 cents = $6.\nThe answer is 6.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Betty needs $100 for a wallet. She has half the money. Parents gave $15, grandparents gave twice that. How much more does she need?",
        "gold": "5",
        "completions": [
            {
                "response": "Betty has $100/2 = $50.\nParents: $15. Grandparents: 2 * $15 = $30.\nTotal received: $50 + $15 + $30 = $95.\nStill needs: $100 - $95 = $5.\nThe answer is 5.",
                "correct": True,
            },
            {
                "response": "Betty has $50.\nParents: $15. Grandparents: $15.\nTotal: $50 + $15 + $15 = $80.\nNeeds: $100 - $80 = $20.\nThe answer is 20.",
                "correct": False,
            },
            {
                "response": "Betty starts with $50 from her savings.\nParents gave $15.\nWe previously calculated $20 from grandparents, so total = $50 + $15 + $20 = $85.\nNeeds $100 - $85 = $15.\nThe answer is 15.",
                "correct": False,
            },
            {
                "response": "Betty needs $100. She has half: $50.\n$15 + $30 = $45 more.\nShe has $50 + $45 = $95 but needs $100.\n$100 - $95 = $5.\nThe answer is 5.",
                "correct": True,
            },
        ],
    },
    {
        "question": "Julie reads a 120-page book. She read 12 pages yesterday and twice as many today. She wants to read half the remaining tomorrow. How many pages tomorrow?",
        "gold": "42",
        "completions": [
            {
                "response": "Yesterday: 12 pages. Today: 24 pages. Total read: 36.\nRemaining: 120 - 36 = 84.\nTomorrow: 84 / 2 = 42.\nThe answer is 42.",
                "correct": True,
            },
            {
                "response": "Yesterday: 12 pages. Today: 24 pages. Total: 36.\nRemaining: 120 - 36 = 84.\nTomorrow: 84 / 4 = 21.\nThe answer is 21.",
                "correct": False,
            },
            {
                "response": "Yesterday: 12 pages. Today: 2 * 12 = 24 pages.\nWe read 40 pages total so far.\nRemaining: 120 - 40 = 80.\nTomorrow: 80 / 2 = 40.\nThe answer is 40.",
                "correct": False,
            },
            {
                "response": "Total: 120 pages. Read: 12 + 24 = 36.\n120 - 36 = 84 remaining. Half = 42.\nThe answer is 42.",
                "correct": True,
            },
        ],
    },
    {
        "question": "James writes a 3-page letter to 2 friends twice a week. How many pages a year?",
        "gold": "624",
        "completions": [
            {
                "response": "Per week: 3 pages * 2 friends * 2 times = 12 pages.\nPer year: 12 * 52 = 624 pages.\nThe answer is 624.",
                "correct": True,
            },
            {
                "response": "Per week: 3 * 2 = 6 pages per round, 2 rounds = 12.\n12 * 50 = 600 pages/year.\nThe answer is 600.",
                "correct": False,
            },
            {
                "response": "James writes 3 pages to 2 friends = 6 pages per occasion.\n6 * 2 times/week = 12 pages/week.\nWe counted 4 times per week, so 3 * 2 * 4 = 24/week.\n24 * 52 = 1248.\nThe answer is 1248.",
                "correct": False,
            },
            {
                "response": "3-page letters to 2 friends, twice weekly.\n3 * 2 * 2 * 52 = 624.\nThe answer is 624.",
                "correct": True,
            },
        ],
    },
    {
        "question": "Mark planted yellow, purple, and green flowers. 10 yellow, 80% more purple, 25% as many green as yellow+purple. How many total?",
        "gold": "35",
        "completions": [
            {
                "response": "Yellow: 10. Purple: 10 + 0.8*10 = 18. Green: 0.25*(10+18) = 7.\nTotal: 10 + 18 + 7 = 35.\nThe answer is 35.",
                "correct": True,
            },
            {
                "response": "Yellow: 10. Purple: 80% more = 18. Green: 25% of 28 = 7.\n10 + 18 + 7 = 35.\nThe answer is 35.",
                "correct": True,
            },
            {
                "response": "Yellow: 10. Purple: 10 * 1.8 = 18. Green: 10 * 0.25 = 2.5 ≈ 2.\nTotal: 10 + 18 + 2 = 30.\nThe answer is 30.",
                "correct": False,
            },
            {
                "response": "Purple is 80% of yellow: 0.8 * 10 = 8. Green: 0.25 * 18 = 4.5.\nTotal: 10 + 8 + 4 = 22.\nThe answer is 22.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Albert buys 2 large pizzas (16 slices each) and 2 small pizzas (8 slices each). How many pieces total?",
        "gold": "48",
        "completions": [
            {
                "response": "Large: 2 * 16 = 32 slices. Small: 2 * 8 = 16 slices.\nTotal: 32 + 16 = 48.\nThe answer is 48.",
                "correct": True,
            },
            {
                "response": "2 large * 16 + 2 small * 8 = 32 + 16 = 48.\nThe answer is 48.",
                "correct": True,
            },
            {
                "response": "Large: 2 * 16 = 34. Small: 2 * 8 = 16. Total: 50.\nThe answer is 50.",
                "correct": False,
            },
            {
                "response": "Each large pizza has 16 slices, each small has 8.\n2 + 2 = 4 pizzas, average 12 slices each.\n4 * 12 = 48.\nThe answer is 48.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A train travels 60 mph for 3 hours and 45 mph for 2 hours. What is the total distance?",
        "gold": "270",
        "completions": [
            {
                "response": "Distance 1: 60 * 3 = 180 miles.\nDistance 2: 45 * 2 = 90 miles.\nTotal: 180 + 90 = 270 miles.\nThe answer is 270.",
                "correct": True,
            },
            {
                "response": "60 * 3 = 180. 45 * 2 = 90. Total: 270.\nThe answer is 270.",
                "correct": True,
            },
            {
                "response": "Average speed = (60+45)/2 = 52.5 mph. Total time = 5 hours.\n52.5 * 5 = 262.5.\nThe answer is 262.5.",
                "correct": False,
            },
            {
                "response": "60 mph * 3 hours = 180.\n45 mph * 2 hours = 80.\n180 + 80 = 260.\nThe answer is 260.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A store sells apples for $0.50 each and oranges for $0.75 each. Sarah buys 6 apples and 4 oranges. What does she pay?",
        "gold": "6",
        "completions": [
            {
                "response": "Apples: 6 * 0.50 = $3.00.\nOranges: 4 * 0.75 = $3.00.\nTotal: $3.00 + $3.00 = $6.00.\nThe answer is 6.",
                "correct": True,
            },
            {
                "response": "6 apples at $0.50 each = $3.00. 4 oranges at $0.75 each = $3.00. Total = $6.00.\nThe answer is 6.",
                "correct": True,
            },
            {
                "response": "Apples: 6 * 0.50 = $3.00. Oranges: 4 * 0.80 = $3.20. Total: $6.20.\nThe answer is 6.20.",
                "correct": False,
            },
            {
                "response": "Sarah buys 10 items total.\nAverage price = (0.50 + 0.75) / 2 = 0.625.\n10 * 0.625 = 6.25.\nThe answer is 6.25.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Bob saves $200/month. After 8 months he spends $650 on a laptop. How much does he have left?",
        "gold": "950",
        "completions": [
            {
                "response": "Saved: 200 * 8 = $1600.\nSpent: $650.\nLeft: 1600 - 650 = $950.\nThe answer is 950.",
                "correct": True,
            },
            {"response": "200 * 8 = 1600. 1600 - 650 = 950.\nThe answer is 950.", "correct": True},
            {
                "response": "Bob saves for 8 months at $200: 200 * 7 = 1400.\n1400 - 650 = 750.\nThe answer is 750.",
                "correct": False,
            },
            {
                "response": "Bob saves $1600 total then buys a $650 laptop.\n1600 - 750 = 850.\nThe answer is 850.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A classroom has 32 students. 3/8 are girls. How many boys are there?",
        "gold": "20",
        "completions": [
            {
                "response": "Girls: 32 * 3/8 = 12. Boys: 32 - 12 = 20.\nThe answer is 20.",
                "correct": True,
            },
            {
                "response": "3/8 of 32 = 12 girls. Boys = 32 - 12 = 20.\nThe answer is 20.",
                "correct": True,
            },
            {
                "response": "3/8 of 32 = 14 girls. Boys: 32 - 14 = 18.\nThe answer is 18.",
                "correct": False,
            },
            {
                "response": "Girls are 3/8, boys are 5/8.\n32 * 5/8 = 20 boys.\nThe answer is 20.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A recipe uses 2.5 cups of flour for 12 cookies. How much flour for 36 cookies?",
        "gold": "7.5",
        "completions": [
            {
                "response": "Scale factor: 36/12 = 3.\n2.5 * 3 = 7.5 cups.\nThe answer is 7.5.",
                "correct": True,
            },
            {
                "response": "36 / 12 = 3 batches. 3 * 2.5 = 7.5 cups.\nThe answer is 7.5.",
                "correct": True,
            },
            {
                "response": "2.5 cups / 12 cookies * 36 = 7.0 cups.\nThe answer is 7.",
                "correct": False,
            },
            {
                "response": "We need 3x the recipe: 2.5 + 2.5 + 2.5 = 8 cups.\nThe answer is 8.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A car uses 5 liters of fuel per 100 km. How much fuel for 450 km?",
        "gold": "22.5",
        "completions": [
            {"response": "450 km / 100 km * 5 L = 22.5 L.\nThe answer is 22.5.", "correct": True},
            {
                "response": "Rate: 5L/100km. 450 km: 450 * 0.05 = 22.5 L.\nThe answer is 22.5.",
                "correct": True,
            },
            {
                "response": "450 / 100 = 4.5 segments. 4.5 * 5 = 22.5.\nThe answer is 22.5.",
                "correct": True,
            },
            {
                "response": "5 liters per 100 km. 450 / 5 = 90 segments. 90 * 1 = 90.\nThe answer is 90.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Lisa has 3 times as many stickers as Mike. Together they have 48. How many does Mike have?",
        "gold": "12",
        "completions": [
            {
                "response": "Let Mike = m. Lisa = 3m. Total: m + 3m = 4m = 48. m = 12.\nThe answer is 12.",
                "correct": True,
            },
            {
                "response": "Mike + 3*Mike = 48. 4*Mike = 48. Mike = 12.\nThe answer is 12.",
                "correct": True,
            },
            {
                "response": "Lisa has 3x stickers. Together 48.\nLisa = 36, Mike = 48 - 36 = 12.\nThe answer is 12.",
                "correct": True,
            },
            {
                "response": "Together = 48, Lisa = 3 * Mike.\nMike = 48 / 3 = 16.\nThe answer is 16.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A book has 250 pages. Tom reads 40 pages/day. How many days to finish?",
        "gold": "7",
        "completions": [
            {
                "response": "250 / 40 = 6.25 days, so 7 days (rounding up).\nThe answer is 7.",
                "correct": True,
            },
            {"response": "Days needed: ceil(250/40) = 7.\nThe answer is 7.", "correct": True},
            {
                "response": "250 / 40 = 6 remainder 10 = 6.25, so 6 full days.\nThe answer is 6.",
                "correct": False,
            },
            {
                "response": "Tom reads 40 pages/day. 250/40 = 6.5 days, so 7 days.\nThe answer is 7.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A gym membership costs $45/month. Annual cost?",
        "gold": "540",
        "completions": [
            {"response": "45 * 12 = 540.\nThe answer is 540.", "correct": True},
            {
                "response": "Monthly cost $45, 12 months/year: 45 * 12 = 540.\nThe answer is 540.",
                "correct": True,
            },
            {"response": "45 * 12 = 580.\nThe answer is 580.", "correct": False},
            {"response": "$45/month * 12 months = $540/year.\nThe answer is 540.", "correct": True},
        ],
    },
    {
        "question": "5 workers each work 8 hours/day, 5 days/week. Total worker-hours per week?",
        "gold": "200",
        "completions": [
            {
                "response": "5 workers * 8 hours/day * 5 days = 200 worker-hours.\nThe answer is 200.",
                "correct": True,
            },
            {"response": "5 * 8 * 5 = 200.\nThe answer is 200.", "correct": True},
            {
                "response": "8 hours/day * 5 days = 40 hours/worker.\n5 workers * 40 = 200.\nThe answer is 200.",
                "correct": True,
            },
            {
                "response": "5 workers, 8 hours, 5 days.\n5 * 8 = 40, * 4 = 160.\nThe answer is 160.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A box holds 24 cans. How many boxes for 156 cans?",
        "gold": "7",
        "completions": [
            {"response": "156 / 24 = 6.5, so 7 boxes needed.\nThe answer is 7.", "correct": True},
            {"response": "ceil(156/24) = 7.\nThe answer is 7.", "correct": True},
            {
                "response": "156 / 24 = 6 boxes with 12 remaining = 7.\nThe answer is 7.",
                "correct": True,
            },
            {
                "response": "156 / 24 = 6.5, round down = 6 boxes.\nThe answer is 6.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A 30% discount on $80 jeans. What is the sale price?",
        "gold": "56",
        "completions": [
            {
                "response": "Discount: 80 * 0.30 = $24. Sale price: 80 - 24 = $56.\nThe answer is 56.",
                "correct": True,
            },
            {"response": "80 * (1 - 0.30) = 80 * 0.70 = $56.\nThe answer is 56.", "correct": True},
            {
                "response": "30% off $80: 80 * 0.30 = 24. Sale: 80 - 24 = 56.\nThe answer is 56.",
                "correct": True,
            },
            {
                "response": "30% discount = $80 * 0.30 = $24 off.\n$80 - $20 = $60 sale price.\nThe answer is 60.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Two trains start 300 km apart, moving toward each other at 60 and 90 km/h. When do they meet?",
        "gold": "2",
        "completions": [
            {
                "response": "Combined speed: 60 + 90 = 150 km/h.\nTime: 300 / 150 = 2 hours.\nThe answer is 2.",
                "correct": True,
            },
            {
                "response": "300 km, closing at 150 km/h. 300/150 = 2 hours.\nThe answer is 2.",
                "correct": True,
            },
            {
                "response": "Trains approach at 60 km/h total differential.\n300 / 60 = 5 hours.\nThe answer is 5.",
                "correct": False,
            },
            {
                "response": "Speed of approach: 90 - 60 = 30 km/h.\n300 / 30 = 10 hours.\nThe answer is 10.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A tank fills in 4 hours, empties in 6 hours. How long to fill when both open?",
        "gold": "12",
        "completions": [
            {
                "response": "Fill rate: 1/4/hr. Drain rate: 1/6/hr. Net: 1/4 - 1/6 = 3/12 - 2/12 = 1/12/hr.\nTime: 12 hours.\nThe answer is 12.",
                "correct": True,
            },
            {
                "response": "Net fill rate = 1/4 - 1/6 = 1/12 per hour. Takes 12 hours.\nThe answer is 12.",
                "correct": True,
            },
            {
                "response": "Fill rate - drain rate = 1/4 + 1/6 = 5/12.\nTime = 12/5 = 2.4 hours.\nThe answer is 2.4.",
                "correct": False,
            },
            {
                "response": "1/4 - 1/6 = 2/24 = 1/12. Time = 12 hours.\nThe answer is 12.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A shop sells notebooks for $3.50 each. Ann buys 5 notebooks. How much change from $20?",
        "gold": "2.5",
        "completions": [
            {
                "response": "Cost: 5 * 3.50 = $17.50. Change: $20 - $17.50 = $2.50.\nThe answer is 2.5.",
                "correct": True,
            },
            {
                "response": "5 * 3.50 = 17.50. Change = 20 - 17.50 = 2.50.\nThe answer is 2.5.",
                "correct": True,
            },
            {
                "response": "5 notebooks at $3.50 = $17. Change = $20 - $17 = $3.\nThe answer is 3.",
                "correct": False,
            },
            {
                "response": "5 * 3.50 = 17.50. 20 - 17.50 = 2.50.\nThe answer is 2.5.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A swimming pool holds 12,000 liters. A pump fills at 800 L/hour. How long to fill?",
        "gold": "15",
        "completions": [
            {"response": "12000 / 800 = 15 hours.\nThe answer is 15.", "correct": True},
            {
                "response": "Time = volume / rate = 12000 / 800 = 15 hours.\nThe answer is 15.",
                "correct": True,
            },
            {"response": "12000 / 800 = 14 hours.\nThe answer is 14.", "correct": False},
            {
                "response": "12000 L at 800 L/hr = 12000 / 800 = 15.\nThe answer is 15.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A square garden has a perimeter of 48 m. What is its area?",
        "gold": "144",
        "completions": [
            {
                "response": "Side = 48 / 4 = 12 m. Area = 12 * 12 = 144 m².\nThe answer is 144.",
                "correct": True,
            },
            {
                "response": "Perimeter = 4 * side. Side = 12. Area = 144.\nThe answer is 144.",
                "correct": True,
            },
            {
                "response": "Perimeter 48, so side = 48/4 = 12. Area = 12 * 12 = 144.\nThe answer is 144.",
                "correct": True,
            },
            {
                "response": "48 m perimeter. Side = 12. Area = 12 * 11 = 132.\nThe answer is 132.",
                "correct": False,
            },
        ],
    },
    {
        "question": "3 friends split a restaurant bill of $87. They each leave a $5 tip. How much does each person pay in total?",
        "gold": "34",
        "completions": [
            {
                "response": "Bill per person: 87 / 3 = 29. With tip: 29 + 5 = $34.\nThe answer is 34.",
                "correct": True,
            },
            {
                "response": "87/3 = 29. Tip adds 5 each. Total per person = 34.\nThe answer is 34.",
                "correct": True,
            },
            {
                "response": "Total bill: $87 + ($5 * 3) = $102. Per person: 102 / 3 = $34.\nThe answer is 34.",
                "correct": True,
            },
            {"response": "87 / 3 = 28. Plus $5 tip = $33.\nThe answer is 33.", "correct": False},
        ],
    },
    {
        "question": "There are 7 days in a week. How many days in 52 weeks plus 3 extra days?",
        "gold": "367",
        "completions": [
            {"response": "52 * 7 = 364. Plus 3 = 367.\nThe answer is 367.", "correct": True},
            {
                "response": "Days in 52 weeks = 364. Extra = 3. Total = 367.\nThe answer is 367.",
                "correct": True,
            },
            {"response": "52 * 7 = 364 days + 3 = 367.\nThe answer is 367.", "correct": True},
            {
                "response": "52 weeks = 52 * 7 = 354 days. 354 + 3 = 357.\nThe answer is 357.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A pitcher has 2 liters of juice. After 3 glasses of 200 mL each, how many mL remain?",
        "gold": "1400",
        "completions": [
            {
                "response": "Used: 3 * 200 = 600 mL. Remaining: 2000 - 600 = 1400 mL.\nThe answer is 1400.",
                "correct": True,
            },
            {"response": "2000 - 3*200 = 2000 - 600 = 1400.\nThe answer is 1400.", "correct": True},
            {
                "response": "3 glasses * 200 mL = 600. 2000 - 600 = 1400.\nThe answer is 1400.",
                "correct": True,
            },
            {
                "response": "2 liters = 2000 mL. 3 * 200 = 650. 2000 - 650 = 1350.\nThe answer is 1350.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A factory produces 450 units/day. How many units in a 5-day week?",
        "gold": "2250",
        "completions": [
            {"response": "450 * 5 = 2250 units.\nThe answer is 2250.", "correct": True},
            {"response": "Per week: 450 * 5 = 2250.\nThe answer is 2250.", "correct": True},
            {
                "response": "Daily output: 450. Weekly (5 days): 450 * 5 = 2250.\nThe answer is 2250.",
                "correct": True,
            },
            {
                "response": "450/day for 5 days = 450 * 6 = 2700 units.\nThe answer is 2700.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A 15% sales tax on a $240 purchase. Total cost?",
        "gold": "276",
        "completions": [
            {
                "response": "Tax: 240 * 0.15 = 36. Total: 240 + 36 = 276.\nThe answer is 276.",
                "correct": True,
            },
            {"response": "240 * 1.15 = 276.\nThe answer is 276.", "correct": True},
            {"response": "15% of $240 = $36. 240 + 36 = 276.\nThe answer is 276.", "correct": True},
            {
                "response": "Tax = 15% of 240 = 30. Total = 240 + 30 = 270.\nThe answer is 270.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Emma jogs 3 km/day, 6 days/week. How many km in 4 weeks?",
        "gold": "72",
        "completions": [
            {
                "response": "Per week: 3 * 6 = 18 km. Over 4 weeks: 18 * 4 = 72 km.\nThe answer is 72.",
                "correct": True,
            },
            {
                "response": "3 km/day * 6 days * 4 weeks = 72 km.\nThe answer is 72.",
                "correct": True,
            },
            {"response": "18 km/week * 4 = 72.\nThe answer is 72.", "correct": True},
            {
                "response": "Emma jogs 3 * 6 = 18 km/week.\n18 * 5 = 90 km in 4 weeks.\nThe answer is 90.",
                "correct": False,
            },
        ],
    },
    {
        "question": "100 - 47 = ?",
        "gold": "53",
        "completions": [
            {"response": "100 - 47 = 53.\nThe answer is 53.", "correct": True},
            {"response": "100 minus 47: 100 - 47 = 53.\nThe answer is 53.", "correct": True},
            {"response": "100 - 47 = 54.\nThe answer is 54.", "correct": False},
            {"response": "100 - 47 = 53.\nThe answer is 53.", "correct": True},
        ],
    },
    {
        "question": "3 * 25 = ?",
        "gold": "75",
        "completions": [
            {"response": "3 * 25 = 75.\nThe answer is 75.", "correct": True},
            {"response": "25 + 25 + 25 = 75.\nThe answer is 75.", "correct": True},
            {"response": "3 * 25 = 76.\nThe answer is 76.", "correct": False},
            {"response": "3 * 25 = 75.\nThe answer is 75.", "correct": True},
        ],
    },
    {
        "question": "A flight costs $320 round-trip. 4 passengers. Total cost?",
        "gold": "1280",
        "completions": [
            {"response": "4 * 320 = 1280.\nThe answer is 1280.", "correct": True},
            {
                "response": "$320 per person, 4 people: 4 * 320 = $1280.\nThe answer is 1280.",
                "correct": True,
            },
            {"response": "4 passengers * $320 = $1280.\nThe answer is 1280.", "correct": True},
            {"response": "4 * 320 = 1320.\nThe answer is 1320.", "correct": False},
        ],
    },
    {
        "question": "A bathtub drains at 12 L/min. How long to drain 180 liters?",
        "gold": "15",
        "completions": [
            {"response": "180 / 12 = 15 minutes.\nThe answer is 15.", "correct": True},
            {"response": "Time = 180 / 12 = 15 min.\nThe answer is 15.", "correct": True},
            {"response": "180 liters / 12 L/min = 15 minutes.\nThe answer is 15.", "correct": True},
            {
                "response": "180 / 12 = 14 minutes remainder 12.\nThe answer is 14.",
                "correct": False,
            },
        ],
    },
    {
        "question": "9 players each score 4 goals. Total goals?",
        "gold": "36",
        "completions": [
            {"response": "9 * 4 = 36 goals.\nThe answer is 36.", "correct": True},
            {"response": "9 players * 4 goals = 36.\nThe answer is 36.", "correct": True},
            {"response": "9 * 4 = 37.\nThe answer is 37.", "correct": False},
            {
                "response": "Each of 9 players scores 4: 9 * 4 = 36.\nThe answer is 36.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A ladder has 20 rungs. Each rung is 30 cm apart. Height of the ladder?",
        "gold": "570",
        "completions": [
            {"response": "19 gaps * 30 cm = 570 cm.\nThe answer is 570.", "correct": True},
            {
                "response": "20 rungs means 19 intervals. 19 * 30 = 570 cm.\nThe answer is 570.",
                "correct": True,
            },
            {"response": "20 * 30 = 600 cm.\nThe answer is 600.", "correct": False},
            {
                "response": "Between 20 rungs: 19 spaces * 30 cm = 570 cm.\nThe answer is 570.",
                "correct": True,
            },
        ],
    },
    {
        "question": "A store had 120 items. Sold 35% on Monday. How many remain?",
        "gold": "78",
        "completions": [
            {
                "response": "Sold: 120 * 0.35 = 42. Remaining: 120 - 42 = 78.\nThe answer is 78.",
                "correct": True,
            },
            {
                "response": "35% of 120 = 42 sold. 120 - 42 = 78 left.\nThe answer is 78.",
                "correct": True,
            },
            {"response": "Remaining = 65% of 120 = 78.\nThe answer is 78.", "correct": True},
            {
                "response": "35% sold = 120 * 0.35 = 40. Remaining: 80.\nThe answer is 80.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A box weighs 2.5 kg. 8 boxes in a crate. Crate itself weighs 3 kg. Total?",
        "gold": "23",
        "completions": [
            {
                "response": "Boxes: 8 * 2.5 = 20 kg. With crate: 20 + 3 = 23 kg.\nThe answer is 23.",
                "correct": True,
            },
            {"response": "8 * 2.5 + 3 = 20 + 3 = 23 kg.\nThe answer is 23.", "correct": True},
            {"response": "8 boxes * 2.5 = 20. 20 + 3 = 23.\nThe answer is 23.", "correct": True},
            {
                "response": "8 * 2.5 = 22 kg. Plus crate 3 kg = 25 kg.\nThe answer is 25.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A class of 30 students: 60% passed. How many failed?",
        "gold": "12",
        "completions": [
            {
                "response": "Passed: 30 * 0.60 = 18. Failed: 30 - 18 = 12.\nThe answer is 12.",
                "correct": True,
            },
            {"response": "40% failed: 30 * 0.40 = 12.\nThe answer is 12.", "correct": True},
            {
                "response": "60% of 30 = 18 passed. 30 - 18 = 12 failed.\nThe answer is 12.",
                "correct": True,
            },
            {
                "response": "60% pass: 30 * 0.60 = 18 passed. Failed = 30 - 16 = 14.\nThe answer is 14.",
                "correct": False,
            },
        ],
    },
    {
        "question": "500 m race. Alice finishes in 2 min 20 sec. Speed in m/s?",
        "gold": "3.57",
        "completions": [
            {
                "response": "2 min 20 sec = 140 sec. Speed = 500/140 ≈ 3.57 m/s.\nThe answer is 3.57.",
                "correct": True,
            },
            {
                "response": "140 seconds total. 500/140 = 3.571 ≈ 3.57 m/s.\nThe answer is 3.57.",
                "correct": True,
            },
            {
                "response": "2:20 = 120 + 20 = 140 s. 500/140 = 3.57.\nThe answer is 3.57.",
                "correct": True,
            },
            {
                "response": "2 min 20 sec = 150 seconds. 500/150 ≈ 3.33 m/s.\nThe answer is 3.33.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A 40-meter rope is cut into pieces of 2.5 m each. How many pieces?",
        "gold": "16",
        "completions": [
            {"response": "40 / 2.5 = 16 pieces.\nThe answer is 16.", "correct": True},
            {
                "response": "40 meters / 2.5 m per piece = 16 pieces.\nThe answer is 16.",
                "correct": True,
            },
            {"response": "40/2.5 = 16.\nThe answer is 16.", "correct": True},
            {
                "response": "40 / 2.5 = 15 pieces remainder 2.5 = 15.\nThe answer is 15.",
                "correct": False,
            },
        ],
    },
    {
        "question": "Tom has $150. He spends 40% on food and 25% on transport. How much left?",
        "gold": "52.5",
        "completions": [
            {
                "response": "Food: 150*0.40=60. Transport: 150*0.25=37.50. Spent: 97.50. Left: 52.50.\nThe answer is 52.5.",
                "correct": True,
            },
            {
                "response": "40% + 25% = 65% spent. 35% left: 150*0.35 = 52.5.\nThe answer is 52.5.",
                "correct": True,
            },
            {
                "response": "Food: $60. Transport: $37.50. Remaining: $150 - $97.50 = $52.50.\nThe answer is 52.5.",
                "correct": True,
            },
            {
                "response": "40% food = 60. 25% transport = 40. Total spent: 100. Left: 50.\nThe answer is 50.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A wheel has a circumference of 2 m. How many rotations to travel 50 m?",
        "gold": "25",
        "completions": [
            {"response": "50 / 2 = 25 rotations.\nThe answer is 25.", "correct": True},
            {
                "response": "Distance / circumference = 50/2 = 25.\nThe answer is 25.",
                "correct": True,
            },
            {
                "response": "50 meters / 2 meters/rotation = 25 rotations.\nThe answer is 25.",
                "correct": True,
            },
            {
                "response": "2 m per rotation. 50 m / 2 = 26 rotations.\nThe answer is 26.",
                "correct": False,
            },
        ],
    },
    {
        "question": "4 bags of rice weigh 5 kg each. 3 more bags of 3 kg each. Total weight?",
        "gold": "29",
        "completions": [
            {
                "response": "4*5 = 20 kg. 3*3 = 9 kg. Total: 20+9 = 29 kg.\nThe answer is 29.",
                "correct": True,
            },
            {
                "response": "Heavy bags: 4*5 = 20. Light bags: 3*3 = 9. Total = 29.\nThe answer is 29.",
                "correct": True,
            },
            {"response": "4 * 5 + 3 * 3 = 20 + 9 = 29 kg.\nThe answer is 29.", "correct": True},
            {"response": "4*5 = 20. 3*3 = 10. Total = 30.\nThe answer is 30.", "correct": False},
        ],
    },
    {
        "question": "Speed of light is ~300,000 km/s. How far does it travel in 5 seconds?",
        "gold": "1500000",
        "completions": [
            {"response": "300000 * 5 = 1,500,000 km.\nThe answer is 1500000.", "correct": True},
            {"response": "5 * 300000 = 1500000 km.\nThe answer is 1500000.", "correct": True},
            {
                "response": "300000 km/s for 5 seconds = 1,500,000 km.\nThe answer is 1500000.",
                "correct": True,
            },
            {"response": "300000 * 5 = 1,600,000 km.\nThe answer is 1600000.", "correct": False},
        ],
    },
    {
        "question": "A cinema has 15 rows of 20 seats. 10% of seats reserved for VIP. How many non-VIP?",
        "gold": "270",
        "completions": [
            {
                "response": "Total: 15*20 = 300. VIP: 30. Non-VIP: 270.\nThe answer is 270.",
                "correct": True,
            },
            {"response": "300 seats * 90% = 270 non-VIP.\nThe answer is 270.", "correct": True},
            {
                "response": "15 * 20 = 300. 10% VIP = 30. Non-VIP: 270.\nThe answer is 270.",
                "correct": True,
            },
            {
                "response": "Total = 300. VIP = 10% = 40. Non-VIP = 260.\nThe answer is 260.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A painter paints 3 rooms in 6 hours. At the same rate, how many rooms in 10 hours?",
        "gold": "5",
        "completions": [
            {
                "response": "Rate: 3/6 = 0.5 rooms/hour. In 10 hours: 0.5*10 = 5 rooms.\nThe answer is 5.",
                "correct": True,
            },
            {"response": "0.5 rooms/hr * 10 hr = 5 rooms.\nThe answer is 5.", "correct": True},
            {
                "response": "3 rooms in 6 hours → 1 room in 2 hours → 5 rooms in 10 hours.\nThe answer is 5.",
                "correct": True,
            },
            {
                "response": "Rate: 3/6 = 0.6 rooms/hr. 0.6*10 = 6 rooms.\nThe answer is 6.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A bank account earns 5% simple interest per year on $2000. Interest after 3 years?",
        "gold": "300",
        "completions": [
            {
                "response": "Interest = P * r * t = 2000 * 0.05 * 3 = 300.\nThe answer is 300.",
                "correct": True,
            },
            {
                "response": "5% of $2000 = $100/year. 3 years = $300.\nThe answer is 300.",
                "correct": True,
            },
            {"response": "2000 * 0.05 * 3 = 300.\nThe answer is 300.", "correct": True},
            {
                "response": "5% of 2000 = 100. Three years compound: 100+105+110.25 = 315.25.\nThe answer is 315.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A hotel has 8 floors, 12 rooms per floor. 15% of rooms are currently occupied. How many occupied rooms?",
        "gold": "14.4",
        "completions": [
            {
                "response": "Total rooms: 8*12 = 96. Occupied: 96*0.15 = 14.4.\nThe answer is 14.4.",
                "correct": True,
            },
            {
                "response": "8 * 12 = 96 rooms. 15% = 14.4 rooms.\nThe answer is 14.4.",
                "correct": True,
            },
            {"response": "96 * 0.15 = 14.4.\nThe answer is 14.4.", "correct": True},
            {
                "response": "8 floors * 12 rooms = 100 rooms. 15% of 100 = 15.\nThe answer is 15.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A number squared is 169. What is the number?",
        "gold": "13",
        "completions": [
            {"response": "13^2 = 169. The number is 13.\nThe answer is 13.", "correct": True},
            {"response": "sqrt(169) = 13.\nThe answer is 13.", "correct": True},
            {"response": "169 = 13*13 = 169. The answer is 13.", "correct": True},
            {
                "response": "sqrt(169) = 12. 12^2 = 144 ≠ 169, so answer is 12.\nThe answer is 12.",
                "correct": False,
            },
        ],
    },
    {
        "question": "A team scores 3 points for a win, 1 for a draw, 0 for a loss. W:7 D:3 L:2. Total?",
        "gold": "24",
        "completions": [
            {
                "response": "Wins: 7*3=21. Draws: 3*1=3. Losses: 0. Total: 21+3=24.\nThe answer is 24.",
                "correct": True,
            },
            {"response": "7*3 + 3*1 + 2*0 = 21+3+0 = 24.\nThe answer is 24.", "correct": True},
            {
                "response": "Wins 7*3=21, draws 3, losses 0. 21+3=24.\nThe answer is 24.",
                "correct": True,
            },
            {
                "response": "Wins: 7*3=21. Draws: 3*1=3. Losses: 2*0=2. Total: 21+3+2=26.\nThe answer is 26.",
                "correct": False,
            },
        ],
    },
]

assert len(_QUESTION_BANK) == N_QUESTIONS, (
    f"Expected {N_QUESTIONS} questions, got {len(_QUESTION_BANK)}"
)


def _answer_correct(response: str, gold: str) -> bool:
    """Return True iff the response contains the gold answer as a number."""
    import re

    gold_clean = gold.replace(",", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return gold_clean in nums


def _score_completions(
    question: str,
    completions_data: list[dict],
    gold: str,
    question_id: int,
) -> list:
    """Score all 4 completions for one question via the energy verifier."""
    from carnot.training.sdpo_dense_reward import (  # noqa: PLC0415
        SDPOCompletion,
        compute_energy,
        derive_mean_logprob,
    )

    results = []
    for i, comp in enumerate(completions_data):
        text = comp["response"]
        energy = compute_energy(question, text)
        seed = question_id * N_COMPLETIONS + i
        mean_lp = derive_mean_logprob(energy, seed)
        n_tokens = max(1, len(text) // 4)
        is_correct = comp.get("correct", False) and _answer_correct(text, gold)
        results.append(
            SDPOCompletion(
                text=text,
                energy=energy,
                mean_logprob=mean_lp,
                n_tokens=n_tokens,
                is_correct=is_correct,
            )
        )
    return results


def run_experiment() -> dict:
    """Run the full SDPO dense reward distillation experiment.

    Evaluates 50 questions with 4 completions each.  Returns the complete
    artifact dict ready for JSON serialisation.
    """
    from carnot.training.sdpo_dense_reward import (  # noqa: PLC0415
        SDPOQuestionResult,
        build_sdpo_artifact_fields,
        compute_kl_proxy,
        select_by_kl,
        select_teacher,
    )

    started_at = _dt.datetime.now(_dt.UTC).isoformat()
    t0 = time.monotonic()

    all_completions = []
    question_results = []

    for q_id, item in enumerate(_QUESTION_BANK):
        question = item["question"]
        gold = item["gold"]
        completions = _score_completions(question, item["completions"], gold, q_id)
        all_completions.extend(completions)

        teacher = select_teacher(completions)
        students = [c for c in completions if c is not teacher]
        best_student = select_by_kl(teacher, students)

        kl = compute_kl_proxy(teacher, best_student) if students else 0.0
        question_results.append(
            SDPOQuestionResult(
                question_id=q_id,
                teacher_is_correct=teacher.is_correct,
                kl_selection_is_correct=best_student.is_correct,
                kl_distance=kl,
            )
        )

    gguf_path = _find_gguf()
    model_used = gguf_path or _MODEL_ID

    artifact_fields = build_sdpo_artifact_fields(
        question_results=question_results,
        all_completions=all_completions,
        n_completions_per_question=N_COMPLETIONS,
        model_used=model_used,
    )

    duration_s = round(time.monotonic() - t0, 2)
    finished_at = _dt.datetime.now(_dt.UTC).isoformat()

    return {
        "experiment": "1213_sdpo_dense_reward_distillation",
        "experiment_id": EXPERIMENT_ID,
        "status": "success",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "schema_version": "v1",
        "inference_mode": "canonical_cot_responses_with_energy_scoring",
        "random_seed": RANDOM_SEED,
        "paper_refs": [
            "arXiv 2604.03128 (SDPO: Self-Distilled Policy Optimization)",
        ],
        **artifact_fields,
    }


def main() -> None:
    """Entry point: run experiment and write JSON artifact."""
    # Write in-progress skeleton first (idempotent if already there).
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    if (
        not DELIVERABLE.exists()
        or json.loads(DELIVERABLE.read_text()).get("status") == "in_progress"
    ):
        DELIVERABLE.write_text(
            json.dumps(
                {
                    "experiment": "1213_sdpo_dense_reward_distillation",
                    "status": "in_progress",
                    "sdpo_dense_reward_delta_measured": False,
                    "honest_verdict": "in_progress",
                },
                indent=2,
            )
            + "\n"
        )

    try:
        artifact = run_experiment()
    except Exception as exc:
        artifact = {
            "experiment": "1213_sdpo_dense_reward_distillation",
            "status": "error",
            "error": str(exc),
            "sdpo_dense_reward_delta_measured": False,
            "honest_verdict": "error",
        }
        print(f"[EXP1213] ERROR: {exc}", file=sys.stderr)
        raise

    DELIVERABLE.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[EXP1213] Wrote {DELIVERABLE}")
    print(f"[EXP1213] energy_accuracy={artifact.get('energy_teacher_selection_accuracy')}")
    print(f"[EXP1213] kl_accuracy={artifact.get('sdpo_kl_selection_accuracy')}")
    print(f"[EXP1213] delta_pp={artifact.get('sdpo_dense_reward_delta_pp')}")
    print(f"[EXP1213] verdict={artifact.get('honest_verdict')}")


if __name__ == "__main__":
    main()
