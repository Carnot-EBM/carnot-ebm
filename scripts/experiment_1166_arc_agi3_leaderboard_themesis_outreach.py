#!/usr/bin/env python3
"""Experiment 1166: ARC-AGI-3 positioning and Themesis outreach draft.

Spec coverage: REQ-KONA-013, SCENARIO-KONA-013
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
EXP1165_PATH = REPO_ROOT / "results" / "experiment_1165_phase4_active_inference_pilot_v1.json"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1166_arc_agi3_leaderboard_themesis_outreach.json"

LEADERBOARD_URL = "https://arcprize.org/leaderboard"
LEADERBOARD_V3_DATA_URL = "https://arcprize.org/media/data/leaderboard/v3.json"

DOCUMENTED_SEED_IQ_SCORE = 1.0
DOCUMENTED_SEED_IQ_PUBLIC_DEMO_SCORE = 0.95
SEED_IQ_ACTION_EFFICIENCY = "115% of human baseline (2674 vs human 7534-8073 actions)"
FRONTIER_LLM_SCORE = "<1% per arXiv 2603.24621"
AUTHOR_EMAIL = "ian@blenke.com"

REQUIRED_ARTIFACT_FIELDS = {
    "seed_iq_score_confirmed",
    "seed_iq_score",
    "seed_iq_action_efficiency",
    "carnot_phase4_action_count_ratio",
    "leaderboard_comparison_table",
    "themesis_email_drafted",
    "themesis_email_text",
    "honest_verdict",
}

HONEST_VERDICTS = {
    "comparison_documented_email_drafted",
    "leaderboard_unavailable_email_drafted",
}


@dataclass(frozen=True)
class LeaderboardEvidence:
    """Seed IQ evidence extracted from the current public leaderboard fetch."""

    seed_iq_score_confirmed: bool
    seed_iq_score: float | None
    seed_iq_action_efficiency: str
    source: str
    note: str
    generated_at: str | None
    honest_verdict: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read().decode("utf-8", errors="replace")


def _json_seed_row(v3_text: str) -> tuple[float | None, str | None] | None:
    try:
        payload = json.loads(v3_text)
    except json.JSONDecodeError:
        return None

    generated_at = payload.get("generatedAt")
    for row in payload.get("evaluations", []):
        row_text = json.dumps(row, sort_keys=True).lower()
        if "seed" in row_text and ("iq" in row_text or "themesis" in row_text):
            score = row.get("score")
            return (float(score) if score is not None else None, generated_at)
    return None


def _generated_at_from_v3(v3_text: str) -> str | None:
    try:
        payload = json.loads(v3_text)
    except json.JSONDecodeError:
        return None
    generated_at = payload.get("generatedAt")
    return str(generated_at) if generated_at else None


def _html_seed_score(html_text: str) -> float | None:
    lower_text = html_text.lower()
    if "seed" not in lower_text or "iq" not in lower_text:
        return None
    match = re.search(r"seed[^0-9]{0,120}(?:score[^0-9]{0,20})?(0?\.\d+|1(?:\.0+)?)", lower_text)
    return float(match.group(1)) if match else None


def fetch_leaderboard_evidence(
    fetch_text: Callable[[str], str] | None = None,
) -> LeaderboardEvidence:
    """Fetch ARC-AGI-3 leaderboard context and return honest Seed IQ evidence."""

    fetch = fetch_text or _default_fetch_text
    html_text = ""
    v3_text = ""
    try:
        html_text = fetch(LEADERBOARD_URL)
    except Exception:
        html_text = ""
    try:
        v3_text = fetch(LEADERBOARD_V3_DATA_URL)
    except Exception:
        v3_text = ""

    json_match = _json_seed_row(v3_text)
    if json_match is not None:
        score, generated_at = json_match
        return LeaderboardEvidence(
            seed_iq_score_confirmed=True,
            seed_iq_score=score,
            seed_iq_action_efficiency=SEED_IQ_ACTION_EFFICIENCY,
            source="arcprize_v3_json",
            note="Seed IQ row found in the current ARC Prize v3 leaderboard data JSON.",
            generated_at=generated_at,
            honest_verdict="comparison_documented_email_drafted",
        )

    html_score = _html_seed_score(html_text)
    if html_score is not None:
        return LeaderboardEvidence(
            seed_iq_score_confirmed=True,
            seed_iq_score=html_score,
            seed_iq_action_efficiency=SEED_IQ_ACTION_EFFICIENCY,
            source="arcprize_html",
            note="Seed IQ row found in the fetched ARC Prize leaderboard HTML.",
            generated_at=_generated_at_from_v3(v3_text),
            honest_verdict="comparison_documented_email_drafted",
        )

    return LeaderboardEvidence(
        seed_iq_score_confirmed=False,
        seed_iq_score=DOCUMENTED_SEED_IQ_SCORE,
        seed_iq_action_efficiency=SEED_IQ_ACTION_EFFICIENCY,
        source="documented_fallback",
        note=(
            "Current ARC Prize fetch did not expose a Seed IQ row; using "
            f"ops/known-issues.md fallback values. The {DOCUMENTED_SEED_IQ_PUBLIC_DEMO_SCORE:.2f} "
            "public demo is documented there as independently verified, but this run did not "
            "independently confirm the reported 1.00 leaderboard score."
        ),
        generated_at=_generated_at_from_v3(v3_text),
        honest_verdict="leaderboard_unavailable_email_drafted",
    )


def _as_float(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload[key])
    except KeyError as exc:
        raise KeyError(f"Exp 1165 artifact missing required field: {key}") from exc


def _action_reduction_percent(action_count_ratio: float) -> float:
    return max(0.0, (1.0 - action_count_ratio) * 100.0)


def build_comparison_table(
    exp1165: dict[str, Any],
    evidence: LeaderboardEvidence,
) -> list[dict[str, Any]]:
    action_count_ratio = _as_float(exp1165, "action_count_ratio")
    solved_rate = _as_float(exp1165, "phase4_solved_rate")
    reduction = _action_reduction_percent(action_count_ratio)
    seed_score = evidence.seed_iq_score

    return [
        {
            "system_name": "Seed IQ (Active Inference)",
            "score": seed_score,
            "action_efficiency": evidence.seed_iq_action_efficiency,
        },
        {
            "system_name": "Carnot Phase 4 pilot",
            "score": f"solved_rate={solved_rate:.3f} on synthetic 5x5 ARC-AGI-3-like puzzles",
            "action_efficiency": (
                f"action_count_ratio={action_count_ratio:.6f}; "
                f"{reduction:.1f}% fewer actions than greedy baseline"
            ),
        },
        {
            "system_name": "Frontier LLMs (autoregressive)",
            "score": FRONTIER_LLM_SCORE,
            "action_efficiency": "Not competitive on ARC-AGI-3; reported scores remain below 1%.",
        },
    ]


def draft_themesis_email(action_count_ratio: float, solved_rate: float) -> str:
    reduction = _action_reduction_percent(action_count_ratio)
    return f"""To: Denise Holt / Denis O. at Themesis
From: Ian Blenke <{AUTHOR_EMAIL}>
Subject: Carnot EBM + Active Inference — Architectural Collaboration?

Denise, Denis,

I'm Ian Blenke, maintainer of Carnot, an Apache 2.0, decentralization-respecting, multi-vendor EBM verification project.

Carnot's k=5 verifier ensemble minimizes a variational free-energy proxy, F(z) = sum_k w_k E_k(z), which appears aligned with Friston-style active inference. Our Phase 4 pilot on synthetic ARC-AGI-3-like puzzles reached solved_rate={solved_rate:.3f} with action_count_ratio={action_count_ratio:.6f} versus a greedy legal-action baseline, a {reduction:.1f}% action reduction.

Themesis appears to have the stronger active-inference/topological-field algorithm; Carnot has an open verifier ensemble that may serve as a calibrated free-energy approximation around LLM infrastructure. This feels complementary, not competitive.

Would you be open to a 30-minute architectural conversation? Concrete options: joint benchmark evaluation, pre-print exchange, or a technical review of whether Carnot's verifier ensemble could plug into Seed IQ as an approximation layer.

Ian
{AUTHOR_EMAIL}
"""


def count_email_words(email_text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_.@=+-]+", email_text))


def build_positioning_narrative(
    action_count_ratio: float,
    solved_rate: float,
    evidence: LeaderboardEvidence,
) -> str:
    reduction = _action_reduction_percent(action_count_ratio)
    confirmation = (
        "independently confirmed by the current ARC Prize fetch"
        if evidence.seed_iq_score_confirmed
        else "not independently confirmed by the current ARC Prize fetch"
    )
    return (
        "Seed IQ is the full ARC-AGI-3 active-inference reference point: "
        f"reported score={evidence.seed_iq_score} with {evidence.seed_iq_action_efficiency}; "
        f"that reported leaderboard row was {confirmation}. Carnot Phase 4 is not yet a "
        "full ARC-AGI-3 result: Exp 1165 is a ten-puzzle synthetic pilot with "
        f"solved_rate={solved_rate:.3f} and action_count_ratio={action_count_ratio:.6f}, "
        f"or {reduction:.1f}% fewer actions than the greedy baseline. Relative to "
        "autoregressive frontier LLMs below 1%, the pilot supports positioning Carnot as "
        "a verifier/free-energy bridge to active inference, not as a Seed IQ competitor."
    )


def build_artifact(exp1165: dict[str, Any], evidence: LeaderboardEvidence) -> dict[str, Any]:
    action_count_ratio = _as_float(exp1165, "action_count_ratio")
    solved_rate = _as_float(exp1165, "phase4_solved_rate")
    email_text = draft_themesis_email(action_count_ratio, solved_rate)

    artifact = {
        "schema": "carnot.arc_agi3_leaderboard_themesis_outreach.v1",
        "experiment": 1166,
        "run_date": "2026-05-02",
        "leaderboard_url": LEADERBOARD_URL,
        "leaderboard_v3_data_url": LEADERBOARD_V3_DATA_URL,
        "leaderboard_data_generated_at": evidence.generated_at,
        "leaderboard_evidence_source": evidence.source,
        "seed_iq_score_confirmed": evidence.seed_iq_score_confirmed,
        "seed_iq_score": evidence.seed_iq_score,
        "seed_iq_independently_documented_demo_score": DOCUMENTED_SEED_IQ_PUBLIC_DEMO_SCORE,
        "seed_iq_confirmation_note": evidence.note,
        "seed_iq_action_efficiency": evidence.seed_iq_action_efficiency,
        "carnot_phase4_action_count_ratio": action_count_ratio,
        "carnot_phase4_solved_rate": solved_rate,
        "leaderboard_comparison_table": build_comparison_table(exp1165, evidence),
        "positioning_narrative": build_positioning_narrative(
            action_count_ratio, solved_rate, evidence
        ),
        "themesis_email_drafted": True,
        "themesis_email_word_count": count_email_words(email_text),
        "themesis_email_text": email_text,
        "honest_verdict": evidence.honest_verdict,
    }
    if not REQUIRED_ARTIFACT_FIELDS <= artifact.keys():
        missing = sorted(REQUIRED_ARTIFACT_FIELDS - artifact.keys())
        raise AssertionError(f"missing required artifact fields: {missing}")
    if artifact["honest_verdict"] not in HONEST_VERDICTS:
        raise AssertionError(f"unsupported honest_verdict: {artifact['honest_verdict']}")
    if artifact["themesis_email_word_count"] >= 300:
        raise AssertionError("Themesis email draft must stay under 300 words")
    return artifact


def write_artifact(artifact: dict[str, Any], path: Path = RESULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_experiment(
    exp1165_path: Path = EXP1165_PATH,
    deliverable_path: Path = RESULT_PATH,
    fetch_text: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    exp1165 = _load_json(exp1165_path)
    evidence = fetch_leaderboard_evidence(fetch_text)
    artifact = build_artifact(exp1165, evidence)
    write_artifact(artifact, deliverable_path)
    return artifact


def main() -> int:
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
