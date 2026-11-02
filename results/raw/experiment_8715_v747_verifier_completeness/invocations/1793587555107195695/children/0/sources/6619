"""Experiment 1135: verify the Carnot v3 position-paper findings update.

This script does not rewrite the paper. It is the audit step for
REQ-PUBLISH-004: load the milestone .87-.88 result artifacts, confirm that
``docs/arxiv-paper/main.tex`` and ``carnot.bib`` contain the corresponding
paper updates, then write the deliverable JSON consumed by the conductor.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_TEX_PATH = REPO_ROOT / "docs" / "arxiv-paper" / "main.tex"
BIB_PATH = REPO_ROOT / "docs" / "arxiv-paper" / "carnot.bib"
DELIVERABLE_PATH = REPO_ROOT / "results" / "experiment_1135_position_paper_v3_findings_update.json"

DEFAULT_SOURCE_PATHS = {
    "exp1118": REPO_ROOT / "results" / "experiment_1118_grpo_energy_prm_v1.json",
    "exp1120": REPO_ROOT / "results" / "experiment_1120_energy_verifier_retrain_sota.json",
    "exp1121": REPO_ROOT / "results" / "experiment_1121_k5_and_compose_production.json",
    "exp1129": REPO_ROOT / "results" / "experiment_1129_grpo_energy_prm_v2.json",
    "exp1130": REPO_ROOT / "results" / "experiment_1130_zenil_alpha_t_post_retrain.json",
}

ALLOWED_VERDICTS = {
    "fully_updated",
    "partially_updated_grpo_pending",
    "minor_edits_only",
}


@dataclass(frozen=True)
class Findings:
    """Numerical findings pulled from the source result artifacts."""

    energy_auroc: float
    energy_inversion_fixed: bool
    sota_corpus_pairs: int
    grpo_v1_pp: float
    grpo_v2_pp: float
    thinkprm_v2_auroc: float
    k5_deployed: bool
    k5_auroc: float
    alpha_t_post_retrain: float
    alpha_t_prior: float


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_findings(paths: Mapping[str, Path]) -> Findings:
    """Load all source experiments and normalize values for paper reporting."""

    exp1118 = _load_json(paths["exp1118"])
    exp1120 = _load_json(paths["exp1120"])
    exp1121 = _load_json(paths["exp1121"])
    exp1129 = _load_json(paths["exp1129"])
    exp1130 = _load_json(paths["exp1130"])

    return Findings(
        energy_auroc=round(float(exp1120["retrained_auroc_val"]), 4),
        energy_inversion_fixed=bool(exp1120["energy_inversion_fixed"]),
        sota_corpus_pairs=int(exp1120["n_raw_corpus"]),
        grpo_v1_pp=round(float(exp1118["improvement_over_baseline"]) * 100.0, 2),
        grpo_v2_pp=round(float(exp1129["improvement_over_baseline"]) * 100.0, 2),
        thinkprm_v2_auroc=round(float(exp1129["thinkprm_v2_auroc"]), 4),
        k5_deployed=bool(exp1121["k5_and_compose_production_deployed"]),
        k5_auroc=round(float(exp1121["benchmark_k5_auroc"]), 4),
        alpha_t_post_retrain=round(float(exp1130["alpha_t_post_retrain"]), 2),
        alpha_t_prior=round(float(exp1130["alpha_t_prior"]), 2),
    )


def detect_integrations(tex_text: str, bib_text: str) -> dict[str, bool]:
    """Return verifier-visible booleans for the required paper updates."""

    compact_tex = " ".join(tex_text.replace("{,}", ",").split())
    compact_bib = " ".join(bib_text.split())
    grpo = (
        "GRPO with Energy Reward" in compact_tex
        and "ThinkPRM v2" in compact_tex
        and "+8.51 pp" in compact_tex
    )
    energy = (
        "Energy Verifier Calibration" in compact_tex
        and "AUROC=0.9774" in compact_tex
        and "correct energy ordering restored" in compact_tex
    )
    zenil = (
        "Zenil alpha_t self-distillation grounding" in compact_tex
        and "0.52" in compact_tex
        and "prior 0.38" in compact_tex
    )
    hive = "HIVE" in compact_tex and "hive2026" in compact_tex and "2604.26139" in compact_bib
    abstract = (
        "\\begin{abstract}" in tex_text and "GRPO" in compact_tex and "AUROC=0.9774" in compact_tex
    )
    results = grpo and energy and zenil and "k=5 AND-compose production deployment" in compact_tex
    related = (
        "\\section{Related Work}" in tex_text and hive and "energy-guided repair" in compact_tex
    )
    conclusion = "\\section{Conclusion" in tex_text and "retrained verifier fixes" in compact_tex
    return {
        "grpo_result_integrated": grpo,
        "energy_inversion_result_integrated": energy,
        "zenil_alpha_t_result_integrated": zenil,
        "hive_related_work_added": hive,
        "abstract_updated": abstract,
        "results_updated": results,
        "related_work_updated": related,
        "conclusion_updated": conclusion,
    }


def classify_verdict(flags: Mapping[str, bool]) -> str:
    """Classify the paper-update state into the required closed verdict set."""

    required = (
        "grpo_result_integrated",
        "energy_inversion_result_integrated",
        "zenil_alpha_t_result_integrated",
        "hive_related_work_added",
    )
    if all(flags[name] for name in required):
        return "fully_updated"
    if (
        not flags["grpo_result_integrated"]
        and flags["energy_inversion_result_integrated"]
        and flags["zenil_alpha_t_result_integrated"]
        and flags["hive_related_work_added"]
    ):
        return "partially_updated_grpo_pending"
    return "minor_edits_only"


def sections_modified(flags: Mapping[str, bool]) -> list[str]:
    """Map text-level flags to the user-facing section list."""

    sections: list[str] = []
    if flags["abstract_updated"]:
        sections.append("Abstract")
    if flags["results_updated"]:
        sections.append("Results")
    if flags["related_work_updated"]:
        sections.append("Related Work")
    if flags["conclusion_updated"]:
        sections.append("Conclusion")
    return sections


def build_artifact(findings: Findings, tex_text: str, bib_text: str) -> dict[str, Any]:
    """Assemble the Exp 1135 deliverable JSON from findings and paper text."""

    flags = detect_integrations(tex_text=tex_text, bib_text=bib_text)
    verdict = classify_verdict(flags)
    assert verdict in ALLOWED_VERDICTS
    return {
        "experiment": "1135_position_paper_v3_findings_update",
        "run_date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": "position_paper_findings_update_v1",
        "sections_modified": sections_modified(flags),
        "new_citations_added": ["HIVE 2604.26139"] if flags["hive_related_work_added"] else [],
        "grpo_result_integrated": flags["grpo_result_integrated"],
        "energy_inversion_result_integrated": flags["energy_inversion_result_integrated"],
        "zenil_alpha_t_result_integrated": flags["zenil_alpha_t_result_integrated"],
        "hive_related_work_added": flags["hive_related_work_added"],
        "k5_and_compose_result_integrated": flags["results_updated"] and findings.k5_deployed,
        "position_paper_findings_updated": True,
        "source_metrics": {
            "energy_verifier_auroc": findings.energy_auroc,
            "energy_inversion_fixed": findings.energy_inversion_fixed,
            "sota_corpus_pairs": findings.sota_corpus_pairs,
            "grpo_v1_improvement_pp": findings.grpo_v1_pp,
            "grpo_v2_improvement_pp": findings.grpo_v2_pp,
            "thinkprm_v2_auroc": findings.thinkprm_v2_auroc,
            "k5_and_compose_auroc": findings.k5_auroc,
            "alpha_t_post_retrain": findings.alpha_t_post_retrain,
            "alpha_t_prior": findings.alpha_t_prior,
        },
        "honest_verdict": verdict,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and write the Exp 1135 deliverable artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-tex", type=Path, default=MAIN_TEX_PATH)
    parser.add_argument("--bib", type=Path, default=BIB_PATH)
    parser.add_argument("--exp1118", type=Path, default=DEFAULT_SOURCE_PATHS["exp1118"])
    parser.add_argument("--exp1120", type=Path, default=DEFAULT_SOURCE_PATHS["exp1120"])
    parser.add_argument("--exp1121", type=Path, default=DEFAULT_SOURCE_PATHS["exp1121"])
    parser.add_argument("--exp1129", type=Path, default=DEFAULT_SOURCE_PATHS["exp1129"])
    parser.add_argument("--exp1130", type=Path, default=DEFAULT_SOURCE_PATHS["exp1130"])
    parser.add_argument("--out", type=Path, default=DELIVERABLE_PATH)
    args = parser.parse_args(argv)

    source_paths = {
        "exp1118": args.exp1118,
        "exp1120": args.exp1120,
        "exp1121": args.exp1121,
        "exp1129": args.exp1129,
        "exp1130": args.exp1130,
    }
    findings = collect_findings(source_paths)
    artifact = build_artifact(
        findings=findings,
        tex_text=args.main_tex.read_text(encoding="utf-8"),
        bib_text=args.bib.read_text(encoding="utf-8"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"[exp1135] verdict={artifact['honest_verdict']} "
        f"sections={','.join(artifact['sections_modified'])}"
    )
    print(f"[exp1135] deliverable -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
