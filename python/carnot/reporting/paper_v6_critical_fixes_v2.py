"""Audit and write the Exp 1269 paper-v6 critical-fixes v2 artifact."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAPER_PATH = REPO_ROOT / "docs" / "arxiv-paper" / "main.tex"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1269_paper_v6_critical_fixes_v2.json"

EXPERIMENT = "1269_paper_v6_critical_fixes_v2"
SCHEMA = "paper_integrity_v2"
RUN_DATE = "20260504"
HONEST_VERDICT_COMPLETE = "paper_v6_critical_fixes_v2_complete"

ISSUES_FIXED_LIST = [
    "estimated_cpu_fpga_speedups",
    "kl_measurement_provenance",
    "hand_typed_cpu_constants",
    "apples_to_oranges_humaneval_latency",
    "sos_kan_auroc_ambiguity",
]

MEASURED_ARTIFACTS = {
    "exp1256": "experiment_1256_verifier_orthogonality_audit_v3.json",
    "exp1264": "experiment_1264_q11_tss_instrumentation_v2.json",
    "exp1265": "experiment_1265_diffutruth_vs_carnot_baseline.json",
    "exp1266": "experiment_1266_quantkan_3bit_lut_kan.json",
}

OLD_CLAIM_PATTERNS = {
    "estimated_cpu_fpga_speedups": re.compile(
        r"13\{,\}061|13,061|13061|12\{,\}000|12,000|12000|11,680|11680",
        re.IGNORECASE,
    ),
    "kl_measurement_provenance": re.compile(
        r"FPGA\s+KL\s*=?\s*3\.07|KL\s*=?\s*3\.07[^.\n]{0,80}FPGA",
        re.IGNORECASE,
    ),
    "hand_typed_cpu_constants": re.compile(
        r"CPU_GIBBS_PER_SWEEP_NS|15\.6\s*(?:x|\\times)|hand-typed CPU",
        re.IGNORECASE,
    ),
    "apples_to_oranges_humaneval_latency": re.compile(
        r"76\s*,\s*130\s*x|76\{,\}130\s*\\times|76130|HumanEval latency",
        re.IGNORECASE,
    ),
    "sos_kan_auroc_ambiguity": re.compile(
        r"SOSKANEnergyV3\s+AUROC\s*=\s*0\.3333[^.\n]{0,120}without corpus context",
        re.IGNORECASE,
    ),
}


def audit_old_claims(tex: str) -> list[str]:
    """Return fixed-order names for old unsupported claim classes still present."""

    return [name for name in ISSUES_FIXED_LIST if OLD_CLAIM_PATTERNS[name].search(tex)]


def find_measured_artifacts_cited(tex: str) -> list[str]:
    """Return the exp1256/1264/1265/1266 citations present in the paper."""

    return [exp_id for exp_id in MEASURED_ARTIFACTS if exp_id in tex]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_artifact(
    tex: str,
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the terminal paper-integrity artifact from paper text and result JSON."""

    results_path = Path(results_dir)
    old_claims_remaining = audit_old_claims(tex)
    measured_artifacts_cited = find_measured_artifacts_cited(tex)
    measured_artifacts_loaded = [
        exp_id
        for exp_id, filename in MEASURED_ARTIFACTS.items()
        if _load_json(results_path / filename).get("status") == "complete"
    ]
    complete = (
        old_claims_remaining == []
        and measured_artifacts_cited == list(MEASURED_ARTIFACTS)
        and measured_artifacts_loaded == list(MEASURED_ARTIFACTS)
    )

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete" if complete else "blocked",
        "critical_issues_fixed": len(ISSUES_FIXED_LIST) if complete else 0,
        "issues_fixed_list": ISSUES_FIXED_LIST if complete else [],
        "measured_artifacts_cited": measured_artifacts_cited,
        "old_claims_remaining": old_claims_remaining,
        "honest_verdict": HONEST_VERDICT_COMPLETE if complete else "paper_v6_critical_fixes_v2_blocked",
    }


def run(
    *,
    paper_path: Path | str = DEFAULT_PAPER_PATH,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """Load the paper, write the Exp 1269 artifact JSON, and return it."""

    tex = Path(paper_path).read_text(encoding="utf-8")
    artifact = build_artifact(tex, results_dir=results_dir)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
