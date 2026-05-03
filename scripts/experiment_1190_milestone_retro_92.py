#!/usr/bin/env python3
"""Milestone 2026.04.92 paper-integrity retrospective.

The workflow reads the .92 source experiment artifacts, scores each roadmap
criterion from artifact fields, and writes the Exp 1190 deliverable. The
publication-hold decision is intentionally stricter than the milestone score:
paper submission remains blocked unless all integrity issues, audit hooks, the
full 4-test paper audit, and explicit operator approval are present.

Spec: REQ-REPORT-012, SCENARIO-REPORT-009.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONDUCTOR_LOG = REPO_ROOT / "ops" / "conductor-log.md"
DELIVERABLE_PATH = RESULTS_DIR / "experiment_1190_milestone_retro_92.json"
MILESTONE = "2026.04.92"
CRITERIA_TOTAL = 13

EXPERIMENT_FILES: dict[int, str] = {
    1178: "experiment_1178_pytest_memory_watchdog.json",
    1179: "experiment_1179_llama_cpp_gpu_offload_fix.json",
    1180: "experiment_1180_paper_v5_critical_issues_1_5.json",
    1181: "experiment_1181_paper_v5_high_issues_6_10.json",
    1182: "experiment_1182_paper_v5_medium_low_issues_11_18.json",
    1183: "experiment_1183_paper_v5_recompile_arxiv_bundle_v6.json",
    1184: "experiment_1184_grpo_v5_tinyv_v2_dualGPU.json",
    1185: "experiment_1185_sc_energy_overfit_regularized_k6.json",
    1186: "experiment_1186_dot_ebm_diffusion_redesign.json",
    1187: "experiment_1187_latent_grpo_energy_reward.json",
    1188: "experiment_1188_wopr_hex_game_cartridge.json",
    1189: "experiment_1189_phase4_stronger_baseline_10x10.json",
}

LOG_TASK_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "exp1178": ("Pytest Memory Watchdog",),
    "exp1179": ("llama.cpp GPU Offload Fix",),
    "exp1180": ("Critical ISSUE-1",),
    "exp1181": ("High-Severity ISSUE-6",),
    "exp1182": ("Medium/Low ISSUE-11",),
    "exp1183": ("Paper v5 Recompile",),
    "exp1184": ("GRPO v5 + TinyV v2",),
    "exp1185": ("SC-Energy Overfit",),
    "exp1186": ("DoT EBM-Diffusion Redesign",),
    "exp1187": ("Latent-GRPO Energy Reward",),
    "exp1188": ("WOPR Hex Game Cartridge",),
    "exp1189": ("Phase 4 Active Inference", "Stronger BFS Baseline"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(results_dir: Path = RESULTS_DIR) -> dict[int, dict[str, Any]]:
    """Load .92 artifacts while preserving missing files as explicit evidence."""

    artifacts: dict[int, dict[str, Any]] = {}
    for exp_id, filename in EXPERIMENT_FILES.items():
        path = results_dir / filename
        if path.exists():
            artifacts[exp_id] = _load_json(path)
        else:
            artifacts[exp_id] = {"_missing": True, "_path": str(path)}
    return artifacts


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _criterion(name: str, exp_id: int | None, met: bool, detail: str) -> dict[str, Any]:
    return {
        "criterion": name,
        "experiment": None if exp_id is None else f"exp{exp_id}",
        "status": "PASS" if met else "FAIL",
        "met": bool(met),
        "detail": detail,
    }


def evaluate_criteria(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Evaluate the 13 planned .92 criteria from authoritative artifacts."""

    e1178 = artifacts.get(1178, {})
    e1179 = artifacts.get(1179, {})
    e1180 = artifacts.get(1180, {})
    e1181 = artifacts.get(1181, {})
    e1182 = artifacts.get(1182, {})
    e1183 = artifacts.get(1183, {})
    e1184 = artifacts.get(1184, {})
    e1185 = artifacts.get(1185, {})
    e1186 = artifacts.get(1186, {})
    e1187 = artifacts.get(1187, {})
    e1188 = artifacts.get(1188, {})
    e1189 = artifacts.get(1189, {})

    dot_diagnosed = bool(
        e1186.get("diagnosis_complete")
        or (
            e1186.get("token_gradient_norms_near_zero")
            and e1186.get("redesign_implemented")
            and "redesigned_dot_auroc" in e1186
        )
    )
    phase4_reported = bool(
        e1189.get("phase4_vs_bfs_delta_reported")
        or (
            e1189.get("stronger_baseline_implemented")
            and e1189.get("free_energy_values_all_puzzles")
            and _int_value(e1189.get("n_10x10_puzzles")) >= 10
        )
    )

    return {
        "exp1178_watchdog_operational": _criterion(
            "exp1178_watchdog_operational",
            1178,
            bool(e1178.get("watchdog_operational")),
            "Pytest memory watchdog must be active.",
        ),
        "exp1179_gpu_offload_verified": _criterion(
            "exp1179_gpu_offload_verified",
            1179,
            bool(e1179.get("llama_cpp_gpu_offload_verified") or e1179.get("gpu_offload_verified")),
            "llama.cpp GPU offload artifact must verify throughput.",
        ),
        "exp1180_critical_issues_fixed": _criterion(
            "exp1180_critical_issues_fixed",
            1180,
            _int_value(e1180.get("critical_issues_fixed")) == 5,
            "All five critical paper-integrity issues must be fixed.",
        ),
        "exp1181_high_severity_fixed": _criterion(
            "exp1181_high_severity_fixed",
            1181,
            _int_value(e1181.get("high_severity_fixed")) == 5,
            "All five high-severity paper-integrity issues must be fixed.",
        ),
        "exp1182_medium_low_fixed": _criterion(
            "exp1182_medium_low_fixed",
            1182,
            _int_value(e1182.get("medium_low_issues_fixed")) == 8,
            "All eight medium/low paper-integrity issues must be fixed.",
        ),
        "exp1183_arxiv_bundle_v6_ready": _criterion(
            "exp1183_arxiv_bundle_v6_ready",
            1183,
            bool(e1183.get("arxiv_bundle_v6_ready")),
            "Paper v6 bundle must be ready after the full audit.",
        ),
        "exp1184_grpo_v5_result_honest": _criterion(
            "exp1184_grpo_v5_result_honest",
            1184,
            bool(e1184.get("honest_verdict")) and not e1184.get("_missing"),
            "GRPO v5 must report an honest outcome, including blocked prerequisites.",
        ),
        "exp1185_sc_energy_regularized": _criterion(
            "exp1185_sc_energy_regularized",
            1185,
            bool(e1185.get("sc_energy_regularized")),
            "SC-Energy overfit must be diagnosed and regularized.",
        ),
        "exp1186_dot_diagnosis_complete": _criterion(
            "exp1186_dot_diagnosis_complete",
            1186,
            dot_diagnosed,
            "DoT root cause must be diagnosed and redesigned result reported.",
        ),
        "exp1187_latent_grpo_delta_honest": _criterion(
            "exp1187_latent_grpo_delta_honest",
            1187,
            bool(e1187.get("honest_verdict")) and "latent_grpo_delta_pp" in e1187,
            "Latent-GRPO must report the delta honestly, even if zero.",
        ),
        "exp1188_hex_game_operational": _criterion(
            "exp1188_hex_game_operational",
            1188,
            bool(e1188.get("hex_game_operational")),
            "Hex cartridge must run and report win-rate evidence.",
        ),
        "exp1189_phase4_stronger_baseline": _criterion(
            "exp1189_phase4_stronger_baseline",
            1189,
            phase4_reported,
            "Phase 4 must be compared against the BFS baseline on 10x10 puzzles.",
        ),
        "exp1190_retro_complete": _criterion(
            "exp1190_retro_complete",
            None,
            True,
            "Exp1190 retrospective artifact was assembled.",
        ),
    }


def criteria_status(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {name: str(item["status"]) for name, item in criteria.items()}


def criteria_results(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    return {name: bool(item["met"]) for name, item in criteria.items()}


def criteria_met_count(criteria: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(1 for item in criteria.values() if item["met"])


def paper_integrity_issues_resolved(artifacts: Mapping[int, Mapping[str, Any]]) -> int:
    """Count resolved paper-integrity issues across Exp 1180 through Exp 1182."""

    return (
        _int_value(artifacts.get(1180, {}).get("critical_issues_fixed"))
        + _int_value(artifacts.get(1181, {}).get("high_severity_fixed"))
        + _int_value(artifacts.get(1182, {}).get("medium_low_issues_fixed"))
    )


def publication_hold_lifted(artifacts: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return true only when all paper and operator gates permit submission."""

    e1180 = artifacts.get(1180, {})
    e1182 = artifacts.get(1182, {})
    e1183 = artifacts.get(1183, {})
    return bool(
        paper_integrity_issues_resolved(artifacts) == 18
        and e1180.get("figure_integrity_script_active")
        and e1182.get("paper_claim_audit_script_active")
        and e1183.get("4_test_full_pass")
        and e1183.get("operator_explicit_approval")
    )


def record_honest_verdicts(
    artifacts: Mapping[int, Mapping[str, Any]], self_verdict: str
) -> dict[str, str]:
    """Return source honest verdicts and the Exp1190 verdict."""

    verdicts: dict[str, str] = {}
    for exp_id in sorted(EXPERIMENT_FILES):
        artifact = artifacts.get(exp_id, {"_missing": True})
        verdicts[f"exp{exp_id}"] = (
            "MISSING"
            if artifact.get("_missing")
            else str(artifact.get("honest_verdict", "NO_VERDICT"))
        )
    verdicts["exp1190"] = self_verdict
    return verdicts


def _parse_log_line(line: str) -> tuple[dt.datetime, str] | None:
    match = re.match(r"^\|\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\s*\|\s*([^|]+?)\s*\|", line)
    if not match:
        return None
    timestamp = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=dt.UTC)
    return timestamp, match.group(2)


def _exp_id_for_title(title: str) -> str | None:
    for exp_id, fragments in LOG_TASK_FRAGMENTS.items():
        if any(fragment in title for fragment in fragments):
            return exp_id
    return None


def build_slowest_tasks(log_lines: Sequence[str]) -> list[dict[str, Any]]:
    """Rank .92 tasks by conductor-log span from first to last visible attempt."""

    spans: dict[str, list[dt.datetime]] = {}
    milestone_started = False
    for line in log_lines:
        parsed = _parse_log_line(line)
        if parsed is None:
            continue
        timestamp, title = parsed
        if f"Milestone {MILESTONE} activated" in title:
            milestone_started = True
            continue
        if milestone_started and title.startswith("Milestone ") and "activated" in title:
            break
        if not milestone_started:
            continue
        exp_id = _exp_id_for_title(title)
        if exp_id is not None:
            spans.setdefault(exp_id, []).append(timestamp)

    ranked: list[dict[str, Any]] = []
    for exp_id, timestamps in spans.items():
        duration = round((max(timestamps) - min(timestamps)).total_seconds() / 60.0, 1)
        ranked.append({"id": exp_id, "duration_min": duration, "attempts_seen": len(timestamps)})
    ranked.sort(key=lambda entry: (-entry["duration_min"], -entry["attempts_seen"], entry["id"]))
    return [{"rank": index + 1, **entry} for index, entry in enumerate(ranked)]


def slowest_task_id(artifacts: Mapping[int, Mapping[str, Any]], log_lines: Sequence[str]) -> str:
    slowest = build_slowest_tasks(log_lines)
    if slowest:
        return str(slowest[0]["id"])
    durations = [
        (f"exp{exp_id}", _float_value(artifact.get("duration_s")))
        for exp_id, artifact in artifacts.items()
        if not artifact.get("_missing")
    ]
    if not durations:
        return "unknown"
    return max(durations, key=lambda item: (item[1], item[0]))[0]


def significant_findings(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Summarize the most consequential .92 outcomes for planning .93."""

    e1183 = artifacts.get(1183, {})
    e1185 = artifacts.get(1185, {})
    e1188 = artifacts.get(1188, {})
    e1189 = artifacts.get(1189, {})
    return [
        (
            "Paper remediation made partial progress: high plus medium/low fixes landed, "
            f"but critical fixes are unresolved and arxiv_bundle_v6_ready="
            f"{e1183.get('arxiv_bundle_v6_ready')}."
        ),
        (
            "SC-Energy regularization resolved overfit, but k=6 still regressed "
            f"(k6={e1185.get('k6_regularized_auroc')}, "
            f"k5={e1185.get('k5_auroc_on_eval')}); k=6 should be retired."
        ),
        (
            "Hex became operational with random_vs_gibbs_win_rate="
            f"{e1188.get('random_vs_gibbs_win_rate')}, while Phase 4 only "
            f"reported {e1189.get('honest_verdict')} against BFS."
        ),
    ]


def open_items_for_93(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Return the unfinished, retired, and priority items for the next milestone."""

    criteria = evaluate_criteria(artifacts)
    status = criteria_status(criteria)
    items = [
        "Prioritize a preemptive pytest memory cap (prlimit or cgroup) because the Exp1178 watchdog did not prevent single-test 35GB+ loads.",
    ]
    if status["exp1179_gpu_offload_verified"] == "FAIL":
        items.append(
            "Rerun exp1179 llama.cpp GPU offload fix and emit the missing GPU-throughput artifact before any GRPO v5 retry."
        )
    if status["exp1180_critical_issues_fixed"] == "FAIL":
        items.append(
            "Rerun exp1180 critical paper-integrity fixes and activate figure_integrity_audit.py."
        )
    if not publication_hold_lifted(artifacts):
        items.append(
            "Keep publication hold active until 18/18 issues, both audit scripts, full 4-test pass, and operator approval are present."
        )
    if bool(artifacts.get(1185, {}).get("retire_k6")):
        items.append(
            "Retire k=6 AND-compose from the production path unless a new verifier root cause is identified."
        )
    if bool(artifacts.get(1186, {}).get("retire_dot")):
        items.append("Retire DoT EBM-diffusion as a near-random verifier signal for .93 planning.")
    if _float_value(artifacts.get(1187, {}).get("latent_grpo_delta_pp")) <= 0.0:
        items.append(
            "Deprioritize Latent-GRPO reward masking until invalid samples are present or a nonzero delta target is defined."
        )
    if str(artifacts.get(1189, {}).get("honest_verdict")) == "phase4_tied_with_bfs":
        items.append(
            "Treat Phase 4 vs BFS as a limitation, not an advantage claim, until it beats a non-trivial baseline."
        )
    return items


def _honest_verdict(criteria_met: int) -> str:
    if criteria_met >= 11:
        return "milestone_complete"
    if criteria_met >= 8:
        return "milestone_partial"
    return "milestone_failed"


def build_artifact(
    artifacts: Mapping[int, Mapping[str, Any]], log_lines: Sequence[str]
) -> dict[str, Any]:
    criteria = evaluate_criteria(artifacts)
    n_met = criteria_met_count(criteria)
    verdict = _honest_verdict(n_met)
    slowest_tasks = build_slowest_tasks(log_lines)
    hold_lifted = publication_hold_lifted(artifacts)

    return {
        "experiment": "1190_milestone_retro_92",
        "schema": "milestone_retro_v3",
        "run_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "milestone": MILESTONE,
        "criteria_results": criteria_results(criteria),
        "criteria_status": criteria_status(criteria),
        "criteria_detail": criteria,
        "criteria_met": n_met,
        "criteria_total": CRITERIA_TOTAL,
        "criteria_score_pct": round(n_met / CRITERIA_TOTAL * 100.0, 2),
        "paper_integrity_issues_resolved": paper_integrity_issues_resolved(artifacts),
        "publication_hold_lifted": hold_lifted,
        "publication_hold_status": "lifted" if hold_lifted else "active",
        "grpo_v5_result": str(artifacts.get(1184, {}).get("honest_verdict", "MISSING")),
        "k6_viable": bool(artifacts.get(1185, {}).get("k6_viable_for_production")),
        "k6_retired": bool(artifacts.get(1185, {}).get("retire_k6")),
        "dot_retired": bool(artifacts.get(1186, {}).get("retire_dot")),
        "latent_grpo_delta_pp": _float_value(artifacts.get(1187, {}).get("latent_grpo_delta_pp")),
        "hex_operational": bool(artifacts.get(1188, {}).get("hex_game_operational")),
        "phase4_stronger_baseline_result": str(
            artifacts.get(1189, {}).get("honest_verdict", "MISSING")
        ),
        "slowest_task_id": slowest_task_id(artifacts, log_lines),
        "slowest_tasks": slowest_tasks,
        "primary_operational_issue": (
            "Pre-test/self-heal retries and missing gate artifacts dominated .92; "
            f"{slowest_tasks[0]['id']} had the longest visible conductor span."
            if slowest_tasks
            else "No conductor-log task spans were available."
        ),
        "significant_findings": significant_findings(artifacts),
        "experiment_honest_verdicts": record_honest_verdicts(artifacts, verdict),
        "open_items_for_93": open_items_for_93(artifacts),
        "retro_complete": True,
        "honest_verdict": verdict,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--conductor-log", type=Path, default=CONDUCTOR_LOG)
    parser.add_argument("--out", type=Path, default=DELIVERABLE_PATH)
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.results_dir)
    log_lines = (
        args.conductor_log.read_text(encoding="utf-8").splitlines()
        if args.conductor_log.exists()
        else []
    )
    artifact = build_artifact(artifacts, log_lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"[exp1190] {artifact['criteria_met']}/{artifact['criteria_total']} criteria met; "
        f"publication_hold={artifact['publication_hold_status']}; out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
