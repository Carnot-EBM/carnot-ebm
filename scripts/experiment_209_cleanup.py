#!/usr/bin/env python3
# ruff: noqa: E501
"""Experiment 209: provenance cleanup for result artifacts and public docs.

REQ-REPORT-001: Audit result artifacts and detect inference provenance.
REQ-REPORT-002: Annotate each artifact with a header and machine-readable status.
REQ-REPORT-003: Rewrite README headline claims with explicit provenance labels.
REQ-REPORT-004: Add a "Simulation vs Reality" disclosure to the report and docs.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULT_GLOB = "experiment_*_results.json"


@dataclass(frozen=True)
class ResultAudit:
    """Normalized provenance summary for one result artifact."""

    experiment_id: str
    path: Path
    detected_mode: str | None
    normalized_mode: str
    source: str
    status: str
    header: str


@dataclass(frozen=True)
class CleanupSummary:
    """Top-level audit summary."""

    total_results: int
    validated_live_gpu: int
    simulated_results: int
    unverified_results: int
    validated_experiments: tuple[str, ...]
    simulated_experiments: tuple[str, ...]
    unverified_experiments: tuple[str, ...]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    path.write_text(json.dumps(payload, indent=2) + "\n")


def normalize_mode(mode: str | None) -> str:
    """Collapse mode aliases into a small reporting vocabulary."""
    if mode is None:
        return "missing"
    lowered = mode.strip().lower()
    if lowered == "live_gpu":
        return "live_gpu"
    if lowered in {"simulated", "simulation"}:
        return "simulated"
    return lowered


def detect_inference_mode(payload: dict[str, Any]) -> tuple[str | None, str]:
    """Return the first detected inference mode and the field that provided it."""
    if payload.get("inference_mode") is not None:
        prior = payload.get("result_provenance")
        if (
            isinstance(prior, dict)
            and prior.get("detected_mode") == payload["inference_mode"]
            and isinstance(prior.get("source"), str)
            and prior["source"] != "missing"
        ):
            return str(payload["inference_mode"]), str(prior["source"])
        return str(payload["inference_mode"]), "inference_mode"

    for field in ("metadata", "statistics", "summary", "results", "benchmark"):
        nested = payload.get(field)
        if isinstance(nested, dict) and nested.get("inference_mode") is not None:
            return str(nested["inference_mode"]), f"{field}.inference_mode"

    return None, "missing"


def build_header(experiment_id: str, normalized_mode: str, source: str) -> tuple[str, str]:
    """Build the machine status and visible header string."""
    if normalized_mode == "live_gpu":
        return (
            "validated_live_gpu",
            (
                f"VALIDATED LIVE RESULT: Experiment {experiment_id} records "
                f"`inference_mode=live_gpu` via `{source}`. This artifact is "
                "treated as validated live evidence in the public docs."
            ),
        )

    if normalized_mode == "simulated":
        return (
            "warning",
            (
                f"WARNING: Experiment {experiment_id} is simulated "
                f"(`{source}`). Keep this artifact for the research record, but "
                "do not present its metrics as validated live performance."
            ),
        )

    return (
        "warning",
        (
            f"WARNING: Experiment {experiment_id} has no explicit live "
            "inference provenance. Keep this artifact for the research record, "
            "but treat its metrics as unverified rather than validated live "
            "performance."
        ),
    )


def audit_result(path: Path) -> ResultAudit:
    """Inspect one result file and return its normalized provenance record."""
    payload = load_json(path)
    experiment_id = "".join(ch for ch in path.stem if ch.isdigit()) or path.stem
    detected_mode, source = detect_inference_mode(payload)
    normalized_mode = normalize_mode(detected_mode)
    status, header = build_header(experiment_id, normalized_mode, source)
    return ResultAudit(
        experiment_id=experiment_id,
        path=path,
        detected_mode=detected_mode,
        normalized_mode=normalized_mode,
        source=source,
        status=status,
        header=header,
    )


def annotate_result(path: Path, audit: ResultAudit) -> None:
    """Write the provenance annotation back into the result artifact."""
    payload = load_json(path)

    if audit.detected_mode is not None:
        payload.setdefault("inference_mode", audit.detected_mode)

    payload["result_header"] = audit.header
    payload["result_provenance"] = {
        "experiment": 209,
        "status": audit.status,
        "detected_mode": audit.detected_mode,
        "normalized_mode": audit.normalized_mode,
        "source": audit.source,
    }

    write_json(path, payload)


def percent(value: float, decimals: int = 1) -> str:
    """Format a ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def signed_pp(value: float, decimals: int = 1) -> str:
    """Format a ratio delta as signed percentage points."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{decimals}f}pp"


def describe_provenance(normalized_mode: str) -> str:
    """Human-readable provenance label."""
    if normalized_mode == "live_gpu":
        return "Validated live_gpu"
    if normalized_mode == "simulated":
        return "Simulated"
    return "Missing explicit inference provenance"


def collect_result_payloads(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all result payloads keyed by experiment id."""
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob(RESULT_GLOB)):
        experiment_id = "".join(ch for ch in path.stem if ch.isdigit()) or path.stem
        payloads[experiment_id] = load_json(path)
    return payloads


def build_cleanup_summary(audits: list[ResultAudit]) -> CleanupSummary:
    """Summarize the audit in a stable, comparable dataclass."""
    validated = tuple(a.experiment_id for a in audits if a.status == "validated_live_gpu")
    simulated = tuple(a.experiment_id for a in audits if a.normalized_mode == "simulated")
    unverified = tuple(a.experiment_id for a in audits if a.normalized_mode == "missing")
    return CleanupSummary(
        total_results=len(audits),
        validated_live_gpu=len(validated),
        simulated_results=len(simulated),
        unverified_results=len(unverified),
        validated_experiments=validated,
        simulated_experiments=simulated,
        unverified_experiments=unverified,
    )


def require_between(text: str, start: str, end_markers: tuple[str, ...], replacement: str) -> str:
    """Replace the body between markers while preserving the markers themselves."""
    start_index = text.find(start)
    if start_index == -1:
        raise ValueError(f"start marker not found: {start}")
    body_start = start_index + len(start)

    end_index = -1
    for marker in end_markers:
        candidate = text.find(marker, body_start)
        if candidate != -1 and (end_index == -1 or candidate < end_index):
            end_index = candidate
    if end_index == -1:
        raise ValueError(f"end marker not found for: {start}")

    return text[:body_start] + "\n\n" + replacement.rstrip() + "\n\n" + text[end_index:]


def replace_block(text: str, start: str, end_markers: tuple[str, ...], replacement: str) -> str:
    """Replace an entire block delimited by a starting marker and an end marker."""
    start_index = text.find(start)
    if start_index == -1:
        raise ValueError(f"start marker not found: {start}")

    end_index = -1
    matched_marker = ""
    for marker in end_markers:
        candidate = text.find(marker, start_index + len(start))
        if candidate != -1 and (end_index == -1 or candidate < end_index):
            end_index = candidate
            matched_marker = marker
    if end_index == -1:
        raise ValueError(f"end marker not found for: {start}")

    remainder = text[end_index + len(matched_marker) :].lstrip("\n")
    return text[:start_index] + replacement.rstrip() + "\n\n" + remainder


def maybe_replace(text: str, old: str, new: str) -> str:
    """Replace exact text when present; otherwise leave the string unchanged."""
    if old in text:
        return text.replace(old, new, 1)
    return text


def get_audit(audits: dict[str, ResultAudit], experiment_id: str) -> ResultAudit | None:
    """Convenience accessor for a known experiment id."""
    return audits.get(experiment_id)


def is_number(value: Any) -> bool:
    """Return True when a value is an int or float."""
    return isinstance(value, (int, float))


def build_readme_intro(
    summary: CleanupSummary, payloads: dict[str, dict[str, Any]]
) -> tuple[str, str]:
    """Generate the updated README overview paragraphs."""
    intro = (
        "Carnot is an Energy-Based Model framework for **verifying and "
        "repairing LLM outputs**. The repository now distinguishes validated "
        "live artifacts from simulated or otherwise unverified ones instead of "
        "presenting them as equivalent evidence."
    )

    exp208 = payloads.get("208", {})
    exp208_stats = exp208.get("statistics", {})
    exp208_baseline = exp208_stats.get("baseline", {}).get("pass_at_1")
    exp208_repair = exp208_stats.get("verify_repair", {}).get("pass_at_1")
    exp208_delta = exp208_stats.get("improvement", {}).get("delta")

    if all(
        isinstance(value, (float, int)) for value in (exp208_baseline, exp208_repair, exp208_delta)
    ):
        evidence = (
            f"**Evidence status:** The provenance audit found "
            f"{summary.validated_live_gpu} validated live artifacts, "
            f"{summary.simulated_results} simulated artifacts, and "
            f"{summary.unverified_results} artifacts missing explicit live "
            f"provenance. The clearest positive live benchmark today is Exp 208 "
            f"on HumanEval: {percent(float(exp208_baseline))} -> "
            f"{percent(float(exp208_repair))} ({signed_pp(float(exp208_delta))}). "
            "The larger GSM8K and adversarial gains remain in the record, but "
            "they are still simulated or otherwise unverified and are labeled "
            "that way below."
        )
    else:
        evidence = (
            f"**Evidence status:** The provenance audit found "
            f"{summary.validated_live_gpu} validated live artifacts, "
            f"{summary.simulated_results} simulated artifacts, and "
            f"{summary.unverified_results} artifacts missing explicit live "
            "provenance."
        )

    return intro, evidence


def build_readme_table(audits: dict[str, ResultAudit], payloads: dict[str, dict[str, Any]]) -> str:
    """Build the README results table with provenance labels."""
    rows = [
        "| Claim | Result | Provenance | Caveat |",
        "|-------|--------|------------|--------|",
    ]

    exp208 = payloads.get("208", {})
    exp208_stats = exp208.get("statistics", {})
    if exp208_stats:
        rows.append(
            "| Live HumanEval (Exp 208) | "
            f"{percent(float(exp208_stats['baseline']['pass_at_1']))} -> "
            f"{percent(float(exp208_stats['verify_repair']['pass_at_1']))} "
            f"({signed_pp(float(exp208_stats['improvement']['delta']))}) | "
            f"{describe_provenance(get_audit(audits, '208').normalized_mode if get_audit(audits, '208') else 'missing')} | "
            "Validated live code benchmark on 30 official problems; modest but real positive delta |"
        )

    exp184 = payloads.get("184", {})
    exp184_stats = exp184.get("standard_statistics", {})
    if exp184_stats:
        rows.append(
            "| Live GSM8K reality check (Exp 184) | "
            f"{percent(float(exp184_stats['baseline']['accuracy']))} -> "
            f"{percent(float(exp184_stats['verify_repair']['accuracy']))} "
            f"({signed_pp(float(exp184_stats['improvement_delta']))}) | "
            f"{describe_provenance(get_audit(audits, '184').normalized_mode if get_audit(audits, '184') else 'missing')} | "
            "Current live math evidence is mixed; this run regressed instead of improving |"
        )

    exp161 = payloads.get("161", {})
    exp161_stats = exp161.get("statistics", {})
    if exp161_stats:
        qwen = exp161_stats.get("Qwen3.5-0.8B", {})
        gemma = exp161_stats.get("Gemma4-E4B-it", {})
        rows.append(
            "| Full GSM8K (Exp 161) | "
            f"Qwen {percent(float(qwen['baseline']['accuracy']))} -> "
            f"{percent(float(qwen['verify_repair']['accuracy']))}; "
            f"Gemma {percent(float(gemma['baseline']['accuracy']))} -> "
            f"{percent(float(gemma['verify_repair']['accuracy']))} | "
            f"{describe_provenance(get_audit(audits, '161').normalized_mode if get_audit(audits, '161') else 'missing')} | "
            "Strong full-dataset benchmark, but still simulated rather than validated live inference |"
        )

    exp178 = payloads.get("178", {})
    exp178_swap = exp178.get("per_variant", {}).get("number_swapped", {})
    deltas = exp178_swap.get("improvement_delta_pp_per_model", {})
    if deltas:
        rows.append(
            "| Adversarial GSM8K (Exp 178) | "
            f"Qwen +{float(deltas['Qwen3.5-0.8B']):.1f}pp; "
            f"Gemma +{float(deltas['Gemma4-E4B-it']):.1f}pp on number-swapped variants | "
            f"{describe_provenance(get_audit(audits, '178').normalized_mode if get_audit(audits, '178') else 'missing')} | "
            "Promising adversarial recovery, but still a simulated benchmark |"
        )

    exp134 = payloads.get("134", {})
    exp134_summary = exp134.get("summary", {})
    if exp134_summary:
        rows.append(
            "| Self-learning (Exp 134) | "
            f"{percent(float(exp134_summary['fixed_overall_accuracy']))} -> "
            f"{percent(float(exp134_summary['adaptive_overall_accuracy']))} | "
            f"{describe_provenance(get_audit(audits, '134').normalized_mode if get_audit(audits, '134') else 'missing')} | "
            "Retained as a research result, but the artifact lacks explicit live inference provenance |"
        )

    exp158 = payloads.get("158", {})
    if "coverage_pct" in exp158:
        rows.append(
            "| Factual coverage (Exp 158) | "
            f"{float(exp158['coverage_pct']):.1f}% claim coverage | "
            f"{describe_provenance(get_audit(audits, '158').normalized_mode if get_audit(audits, '158') else 'missing')} | "
            "Coverage study preserved as historical evidence; not a validated live end-to-end repair benchmark |"
        )

    return (
        "The table keeps the strongest historical numbers in the research "
        "record while making the evidence status explicit.\n\n" + "\n".join(rows)
    )


def update_readme(
    root: Path,
    summary: CleanupSummary,
    audits: dict[str, ResultAudit],
    payloads: dict[str, dict[str, Any]],
) -> None:
    """Rewrite the README overview and key-results section."""
    path = root / "README.md"
    text = path.read_text()

    intro_old = (
        "Carnot is an Energy-Based Model framework for **verifying and repairing "
        "LLM outputs**. Through 160+ experiments across 11 milestones, we proved "
        "that structural constraint verification via Ising models catches "
        "hallucinations that activation-based approaches miss — and that a "
        "verify-repair loop can fix them automatically."
    )
    breakthrough_old = (
        "**The breakthrough:** LLM proposes → Ising verifies → repair loop fixes. "
        "Full GSM8K (1,319 questions): +10-14% accuracy. Adversarial GSM8K "
        "(Apple methodology): +24-28% on number-swapped variants. HumanEval "
        "pass@1: 90%→96%. Self-learning pipeline: 67.6%→97.0% over 500 "
        "questions. 0.006ms per constraint check enables real-time guided "
        "decoding."
    )
    intro_new, breakthrough_new = build_readme_intro(summary, payloads)

    text = maybe_replace(text, intro_old, intro_new)
    text = maybe_replace(text, breakthrough_old, breakthrough_new)
    if "### What actually works in practice" in text:
        text = text.replace(
            "### What actually works in practice",
            "### Headline results with provenance",
            1,
        )
    text = require_between(
        text,
        "### Headline results with provenance",
        ("### What works on test sets but fails in practice",),
        build_readme_table(audits, payloads),
    )
    path.write_text(text)


def build_report_phase2_paragraph(
    audits: dict[str, ResultAudit], payloads: dict[str, dict[str, Any]]
) -> str:
    """Generate the revised Phase 2 abstract paragraph."""
    exp161 = payloads.get("161", {}).get("statistics", {})
    exp178 = payloads.get("178", {}).get("per_variant", {}).get("number_swapped", {})
    exp208 = payloads.get("208", {}).get("statistics", {})
    exp134 = payloads.get("134", {}).get("summary", {})
    exp158 = payloads.get("158", {})

    qwen = exp161.get("Qwen3.5-0.8B", {})
    gemma = exp161.get("Gemma4-E4B-it", {})
    deltas = exp178.get("improvement_delta_pp_per_model", {})

    if not (
        qwen.get("baseline")
        and qwen.get("verify_repair")
        and gemma.get("baseline")
        and gemma.get("verify_repair")
        and deltas
        and exp208.get("baseline")
        and exp208.get("verify_repair")
        and exp134
        and "coverage_pct" in exp158
    ):
        return (
            "**Phase 2 (Constraint-based, Experiments 39-160+):** The paradigm "
            "shift from detection to verification produced a mix of validated "
            "live, simulated, and unverified evidence. Exp 209 preserves the "
            "historical record while requiring public docs to label whether a "
            "headline number is validated live, simulated, or missing explicit "
            "live inference provenance."
        )

    return (
        "**Phase 2 (Constraint-based, Experiments 39-160+):** The paradigm shift "
        "from detection to verification produced a mix of validated live, "
        "simulated, and unverified evidence. The strongest full-dataset GSM8K "
        f"numbers remain {describe_provenance(get_audit(audits, '161').normalized_mode if get_audit(audits, '161') else 'missing').lower()} "
        f"(Exp 161: Qwen {percent(float(qwen['baseline']['accuracy']))} -> "
        f"{percent(float(qwen['verify_repair']['accuracy']))}, Gemma "
        f"{percent(float(gemma['baseline']['accuracy']))} -> "
        f"{percent(float(gemma['verify_repair']['accuracy']))}); the strongest "
        "adversarial recovery remains "
        f"{describe_provenance(get_audit(audits, '178').normalized_mode if get_audit(audits, '178') else 'missing').lower()} "
        f"(Exp 178: Qwen +{float(deltas['Qwen3.5-0.8B']):.1f}pp, Gemma "
        f"+{float(deltas['Gemma4-E4B-it']):.1f}pp on number-swapped variants); "
        "the clearest positive validated live benchmark is Exp 208 on "
        f"HumanEval ({percent(float(exp208['baseline']['pass_at_1']))} -> "
        f"{percent(float(exp208['verify_repair']['pass_at_1']))}); the "
        f"self-learning result ({percent(float(exp134['fixed_overall_accuracy']))} -> "
        f"{percent(float(exp134['adaptive_overall_accuracy']))}) and factual "
        f"coverage ({float(exp158['coverage_pct']):.1f}%) remain in the record "
        "but lack explicit live inference provenance."
    )


def build_simulation_vs_reality(
    summary: CleanupSummary, audits: dict[str, ResultAudit], payloads: dict[str, dict[str, Any]]
) -> str:
    """Build the technical-report provenance disclosure section."""
    live_list = ", ".join(f"Exp {exp}" for exp in summary.validated_experiments) or "none"
    simulated_list = ", ".join(f"Exp {exp}" for exp in summary.simulated_experiments) or "none"

    lines = [
        "## Simulation vs Reality",
        "",
        f"This provenance audit found {summary.validated_live_gpu} validated live_gpu artifacts, {summary.simulated_results} explicitly simulated artifacts, and {summary.unverified_results} artifacts missing explicit live inference provenance.",
        "",
        f"- Validated live artifacts: {live_list}",
        f"- Simulated artifacts: {simulated_list}",
        f"- Unverified artifacts: {summary.unverified_results} result files without explicit live provenance",
        "",
        "| Headline claim | Current number | Provenance | Interpretation |",
        "|---------------|----------------|------------|----------------|",
    ]

    exp208 = payloads.get("208", {}).get("statistics", {})
    exp161 = payloads.get("161", {}).get("statistics", {})
    exp178 = payloads.get("178", {}).get("per_variant", {}).get("number_swapped", {})
    exp134 = payloads.get("134", {}).get("summary", {})
    exp158 = payloads.get("158", {})

    if exp208:
        lines.append(
            "| Live HumanEval (Exp 208) | "
            f"{percent(float(exp208['baseline']['pass_at_1']))} -> "
            f"{percent(float(exp208['verify_repair']['pass_at_1']))} "
            f"({signed_pp(float(exp208['improvement']['delta']))}) | "
            f"{describe_provenance(get_audit(audits, '208').normalized_mode if get_audit(audits, '208') else 'missing')} | "
            "Best current validated live benchmark improvement |"
        )

    if exp161:
        qwen = exp161["Qwen3.5-0.8B"]
        gemma = exp161["Gemma4-E4B-it"]
        lines.append(
            "| Full GSM8K (Exp 161) | "
            f"Qwen {percent(float(qwen['baseline']['accuracy']))} -> "
            f"{percent(float(qwen['verify_repair']['accuracy']))}; "
            f"Gemma {percent(float(gemma['baseline']['accuracy']))} -> "
            f"{percent(float(gemma['verify_repair']['accuracy']))} | "
            f"{describe_provenance(get_audit(audits, '161').normalized_mode if get_audit(audits, '161') else 'missing')} | "
            "Promising full benchmark, but not yet validated as a full live benchmark |"
        )

    if exp178:
        deltas = exp178["improvement_delta_pp_per_model"]
        lines.append(
            "| Adversarial GSM8K (Exp 178) | "
            f"Qwen +{float(deltas['Qwen3.5-0.8B']):.1f}pp; Gemma +{float(deltas['Gemma4-E4B-it']):.1f}pp | "
            f"{describe_provenance(get_audit(audits, '178').normalized_mode if get_audit(audits, '178') else 'missing')} | "
            "Strong adversarial signal, but still simulated |"
        )

    if exp134:
        lines.append(
            "| Self-learning (Exp 134) | "
            f"{percent(float(exp134['fixed_overall_accuracy']))} -> "
            f"{percent(float(exp134['adaptive_overall_accuracy']))} | "
            f"{describe_provenance(get_audit(audits, '134').normalized_mode if get_audit(audits, '134') else 'missing')} | "
            "Useful research result, but not explicit live inference evidence |"
        )

    if exp158:
        lines.append(
            "| Factual coverage (Exp 158) | "
            f"{float(exp158['coverage_pct']):.1f}% | "
            f"{describe_provenance(get_audit(audits, '158').normalized_mode if get_audit(audits, '158') else 'missing')} | "
            "Coverage study retained with caveat rather than deleted |"
        )

    return "\n".join(lines)


def build_report_conclusion(
    audits: dict[str, ResultAudit], payloads: dict[str, dict[str, Any]]
) -> str:
    """Generate the revised conclusion bullets with provenance labels."""
    exp161 = payloads.get("161", {}).get("statistics", {})
    exp178 = payloads.get("178", {}).get("per_variant", {}).get("number_swapped", {})
    exp208 = payloads.get("208", {}).get("statistics", {})
    exp134 = payloads.get("134", {}).get("summary", {})
    exp158 = payloads.get("158", {})

    qwen = exp161.get("Qwen3.5-0.8B", {})
    gemma = exp161.get("Gemma4-E4B-it", {})
    deltas = exp178.get("improvement_delta_pp_per_model", {})

    if not (
        qwen.get("baseline")
        and qwen.get("verify_repair")
        and gemma.get("baseline")
        and gemma.get("verify_repair")
        and deltas
        and exp208.get("baseline")
        and exp208.get("verify_repair")
        and exp134
        and "coverage_pct" in exp158
    ):
        return (
            "- Headline benchmark claims are now labeled as validated live, "
            "simulated, or missing explicit inference provenance\n"
            "- Simulated and unverified artifacts are preserved as research "
            "history rather than deleted\n"
            "- The public docs now distinguish exploratory benchmarks from "
            "validated live evidence"
        )

    lines = [
        "- **Validated live HumanEval (Exp 208):** "
        f"{percent(float(exp208['baseline']['pass_at_1']))} -> "
        f"{percent(float(exp208['verify_repair']['pass_at_1']))} "
        f"({signed_pp(float(exp208['improvement']['delta']))})",
        "- **Full GSM8K (Exp 161, simulated):** "
        f"Qwen {percent(float(qwen['baseline']['accuracy']))} -> "
        f"{percent(float(qwen['verify_repair']['accuracy']))}, Gemma "
        f"{percent(float(gemma['baseline']['accuracy']))} -> "
        f"{percent(float(gemma['verify_repair']['accuracy']))}",
        "- **Adversarial GSM8K (Exp 178, simulated):** "
        f"Qwen +{float(deltas['Qwen3.5-0.8B']):.1f}pp, Gemma "
        f"+{float(deltas['Gemma4-E4B-it']):.1f}pp on number-swapped variants",
        "- **Self-learning Tier 1 (Exp 134, unverified provenance):** "
        f"{percent(float(exp134['fixed_overall_accuracy']))} -> "
        f"{percent(float(exp134['adaptive_overall_accuracy']))}",
        "- **Factual coverage (Exp 158, unverified provenance):** "
        f"{float(exp158['coverage_pct']):.1f}% via Wikidata knowledge base integration",
        "- **Experiment 56 (live small-sample):** 100% wrong-answer detection on a 20-question live study",
        "- **HumanEval pass@1 90% -> 96% (Experiment 68):** retained as a historical result, but not currently validated as a full live benchmark",
    ]
    return "\n".join(lines)


def update_technical_report(
    root: Path,
    summary: CleanupSummary,
    audits: dict[str, ResultAudit],
    payloads: dict[str, dict[str, Any]],
) -> None:
    """Rewrite the report's headline claims with provenance labels."""
    path = root / "docs" / "technical-report.md"
    text = path.read_text()

    phase2_old = (
        "**Phase 2 (Constraint-based, Experiments 39-160+):** The paradigm shift "
        "from detection to verification yielded: (1) full GSM8K (1,319 "
        "questions) showing Qwen3.5 improving from 70.6% to 84.4% and Gemma4 "
        "from 77.1% to 87.8% with verify-repair, (2) adversarial GSM8K (Apple "
        "methodology) recovering +24-28% accuracy on number-swapped variants, "
        "(3) self-learning Tier 1 improving from 67.6% to 97.0% accuracy over "
        "500 questions via online constraint generation, (4) 96% factual claim "
        "coverage via Wikidata knowledge base integration, (5) KAN energy tier "
        "achieving 0.994 AUROC with 8.7x fewer parameters than Ising, (6) JEPA "
        "predictive verification as a multi-domain predictor, (7) constraint "
        "state machine for agentic workflows catching 60/60 violations, (8) "
        "Ising constraint verification achieving 100% hallucination detection "
        "on live LLM output (Experiment 56), (9) HumanEval pass@1 improving "
        "from 90% to 96% (Experiment 68), (10) SAT solving scaling to 5000 "
        "variables in 0.7 seconds (Experiment 46b), and (11) the constraint "
        "pipeline transferring across model families without retraining (Experiment 69)."
    )
    text = maybe_replace(text, phase2_old, build_report_phase2_paragraph(audits, payloads))

    if "## Simulation vs Reality" not in text:
        insert_at = text.find("## 1. Introduction")
        if insert_at == -1:
            raise ValueError("could not insert Simulation vs Reality section")
        block = build_simulation_vs_reality(summary, audits, payloads) + "\n\n"
        text = text[:insert_at] + block + text[insert_at:]

    paradigm_old = (
        "The resulting architecture — LLM proposes, Ising verifies, repair loop "
        "fixes — proved dramatically more effective than any activation-based "
        "approach. On live LLM output, it achieves 100% hallucination detection "
        "(vs 50% practical for activation EBMs), +27% accuracy improvement via "
        "repair (vs -3% to -6% for activation-based rejection), and 96% pass@1 "
        "on HumanEval (vs 90% baseline)."
    )
    paradigm_new = (
        "The resulting architecture — LLM proposes, Ising verifies, repair loop "
        "fixes — clearly works as a live end-to-end pattern, but the evidence is "
        "more mixed than the earlier summaries implied. The live small-sample "
        "studies (Experiments 56-57) remain encouraging, the currently validated "
        "live benchmark gain is Exp 208 on HumanEval (16.7% -> 20.0%), and the "
        "larger GSM8K and adversarial gains elsewhere in this report remain "
        "simulated until rerun with explicit live provenance."
    )
    text = maybe_replace(text, paradigm_old, paradigm_new)

    phase3_old = (
        "Phase 2 validated individual components with simulated LLM outputs. "
        "Phase 3 connects a real LLM (Qwen3.5-0.8B, local) to the constraint "
        "pipeline and runs everything end-to-end. This is where the paradigm "
        "shift delivers its payoff."
    )
    phase3_new = (
        "Phase 2 validated individual components with simulated LLM outputs. "
        "Phase 3 connects a real LLM (Qwen3.5-0.8B, local) to the constraint "
        "pipeline and runs everything end-to-end. These live Experiments 53-64 "
        "are the strongest evidence that the architecture works at all, but "
        "they should not be conflated with the later simulated full-benchmark "
        "numbers."
    )
    text = maybe_replace(text, phase3_old, phase3_new)

    text = require_between(
        text,
        "### Part 2: Constraint-based verification works",
        ("### The story", "## 13. Pre-trained Models", "## 15. Limitations"),
        build_report_conclusion(audits, payloads),
    )

    text = maybe_replace(
        text,
        "validated it on published benchmarks (HumanEval, GSM8K)",
        "benchmarked it on published datasets, while the largest GSM8K-style gains remain simulated or otherwise unverified",
    )
    text = maybe_replace(
        text,
        "3. **Simulated fallbacks.** Some benchmark experiments (GSM8K, HumanEval) used simulated LLM outputs when model loading failed. Live results are available for Experiments 56-57 but not all benchmarks have been validated at full scale with live models.",
        (
            f"3. **Simulation vs reality gap.** The Exp 209 provenance audit found "
            f"{summary.validated_live_gpu} validated live_gpu artifacts, "
            f"{summary.simulated_results} explicitly simulated artifacts, and "
            f"{summary.unverified_results} artifacts missing explicit live "
            "provenance. Large GSM8K and adversarial improvements remain in the "
            "research record, but they are not yet validated as full live "
            "benchmarks."
        ),
    )
    text = maybe_replace(
        text,
        "**Result:** Starting from 60% accuracy on tricky questions, the verify-repair loop achieves 87% (+27% improvement). The architecture works; constraint coverage is the bottleneck (1/6 repair attempts triggered).",
        "**Result:** Starting from 60% accuracy on tricky questions, the verify-repair loop reaches 87% (+27% improvement) on this small live study. The architecture works, but the sample is too small to treat as a validated full benchmark and constraint coverage remains the bottleneck (1/6 repair attempts triggered).",
    )
    text = maybe_replace(
        text,
        "**HumanEval (Experiment 68):** 50 HumanEval-style problems through the full pipeline (extract -> instrument -> test -> fuzz -> repair). Pass@1 improves from 90% to 96% with Ising-guided fuzzing and repair. Bug detection breaks down across test execution, runtime instrumentation, and Ising-guided fuzzing — each catches bugs the others miss.",
        "**HumanEval (Experiment 68):** 50 HumanEval-style problems through the full pipeline (extract -> instrument -> test -> fuzz -> repair). This historical benchmark reported pass@1 improving from 90% to 96%, but it is not currently validated as a full live benchmark. Bug detection breaks down across test execution, runtime instrumentation, and Ising-guided fuzzing — each catches bugs the others miss.",
    )

    path.write_text(text)


def build_index_results(summary: CleanupSummary, payloads: dict[str, dict[str, Any]]) -> str:
    """Build the docs landing-page results section."""
    exp208 = payloads.get("208", {}).get("statistics", {})
    exp184 = payloads.get("184", {}).get("standard_statistics", {})
    exp161 = payloads.get("161", {}).get("statistics", {})
    exp178 = payloads.get("178", {}).get("per_variant", {}).get("number_swapped", {})
    exp134 = payloads.get("134", {}).get("summary", {})
    exp158 = payloads.get("158", {})

    qwen = exp161.get("Qwen3.5-0.8B", {})
    deltas = exp178.get("improvement_delta_pp_per_model", {})

    cards: list[str] = []

    if exp208.get("baseline") and exp208.get("verify_repair"):
        cards.append(
            f"""      <div class="r-card">
        <span class="r-tag">Validated live GPU</span>
        <h3 class="r-title">Live HumanEval (Exp 208)</h3>
        <div class="r-stats"><span class="r-before">Baseline: {percent(float(exp208["baseline"]["pass_at_1"]))}</span> <span class="r-after">+Repair: {percent(float(exp208["verify_repair"]["pass_at_1"]))}</span></div>
      </div>"""
        )

    if exp184.get("baseline") and exp184.get("verify_repair"):
        cards.append(
            f"""      <div class="r-card">
        <span class="r-tag">Validated live GPU</span>
        <h3 class="r-title">Live GSM8K reality check (Exp 184)</h3>
        <div class="r-stats"><span class="r-before">Baseline: {percent(float(exp184["baseline"]["accuracy"]))}</span> <span class="r-after">+Repair: {percent(float(exp184["verify_repair"]["accuracy"]))}</span></div>
      </div>"""
        )

    if qwen.get("baseline") and qwen.get("verify_repair"):
        cards.append(
            f"""      <div class="r-card">
        <span class="r-tag">Simulated benchmark</span>
        <h3 class="r-title">Full GSM8K (Exp 161)</h3>
        <div class="r-stats"><span class="r-before">Qwen: {percent(float(qwen["baseline"]["accuracy"]))}</span> <span class="r-after">Repair: {percent(float(qwen["verify_repair"]["accuracy"]))}</span></div>
      </div>"""
        )

    if deltas:
        cards.append(
            f"""      <div class="r-card">
        <span class="r-tag">Simulated benchmark</span>
        <h3 class="r-title">Adversarial GSM8K (Exp 178)</h3>
        <div class="r-stats"><span class="r-before">Qwen: +{float(deltas["Qwen3.5-0.8B"]):.1f}pp</span> <span class="r-after">Gemma: +{float(deltas["Gemma4-E4B-it"]):.1f}pp</span></div>
      </div>"""
        )

    if exp134:
        cards.append(
            f"""      <div class="r-card">
        <span class="r-tag">Missing explicit inference provenance</span>
        <h3 class="r-title">Self-learning (Exp 134)</h3>
        <div class="r-stats"><span class="r-before">Fixed: {percent(float(exp134["fixed_overall_accuracy"]))}</span> <span class="r-after">Adaptive: {percent(float(exp134["adaptive_overall_accuracy"]))}</span></div>
      </div>"""
        )

    if "coverage_pct" in exp158:
        cards.append(
            f"""      <div class="r-card">
        <span class="r-tag">Missing explicit inference provenance</span>
        <h3 class="r-title">Factual coverage (Exp 158)</h3>
        <div class="r-stats"><span class="r-before">Coverage</span> <span class="r-after">{float(exp158["coverage_pct"]):.1f}%</span></div>
      </div>"""
        )

    if not cards:
        cards.append(
            """      <div class="r-card">
        <span class="r-tag">Provenance audit</span>
        <h3 class="r-title">No benchmark rows available</h3>
        <div class="r-stats"><span class="r-before">Audit complete</span> <span class="r-after">Results labeled</span></div>
      </div>"""
        )

    return f"""<section id="results">
  <div class="container">
    <span class="section-label">Evidence</span>
    <h2 class="section-title">Measured Performance With Provenance</h2>
    <p class="section-desc">This provenance audit found {summary.validated_live_gpu} validated live GPU artifacts, {summary.simulated_results} simulated artifacts, and {summary.unverified_results} artifacts missing explicit live provenance. The cards below keep the strongest historical numbers while labeling what is and is not validated live evidence.</p>
    <div class="results-grid">
{chr(10).join(cards)}
    </div>
  </div>
</section>"""


def update_index(root: Path, summary: CleanupSummary, payloads: dict[str, dict[str, Any]]) -> None:
    """Rewrite the landing-page copy that previously implied uncaveated claims."""
    path = root / "docs" / "index.html"
    text = path.read_text()

    text = maybe_replace(
        text,
        "Constraint verification that catches what LLMs miss. Adversarial-robust, self-learning,\n      real-time guided decoding. Built in Rust + JAX. <code>pip install carnot</code>",
        "Constraint verification that catches what LLMs miss. Live and simulated research evidence are labeled separately after the Exp 209 provenance audit. Built in Rust + JAX. <code>pip install carnot</code>",
    )
    text = maybe_replace(
        text,
        '<div class="stat"><div class="stat-num">160+</div><div class="stat-label">Experiments</div></div>',
        '<div class="stat"><div class="stat-num">186</div><div class="stat-label">Experiments</div></div>',
    )
    text = maybe_replace(
        text,
        "VerifyRepairPipeline: extract constraints, verify via Ising, repair with LLM feedback. +15% on adversarial math, +6% on HumanEval. 5 lines of Python.",
        "VerifyRepairPipeline: extract constraints, verify via Ising, repair with LLM feedback. The clearest validated live benchmark is Exp 208 on HumanEval: 16.7% -> 20.0% on 30 official problems.",
    )
    text = maybe_replace(
        text,
        "Apple proved LLMs can't do math (accuracy drops 21% with irrelevant sentences). Carnot recovers +15% because Ising verification ignores irrelevant context entirely.",
        "The largest adversarial recovery numbers are currently simulated (Exp 178: +28.3pp Qwen, +24.0pp Gemma on number-swapped variants). They remain in the record, but are no longer presented as validated live results.",
    )
    text = maybe_replace(
        text,
        "Gets smarter with use: 67.6% &rarr; 97.0% over 500 questions. Online constraint generation from memory patterns, persistent across sessions. 96% factual claim coverage via Wikidata.",
        "Self-learning (Exp 134) still shows 67.6% &rarr; 97.0%, and factual coverage (Exp 158) still shows 96%, but both artifacts currently lack explicit live inference provenance and are labeled accordingly.",
    )

    text = replace_block(
        text,
        '<section id="results">',
        ('<section id="quickstart"', "<footer>", "</section>"),
        build_index_results(summary, payloads),
    )
    path.write_text(text)


def run_cleanup(root: Path) -> CleanupSummary:
    """Run the full provenance audit and rewrite the public docs."""
    results_dir = root / "results"
    audits_list = [audit_result(path) for path in sorted(results_dir.glob(RESULT_GLOB))]
    for audit in audits_list:
        annotate_result(audit.path, audit)

    summary = build_cleanup_summary(audits_list)
    audits = {audit.experiment_id: audit for audit in audits_list}
    payloads = collect_result_payloads(results_dir)

    update_readme(root, summary, audits, payloads)
    update_technical_report(root, summary, audits, payloads)
    update_index(root, summary, payloads)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    summary = run_cleanup(args.root)
    print(
        "Exp 209 cleanup complete:",
        f"{summary.total_results} artifacts scanned,",
        f"{summary.validated_live_gpu} validated live_gpu,",
        f"{summary.simulated_results} simulated,",
        f"{summary.unverified_results} unverified.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
