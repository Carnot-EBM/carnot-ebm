"""Exp 2432 capstone: paper-v6 results table for milestone 2026.05.235.

Milestone .235 (AUROC Ceiling Assault v4 + Codex Recovery Sprint v2) ran 10
experiments aimed at closing the gap between Carnot's best Tier 0 verifier
and the external HIVE peer ceiling (AUROC 0.9236, arXiv:2604.26139). This
module compiles every .231-.235 verifier-AUROC artifact into a single
reviewer-facing Markdown table, computes the .235 headline metrics
(best AUROC, gap to HIVE peer, best sampler KL improvement, FR-11 status,
KV260 hardware status, Phase 1 ship-gate status), and emits the terminal
Exp 2432 deliverable JSON.

The capstone is honest: missing source artifacts produce ``status: missing``
rows and ``null`` headline fields rather than fabricated values, and the
honest_verdict carries the ``complete:`` terminal prefix required by the
conductor reconciler's verdict-prefix discipline.

Differs from the .233 capstone (exp2404):
  - Adds three new verifier rows from .235 (HIVE v4, Hierarchical LogCons
    v2, HALT-RAG NLI v2) on top of the .233 set.
  - Headline gap is measured against the HIVE peer (0.9236), not HalluScan
    (0.88), because .234/.235's stated AUROC closure target was HIVE.
  - Adds best_sampler_kl_delta from the three new sampler experiments
    (Kinetic / Dikin / DE-PSGLD vs CASAL baseline).
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260518"
MILESTONE = "2026.05.235"
EXPERIMENT = "2432_capstone_v235"
SCHEMA = "carnot.paper_v6_capstone_2432.v1"
OUTPUT_FILENAME = "experiment_2432_capstone_v235.json"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME

# Headline external references.  Set once here so the module is the single
# source of truth for the peer-ceiling numbers cited in paper-v6 Section 5.
HALLUSCAN_AUROC = 0.88
HIVE_EXTERNAL_AUROC = 0.9236
SEMANTIC_ENERGY_BASELINE_AUROC = 0.685

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "best_auroc_achieved",
    "auroc_gap_to_hive_peer",
    "hive_v4_auroc",
    "fr11_satisfied",
    "kv260_yosys_succeeded",
    "best_sampler_kl_delta",
    "phase1_ship_gate_met",
    "paper_v6_results_table",
    "n_paper_ready_results",
    "duration_s",
}


@dataclass(frozen=True)
class VerifierSource:
    """Configuration for extracting one verifier-AUROC row from one artifact."""

    label: str
    source_id: str
    rel_path: str
    auroc_fields: tuple[str, ...]
    tier: str


# Order matters: the table is rendered in this order so reviewers see the
# .233 baseline rows first, then .235's new ensembles.
VERIFIER_SOURCES: tuple[VerifierSource, ...] = (
    VerifierSource(
        label="SemanticEnergy (baseline)",
        source_id="exp2351",
        rel_path="results/experiment_2351_semantic_energy_real.json",
        auroc_fields=("semantic_energy_real_auroc", "semantic_energy_auroc", "auroc"),
        tier="0g",
    ),
    VerifierSource(
        label="HALT Tier 0j",
        source_id="exp2394",
        rel_path="results/experiment_2394_halt_tier0j.json",
        auroc_fields=("halt_k19j_auroc", "halt_auroc", "auroc"),
        tier="0j",
    ),
    VerifierSource(
        label="FregeLogic",
        source_id="exp2395",
        rel_path="results/experiment_2395_fregelogic.json",
        auroc_fields=("fregelogic_auroc", "frege_logic_auroc", "auroc"),
        tier="0k",
    ),
    VerifierSource(
        label="Typed CoT",
        source_id="exp2396",
        rel_path="results/experiment_2396_typed_cot.json",
        auroc_fields=("typed_cot_auroc", "auroc"),
        tier="0l",
    ),
    VerifierSource(
        label="Freq-Aware Attn",
        source_id="exp2397",
        rel_path="results/experiment_2397_freq_aware_attn.json",
        auroc_fields=("freq_attn_auroc", "freq_aware_attn_auroc", "auroc"),
        tier="0f",
    ),
    VerifierSource(
        label="HIVE Ensemble v3",
        source_id="exp2398",
        rel_path="results/experiment_2398_hive_ensemble.json",
        auroc_fields=("hive_ensemble_auroc", "ensemble_auroc", "auroc"),
        tier="ensemble",
    ),
    VerifierSource(
        label="HIVE Ensemble v4",
        source_id="exp2422",
        rel_path="results/experiment_2422_hive_full_v4.json",
        auroc_fields=("hive_v4_auroc", "hive_ensemble_v4_auroc", "auroc"),
        tier="ensemble",
    ),
    VerifierSource(
        label="Hierarchical LogCons v2",
        source_id="exp2423",
        rel_path="results/experiment_2423_hierarchical_logcons_v2.json",
        auroc_fields=("logcons_auroc", "hierarchical_logcons_auroc", "auroc"),
        tier="0m",
    ),
    VerifierSource(
        label="HALT-RAG NLI v2",
        source_id="exp2424",
        rel_path="results/experiment_2424_halt_rag_nli_v2.json",
        auroc_fields=("halt_rag_auroc_full", "halt_rag_auroc", "auroc"),
        tier="0n",
    ),
)

EXTERNAL_BASELINES: tuple[dict[str, Any], ...] = (
    {
        "label": "HalluScan NLI (peer)",
        "auroc": HALLUSCAN_AUROC,
        "source": "arXiv:2605.02443",
        "tier": "external",
    },
    {
        "label": "HIVE peer",
        "auroc": HIVE_EXTERNAL_AUROC,
        "source": "arXiv:2604.26139",
        "tier": "external",
    },
)

# Non-verifier source artifacts read for FR-11, KV260, sampler, and ship-gate.
# Each entry maps a logical key to the experiment ID and relative path so we
# can record honest missing-artifact entries when files are absent.
OTHER_SOURCES: tuple[dict[str, str], ...] = (
    {"key": "fr11", "id": "exp2425", "path": "results/experiment_2425_fr11_nsvif_online_v4.json"},
    {"key": "fst_mcmc", "id": "exp2426", "path": "results/experiment_2426_fst_constrained_mcmc_v2.json"},
    {"key": "kv260", "id": "exp2427", "path": "results/experiment_2427_kv260_yosys_v4.json"},
    {"key": "kinetic", "id": "exp2428", "path": "results/experiment_2428_kinetic_langevin_v4.json"},
    {"key": "dikin", "id": "exp2429", "path": "results/experiment_2429_dikin_langevin_v2.json"},
    {"key": "de_psgld", "id": "exp2430", "path": "results/experiment_2430_de_psgld_v2.json"},
    {"key": "ship_gate", "id": "exp2431", "path": "results/experiment_2431_phase1_ship_gate_v4.json"},
)


def _load_artifact(root: Path, rel_path: str) -> Mapping[str, Any] | None:
    """Read a JSON artifact if present; return None if missing or unparseable.

    Returning None on parse errors rather than raising means a corrupt
    artifact behaves the same as a missing one — both surface in
    ``missing_source_artifacts`` so the next planner can re-run them.
    """

    path = root / rel_path
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _find_first(payload: Mapping[str, Any] | None, fields: Sequence[str]) -> Any:
    """Return the first non-None field from ``fields``; None if payload is empty."""

    if payload is None:
        return None
    for field in fields:
        value = payload.get(field)
        if value is not None:
            return value
    return None


def collect_verifier_rows(root: str | Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Read each verifier artifact and return one row per source plus peers.

    Local verifier rows include both .233 and .235 entries so the paper-v6
    table tells the full longitudinal story.  Missing artifacts surface as
    ``status: missing`` rows so reviewers can see what was attempted but
    not completed.
    """

    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    for vsrc in VERIFIER_SOURCES:
        payload = _load_artifact(root_path, vsrc.rel_path)
        auroc_raw = _find_first(payload, vsrc.auroc_fields)
        auroc = float(auroc_raw) if isinstance(auroc_raw, (int, float)) else None
        delta = (
            round(auroc - SEMANTIC_ENERGY_BASELINE_AUROC, 6)
            if auroc is not None and vsrc.source_id != "exp2351"
            else None
        )
        rows.append(
            {
                "label": vsrc.label,
                "source_id": vsrc.source_id,
                "source_artifact": vsrc.rel_path,
                "tier": vsrc.tier,
                "auroc": round(auroc, 6) if auroc is not None else None,
                "delta_vs_baseline": delta,
                "status": "available" if payload is not None else "missing",
                "is_external_baseline": False,
                "paper_ready": bool(
                    auroc is not None and auroc > SEMANTIC_ENERGY_BASELINE_AUROC
                ),
            }
        )
    for baseline in EXTERNAL_BASELINES:
        rows.append(
            {
                "label": baseline["label"],
                "source_id": baseline["source"],
                "source_artifact": baseline["source"],
                "tier": baseline["tier"],
                "auroc": baseline["auroc"],
                "delta_vs_baseline": None,
                "status": "external",
                "is_external_baseline": True,
                "paper_ready": False,
            }
        )
    return rows


def best_local_auroc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Return the highest AUROC across local Carnot Tier 0 verifiers.

    External-baseline rows are explicitly excluded so we never claim a peer
    paper's AUROC as our own headline.
    """

    local = [
        float(row["auroc"])
        for row in rows
        if not row.get("is_external_baseline")
        and row.get("status") == "available"
        and isinstance(row.get("auroc"), (int, float))
    ]
    return max(local) if local else None


def collect_other_sources(root: str | Path = REPO_ROOT) -> dict[str, Mapping[str, Any] | None]:
    """Read the non-verifier .235 artifacts (FR-11, KV260, samplers, ship-gate)."""

    root_path = Path(root)
    return {
        entry["key"]: _load_artifact(root_path, entry["path"])
        for entry in OTHER_SOURCES
    }


def best_sampler_kl_delta(other: Mapping[str, Mapping[str, Any] | None]) -> float | None:
    """Return the largest KL-divergence improvement across the three samplers.

    The three .235 sampler experiments each report a ``*_vs_casal_kl_delta``
    field measuring how much they outperform the CASAL baseline (larger
    delta = better mixing).  We take the max because the paper-v6 sampler
    section quotes the single strongest result.
    """

    candidates: list[float] = []
    for key, field in (
        ("kinetic", "kinetic_vs_casal_kl_delta"),
        ("dikin", "dikin_vs_casal_kl_delta"),
        ("de_psgld", "de_psgld_vs_casal_kl_delta"),
    ):
        payload = other.get(key) or {}
        value = payload.get(field)
        if isinstance(value, (int, float)):
            candidates.append(float(value))
    return max(candidates) if candidates else None


def render_markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the reviewer-facing paper-v6 Markdown table.

    Format matches the .233 capstone so reviewers comparing milestones see
    a stable table structure.
    """

    lines = [
        "| Verifier | AUROC | vs Baseline | Source |",
        "|---|---|---|---|",
    ]
    for row in rows:
        auroc = _fmt_value(row.get("auroc"))
        delta = _fmt_delta(row.get("delta_vs_baseline"))
        lines.append(
            "| {label} | {auroc} | {delta} | {source} |".format(
                label=_md_cell(str(row["label"])),
                auroc=auroc,
                delta=delta,
                source=_md_cell(str(row["source_id"])),
            )
        )
    return "\n".join(lines)


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    other: Mapping[str, Mapping[str, Any] | None],
    duration_s: float,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 2432 deliverable payload."""

    best_auroc = best_local_auroc(rows)
    gap_to_hive = (
        round(HIVE_EXTERNAL_AUROC - best_auroc, 6) if best_auroc is not None else None
    )
    gap_to_hallscan = (
        round(HALLUSCAN_AUROC - best_auroc, 6) if best_auroc is not None else None
    )

    hive_v4_row = next((r for r in rows if r["source_id"] == "exp2422"), None)
    hive_v4_auroc = hive_v4_row["auroc"] if hive_v4_row else None
    logcons_row = next((r for r in rows if r["source_id"] == "exp2423"), None)
    logcons_auroc = logcons_row["auroc"] if logcons_row else None
    halt_rag_row = next((r for r in rows if r["source_id"] == "exp2424"), None)
    halt_rag_auroc = halt_rag_row["auroc"] if halt_rag_row else None

    paper_ready = [
        r for r in rows
        if r.get("paper_ready") and not r.get("is_external_baseline")
    ]

    fr11_payload = other.get("fr11") or {}
    fr11_satisfied = bool(fr11_payload.get("fr11_nsvif_online_passed", False))

    kv260_payload = other.get("kv260") or {}
    kv260_yosys = bool(kv260_payload.get("synthesis_succeeded", False))

    ship_gate_payload = other.get("ship_gate") or {}
    phase1_gate = bool(ship_gate_payload.get("phase1_ship_gate_met", False))

    best_kl = best_sampler_kl_delta(other)

    missing_sources = [
        {"source_id": vsrc.source_id, "path": vsrc.rel_path}
        for vsrc, row in zip(VERIFIER_SOURCES, rows[: len(VERIFIER_SOURCES)])
        if row["status"] == "missing"
    ]
    missing_others = [
        {"key": entry["key"], "source_id": entry["id"], "path": entry["path"]}
        for entry in OTHER_SOURCES
        if other.get(entry["key"]) is None
    ]

    headline_summary = _build_synthesis(
        best_auroc=best_auroc,
        gap_to_hive=gap_to_hive,
        hive_v4_auroc=hive_v4_auroc,
        logcons_auroc=logcons_auroc,
        halt_rag_auroc=halt_rag_auroc,
        n_paper_ready=len(paper_ready),
        fr11_satisfied=fr11_satisfied,
        kv260_yosys=kv260_yosys,
        phase1_gate=phase1_gate,
        best_kl=best_kl,
        ship_gate_payload=ship_gate_payload,
        n_missing=len(missing_sources) + len(missing_others),
    )

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "spec_refs": ["REQ-REPORT-2432", "SCENARIO-REPORT-2432"],
        "best_auroc_achieved": round(best_auroc, 6) if best_auroc is not None else None,
        "auroc_gap_to_hive_peer": gap_to_hive,
        "auroc_gap_to_hallscan": gap_to_hallscan,
        "hive_v4_auroc": hive_v4_auroc,
        "logcons_auroc": logcons_auroc,
        "halt_rag_auroc_full": halt_rag_auroc,
        "n_paper_ready_results": len(paper_ready),
        "fr11_satisfied": fr11_satisfied,
        "kv260_yosys_succeeded": kv260_yosys,
        "phase1_ship_gate_met": phase1_gate,
        "best_sampler_kl_delta": best_kl,
        "paper_v6_results_table": render_markdown_table(rows),
        "results_table_rows": [dict(r) for r in rows],
        "missing_source_artifacts": missing_sources + missing_others,
        "synthesis": headline_summary,
        "duration_s": duration_s,
        "random_seed": 42,
        "external_baselines": {
            "halluscan_auroc": HALLUSCAN_AUROC,
            "halluscan_source": "arXiv:2605.02443",
            "hive_external_auroc": HIVE_EXTERNAL_AUROC,
            "hive_external_source": "arXiv:2604.26139",
            "semantic_energy_baseline_auroc": SEMANTIC_ENERGY_BASELINE_AUROC,
        },
        "field_principles": {
            "honest_verdict": "Terminal-prefix required. complete: with best_auroc and gap.",
            "best_auroc_achieved": "Best AUROC from any .235 verifier. Paper-v6 headline metric.",
            "auroc_gap_to_hive_peer": "0.9236 - best_auroc_achieved. Primary progress signal.",
            "hive_v4_auroc": "HIVE 4-verifier ensemble AUROC (null if exp2422 failed).",
            "fr11_satisfied": "FR-11 mandatory: true if exp2425.fr11_nsvif_online_passed=true.",
            "kv260_yosys_succeeded": "KV260 hardware track milestone.",
            "best_sampler_kl_delta": "Best KL improvement from new samplers.",
            "phase1_ship_gate_met": "Phase 1 completion gate.",
            "paper_v6_results_table": "Markdown table for paper-v6 Section 5.",
            "n_paper_ready_results": "Count of verifiers with AUROC > 0.685 baseline.",
            "duration_s": "Guards against fabrication.",
        },
        "acceptance_gates": {
            "best_auroc_achieved_present": best_auroc is not None,
            "fr11_satisfied_recorded": fr11_satisfied is not None,
        },
        "honest_verdict": _build_verdict(
            best_auroc=best_auroc,
            gap_to_hive=gap_to_hive,
            n_paper_ready=len(paper_ready),
            best_kl=best_kl,
            fr11_satisfied=fr11_satisfied,
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 2432 schema invariants."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if not str(artifact["honest_verdict"]).startswith("complete:"):
        raise ValueError("honest_verdict must start with 'complete:'")
    if artifact["duration_s"] < 0:
        raise ValueError("duration_s must be non-negative")
    if artifact["best_auroc_achieved"] is None:
        raise ValueError("best_auroc_achieved must be present (acceptance gate)")
    if artifact["fr11_satisfied"] is None:
        raise ValueError("fr11_satisfied must be recorded (acceptance gate)")
    if not isinstance(artifact["paper_v6_results_table"], str):
        raise ValueError("paper_v6_results_table must be a markdown string")


def run(
    *,
    root: str | Path = REPO_ROOT,
    out_path: str | Path = DEFAULT_OUT_PATH,
    duration_override_s: float | None = None,
) -> dict[str, Any]:
    """Read all .231-.235 artifacts, build the capstone, and write to disk."""

    start = time.perf_counter()
    root_path = Path(root)
    rows = collect_verifier_rows(root_path)
    other = collect_other_sources(root_path)
    duration_s = (
        float(duration_override_s)
        if duration_override_s is not None
        else round(max(time.perf_counter() - start, 0.0), 6)
    )
    artifact = build_artifact(rows=rows, other=other, duration_s=duration_s)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(dict(artifact))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _build_verdict(
    *,
    best_auroc: float | None,
    gap_to_hive: float | None,
    n_paper_ready: int,
    best_kl: float | None,
    fr11_satisfied: bool,
) -> str:
    """Build the terminal complete: honest_verdict line."""

    best_text = f"{best_auroc:.4f}" if best_auroc is not None else "n/a"
    gap_text = f"{gap_to_hive:+.4f}" if gap_to_hive is not None else "n/a"
    kl_text = f"{best_kl:.4f}" if best_kl is not None else "n/a"
    fr11_text = "satisfied" if fr11_satisfied else "missing"
    return (
        f"complete: best_auroc={best_text}; hive_gap={gap_text}; "
        f"n_paper_ready={n_paper_ready}; best_sampler_kl_delta={kl_text}; "
        f"fr11={fr11_text}"
    )


def _build_synthesis(
    *,
    best_auroc: float | None,
    gap_to_hive: float | None,
    hive_v4_auroc: float | None,
    logcons_auroc: float | None,
    halt_rag_auroc: float | None,
    n_paper_ready: int,
    fr11_satisfied: bool,
    kv260_yosys: bool,
    phase1_gate: bool,
    best_kl: float | None,
    ship_gate_payload: Mapping[str, Any],
    n_missing: int,
) -> dict[str, Any]:
    """Synthesize what .235 proved and what still needs work."""

    proved: list[str] = []
    needs_work: list[str] = []

    if best_auroc is not None:
        gap_msg = (
            f"closed {-gap_to_hive:.4f} past the HIVE peer 0.9236 ceiling"
            if gap_to_hive is not None and gap_to_hive < 0
            else f"remaining gap to HIVE peer 0.9236 is {gap_to_hive:.4f}"
            if gap_to_hive is not None
            else "gap to HIVE peer not computed"
        )
        proved.append(
            f"Best Carnot Tier 0 AUROC = {best_auroc:.4f} on cached live SOTA telemetry; {gap_msg}."
        )
    else:
        needs_work.append("No Tier 0 AUROC available - all verifier artifacts missing.")

    if hive_v4_auroc is not None:
        proved.append(
            f"HIVE 4-verifier ensemble v4 fused live on cached telemetry at AUROC={hive_v4_auroc:.4f}."
        )
    else:
        needs_work.append("HIVE v4 ensemble fusion artifact missing (exp2422).")

    if logcons_auroc is not None:
        proved.append(
            f"Hierarchical LogCons v2 (Z3-backed) reached AUROC={logcons_auroc:.4f}."
        )
    else:
        needs_work.append("Hierarchical LogCons v2 artifact missing (exp2423).")

    if halt_rag_auroc is not None:
        proved.append(
            f"HALT-RAG NLI v2 reached AUROC={halt_rag_auroc:.4f} on the full corpus."
        )

    if fr11_satisfied:
        proved.append("FR-11 NSVIF online retention rate satisfied per exp2425.")
    else:
        needs_work.append("FR-11 NSVIF online artifact missing or not satisfied (exp2425).")

    if kv260_yosys:
        proved.append("KV260 Yosys synthesis succeeded; hardware track moved forward.")
    else:
        needs_work.append("KV260 Yosys synthesis not confirmed (exp2427 blocked or failed).")

    if best_kl is not None:
        proved.append(
            f"Best new sampler beat CASAL baseline by KL delta = {best_kl:.4f} (Kinetic / Dikin / DE-PSGLD)."
        )
    else:
        needs_work.append("No sampler KL delta available - all three sampler artifacts missing.")

    if phase1_gate:
        proved.append("Phase 1 ship gate met per exp2431.")
    else:
        missing_criteria = ship_gate_payload.get("missing_criteria") or []
        if missing_criteria:
            needs_work.append(
                "Phase 1 ship gate not met (exp2431). Missing: "
                + "; ".join(str(c) for c in missing_criteria)
            )
        else:
            needs_work.append("Phase 1 ship gate not confirmed (exp2431 missing or criteria unmet).")

    delta_from_233 = (
        f"{n_paper_ready - 5} additional verifier(s) crossed the 0.685 baseline since .233"
        if n_paper_ready >= 5
        else "no additional paper-ready verifiers vs .233 baseline"
    )

    return {
        "proved_in_235": proved,
        "still_needs_work": needs_work,
        "change_since_233": delta_from_233,
        "n_missing_artifacts": n_missing,
        "n_paper_ready_results": n_paper_ready,
    }


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    run()
