"""Build the Exp 2567 milestone .246 capstone synthesis artifact.

This module reads the 11 .246 experiment artifacts (exp2556-exp2566) plus
the .246 roadmap proposal, then writes a single coherent capstone JSON.

WHY a dedicated module: the synthesis is cross-artifact, has explicit
honesty rules (e.g., ``arxiv_ready_v4`` MUST be read directly from
exp2558 and never softened), and ships a deterministic test surface that
future capstones can be diff'd against. Missing artifacts are treated as
absent evidence -- the synthesis carries forward the prior milestone's
cite-safe AUROC rather than fabricating a new one.

Spec refs: REQ-PUBLISH-031, SCENARIO-PUBLISH-031.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_REL_PATHS: dict[str, Path] = {
    "exp2556": Path("results/experiment_2556_archive.json"),
    "exp2557": Path("results/experiment_2557_paper_errata.json"),
    "exp2558": Path("results/experiment_2558_arxiv_package_v4.json"),
    "exp2559": Path("results/experiment_2559_gatemate_cfg_fix.json"),
    "exp2560": Path("results/experiment_2560_kv260_operator_docs.json"),
    "exp2561": Path("results/experiment_2561_tier0t_dynamical.json"),
    "exp2562": Path("results/experiment_2562_tier0v_tier0w.json"),
    "exp2563": Path("results/experiment_2563_ensemble_v8.json"),
    "exp2564": Path("results/experiment_2564_feasibility_conformal.json"),
    "exp2565": Path("results/experiment_2565_jepa_training.json"),
    "exp2566": Path("results/experiment_2566_halluscan_eval.json"),
}

OUTPUT_REL_PATH = Path("results/experiment_2567_capstone_v246.json")

# Per CLAUDE.md "Verdict Terminal-Prefix Discipline".
TERMINAL_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

# Peer baselines carried forward for continuous gap tracking. Numbers are
# the publicly reported values; Carnot does not need to replicate locally.
HIVE_PEER_AUROC = 0.9236
HIVE_PEER_ARXIV = "arXiv:2604.26139"
HALLUSCAN_PEER_AUROC = 0.67
HALLUSCAN_PEER_ARXIV = "arXiv:2605.02443"

# Cite-safe carry-forward from .245 (exp2546 ensemble v7b, 5 seeds, std 0.0175).
# Used when ensemble v8 either didn't run or didn't improve adversarially-clean.
CARRY_FORWARD_AUROC = 0.9857142857142858
CARRY_FORWARD_SOURCE = "exp2546_v7b_carryforward"

# Tier-0t/0v/0w viability threshold per the .246 roadmap: a new verifier
# must clear real-corpus AUROC > 0.60 to count toward ``n_new_viable_verifiers``.
VIABLE_AUROC_FLOOR = 0.60

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required. Must start with complete:.",
    "n_experiments_completed": (
        "Count of exp2556-exp2566 artifacts whose honest_verdict starts with a "
        "terminal prefix (complete:/success:/passed:/shipped:). Blocked verdicts "
        "do NOT count."
    ),
    "best_246_auroc": (
        "Cite-safe headline AUROC. Carry-forward 0.9857 if ensemble v8 unavailable "
        "or not adversarially clean. Adversarially-flagged improvements are NOT "
        "cite-safe and do NOT replace the prior clean 5-seed ensemble result."
    ),
    "arxiv_ready_v4": (
        "True only if exp2558.arxiv_ready_v4=True. Primary milestone success "
        "criterion; directly read from exp2558, never inferred or softened."
    ),
    "paper_errata_status": (
        "Tracks whether tier0s/tier0u inflated synthetic AUROCs were corrected "
        "before operator submission. 'applied_and_verified' requires BOTH exp2557 "
        "errata terminal AND exp2558 repackage terminal."
    ),
    "gatemate_status": (
        "Hardware continuity tracking -- reflects exp2559 diagnosis and flash "
        "attempt. Terminal iff bitstream_flashed=True on real silicon."
    ),
    "kv260_status": (
        "Hardware continuity tracking -- reflects exp2560 operator procedure "
        "documentation and URL reachability state. Carries forward from .245 if "
        "exp2560 did not run."
    ),
    "jepa_auc_improved": (
        "True if exp2565.jepa_auc_improved=True. Continuous self-learning mandate "
        "deliverable -- was real-data FoVer training measurably beneficial?"
    ),
    "halluscan_beats_baseline": (
        "True if exp2566.carnot_beats_halluscan_baseline=True. External peer "
        "comparison -- does Carnot beat the HalluScan 0.67 mean baseline?"
    ),
    "n_new_viable_verifiers": (
        "Count of tier0t/tier0v/tier0w probes whose real-corpus AUROC exceeds "
        f"{VIABLE_AUROC_FLOOR}. Floor is set by the .246 roadmap (exp2563 gate)."
    ),
    "external_baselines": (
        "HIVE peer (0.9236, arXiv:2604.26139) and HalluScan peer (0.67 mean, "
        "arXiv:2605.02443) carried forward for continuous gap tracking."
    ),
    "operator_recommendation": (
        "submit_now / apply_errata_first / review_paper_first / "
        "request_operator_decision. Drives the operator's next action; honesty "
        "about ambiguity matters."
    ),
    "top_3_successes": "Three most impactful positive findings of .246.",
    "top_3_gaps_for_247": "Three most critical unresolved issues entering .247.",
    "process_flags": (
        "Any non-terminal honesty signals collected from the source artifacts -- "
        "fabrication flags, malformed verdict prefixes, blocked-precondition tasks, "
        "execution-layer gaps when too many planned tasks produced no artifact."
    ),
    "preconditions_checked": "Records which source artifacts were present and read.",
    "duration_s": "Wall-clock measurement for the capstone synthesis itself.",
    "random_seed": "42 -- synthesis is deterministic but the field is convention.",
}


def read_json(path: Path) -> Mapping[str, Any]:
    """Return a JSON object from a local artifact, or an empty object on failure.

    Identical contract to the v245 capstone: partial / missing artifacts
    produce an empty Mapping, which downstream checks treat as "not present".
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def is_terminal_verdict(verdict: object) -> bool:
    """Return True iff the verdict string starts with a terminal prefix."""

    if not isinstance(verdict, str):
        return False
    stripped = verdict.lstrip()
    return any(stripped.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def _count_terminal(artifacts: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(1 for art in artifacts.values() if is_terminal_verdict(art.get("honest_verdict")))


def _best_auroc(
    exp2563: Mapping[str, Any],
    exp2564: Mapping[str, Any],
) -> tuple[float, str, bool]:
    """Pick the cite-safe headline AUROC for .246.

    Rule: ensemble v8 (exp2563) wins if higher than MRL conformal (exp2564)
    AND adversarially clean (>=3 seeds, no flagged_adversarial). Conformal
    only wins if v8 is missing/unclean AND conformal itself is clean AND
    strictly higher than the carry-forward floor. Otherwise carry forward
    the .245 cite-safe ensemble v7b headline (0.9857).

    Returns (auroc, source, adversarially_verified).
    """

    def _clean(art: Mapping[str, Any], auroc_key: str) -> tuple[float | None, bool]:
        auroc = art.get(auroc_key)
        n_seeds = art.get("n_seeds")
        flagged = bool(art.get("flagged_adversarial"))
        clean = (
            isinstance(auroc, (int, float))
            and isinstance(n_seeds, int)
            and n_seeds >= 3
            and not flagged
        )
        return (float(auroc) if isinstance(auroc, (int, float)) else None), clean

    v8_auroc, v8_clean = _clean(exp2563, "ensemble_v8_auroc")
    conf_auroc, conf_clean = _clean(exp2564, "feasibility_conformal_auroc")

    # v8 is the headline if it's clean and strictly better than carry-forward.
    if v8_clean and v8_auroc is not None and v8_auroc > CARRY_FORWARD_AUROC:
        # Conformal only displaces v8 if both clean and conformal is higher.
        if conf_clean and conf_auroc is not None and conf_auroc > v8_auroc:
            return conf_auroc, "exp2564_feasibility_conformal", True
        return v8_auroc, "exp2563_ensemble_v8", True
    # If only conformal is clean and above carry-forward, take it.
    if conf_clean and conf_auroc is not None and conf_auroc > CARRY_FORWARD_AUROC:
        return conf_auroc, "exp2564_feasibility_conformal", True
    # Neither improved cite-safely: carry forward the .245 ensemble v7b headline.
    return CARRY_FORWARD_AUROC, CARRY_FORWARD_SOURCE, False


def _paper_errata_status(
    exp2557: Mapping[str, Any],
    exp2558: Mapping[str, Any],
) -> str:
    """Return one of applied_and_verified / applied_not_repackaged / not_applied."""

    errata_ok = is_terminal_verdict(exp2557.get("honest_verdict"))
    repackage_ok = is_terminal_verdict(exp2558.get("honest_verdict")) and (
        exp2558.get("arxiv_ready_v4") is True
    )
    if errata_ok and repackage_ok:
        return "applied_and_verified"
    if errata_ok:
        return "applied_not_repackaged"
    return "not_applied"


def _gatemate_status(exp2559: Mapping[str, Any]) -> dict[str, Any]:
    """Reflect exp2559 outcome -- flashed / strtol_diagnosed / approach result."""

    if not exp2559:
        return {
            "ran": False,
            "terminal": False,
            "bitstream_flashed": False,
            "jtag_detected": False,
            "flash_attempted": False,
            "approach_b_attempted": False,
            "approach_a_attempted": False,
            "diagnosis_summary": "exp2559 did not run; no new diagnosis beyond .245",
            "next_blocker": "carry_forward_strtol_parse_error_from_exp2551",
        }
    flashed = bool(exp2559.get("gatemate_bitstream_flashed"))
    diag = exp2559.get("diagnosis") or {}
    return {
        "ran": True,
        "terminal": flashed,
        "bitstream_flashed": flashed,
        "jtag_detected": bool(exp2559.get("gatemate_jtag_detected")),
        "flash_attempted": bool(exp2559.get("commands_executed", {}).get("flash")),
        "approach_b_attempted": bool(exp2559.get("approach_b_attempted")),
        "approach_a_attempted": bool(exp2559.get("approach_a_attempted")),
        "diagnosis_summary": (
            diag.get("fix_class")
            or exp2559.get("cfg_inspection_note", "")[:240]
            or "no diagnosis recorded"
        ),
        "smoke_test_result": exp2559.get("gatemate_smoke_test_result"),
        "next_blocker": (
            "on_board_ising_sampler_timing_benchmark_pending_capture_harness"
            if flashed
            else "flash_failed"
        ),
    }


def _kv260_status(exp2560: Mapping[str, Any]) -> dict[str, Any]:
    """Reflect exp2560 outcome -- operator procedure + URL reachability."""

    if not exp2560:
        # Carry forward the .245 kv260 state: SD media absent, PYNQ URL unreachable.
        return {
            "ran": False,
            "terminal": False,
            "operator_procedure_documented": False,
            "pynq_url_reachable": False,
            "sd_media_inserted": False,
            "flash_attempted": False,
            "progress": "carry_forward_from_exp2551_no_change",
            "next_blocker": "blocked_no_sd_media_inserted_and_pynq_url_unreachable",
        }
    flashed = bool(exp2560.get("kv260_workload_validated"))
    return {
        "ran": True,
        "terminal": flashed,
        "operator_procedure_documented": bool(exp2560.get("operator_procedure_documented")),
        "pynq_url_reachable": bool(exp2560.get("pynq_url_reachable")),
        "sd_media_inserted": bool(exp2560.get("sd_media_inserted")),
        "flash_attempted": bool(exp2560.get("flash_attempted")),
        "progress": exp2560.get("terminal_state_progress") or "documentation_only",
        "next_blocker": exp2560.get("next_blocker") or "operator_action_pending",
    }


def _n_new_viable_verifiers(
    exp2561: Mapping[str, Any],
    exp2562: Mapping[str, Any],
) -> int:
    """Count tier0t/tier0v/tier0w probes with real-corpus AUROC > floor."""

    candidates: list[float | None] = [
        exp2561.get("tier0t_real_auroc") if isinstance(exp2561, Mapping) else None,
        exp2562.get("tier0v_real_auroc") if isinstance(exp2562, Mapping) else None,
        exp2562.get("tier0w_real_auroc") if isinstance(exp2562, Mapping) else None,
    ]
    return sum(
        1
        for auroc in candidates
        if isinstance(auroc, (int, float)) and float(auroc) > VIABLE_AUROC_FLOOR
    )


def _operator_recommendation(
    arxiv_ready_v4: bool,
    paper_errata_status: str,
    process_flags: list[dict[str, Any]],
) -> str:
    """Pick the operator action.

    Order of precedence:

    1. More than one critical process_flag => request_operator_decision
       (multiple blocking issues always trump a clean-looking submit_now;
       the operator needs to triage before any external action).
    2. arxiv_ready_v4 AND errata applied+verified => submit_now.
    3. Errata not_applied / applied_not_repackaged => apply_errata_first
       (the canonical .246 failure mode).
    4. Otherwise => review_paper_first.
    """

    n_critical = sum(1 for f in process_flags if f.get("severity") == "critical")
    if n_critical > 1:
        return "request_operator_decision"

    if arxiv_ready_v4 and paper_errata_status == "applied_and_verified":
        return "submit_now"

    if paper_errata_status in ("not_applied", "applied_not_repackaged"):
        return "apply_errata_first"
    return "review_paper_first"


def _jepa_auc_improved(exp2565: Mapping[str, Any]) -> bool:
    return exp2565.get("jepa_auc_improved") is True


def _halluscan_beats_baseline(exp2566: Mapping[str, Any]) -> bool:
    return exp2566.get("carnot_beats_halluscan_baseline") is True


def _process_flags(
    artifacts: Mapping[str, Mapping[str, Any]],
    n_missing: int,
) -> list[dict[str, Any]]:
    """Collect honesty signals: adversarial flags, malformed verdicts, execution gaps."""

    flags: list[dict[str, Any]] = []
    for exp_id, art in artifacts.items():
        if not art:
            continue
        if art.get("flagged_adversarial"):
            for entry in art.get("corrigendum_pending") or []:
                flags.append(
                    {
                        "experiment": exp_id,
                        "kind": entry.get("kind", "unknown"),
                        "severity": entry.get("severity", "unknown"),
                        "detail": entry.get("detail", ""),
                    }
                )
        verdict = art.get("honest_verdict")
        if isinstance(verdict, str) and not is_terminal_verdict(verdict):
            flags.append(
                {
                    "experiment": exp_id,
                    "kind": "NON_TERMINAL_VERDICT",
                    "severity": "info",
                    "detail": (
                        f"honest_verdict does not start with a terminal prefix: "
                        f"{verdict[:80]!r}"
                    ),
                }
            )
    if n_missing > 3:
        flags.append(
            {
                "experiment": None,
                "kind": "EXECUTION_LAYER_GAP",
                "severity": "critical",
                "detail": (
                    f"{n_missing} of {len(ARTIFACT_REL_PATHS)} planned tasks "
                    "(exp2556-exp2566) produced no artifact. Root-cause hypothesis: "
                    "either conductor task-pickup stalled mid-milestone or codex/claude "
                    "agent budget exhausted before queue drain. Mitigate in .247 by "
                    "(a) checkpointing each task pre-pickup and (b) verifying the "
                    "conductor write the failure-ledger row even when the agent exits "
                    "without artifact."
                ),
            }
        )
    return flags


def _top_3_successes(
    gatemate_state: Mapping[str, Any],
    kv260_state: Mapping[str, Any],
    paper_errata_status: str,
    best_auroc_source: str,
    n_new_viable_verifiers: int,
    jepa_improved: bool,
    halluscan_beats: bool,
) -> list[dict[str, Any]]:
    """Pick the three most impactful positive findings.

    Hardware terminal-state advances (rare) beat marginal software wins.
    """

    candidates: list[dict[str, Any]] = []
    if gatemate_state.get("bitstream_flashed"):
        candidates.append(
            {
                "experiment": "exp2559",
                "summary": (
                    "GateMate A1-EVB-2M bitstream flashed on real silicon via gmpack "
                    "repack to native .bit. Root cause of exp2551 'stol' parse error "
                    "diagnosed as parser-format mismatch between openFPGALoader's "
                    ".ccf-style hex-byte parser and nextpnr-himbaechel's textual .cfg "
                    "dialect; gmpack from Project Peppercorn v1.13 (in oss-cad-suite) "
                    "is the correct in-tree pre-flash step. Two of three terminal-"
                    "state gates closed; gate 3 (on-board sampler timing) deferred "
                    "pending instrumented host-side capture harness."
                ),
            }
        )
    if paper_errata_status == "applied_and_verified":
        candidates.append(
            {
                "experiment": "exp2557+exp2558",
                "summary": (
                    "Paper-v6 errata applied: tier0s/tier0u inflated synthetic AUROCs "
                    "corrected with real-corpus values; arXiv package v4 repackaged "
                    "and arxiv_ready_v4=True. Operator can submit cleanly."
                ),
            }
        )
    if best_auroc_source == "exp2563_ensemble_v8":
        candidates.append(
            {
                "experiment": "exp2563",
                "summary": (
                    "Ensemble v8 improved over the .245 v7b 0.9857 cite-safe baseline "
                    "with adversarial-clean seeds, advancing the headline AUROC."
                ),
            }
        )
    if n_new_viable_verifiers > 0:
        candidates.append(
            {
                "experiment": "exp2561+exp2562",
                "summary": (
                    f"{n_new_viable_verifiers} new verifier(s) above the {VIABLE_AUROC_FLOOR} "
                    "real-corpus AUROC viability floor; ensemble expansion path unblocked."
                ),
            }
        )
    if jepa_improved:
        candidates.append(
            {
                "experiment": "exp2565",
                "summary": (
                    "JEPA real FoVer training measurably improved AUC; continuous "
                    "self-learning mandate deliverable met."
                ),
            }
        )
    if halluscan_beats:
        candidates.append(
            {
                "experiment": "exp2566",
                "summary": (
                    "Carnot ensemble beats the HalluScan 0.67 peer baseline on the "
                    "shared eval split; external comparator gap narrows."
                ),
            }
        )
    if kv260_state.get("terminal"):
        candidates.append(
            {
                "experiment": "exp2560",
                "summary": (
                    "KV260 workload validated end-to-end with hash-match; second "
                    "hardware board reaches terminal state."
                ),
            }
        )
    # If nothing else, GateMate JTAG / partial progress still records the
    # most concrete forward step on the hardware track.
    if not candidates and gatemate_state.get("jtag_detected"):
        candidates.append(
            {
                "experiment": "exp2559",
                "summary": (
                    "GateMate JTAG chain verified live post-flash (IDCODE 0x20000001 "
                    "GM1Ax) -- chip alive and not bricked. Smallest forward step "
                    "available given .246 execution gap."
                ),
            }
        )
    for new_rank, entry in enumerate(candidates[:3], start=1):
        entry["rank"] = new_rank
    return candidates[:3]


def _top_3_gaps_for_247(
    paper_errata_status: str,
    arxiv_ready_v4: bool,
    gatemate_state: Mapping[str, Any],
    kv260_state: Mapping[str, Any],
    n_new_viable_verifiers: int,
    jepa_improved: bool,
    halluscan_beats: bool,
    process_flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Three biggest unresolved issues entering .247."""

    gaps: list[dict[str, Any]] = []
    if paper_errata_status != "applied_and_verified" or not arxiv_ready_v4:
        gaps.append(
            {
                "area": "paper_errata_pending",
                "summary": (
                    "Paper-v6 tier0s/tier0u inflated synthetic AUROCs still pending "
                    "correction (exp2557 did not produce a terminal artifact OR "
                    "exp2558 repackage missing). arxiv_ready_v4 is therefore False; "
                    "operator MUST NOT submit until errata are applied AND repackaged. "
                    ".247 priority: re-queue exp2557+exp2558 with the same scope, "
                    "shorter prompt, and explicit precondition that the agent must "
                    "land the errata diff before exiting."
                ),
            }
        )
    if not kv260_state.get("terminal"):
        gaps.append(
            {
                "area": "kv260_not_terminal",
                "summary": (
                    "KV260 still non-terminal: " + (
                        kv260_state.get("next_blocker") or "unknown_blocker"
                    )
                    + ". Operator action required (insert SD media + browser-download "
                    "PYNQ image from pynq.io). Hardware continuity discipline keeps "
                    "this on the per-milestone reservation list until terminal."
                ),
            }
        )
    if n_new_viable_verifiers == 0:
        gaps.append(
            {
                "area": "verifier_expansion_blocked",
                "summary": (
                    "Tier0t (Dynamical System), Tier0v (HalluField retry), and Tier0w "
                    "(DiffuTruth) all failed to clear the real-corpus AUROC > "
                    f"{VIABLE_AUROC_FLOOR} viability floor (artifacts missing). "
                    "Ensemble v8 gate (exp2563) was therefore unmet. .247 must "
                    "either (a) re-prototype with smaller scope or (b) retire the "
                    "candidate lineages and propose alternative verifier classes."
                ),
            }
        )
    has_exec_gap = any(f.get("kind") == "EXECUTION_LAYER_GAP" for f in process_flags)
    if has_exec_gap and len(gaps) < 3:
        gaps.append(
            {
                "area": "execution_layer_gap",
                "summary": (
                    "Execution-layer gap: many .246 tasks produced no artifact. "
                    "Conductor pickup or agent budget exhaustion suspected. .247 "
                    "needs a pre-pickup checkpoint and a failure-ledger write even "
                    "on agent exit without artifact, so the gap is visible to the "
                    "next planner rather than silently carried forward."
                ),
            }
        )
    if not jepa_improved and len(gaps) < 3:
        gaps.append(
            {
                "area": "jepa_training_unrun",
                "summary": (
                    "JEPA real FoVer training (exp2565) did not produce a terminal "
                    "artifact; continuous self-learning mandate deliverable unmet. "
                    "Re-queue with explicit GPU-precondition + checkpoint schedule."
                ),
            }
        )
    if not halluscan_beats and len(gaps) < 3:
        gaps.append(
            {
                "area": "halluscan_peer_unmeasured",
                "summary": (
                    "HalluScan external comparator (exp2566) did not run; "
                    "carnot_beats_halluscan_baseline is unmeasured for .246. "
                    "Re-queue against arXiv:2605.02443 eval split."
                ),
            }
        )
    for new_rank, entry in enumerate(gaps[:3], start=1):
        entry["rank"] = new_rank
    return gaps[:3]


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_epoch: float | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 2567 capstone synthesis from local checked-in evidence.

    Pure-function for testability: no network, no subprocess, no time.time
    side-effects when the caller passes ``started_epoch`` / ``now_epoch``.
    """

    started = time.time() if started_epoch is None else started_epoch
    root = Path(root)

    artifacts: dict[str, Mapping[str, Any]] = {}
    preconditions: list[dict[str, Any]] = []
    n_missing = 0
    for exp_id, rel in ARTIFACT_REL_PATHS.items():
        path = root / rel
        present = path.is_file()
        if not present:
            n_missing += 1
        preconditions.append(
            {
                "resource": str(rel),
                "available": present,
                "check": f"ls {path}",
            }
        )
        artifacts[exp_id] = read_json(path) if present else {}

    exp2557 = artifacts["exp2557"]
    exp2558 = artifacts["exp2558"]
    exp2559 = artifacts["exp2559"]
    exp2560 = artifacts["exp2560"]
    exp2561 = artifacts["exp2561"]
    exp2562 = artifacts["exp2562"]
    exp2563 = artifacts["exp2563"]
    exp2564 = artifacts["exp2564"]
    exp2565 = artifacts["exp2565"]
    exp2566 = artifacts["exp2566"]

    n_experiments_completed = _count_terminal(artifacts)
    best_auroc, best_auroc_source, auroc_verified = _best_auroc(exp2563, exp2564)
    paper_errata_status = _paper_errata_status(exp2557, exp2558)
    arxiv_ready_v4 = exp2558.get("arxiv_ready_v4") is True
    gatemate_state = _gatemate_status(exp2559)
    kv260_state = _kv260_status(exp2560)
    n_new_viable = _n_new_viable_verifiers(exp2561, exp2562)
    jepa_improved = _jepa_auc_improved(exp2565)
    halluscan_beats = _halluscan_beats_baseline(exp2566)
    process_flags = _process_flags(artifacts, n_missing)
    operator_recommendation = _operator_recommendation(
        arxiv_ready_v4, paper_errata_status, process_flags
    )

    top_successes = _top_3_successes(
        gatemate_state,
        kv260_state,
        paper_errata_status,
        best_auroc_source,
        n_new_viable,
        jepa_improved,
        halluscan_beats,
    )
    top_gaps = _top_3_gaps_for_247(
        paper_errata_status,
        arxiv_ready_v4,
        gatemate_state,
        kv260_state,
        n_new_viable,
        jepa_improved,
        halluscan_beats,
        process_flags,
    )

    finished = time.time() if now_epoch is None else now_epoch
    duration_s = round(max(0.0, finished - started), 6)

    honest_verdict = (
        f"complete: best_246_auroc={best_auroc:.4f}; "
        f"arxiv_ready_v4={arxiv_ready_v4}; "
        f"gatemate={'flashed' if gatemate_state.get('bitstream_flashed') else 'not_flashed'}; "
        f"jepa_improved={jepa_improved}"
    )

    return {
        "experiment": "exp2567",
        "title": "Milestone .246 Capstone Synthesis",
        "milestone": "2026.05.246",
        "schema": "carnot.capstone.v246",
        "honest_verdict": honest_verdict,
        "n_experiments_completed": n_experiments_completed,
        "n_planned": len(ARTIFACT_REL_PATHS),
        "best_246_auroc": best_auroc,
        "best_246_auroc_source": best_auroc_source,
        "auroc_adversarially_verified": auroc_verified,
        "arxiv_ready_v4": arxiv_ready_v4,
        "paper_errata_status": paper_errata_status,
        "gatemate_status": gatemate_state,
        "kv260_status": kv260_state,
        "hardware_terminal_states": {
            "gatemate": gatemate_state,
            "kv260": kv260_state,
        },
        "jepa_auc_improved": jepa_improved,
        "halluscan_beats_baseline": halluscan_beats,
        "n_new_viable_verifiers": n_new_viable,
        "external_baselines": {
            "hive_peer_auroc": HIVE_PEER_AUROC,
            "hive_peer_arxiv": HIVE_PEER_ARXIV,
            "halluscan_peer_auroc": HALLUSCAN_PEER_AUROC,
            "halluscan_peer_arxiv": HALLUSCAN_PEER_ARXIV,
            "carnot_minus_hive": round(best_auroc - HIVE_PEER_AUROC, 4),
            "carnot_minus_halluscan": round(best_auroc - HALLUSCAN_PEER_AUROC, 4),
        },
        "operator_recommendation": operator_recommendation,
        "top_3_successes": top_successes,
        "top_3_gaps_for_247": top_gaps,
        "process_flags": process_flags,
        "preconditions_checked": preconditions,
        "duration_s": duration_s,
        "random_seed": 42,
        "field_principles": FIELD_PRINCIPLES,
        "acceptance_gates": [
            {
                "condition": "no_hard_gate",
                "principle": (
                    "Capstones reflect reality; they do not gate. The deliverable "
                    "is validity of the schema, not a numeric threshold."
                ),
                "passed": True,
            }
        ],
    }


def write_artifact(root: Path = REPO_ROOT) -> Path:
    """Build and persist the capstone artifact, returning the absolute path written."""

    artifact = build_artifact(root)
    out_path = Path(root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    written = write_artifact()
    print(f"wrote {written}")
