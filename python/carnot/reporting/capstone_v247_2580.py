"""Build the Exp 2580 milestone .247 capstone synthesis artifact.

This module reads the 11 .247 experiment artifacts (exp2569-exp2579) and the
roadmap proposal, then writes a single coherent capstone JSON.

WHY a dedicated module: the synthesis is cross-artifact, has explicit
honesty rules (e.g., ``best_247_auroc`` MUST carry forward 0.9857 unless
ensemble v9 both improved AND is adversarially clean), and ships a
deterministic test surface that future capstones can be diff'd against.
Missing artifacts are treated as absent evidence -- the synthesis carries
forward the prior milestone's cite-safe values rather than fabricating new
ones.

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
    "exp2569": Path("results/experiment_2569_archive.json"),
    "exp2570": Path("results/experiment_2570_hf_model_cards.json"),
    "exp2571": Path("results/experiment_2571_ipfs_mirror.json"),
    "exp2572": Path("results/experiment_2572_tier0s_retrain.json"),
    "exp2573": Path("results/experiment_2573_tier0u_fix.json"),
    "exp2574": Path("results/experiment_2574_safety_corpus.json"),
    "exp2575": Path("results/experiment_2575_safety_ensemble.json"),
    "exp2576": Path("results/experiment_2576_jepa_v3_online.json"),
    "exp2577": Path("results/experiment_2577_gatemate_continuity.json"),
    "exp2578": Path("results/experiment_2578_kv260_continuity.json"),
    "exp2579": Path("results/experiment_2579_ensemble_v9.json"),
}

# The .246 capstone is read for carry-forward of hardware terminal states
# and arxiv_ready_v4 (which is a publication-track precondition, not a
# .247 deliverable). Treated as best-effort -- absence is fine.
PRIOR_CAPSTONE_REL_PATH = Path("results/experiment_2567_capstone_v246.json")

OUTPUT_REL_PATH = Path("results/experiment_2580_capstone_v247.json")

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

# Cite-safe carry-forward from .245 ensemble v7b (5 seeds, std 0.0175).
# Used when ensemble v9 either didn't run, didn't improve, or regressed
# below the v7b baseline. Adversarially-flagged improvements do NOT
# displace the prior clean result.
CARRY_FORWARD_AUROC = 0.9857142857142858
CARRY_FORWARD_SOURCE = "exp2546_v7b_carryforward"

# Per the .247 honesty protocol in the roadmap: EXECUTION_LAYER_GAP fires
# when more than this many planned tasks produced no artifact. The .246
# capstone fired this at >3 missing; the .247 plan tightened it to >4 to
# reduce noise on small-shortfall milestones while preserving the signal
# on actual stall cascades.
EXECUTION_GAP_THRESHOLD = 4

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required. Must start with complete:.",
    "n_experiments_completed": (
        "Count of exp2569-exp2579 artifacts whose honest_verdict starts with a "
        "terminal prefix (complete:/success:/passed:/shipped:). Blocked verdicts "
        "do NOT count."
    ),
    "best_247_auroc": (
        "Carry-forward or improved headline AUROC. Carry-forward 0.9857 if "
        "ensemble v9 regressed, was adversarially flagged, or didn't run. Only "
        "update if exp2579.ensemble_v9_viable=True AND no regression detected."
    ),
    "tier0s_real_improvement": (
        "Documents whether the real-corpus verifier gap narrowed for tier0s. "
        "True iff exp2572.tier0s_improved=True."
    ),
    "tier0u_real_improvement": (
        "Documents whether the real-corpus verifier gap narrowed for tier0u. "
        "True iff exp2573.tier0u_improved=True."
    ),
    "safety_classifier_viable": (
        "Tier B product milestone deliverable -- is the safety classifier viable? "
        "True iff exp2574.safety_verifier_viable=True AND "
        "exp2575.safety_integration_complete=True. Both gates must hold; the "
        "field is read from artifacts, never inferred from sibling state."
    ),
    "ipfs_mirror_status": (
        "Rule 3 (mandatory mirroring) compliance tracking. "
        "'pinned_cid_known' iff exp2571.ipfs_cid IS NOT NULL; "
        "'documented_operator_needed' otherwise."
    ),
    "gatemate_status": (
        "Hardware continuity tracking per CLAUDE.md Hardware-Task Continuity "
        "Discipline. Reflects exp2577 outcome (terminal / manual_repair_attempted "
        "/ jtag_not_detected); carries forward .246 terminal state if exp2577 "
        "did not run."
    ),
    "kv260_status": (
        "Hardware continuity tracking per CLAUDE.md Hardware-Task Continuity "
        "Discipline. Reflects exp2578 outcome (sd_flashed / sd_absent_prep_updated); "
        "carries forward .246 non-terminal state if exp2578 did not run."
    ),
    "jepa_online_active": (
        "FR-11 Tier 3 mandate deliverable -- continuous self-learning active? "
        "True iff exp2576.jepa_online_learning_active=True."
    ),
    "external_baselines": (
        "HIVE peer (0.9236, arXiv:2604.26139) and HalluScan peer (0.67 mean, "
        "arXiv:2605.02443) carried forward for continuous gap tracking."
    ),
    "preconditions_checked": (
        "Records which source artifacts were present and read."
    ),
    "duration_s": "Wall-clock measurement for the capstone synthesis itself.",
    "random_seed": "42 -- synthesis is deterministic but the field is convention.",
    "operator_recommendation": (
        "submit_arxiv_now / update_hf_cards_push_ipfs / continue_safety_classifier / "
        "hardware_terminal_pending / all_tracks_advancing. Drives the operator's "
        "next action; honesty about ambiguity matters."
    ),
    "top_3_successes": "Three most impactful positive findings of .247.",
    "top_3_gaps_for_248": "Three most critical unresolved issues entering .248.",
    "process_flags": (
        "Any non-terminal honesty signals collected from the source artifacts -- "
        "fabrication flags, malformed verdict prefixes, blocked-precondition tasks, "
        "execution-layer gaps when too many planned tasks produced no artifact."
    ),
}


def read_json(path: Path) -> Mapping[str, Any]:
    """Return a JSON object from a local artifact, or an empty object on failure.

    Identical contract to the v245/v246 capstones: partial / missing
    artifacts produce an empty Mapping, which downstream checks treat as
    "not present".
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
    return sum(
        1 for art in artifacts.values() if is_terminal_verdict(art.get("honest_verdict"))
    )


def _best_auroc(
    exp2579: Mapping[str, Any],
) -> tuple[float, str, bool]:
    """Pick the cite-safe headline AUROC for .247.

    Rule: ensemble v9 (exp2579) wins iff:
      - the artifact is present AND
      - ``ensemble_v9_viable`` is True AND
      - ``regression_detected`` is False AND
      - ``ensemble_v9_auroc`` is a number strictly greater than the
        carry-forward floor (0.9857) AND
      - the adversarial-flag is absent AND n_seeds >= 3.

    Otherwise carry forward the .245 ensemble v7b headline (0.9857).

    Returns (auroc, source, adversarially_verified).
    """

    auroc = exp2579.get("ensemble_v9_auroc")
    viable = exp2579.get("ensemble_v9_viable") is True
    regressed = bool(exp2579.get("regression_detected"))
    flagged = bool(exp2579.get("flagged_adversarial"))
    n_seeds = exp2579.get("n_seeds")
    seeds_ok = isinstance(n_seeds, int) and n_seeds >= 3

    auroc_clean = (
        viable
        and not regressed
        and not flagged
        and seeds_ok
        and isinstance(auroc, (int, float))
        and float(auroc) > CARRY_FORWARD_AUROC
    )
    if auroc_clean:
        return float(auroc), "exp2579_ensemble_v9", True
    return CARRY_FORWARD_AUROC, CARRY_FORWARD_SOURCE, False


def _safety_classifier_viable(
    exp2574: Mapping[str, Any],
    exp2575: Mapping[str, Any],
) -> bool:
    """Tier B safety classifier requires BOTH viability gates.

    Read directly from artifacts -- never inferred from sibling fields.
    """

    corpus_ok = exp2574.get("safety_verifier_viable") is True
    integration_ok = exp2575.get("safety_integration_complete") is True
    return corpus_ok and integration_ok


def _ipfs_mirror_status(exp2571: Mapping[str, Any]) -> str:
    """Rule 3 compliance: 'pinned_cid_known' iff exp2571.ipfs_cid is non-null."""

    cid = exp2571.get("ipfs_cid")
    if isinstance(cid, str) and cid.strip():
        return "pinned_cid_known"
    return "documented_operator_needed"


def _gatemate_status(
    exp2577: Mapping[str, Any],
    prior_capstone: Mapping[str, Any],
) -> dict[str, Any]:
    """Reflect exp2577 outcome, falling back to .246 carry-forward."""

    if exp2577:
        status = exp2577.get("gatemate_status")
        if isinstance(status, Mapping):
            return dict(status)
        flashed = bool(exp2577.get("gatemate_bitstream_flashed"))
        smoke = bool(exp2577.get("gatemate_smoke_test_passed"))
        repair = bool(exp2577.get("manual_repair_attempted"))
        jtag = bool(exp2577.get("gatemate_jtag_detected"))
        if smoke:
            outcome = "terminal"
        elif repair:
            outcome = "manual_repair_attempted"
        elif not jtag:
            outcome = "jtag_not_detected"
        elif flashed:
            outcome = "terminal"
        else:
            outcome = "no_change"
        return {
            "ran": True,
            "outcome": outcome,
            "terminal": flashed or smoke,
            "bitstream_flashed": flashed,
            "smoke_test_passed": smoke,
            "manual_repair_attempted": repair,
            "jtag_detected": jtag,
        }
    # Carry-forward from .246 capstone if available.
    prior = prior_capstone.get("gatemate_status") if prior_capstone else None
    if isinstance(prior, Mapping):
        carry = dict(prior)
        carry.setdefault("ran", False)
        carry["outcome"] = "carry_forward_from_246"
        return carry
    return {
        "ran": False,
        "outcome": "carry_forward_no_prior",
        "terminal": False,
        "bitstream_flashed": False,
        "next_blocker": "no_prior_capstone_state",
    }


def _kv260_status(
    exp2578: Mapping[str, Any],
    prior_capstone: Mapping[str, Any],
) -> dict[str, Any]:
    """Reflect exp2578 outcome, falling back to .246 carry-forward."""

    if exp2578:
        status = exp2578.get("kv260_status")
        if isinstance(status, Mapping):
            return dict(status)
        flashed = bool(exp2578.get("sd_card_flashed") or exp2578.get("kv260_workload_validated"))
        prep_updated = bool(exp2578.get("prep_script_updated"))
        if flashed:
            outcome = "sd_flashed"
        elif prep_updated:
            outcome = "sd_absent_prep_updated"
        else:
            outcome = "no_change"
        return {
            "ran": True,
            "outcome": outcome,
            "terminal": flashed,
            "sd_card_flashed": flashed,
            "prep_script_updated": prep_updated,
        }
    prior = prior_capstone.get("kv260_status") if prior_capstone else None
    if isinstance(prior, Mapping):
        carry = dict(prior)
        carry.setdefault("ran", False)
        carry["outcome"] = "carry_forward_from_246"
        return carry
    return {
        "ran": False,
        "outcome": "carry_forward_no_prior",
        "terminal": False,
        "sd_card_flashed": False,
        "next_blocker": "no_prior_capstone_state",
    }


def _operator_recommendation(
    *,
    arxiv_ready_v4: bool,
    paper_errata_applied: bool,
    ipfs_status: str,
    hf_cards_updated: bool,
    safety_viable: bool,
    safety_needs_more: bool,
    gatemate_terminal: bool,
    kv260_terminal: bool,
    n_terminal: int,
    n_planned: int,
    process_flags: list[dict[str, Any]],
) -> str:
    """Pick the operator action.

    Order of precedence:

      1. arxiv_ready_v4 AND errata applied => submit_arxiv_now.
      2. HF cards updated but IPFS pin not yet pushed => update_hf_cards_push_ipfs.
      3. Safety classifier viable but needs more corpus/calibration =>
         continue_safety_classifier.
      4. Both hardware boards still need operator action =>
         hardware_terminal_pending.
      5. Most tracks advancing (>=50% terminal AND no critical flags) =>
         all_tracks_advancing.
      6. Fallback when none of the above hold AND at least one board is
         still operator-pending => hardware_terminal_pending (keeps the
         board visible until terminal per the continuity discipline).

    The fallback is a stable default rather than introducing a new enum
    value; the EXECUTION_LAYER_GAP critical flag carries the urgency
    signal independently.
    """

    if arxiv_ready_v4 and paper_errata_applied:
        return "submit_arxiv_now"

    # HF cards updated locally but IPFS pin not done yet -- this is the
    # canonical Rule-3 follow-up sequence the operator must push.
    if hf_cards_updated and ipfs_status != "pinned_cid_known":
        return "update_hf_cards_push_ipfs"

    if safety_viable and safety_needs_more:
        return "continue_safety_classifier"

    if not gatemate_terminal and not kv260_terminal:
        return "hardware_terminal_pending"

    n_critical = sum(1 for f in process_flags if f.get("severity") == "critical")
    if n_planned > 0 and n_terminal * 2 >= n_planned and n_critical == 0:
        return "all_tracks_advancing"

    # Stable fallback: prefer surfacing the hardware-pending state since
    # KV260 needing operator SD insertion is the most actionable
    # outstanding step regardless of milestone.
    if not kv260_terminal or not gatemate_terminal:
        return "hardware_terminal_pending"

    return "all_tracks_advancing"


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
    if n_missing > EXECUTION_GAP_THRESHOLD:
        flags.append(
            {
                "experiment": None,
                "kind": "EXECUTION_LAYER_GAP",
                "severity": "critical",
                "detail": (
                    f"{n_missing} of {len(ARTIFACT_REL_PATHS)} planned tasks "
                    "(exp2569-exp2579) produced no artifact. Root-cause hypothesis: "
                    "either conductor task-pickup stalled mid-milestone or codex/claude "
                    "agent budget exhausted before queue drain. Mitigate in .248 by "
                    "(a) checkpointing each task pre-pickup and (b) verifying the "
                    "conductor write the failure-ledger row even when the agent exits "
                    "without artifact."
                ),
            }
        )
    return flags


def _top_3_successes(
    *,
    tier0s_improved: bool,
    tier0u_improved: bool,
    safety_viable: bool,
    ipfs_status: str,
    hf_cards_updated: bool,
    gatemate_state: Mapping[str, Any],
    kv260_state: Mapping[str, Any],
    jepa_online: bool,
    best_auroc_source: str,
    n_terminal: int,
) -> list[dict[str, Any]]:
    """Pick the three most impactful positive findings.

    Real-corpus verifier improvements + safety classifier viability + IPFS
    pin are the headline tracks; hardware progress is secondary unless a
    board reaches a new terminal state.
    """

    candidates: list[dict[str, Any]] = []
    if best_auroc_source == "exp2579_ensemble_v9":
        candidates.append(
            {
                "experiment": "exp2579",
                "summary": (
                    "Ensemble v9 improved over the .245 v7b 0.9857 cite-safe baseline "
                    "with adversarial-clean seeds; real-corpus-validated tier0s/tier0u "
                    "substitutions paid off in the headline AUROC."
                ),
            }
        )
    if tier0s_improved and tier0u_improved:
        candidates.append(
            {
                "experiment": "exp2572+exp2573",
                "summary": (
                    "Both tier0s and tier0u verifiers retrained on real FoVer corpus "
                    "cleared the viability gate. The real-corpus AUROC gap that drove "
                    "the .246 paper errata is closed; the ensemble no longer wastes "
                    "calibration signal on near-random verifiers."
                ),
            }
        )
    elif tier0s_improved:
        candidates.append(
            {
                "experiment": "exp2572",
                "summary": (
                    "Tier0s retrained on real FoVer corpus cleared the viability gate "
                    "(real AUROC > 0.65). Partial closure of the .246 verifier gap."
                ),
            }
        )
    elif tier0u_improved:
        candidates.append(
            {
                "experiment": "exp2573",
                "summary": (
                    "Tier0u self-consistency retrained on real text cleared the "
                    "viability gate (real AUROC > 0.60). Partial closure of the .246 "
                    "verifier gap."
                ),
            }
        )
    if safety_viable:
        candidates.append(
            {
                "experiment": "exp2574+exp2575",
                "summary": (
                    "Tier B safety/jailbreak classifier viable: Ising safety verifier "
                    "implemented on a 200-pair safe/unsafe corpus and integrated into "
                    "the ensemble pipeline. First commercially viable product surface "
                    "outside hallucination verification."
                ),
            }
        )
    if ipfs_status == "pinned_cid_known":
        candidates.append(
            {
                "experiment": "exp2571",
                "summary": (
                    "IPFS mirror established for the arXiv preprint with a known CID. "
                    "Rule 3 (mandatory mirroring) compliance reaches a new structural "
                    "milestone: the publication artifact is now content-addressed and "
                    "user-base-mirror-able, not single-host."
                ),
            }
        )
    if hf_cards_updated:
        candidates.append(
            {
                "experiment": "exp2570",
                "summary": (
                    "HuggingFace model cards updated with arXiv preprint citation; "
                    "discoverability link between weights mirror and paper landed."
                ),
            }
        )
    if jepa_online:
        candidates.append(
            {
                "experiment": "exp2576",
                "summary": (
                    "JEPA v3 online integration active: real-data checkpoint wired "
                    "into VerifyRepairPipeline with session-level update enabled. "
                    "FR-11 Tier 3 continuous self-learning mandate met."
                ),
            }
        )
    if gatemate_state.get("smoke_test_passed"):
        candidates.append(
            {
                "experiment": "exp2577",
                "summary": (
                    "GateMate on-board Ising sampler smoke test passed; gate 3 of "
                    "the hardware terminal-state checklist closed."
                ),
            }
        )
    if kv260_state.get("sd_card_flashed"):
        candidates.append(
            {
                "experiment": "exp2578",
                "summary": (
                    "KV260 SD card flashed and Carnot validation completed; second "
                    "hardware board reaches terminal state."
                ),
            }
        )
    # If nothing else landed, surface the most concrete forward step
    # available. With zero terminal artifacts in .247 this branch is the
    # only one populating top_3_successes.
    if not candidates:
        carryforward_gatemate = gatemate_state.get("bitstream_flashed") or gatemate_state.get(
            "terminal"
        )
        if carryforward_gatemate:
            candidates.append(
                {
                    "experiment": "exp2567",
                    "summary": (
                        "Carry-forward from .246: GateMate A1-EVB-2M bitstream "
                        "remains flashed; chip alive and not bricked. No new "
                        "forward progress in .247 due to execution layer gap."
                    ),
                }
            )
        if n_terminal == 0:
            candidates.append(
                {
                    "experiment": None,
                    "summary": (
                        "No new experiment artifacts in .247 (execution-layer gap). "
                        "Cite-safe headline AUROC carries forward at 0.9857 from "
                        ".245 ensemble v7b; the prior milestone's results remain "
                        "the operator's best public claim."
                    ),
                }
            )
    for new_rank, entry in enumerate(candidates[:3], start=1):
        entry["rank"] = new_rank
    return candidates[:3]


def _top_3_gaps_for_248(
    *,
    tier0s_improved: bool,
    tier0u_improved: bool,
    safety_viable: bool,
    ipfs_status: str,
    hf_cards_updated: bool,
    gatemate_state: Mapping[str, Any],
    kv260_state: Mapping[str, Any],
    jepa_online: bool,
    process_flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Three biggest unresolved issues entering .248."""

    gaps: list[dict[str, Any]] = []
    has_exec_gap = any(f.get("kind") == "EXECUTION_LAYER_GAP" for f in process_flags)
    if has_exec_gap:
        gaps.append(
            {
                "area": "execution_layer_gap",
                "summary": (
                    "Execution-layer gap continues: many .247 tasks produced no "
                    "artifact (same failure mode as .246). Conductor pickup or agent "
                    "budget exhaustion suspected. .248 needs a pre-pickup checkpoint "
                    "and a failure-ledger write even on agent exit without artifact, "
                    "so the gap is visible to the next planner rather than silently "
                    "carried forward through successive milestones."
                ),
            }
        )
    if not (tier0s_improved and tier0u_improved):
        gaps.append(
            {
                "area": "real_corpus_verifier_gap",
                "summary": (
                    "Tier0s/tier0u real-corpus AUROC gap still open: the .246 paper "
                    "errata corrected the headline claim but the underlying verifiers "
                    "remain near-random on natural text. .248 must re-queue exp2572 "
                    "(tier0s retrain) and exp2573 (tier0u fix) with explicit GPU "
                    "preconditions and corpus-source assertions."
                ),
            }
        )
    if not safety_viable:
        gaps.append(
            {
                "area": "safety_classifier_unviable",
                "summary": (
                    "Tier B safety/jailbreak classifier not yet viable: exp2574 "
                    "(safety corpus + Ising verifier) and/or exp2575 (ensemble "
                    "integration) did not produce terminal artifacts. .248 must "
                    "re-prototype with a smaller initial corpus (50 pairs) or retire "
                    "the lineage in favor of an alternative product surface."
                ),
            }
        )
    if not jepa_online and len(gaps) < 3:
        gaps.append(
            {
                "area": "jepa_online_inactive",
                "summary": (
                    "JEPA v3 online integration (exp2576) did not produce a terminal "
                    "artifact; FR-11 Tier 3 continuous self-learning mandate still "
                    "deferred. Re-queue with explicit checkpoint-load + pipeline-wire "
                    "verification steps."
                ),
            }
        )
    if ipfs_status != "pinned_cid_known" and len(gaps) < 3:
        gaps.append(
            {
                "area": "ipfs_mirror_pending",
                "summary": (
                    "IPFS mirror for the arXiv preprint not yet pinned with a known "
                    "CID. Rule 3 (mandatory mirroring) compliance incomplete -- "
                    "preprint is still single-host. Operator step: pin via web3.storage "
                    "or equivalent Filecoin-backed service and record CID."
                ),
            }
        )
    if not kv260_state.get("terminal") and len(gaps) < 3:
        gaps.append(
            {
                "area": "kv260_not_terminal",
                "summary": (
                    "KV260 still non-terminal: operator action required (insert SD "
                    "media + browser-download PYNQ image from pynq.io). Hardware "
                    "continuity discipline keeps this on the per-milestone "
                    "reservation list until terminal."
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
    """Build the Exp 2580 capstone synthesis from local checked-in evidence.

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

    prior_capstone_path = root / PRIOR_CAPSTONE_REL_PATH
    prior_capstone = read_json(prior_capstone_path) if prior_capstone_path.is_file() else {}
    preconditions.append(
        {
            "resource": str(PRIOR_CAPSTONE_REL_PATH),
            "available": prior_capstone_path.is_file(),
            "check": f"ls {prior_capstone_path}",
        }
    )

    exp2570 = artifacts["exp2570"]
    exp2571 = artifacts["exp2571"]
    exp2572 = artifacts["exp2572"]
    exp2573 = artifacts["exp2573"]
    exp2574 = artifacts["exp2574"]
    exp2575 = artifacts["exp2575"]
    exp2576 = artifacts["exp2576"]
    exp2577 = artifacts["exp2577"]
    exp2578 = artifacts["exp2578"]
    exp2579 = artifacts["exp2579"]

    n_experiments_completed = _count_terminal(artifacts)
    best_auroc, best_auroc_source, auroc_verified = _best_auroc(exp2579)
    tier0s_improved = exp2572.get("tier0s_improved") is True
    tier0u_improved = exp2573.get("tier0u_improved") is True
    safety_viable = _safety_classifier_viable(exp2574, exp2575)
    safety_needs_more = bool(exp2575.get("needs_more_corpus") or exp2575.get("needs_more_calibration"))
    ipfs_status = _ipfs_mirror_status(exp2571)
    hf_cards_updated = exp2570.get("hf_model_cards_updated") is True
    jepa_online = exp2576.get("jepa_online_learning_active") is True
    gatemate_state = _gatemate_status(exp2577, prior_capstone)
    kv260_state = _kv260_status(exp2578, prior_capstone)
    arxiv_ready_v4 = bool(prior_capstone.get("arxiv_ready_v4"))
    paper_errata_applied = prior_capstone.get("paper_errata_status") == "applied_and_verified"
    process_flags = _process_flags(artifacts, n_missing)

    operator_recommendation = _operator_recommendation(
        arxiv_ready_v4=arxiv_ready_v4,
        paper_errata_applied=paper_errata_applied,
        ipfs_status=ipfs_status,
        hf_cards_updated=hf_cards_updated,
        safety_viable=safety_viable,
        safety_needs_more=safety_needs_more,
        gatemate_terminal=bool(gatemate_state.get("terminal")),
        kv260_terminal=bool(kv260_state.get("terminal")),
        n_terminal=n_experiments_completed,
        n_planned=len(ARTIFACT_REL_PATHS),
        process_flags=process_flags,
    )

    top_successes = _top_3_successes(
        tier0s_improved=tier0s_improved,
        tier0u_improved=tier0u_improved,
        safety_viable=safety_viable,
        ipfs_status=ipfs_status,
        hf_cards_updated=hf_cards_updated,
        gatemate_state=gatemate_state,
        kv260_state=kv260_state,
        jepa_online=jepa_online,
        best_auroc_source=best_auroc_source,
        n_terminal=n_experiments_completed,
    )
    top_gaps = _top_3_gaps_for_248(
        tier0s_improved=tier0s_improved,
        tier0u_improved=tier0u_improved,
        safety_viable=safety_viable,
        ipfs_status=ipfs_status,
        hf_cards_updated=hf_cards_updated,
        gatemate_state=gatemate_state,
        kv260_state=kv260_state,
        jepa_online=jepa_online,
        process_flags=process_flags,
    )

    finished = time.time() if now_epoch is None else now_epoch
    duration_s = round(max(0.0, finished - started), 6)

    gatemate_summary = (
        "terminal"
        if gatemate_state.get("terminal")
        else gatemate_state.get("outcome", "no_change")
    )
    honest_verdict = (
        f"complete: best_247_auroc={best_auroc:.4f}; "
        f"tier0s_improved={tier0s_improved}; "
        f"safety_viable={safety_viable}; "
        f"gatemate={gatemate_summary}; "
        f"jepa_online={jepa_online}"
    )

    return {
        "experiment": "exp2580",
        "title": "Milestone .247 Capstone Synthesis",
        "milestone": "2026.05.247",
        "schema": "carnot.capstone.v247",
        "honest_verdict": honest_verdict,
        "n_experiments_completed": n_experiments_completed,
        "n_planned": len(ARTIFACT_REL_PATHS),
        "best_247_auroc": best_auroc,
        "best_247_auroc_source": best_auroc_source,
        "auroc_adversarially_verified": auroc_verified,
        "tier0s_real_improvement": tier0s_improved,
        "tier0u_real_improvement": tier0u_improved,
        "safety_classifier_viable": safety_viable,
        "ipfs_mirror_status": ipfs_status,
        "gatemate_status": gatemate_state,
        "kv260_status": kv260_state,
        "hardware_terminal_states": {
            "gatemate": gatemate_state,
            "kv260": kv260_state,
        },
        "jepa_online_active": jepa_online,
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
        "top_3_gaps_for_248": top_gaps,
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
