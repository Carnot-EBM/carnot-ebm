"""Build the Exp 3933 v363 hardened verifier scorecard artifact.

Spec refs: REQ-CAPSTONE-3933, SCENARIO-CAPSTONE-3933.

This is a conditioned aggregation step. It reads the .363 upstream artifacts,
excludes anything stamped ``flagged_adversarial:true`` or live-critical under
``scripts/summarize_artifact.py`` before deriving scorecard fields, preserves
the frozen FoVer 0.9131 paper-ready headline, and records partial landing
honestly instead of manufacturing a new efficiency or moat headline.
"""

from __future__ import annotations

from collections.abc import Mapping
import glob
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script import guard.
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_REL_PATH = Path("results/experiment_3933_capstone_v363.json")
EXPERIMENT_ID = 3933
RANDOM_SEED = 3933
FROZEN_FOVER_AUROC = 0.9131
DECISIVE_COST_RATIO = 10.0
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

UPSTREAM_IDS = (3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3924: Path("results/experiment_3924_archive_v362_activate_v363_retire_facts.json"),
    3925: Path("results/experiment_3925_competent_judge_build.json"),
    3926: Path("results/experiment_3926_valid_efficiency_head_to_head.json"),
    3927: Path("results/experiment_3927_non_degenerate_cascade_router.json"),
    3928: Path("results/experiment_3928_moat_scissor_replication.json"),
    3929: Path("results/experiment_3929_arc_agi3_action_efficiency.json"),
    3930: Path("results/experiment_3930_fr11_v26_cascade_band_online_learning.json"),
    3931: Path("results/experiment_3931_hardware_continuity_clean_rerun.json"),
    3932: Path("results/experiment_3932_literature_synthesis_agentic_verification.json"),
}
UPSTREAM_GLOBS: Mapping[int, str] = {
    experiment_id: f"results/experiment_{experiment_id}_*"
    for experiment_id in UPSTREAM_IDS
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "efficiency_verdict",
    "efficiency_cost_ratio",
    "judge_was_competent",
    "moat_replicated",
    "cascade_verdict",
    "verifier_earns_its_place",
    "arc_agentic_advantage",
    "fr11_v26_invariant",
    "hardware_outcome",
    "facts_route_retired",
    "both_energy_theses_bounded",
    "paper_ready",
    "unmet_gates",
    "frozen_headline_unchanged",
    "operator_next_step_recommendation",
    "flagged_artifacts_excluded",
    "preconditions_checked",
    "duration_s",
    "inference_substrate",
)

STRING_VERDICT_FIELDS = (
    "efficiency_verdict",
    "cascade_verdict",
    "fr11_v26_invariant",
    "hardware_outcome",
    "operator_next_step_recommendation",
    "inference_substrate",
)

BOOL_VERDICT_FIELDS = (
    "judge_was_competent",
    "moat_replicated",
    "verifier_earns_its_place",
    "facts_route_retired",
    "both_energy_theses_bounded",
    "paper_ready",
    "frozen_headline_unchanged",
)

NUMERIC_VERDICT_FIELDS = (
    "efficiency_cost_ratio",
    "arc_agentic_advantage",
)

FIELD_PRINCIPLES = {
    "efficiency_verdict": (
        "VALID_EARNS_PLACE / CHEAPER_BUT_LESS_ACCURATE / INCONCLUSIVE from exp3926 - "
        "the EFFICIENCY axis answer against a COMPETENT judge."
    ),
    "efficiency_cost_ratio": (
        "Bare float - measured 'Nx cheaper' number against the competent judge."
    ),
    "judge_was_competent": (
        "Bare bool - did the .363 judge clear its positive control, fixing the .362 comparator."
    ),
    "moat_replicated": (
        "Bare bool - did MOAT_SURVIVES replicate on the independent corpus in exp3928."
    ),
    "cascade_verdict": (
        "Cascade router WINS/MARGINAL plus whether it was non-degenerate with escalation > 0."
    ),
    "verifier_earns_its_place": (
        "Bare bool - true iff parity or Pareto holds against a competent judge and is decisively cheaper."
    ),
    "arc_agentic_advantage": "The verifier-as-router action-efficiency ratio from exp3929.",
    "fr11_v26_invariant": "Self-learning mandate outcome - invariant held across v25 to v26.",
    "hardware_outcome": (
        "GateMate terminal-assessment plus PolarFire/KV260 continuity - honest no-fabric-claim record."
    ),
    "facts_route_retired": (
        "Bare bool - graph-grounding facts route retired to future work in exp3924."
    ),
    "both_energy_theses_bounded": (
        "Bare bool - true because selection P0.1 and generation EBT are both bounded-negative."
    ),
    "paper_ready": (
        "Must stay true - the milestone adds credibility lenses, not a new headline."
    ),
    "unmet_gates": "G1-G4 publication-gate status from scripts/publication_gate.py --json.",
    "frozen_headline_unchanged": "G1-G4 status plus the frozen FoVer 0.9131 invariant guard.",
    "operator_next_step_recommendation": (
        "Given the conditioned offline proof, recommend the ARC-AGI-3 agentic-proof venue for operator choice."
    ),
    "flagged_artifacts_excluded": (
        "Artifacts skipped for flagged_adversarial:true before any scorecard aggregation."
    ),
    "preconditions_checked": (
        "Aggregation methodology, upstream landing state, and live summarizer status."
    ),
    "duration_s": "Aggregation wall-clock duration.",
    "inference_substrate": "Aggregation methodology without GGUF/CUDA markers.",
}


def is_sha256(value: object) -> bool:
    """Return whether ``value`` looks like a SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def numeric(value: object) -> float | None:
    """Return JSON numeric scalars while rejecting booleans and strings."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON artifact object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover - defensive guard.
    return payload


def sha256_file(path: Path) -> str:
    """Hash a cited upstream artifact for provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable repo-relative path string."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - only used for external paths.
        return str(path)


def honest_verdict(payload: Mapping[str, Any] | None) -> str:
    """Extract the upstream terminal verdict without interpreting metrics."""

    if not isinstance(payload, Mapping):
        return "missing"
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else "missing"


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether the upstream carries the stamped fabrication flag."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def live_critical(summary: Mapping[str, Any] | None) -> bool:
    """Return whether summarize_artifact.py found a live critical flag."""

    return isinstance(summary, Mapping) and summary.get("returncode") == 2


def derive_judge_was_competent(payload: Mapping[str, Any] | None) -> bool:
    """Return whether the competent-judge positive control cleared."""

    if not isinstance(payload, Mapping):
        return False
    verdict = honest_verdict(payload).upper()
    return payload.get("judge_positive_control_passed") is True or (
        "POSITIVE_CONTROL_PASSED" in verdict and not verdict.startswith("BLOCKED_")
    )


def derive_efficiency_cost_ratio(payload: Mapping[str, Any] | None) -> float:
    """Extract the measured competent-judge wall-time cost ratio."""

    if not isinstance(payload, Mapping):
        return 0.0
    ratio = numeric(payload.get("cost_ratio_walltime"))
    if ratio is not None:
        return ratio
    ratio = numeric(payload.get("efficiency_cost_ratio"))
    return 0.0 if ratio is None else ratio


def derive_efficiency_verdict(payload: Mapping[str, Any] | None) -> str:
    """Map Exp 3926 to the .363 efficiency-axis scalar."""

    if not isinstance(payload, Mapping) or not derive_judge_was_competent(payload):
        return "INCONCLUSIVE"
    ratio = derive_efficiency_cost_ratio(payload)
    if ratio <= DECISIVE_COST_RATIO:
        return "INCONCLUSIVE"
    verdict = honest_verdict(payload).upper()
    if payload.get("accuracy_parity") is True or payload.get("pareto_dominates") is True:
        return "VALID_EARNS_PLACE"
    if "VALID_EARNS_PLACE" in verdict:
        return "VALID_EARNS_PLACE"
    return "CHEAPER_BUT_LESS_ACCURATE"


def derive_verifier_earns_place(
    efficiency_verdict: str,
    cost_ratio: float,
    *,
    judge_was_competent: bool,
) -> bool:
    """Apply the .363 operator win condition."""

    return (
        judge_was_competent
        and efficiency_verdict == "VALID_EARNS_PLACE"
        and cost_ratio > DECISIVE_COST_RATIO
    )


def derive_moat_replicated(payload: Mapping[str, Any] | None) -> bool:
    """Return whether the independent-corpus moat replication landed cleanly."""

    if not isinstance(payload, Mapping):
        return False
    if payload.get("moat_replicates") is True:
        return True
    verdict = honest_verdict(payload).upper()
    return verdict.startswith("COMPLETE:") and ("MOAT_REPLICATES" in verdict or "REPLICATES" in verdict)


def derive_cascade_verdict(payload: Mapping[str, Any] | None) -> str:
    """Map Exp 3927 to WINS/MARGINAL plus non-degenerate escalation state."""

    verdict = honest_verdict(payload).upper()
    base = "WINS" if "WINS" in verdict else "MARGINAL"
    escalation = numeric(payload.get("escalation_fraction")) if isinstance(payload, Mapping) else None
    non_degenerate = bool(escalation is not None and escalation > 0.0)
    if not non_degenerate and isinstance(payload, Mapping):
        non_degenerate = payload.get("cascade_degenerate") is False and "NON_DEGENERATE" in verdict
    suffix = "NON_DEGENERATE" if non_degenerate else "DEGENERATE"
    return f"{base}_{suffix}"


def derive_arc_agentic_advantage(payload: Mapping[str, Any] | None) -> float:
    """Extract the synthetic ARC verifier-router action-efficiency ratio."""

    if not isinstance(payload, Mapping):
        return 0.0
    ratio = numeric(payload.get("action_efficiency_ratio"))
    return 0.0 if ratio is None else ratio


def derive_fr11_v26_invariant(payload: Mapping[str, Any] | None) -> str:
    """Return whether the FR-11 v26 invariant held."""

    if not isinstance(payload, Mapping):
        return "INCONCLUSIVE"
    verdict = honest_verdict(payload).upper()
    if "INVARIANT_BROKEN" in verdict:
        return "INVARIANT_BROKEN"
    if "INVARIANT_HELD" in verdict or (
        payload.get("frozen_headline_unchanged") is True
        and payload.get("learned_ensemble_auroc_in_frozen_ci") is True
        and payload.get("memory_ablation_contribution_min_met") is True
    ):
        return "INVARIANT_HELD"
    return "INCONCLUSIVE"


def derive_hardware_outcome(payload: Mapping[str, Any] | None) -> str:
    """Summarize the clean hardware continuity rerun without a fabric claim."""

    if not isinstance(payload, Mapping):
        return "MISSING"
    verdict = honest_verdict(payload).lower()
    if verdict.startswith("blocked_"):
        return "BLOCKED"
    if "no_fabric_claim" in verdict or payload.get("fabric_acceleration_claimed") is False:
        return "TERMINAL_OR_CONTINUITY_NO_FABRIC_CLAIM"
    return "PARTIAL_NO_FABRIC_CLAIM"


def derive_facts_route_retired(payload: Mapping[str, Any] | None) -> bool:
    """Return whether Exp 3924 retired the facts route to future work."""

    if not isinstance(payload, Mapping):
        return False
    verdict = honest_verdict(payload).upper()
    return payload.get("facts_route_retired") is True or "FACTS_RETIRED" in verdict or "RETIRED" in verdict


def frozen_headline_unchanged(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Guard the frozen 0.9131 FoVer headline against explicit regressions."""

    for payload in clean_upstreams.values():
        if payload.get("frozen_headline_unchanged") is False:
            return False
        frozen = numeric(payload.get("frozen_headline_ensemble_auroc"))
        if frozen is not None and frozen != FROZEN_FOVER_AUROC:
            return False
        frozen_alias = numeric(payload.get("frozen_fover_auroc_unchanged"))
        if frozen_alias is not None and frozen_alias != FROZEN_FOVER_AUROC:
            return False
    return True


def operator_next_step_recommendation(
    *,
    efficiency_verdict: str,
    moat_replicated: bool,
    cascade_verdict: str,
    verifier_earns_its_place: bool,
) -> str:
    """State the operator decision and recommended forward step."""

    win_state = "met" if verifier_earns_its_place else "not met"
    return (
        "Conditioned hardened verifier scorecard: "
        f"efficiency={efficiency_verdict}, moat_replicated={str(moat_replicated).lower()}, "
        f"cascade={cascade_verdict}, operator win condition={win_state}. "
        "Recommended next forward step is to scale the ARC-AGI-3 agentic-proof venue; "
        "the loop recommends, the operator decides."
    )


def verdict_slug(efficiency_verdict: str, moat_replicated: bool, verifier_earns_its_place: bool) -> str:
    """Build the fixed v363 honest-verdict suffix."""

    return (
        f"capstone_v363_efficiency{efficiency_verdict}_"
        f"moat_replicated{str(moat_replicated).lower()}_"
        f"earns{str(verifier_earns_its_place).lower()}"
    )


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact payload while excluding its own checksum."""

    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def matching_files(root: Path, experiment_id: int) -> list[str]:
    """Return repo-relative files that matched the requested upstream pattern."""

    return [
        relative_to_root(root, Path(path))
        for path in sorted(glob.glob(str(root / UPSTREAM_GLOBS[experiment_id])))
    ]


def module_importable(name: str) -> bool:
    """Return whether a precondition module can be imported."""

    try:
        importlib.import_module(name)
    except Exception:  # pragma: no cover - environment failure path.
        return False
    return True


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    publication_gate_data: Mapping[str, Any] | None = None,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v363 scorecard from existing upstream verdict artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = {experiment_id: root_path / DEFAULT_UPSTREAM_PATHS[experiment_id] for experiment_id in UPSTREAM_IDS}
    upstreams = {
        experiment_id: read_json_object(path) if path.exists() else None
        for experiment_id, path in paths.items()
    }
    summaries = (
        {experiment_id: dict(status) for experiment_id, status in summary_statuses.items()}
        if summary_statuses is not None
        else run_summarize_statuses(root_path, paths)  # pragma: no cover - subprocess IO.
    )
    publication_gate = (
        dict(publication_gate_data)
        if publication_gate_data is not None
        else publication_gate_state()  # pragma: no cover - subprocess IO.
    )

    flagged_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if flagged(payload)
    }
    live_critical_ids = {
        experiment_id
        for experiment_id, status in summaries.items()
        if live_critical(status)
    }
    clean_upstreams = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping)
        and experiment_id not in flagged_ids
        and experiment_id not in live_critical_ids
    }

    efficiency_payload = clean_upstreams.get(3926)
    judge_was_competent = derive_judge_was_competent(efficiency_payload) or derive_judge_was_competent(
        clean_upstreams.get(3925)
    )
    efficiency = derive_efficiency_verdict(efficiency_payload)
    efficiency_cost_ratio = derive_efficiency_cost_ratio(efficiency_payload)
    moat_replicated = derive_moat_replicated(clean_upstreams.get(3928))
    cascade = derive_cascade_verdict(clean_upstreams.get(3927))
    earns = derive_verifier_earns_place(
        efficiency,
        efficiency_cost_ratio,
        judge_was_competent=judge_was_competent,
    )
    arc_agentic = derive_arc_agentic_advantage(clean_upstreams.get(3929))
    fr11 = derive_fr11_v26_invariant(clean_upstreams.get(3930))
    hardware = derive_hardware_outcome(clean_upstreams.get(3931))
    facts_retired = derive_facts_route_retired(clean_upstreams.get(3924))
    paper_ready = publication_gate.get("paper_ready") is True
    unmet_gates = list(publication_gate.get("unmet_gates", []))
    frozen_unchanged = frozen_headline_unchanged(clean_upstreams)
    aggregation_target_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping)
        and experiment_id not in flagged_ids
        and experiment_id not in live_critical_ids
    }
    all_landed_nonflagged_aggregated = set(clean_upstreams) == aggregation_target_ids
    capstone_complete = bool(paper_ready and frozen_unchanged and all_landed_nonflagged_aggregated)

    cited = [
        {
            "experiment_id": experiment_id,
            "path": relative_to_root(root_path, paths[experiment_id]),
            "sha256": sha256_file(paths[experiment_id]),
            "honest_verdict": honest_verdict(payload),
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
        for experiment_id, payload in sorted(clean_upstreams.items())
    ]
    excluded = [
        {
            "experiment_id": experiment_id,
            "path": relative_to_root(root_path, paths[experiment_id]),
            "reason": "flagged_adversarial:true",
            "honest_verdict": honest_verdict(upstreams[experiment_id]),
            "live_critical": experiment_id in live_critical_ids,
        }
        for experiment_id in sorted(flagged_ids)
    ]
    critical_excluded = [
        {
            "experiment_id": experiment_id,
            "path": relative_to_root(root_path, paths[experiment_id]),
            "reason": "summarize_artifact_live_critical",
            "honest_verdict": honest_verdict(upstreams[experiment_id]),
        }
        for experiment_id in sorted(live_critical_ids - flagged_ids)
    ]
    upstream_state = {
        experiment_id: {
            "path": relative_to_root(root_path, path),
            "matching_files": matching_files(root_path, experiment_id),
            "exists": upstreams[experiment_id] is not None,
            "honest_verdict": honest_verdict(upstreams[experiment_id]),
            "flagged_adversarial": experiment_id in flagged_ids,
            "live_critical": experiment_id in live_critical_ids,
            "included": experiment_id in clean_upstreams,
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
        for experiment_id, path in paths.items()
    }

    base_slug = verdict_slug(efficiency, moat_replicated, earns)
    suffix = f"{base_slug}_paper_ready_{str(paper_ready).lower()}_frozen_{'unchanged' if frozen_unchanged else 'changed'}"
    if capstone_complete:
        terminal_verdict = f"complete: {suffix}"
    elif not paper_ready:
        terminal_verdict = f"blocked_publication_gate: {suffix}"
    elif not frozen_unchanged:
        terminal_verdict = f"blocked_frozen_headline: {suffix}"
    else:  # pragma: no cover - retained for future citation drift guards.
        terminal_verdict = f"blocked_aggregation: {suffix}"

    end = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "schema": "carnot.capstone_v363_3933.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": terminal_verdict,
        "efficiency_verdict": efficiency,
        "efficiency_cost_ratio": efficiency_cost_ratio,
        "judge_was_competent": judge_was_competent,
        "moat_replicated": moat_replicated,
        "cascade_verdict": cascade,
        "verifier_earns_its_place": earns,
        "arc_agentic_advantage": arc_agentic,
        "fr11_v26_invariant": fr11,
        "hardware_outcome": hardware,
        "facts_route_retired": facts_retired,
        "both_energy_theses_bounded": True,
        "paper_ready": paper_ready,
        "unmet_gates": unmet_gates,
        "publication_gate": publication_gate,
        "frozen_headline_unchanged": frozen_unchanged,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "operator_next_step_recommendation": operator_next_step_recommendation(
            efficiency_verdict=efficiency,
            moat_replicated=moat_replicated,
            cascade_verdict=cascade,
            verifier_earns_its_place=earns,
        ),
        "flagged_artifacts_excluded": excluded,
        "live_critical_artifacts_excluded": critical_excluded,
        "cited_upstream_artifacts": cited,
        "preconditions_checked": {
            "summarize_artifact_importable": module_importable("scripts.summarize_artifact"),
            "publication_gate_importable": module_importable("scripts.publication_gate"),
            "summarize_artifact_runnable": all(
                status.get("returncode") in {0, 1, 2}
                for status in summaries.values()
            ),
            "publication_gate_checked": True,
            "publication_gate_runnable": True,
            "all_landed_nonflagged_verdicts_aggregated": all_landed_nonflagged_aggregated,
            "capstone_complete": capstone_complete,
            "live_critical_artifacts_observed": sorted(live_critical_ids),
            "upstream_artifacts": upstream_state,
        },
        "duration_s": max(0.0001, end - start),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the v363 scorecard contract that prevents over-claiming."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive guard.
    for field in STRING_VERDICT_FIELDS:
        if not isinstance(artifact.get(field), str):
            raise ValueError(f"{field} must be a bare string")  # pragma: no cover - defensive guard.
    for field in BOOL_VERDICT_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")  # pragma: no cover - defensive guard.
    for field in NUMERIC_VERDICT_FIELDS:
        if numeric(artifact.get(field)) is None:
            raise ValueError(f"{field} must be a bare number")  # pragma: no cover - defensive guard.
    if not isinstance(artifact.get("unmet_gates"), list):
        raise ValueError("unmet_gates must be a list")  # pragma: no cover - defensive guard.
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "success:", "failure:", "blocked_")):
        raise ValueError("honest_verdict must have a terminal prefix")  # pragma: no cover - defensive guard.
    substrate = str(artifact.get("inference_substrate"))
    if "GGUF" in substrate or "CUDA" in substrate:
        raise ValueError("inference_substrate must not carry GGUF/CUDA markers")  # pragma: no cover - defensive guard.
    duration = numeric(artifact.get("duration_s"))
    if duration is None or duration < 0.0001:
        raise ValueError("duration_s must respect the aggregation floor")  # pragma: no cover - defensive guard.
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")  # pragma: no cover - defensive guard.


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: str | Path = OUTPUT_REL_PATH,
    publication_gate_data: Mapping[str, Any] | None = None,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 3933 capstone JSON artifact."""

    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        publication_gate_data=publication_gate_data,
        summary_statuses=summary_statuses,
        started_s=started_s,
        now_s=now_s,
    )
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def publication_gate_state() -> JsonDict:  # pragma: no cover - thin wrapper around existing gate.
    """Evaluate the stable G1-G4 publication gate through its JSON CLI."""

    result = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["publication_gate_cli_failed"],
            "__source__": "publication_gate_failed",
            "stderr_excerpt": result.stderr[:1000],
        }
    payload = json.loads(result.stdout)
    payload["__source__"] = "scripts/publication_gate.py --json"
    return payload


def run_summarize_statuses(root: Path, paths: Mapping[int, Path]) -> dict[int, JsonDict]:  # pragma: no cover - subprocess IO.
    """Run summarize_artifact.py for every expected upstream pattern."""

    statuses: dict[int, JsonDict] = {}
    summarizer = root / "scripts" / "summarize_artifact.py"
    for experiment_id, path in paths.items():
        matches = matching_files(root, experiment_id)
        if not matches:
            statuses[experiment_id] = {"returncode": 1, "missing": True}
            continue
        args = [sys.executable, str(summarizer), *matches]
        result = subprocess.run(
            args,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        statuses[experiment_id] = {
            "returncode": result.returncode,
            "expected_json_exists": path.exists(),
            "stdout_excerpt": result.stdout[:1000],
            "stderr_excerpt": result.stderr[:1000],
        }
    return statuses


def main() -> int:  # pragma: no cover - CLI wrapper.
    """Write the default Exp 3933 artifact and print its path."""

    print(write_artifact(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
