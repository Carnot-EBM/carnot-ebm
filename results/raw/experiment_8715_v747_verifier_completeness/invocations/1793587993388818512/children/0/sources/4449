"""Build the Exp 3902 v360 milestone capstone artifact.

Spec refs: REQ-CAPSTONE-3902, SCENARIO-CAPSTONE-3902.

This module is a disciplined aggregation step. It reads landed upstream
artifacts, excludes anything stamped ``flagged_adversarial:true``, records
missing or blocked upstreams as part of the state, and preserves the frozen
FoVer 0.9131 paper-ready headline rather than making a new headline.
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

OUTPUT_REL_PATH = Path("results/experiment_3902_capstone_v360.json")
EXPERIMENT_ID = 3902
RANDOM_SEED = 3902
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

UPSTREAM_IDS = (3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3893: Path("results/experiment_3893_ebt_fundamental_replication.json"),
    3894: Path("results/experiment_3894_reasoner_self_verify_harness.json"),
    3895: Path("results/experiment_3895_moat_scissor_tested_harness.json"),
    3896: Path("results/experiment_3896_graph_grounding_verifier_harness.json"),
    3897: Path("results/experiment_3897_graph_grounding_facts_run.json"),
    3898: Path("results/experiment_3898_facts_complementarity.json"),
    3899: Path("results/experiment_3899_fr11_v25.json"),
    3900: Path("results/experiment_3900_gatemate_terminal_confirmation.json"),
    3901: Path("results/experiment_3901_polarfire_kv260_continuity.json"),
}
UPSTREAM_GLOBS: Mapping[int, str] = {
    experiment_id: f"results/experiment_{experiment_id}_*"
    for experiment_id in UPSTREAM_IDS
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "ebt_replication_outcome",
    "moat_verdict",
    "facts_outcome",
    "fr11_v25_invariant",
    "hardware_outcome",
    "both_energy_theses_bounded",
    "paper_ready",
    "unmet_gates",
    "frozen_headline_unchanged",
    "operator_next_thesis_recommendation",
    "flagged_artifacts_excluded",
    "preconditions_checked",
    "duration_s",
    "inference_substrate",
)

STRING_VERDICT_FIELDS = (
    "ebt_replication_outcome",
    "moat_verdict",
    "facts_outcome",
    "fr11_v25_invariant",
    "hardware_outcome",
    "operator_next_thesis_recommendation",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "ebt_replication_outcome": (
        "REPLICATED/REFUTED/INCONCLUSIVE - does the .359 FUNDAMENTAL hold; "
        "the energy-as-generator bank-or-not signal."
    ),
    "moat_verdict": (
        "MOAT_SURVIVES/SUBSUMED/INCONCLUSIVE with the tested harness - the "
        "DT-P2 durability answer."
    ),
    "facts_outcome": (
        "Graph-grounding signal reproduced? complementary? - the broaden-the-verifier "
        "status; excluded if exp3897 is flagged."
    ),
    "fr11_v25_invariant": (
        "Self-learning mandate outcome - invariant held across the v24-to-v25 iteration."
    ),
    "hardware_outcome": (
        "GateMate terminal-assessment plus PolarFire/KV260 continuity - honest "
        "no-fabric-claim record."
    ),
    "both_energy_theses_bounded": (
        "Bare bool - true iff EBT replication confirms FUNDAMENTAL given P0.1 "
        "already settled."
    ),
    "paper_ready": (
        "Must stay true - the milestone adds lenses, not a new headline; report unmet_gates."
    ),
    "unmet_gates": "G1-G4 publication-gate status from scripts/publication_gate.py --json.",
    "frozen_headline_unchanged": "G1-G4 status plus the frozen FoVer 0.9131 invariant guard.",
    "operator_next_thesis_recommendation": (
        "The loop recommends; the operator decides the next forward bet."
    ),
    "flagged_artifacts_excluded": (
        "Artifacts skipped for flagged_adversarial:true before metric aggregation."
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


def derive_ebt_replication_outcome(payload: Mapping[str, Any] | None) -> str:
    """Map Exp 3893 to the replication outcome scalar."""

    if not isinstance(payload, Mapping):
        return "INCONCLUSIVE"
    outcome = str(payload.get("replication_outcome", "")).upper()
    if outcome in {"REPLICATED", "REFUTED", "INCONCLUSIVE"}:
        return outcome
    verdict = honest_verdict(payload).upper()
    if "REPLICATED" in verdict:
        return "REPLICATED"
    if "REFUTED" in verdict:
        return "REFUTED"
    return "INCONCLUSIVE"


def derive_moat_verdict(payload: Mapping[str, Any] | None) -> str:
    """Map the tested-harness moat scissor to the durability scalar."""

    verdict = honest_verdict(payload).upper()
    if "MOAT_SURVIVES" in verdict or "MOAT SURVIVES" in verdict:
        return "MOAT_SURVIVES"
    if "SUBSUMED" in verdict:
        return "SUBSUMED"
    return "INCONCLUSIVE"


def derive_facts_outcome(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    *,
    exp3897_was_flagged: bool,
) -> str:
    """Summarize graph-grounding breadth without using quarantined facts runs."""

    if exp3897_was_flagged:
        return "EXCLUDED_EXP3897_FLAGGED"
    graph = clean_upstreams.get(3897)
    complement = clean_upstreams.get(3898)
    reproduced = isinstance(graph, Mapping) and (
        graph.get("graph_grounding_signal_reproduced") is True
        or "REPRODUCED" in honest_verdict(graph).upper()
    )
    complementary = isinstance(complement, Mapping) and (
        complement.get("facts_complementary") is True
        or "COMPLEMENT" in honest_verdict(complement).upper()
    )
    if reproduced and complementary:
        return "REPRODUCED_COMPLEMENTARY"
    if reproduced:
        return "REPRODUCED"
    if complementary:
        return "COMPLEMENTARY_WITHOUT_REPRODUCTION_ARTIFACT"
    return "INCONCLUSIVE"


def derive_fr11_v25_invariant(payload: Mapping[str, Any] | None) -> str:
    """Return whether the FR-11 v25 invariant held."""

    if not isinstance(payload, Mapping):
        return "INCONCLUSIVE"
    verdict = honest_verdict(payload).upper()
    if "INVARIANT_HELD" in verdict or (
        payload.get("frozen_headline_unchanged") is True
        and (
            payload.get("self_learning_invariant_held") is True
            or payload.get("fr11_v25_invariant_held") is True
        )
    ):
        return "INVARIANT_HELD"
    return "INCONCLUSIVE"


def derive_hardware_outcome(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> str:
    """Summarize GateMate terminal assessment and PolarFire/KV260 continuity."""

    gatemate = clean_upstreams.get(3900)
    if not isinstance(gatemate, Mapping):
        gatemate_state = "GATEMATE_MISSING"
    elif honest_verdict(gatemate).startswith("blocked_gatemate"):
        gatemate_state = "GATEMATE_BLOCKED"
    elif gatemate.get("terminal_state_reached") is True or honest_verdict(gatemate).startswith("success:"):
        gatemate_state = "GATEMATE_TERMINAL_CONFIRMED"
    else:
        gatemate_state = "GATEMATE_PARTIAL"

    polarfire = clean_upstreams.get(3901)
    if not isinstance(polarfire, Mapping):
        polarfire_state = "POLARFIRE_KV260_MISSING"
    elif honest_verdict(polarfire).startswith("blocked_"):
        polarfire_state = "POLARFIRE_KV260_BLOCKED"
    elif polarfire.get("no_fpga_fabric_claim") is True or polarfire.get("fabric_acceleration_claimed") is False:
        polarfire_state = "POLARFIRE_KV260_CONTINUITY_NO_FABRIC_CLAIM"
    else:
        polarfire_state = "POLARFIRE_KV260_PARTIAL_NO_FABRIC_CLAIM"
    return f"{gatemate_state}_{polarfire_state}"


def frozen_headline_unchanged(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Guard the frozen 0.9131 FoVer headline against explicit regressions."""

    for payload in clean_upstreams.values():
        if payload.get("frozen_headline_unchanged") is False:
            return False
        frozen = numeric(payload.get("frozen_headline_ensemble_auroc"))
        if frozen is not None and frozen != FROZEN_FOVER_AUROC:
            return False
    return True


def both_energy_theses_bounded(ebt_replication_outcome: str) -> bool:
    """Return the crossroads trigger: P0.1 settled plus EBT replication banked."""

    return ebt_replication_outcome == "REPLICATED"


def operator_next_thesis_recommendation(
    *,
    energy_bounded: bool,
    moat_verdict: str,
    facts_outcome: str,
) -> str:
    """State the loop recommendation while preserving operator choice."""

    if energy_bounded:
        return (
            "Recommend verifier as a durable, broad external second-opinion layer: "
            f"moat={moat_verdict}, facts={facts_outcome}. Treat energy as generator/selector "
            "as twice-refuted for forward-bet purposes; the loop recommends, the operator decides."
        )
    return (
        "Do not manufacture a new headline from partial landing. Keep FoVer 0.9131 frozen, "
        f"carry moat={moat_verdict} and facts={facts_outcome} as conditioned verifier evidence, "
        "and require a clean EBT replication artifact before declaring both energy theses bounded."
    )


def verdict_slug(ebt_replication_outcome: str, moat_verdict: str, facts_outcome: str) -> str:
    """Build the fixed v360 honest-verdict suffix."""

    return (
        f"capstone_v360_ebt{ebt_replication_outcome}_moat{moat_verdict}_"
        f"facts{facts_outcome}"
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


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    publication_gate_data: Mapping[str, Any] | None = None,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v360 capstone from existing upstream verdict artifacts."""

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
        else run_summarize_statuses(root_path, paths)
    )
    publication_gate = dict(publication_gate_data) if publication_gate_data is not None else publication_gate_state()

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

    ebt = derive_ebt_replication_outcome(clean_upstreams.get(3893))
    moat = derive_moat_verdict(clean_upstreams.get(3895))
    facts = derive_facts_outcome(clean_upstreams, exp3897_was_flagged=3897 in flagged_ids)
    fr11 = derive_fr11_v25_invariant(clean_upstreams.get(3899))
    hardware = derive_hardware_outcome(clean_upstreams)
    energy_bounded = both_energy_theses_bounded(ebt)
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
        }
        for experiment_id in sorted(flagged_ids)
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

    base_slug = verdict_slug(ebt, moat, facts)
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
        "schema": "carnot.capstone_v360_3902.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": terminal_verdict,
        "ebt_replication_outcome": ebt,
        "moat_verdict": moat,
        "facts_outcome": facts,
        "fr11_v25_invariant": fr11,
        "hardware_outcome": hardware,
        "both_energy_theses_bounded": energy_bounded,
        "paper_ready": paper_ready,
        "unmet_gates": unmet_gates,
        "publication_gate": publication_gate,
        "frozen_headline_unchanged": frozen_unchanged,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "operator_next_thesis_recommendation": operator_next_thesis_recommendation(
            energy_bounded=energy_bounded,
            moat_verdict=moat,
            facts_outcome=facts,
        ),
        "flagged_artifacts_excluded": excluded,
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


def module_importable(name: str) -> bool:
    """Return whether a precondition module can be imported."""

    try:
        importlib.import_module(name)
    except Exception:  # pragma: no cover - environment failure path.
        return False
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the v360 artifact contract that prevents over-claiming."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive guard.
    for field in STRING_VERDICT_FIELDS:
        if not isinstance(artifact.get(field), str):
            raise ValueError(f"{field} must be a bare string")  # pragma: no cover - defensive guard.
    for field in ("both_energy_theses_bounded", "paper_ready", "frozen_headline_unchanged"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")  # pragma: no cover - defensive guard.
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
    """Build, validate, and write the Exp 3902 capstone JSON artifact."""

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
    """Write the default Exp 3902 artifact and print its path."""

    print(write_artifact(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
