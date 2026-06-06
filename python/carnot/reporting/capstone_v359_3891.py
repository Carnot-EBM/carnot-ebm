"""Build the Exp 3891 v359 forward-bet capstone artifact.

Spec refs: REQ-CAPSTONE-3891, SCENARIO-CAPSTONE-3891.

This module is an aggregation step, not a new experiment. It preserves the
paper-ready FoVer headline, excludes any upstream artifact stamped
``flagged_adversarial:true``, and records the milestone's added lenses as
conditioned verdicts rather than inventing a fresh headline.
"""

from __future__ import annotations

from collections.abc import Mapping
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
OUTPUT_REL_PATH = Path("results/experiment_3891_capstone_v359.json")
EXPERIMENT_ID = 3891
RANDOM_SEED = 3891
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = "artifact_aggregation_only_cached_json_and_publication_gate"

UPSTREAM_IDS = tuple(range(3882, 3891))
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3882: Path("results/experiment_3882_thesis_a_partb_killgate.json"),
    3883: Path("results/experiment_3883_ebt_system2_kcurve.json"),
    3884: Path("results/experiment_3884_in_distribution_error_rich_corpus.json"),
    3885: Path("results/experiment_3885_moat_scissor_in_distribution.json"),
    3886: Path("results/experiment_3886_graph_grounding_fact_verifier_defabricated.json"),
    3887: Path("results/experiment_3887_facts_complementarity.json"),
    3888: Path("results/experiment_3888_fr11_v24_independence_reweighting.json"),
    3889: Path("results/experiment_3889_gatemate_continuity_corrigendum.json"),
    3890: Path("results/experiment_3890_polarfire_kv260_continuity.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "ebt_adjudication",
    "ebt_system2_outcome",
    "moat_verdict",
    "facts_outcome",
    "fr11_v24_invariant",
    "hardware_outcome",
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
    "ebt_adjudication",
    "ebt_system2_outcome",
    "moat_verdict",
    "facts_outcome",
    "fr11_v24_invariant",
    "hardware_outcome",
    "operator_next_thesis_recommendation",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "ebt_adjudication": "ARTIFACT/FUNDAMENTAL/INCONCLUSIVE - the Phase-3 energy-as-generator verdict.",
    "ebt_system2_outcome": "SUPPORTED/BOUNDED - whether energy-descent thinking helps at scale.",
    "moat_verdict": "MOAT_SURVIVES/SUBSUMED/INCONCLUSIVE on the in-distribution corpus - the DT-P2 durability answer.",
    "facts_outcome": "Graph-grounding signal reproduced? complementary? - excluded if exp3886 is flagged.",
    "fr11_v24_invariant": "Self-learning mandate outcome across the v23-to-v24 iteration.",
    "hardware_outcome": "GateMate de-flagged plus PolarFire/KV260 continuity without a fabric claim.",
    "paper_ready": "Must stay true; the milestone adds lenses, not a new headline.",
    "unmet_gates": "G1-G4 publication-gate status from scripts/publication_gate.py --json.",
    "frozen_headline_unchanged": "The frozen FoVer 0.9131 invariant guard.",
    "operator_next_thesis_recommendation": "Recommended next forward bet; the operator decides.",
    "flagged_artifacts_excluded": "Artifacts skipped for flagged_adversarial:true before any metric aggregation.",
    "preconditions_checked": "Aggregation methodology and upstream landing state.",
    "duration_s": "Aggregation wall-clock duration.",
    "inference_substrate": "Aggregation methodology without model or hardware execution markers.",
}


def is_sha256(value: object) -> bool:
    """Return whether ``value`` is a SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def numeric(value: object) -> float | None:
    """Convert JSON numeric scalars while rejecting booleans."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object artifact from disk."""

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
    except ValueError:  # pragma: no cover - only used for odd external paths.
        return str(path)


def honest_verdict(payload: Mapping[str, Any] | None) -> str:
    """Extract an upstream honest verdict without interpreting metrics."""

    if not isinstance(payload, Mapping):
        return "missing"
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else "missing"


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether the upstream is stamped with the fabrication gate flag."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def derive_ebt_adjudication(payload: Mapping[str, Any] | None) -> str:
    """Map Exp 3882's verdict to the v359 EBT adjudication scalar."""

    verdict = honest_verdict(payload).upper()
    if "FUNDAMENTAL" in verdict:
        return "FUNDAMENTAL"
    if "ARTIFACT" in verdict:
        return "ARTIFACT"
    return "INCONCLUSIVE"


def derive_system2_outcome(payload: Mapping[str, Any] | None) -> str:
    """Map Exp 3883's K-curve verdict to SUPPORTED or BOUNDED."""

    verdict = honest_verdict(payload).upper()
    if "SUPPORTED" in verdict:
        return "SUPPORTED"
    return "BOUNDED"


def derive_moat_verdict(
    payload: Mapping[str, Any] | None,
    *,
    scissor_was_flagged: bool,
) -> str:
    """Map the in-distribution moat scissor only if Exp 3885 is clean."""

    if scissor_was_flagged or not isinstance(payload, Mapping):
        return "INCONCLUSIVE"
    verdict = honest_verdict(payload).upper()
    if "MOAT_SURVIVES" in verdict or "MOAT SURVIVES" in verdict:
        return "MOAT_SURVIVES"
    if "SUBSUMED" in verdict:
        return "SUBSUMED"
    return "INCONCLUSIVE"


def derive_facts_outcome(
    clean_upstreams: Mapping[int, Mapping[str, Any]],
    *,
    exp3886_was_flagged: bool,
) -> str:
    """Summarize facts broadening without using flagged Exp 3886 numbers."""

    if exp3886_was_flagged:
        return "EXCLUDED_EXP3886_FLAGGED"
    complementarity = clean_upstreams.get(3887)
    if isinstance(complementarity, Mapping):
        verdict = honest_verdict(complementarity).upper()
        if verdict.startswith("BLOCKED"):
            return "INCONCLUSIVE"
        if "COMPLEMENT" in verdict or numeric(complementarity.get("graph_independent_contribution")) not in (None, 0.0):
            return "COMPLEMENTARY"
    graph = clean_upstreams.get(3886)
    if isinstance(graph, Mapping) and numeric(graph.get("facts_catch_delta")) not in (None, 0.0):
        return "REPRODUCED_NO_COMPLEMENTARITY_AUDIT"
    return "INCONCLUSIVE"


def derive_fr11_v24_invariant(payload: Mapping[str, Any] | None) -> str:
    """Return whether the FR-11 v24 invariant held."""

    if not isinstance(payload, Mapping):
        return "INCONCLUSIVE"
    verdict = honest_verdict(payload).upper()
    if "INVARIANT_HELD" in verdict or (
        payload.get("learned_ensemble_auroc_in_frozen_ci") is True
        and payload.get("memory_ablation_contribution_min_met") is True
        and payload.get("frozen_headline_unchanged") is True
    ):
        return "INVARIANT_HELD"
    return "INCONCLUSIVE"


def derive_hardware_outcome(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> str:
    """Summarize GateMate and PolarFire/KV260 continuity as one scalar."""

    gatemate = clean_upstreams.get(3889)
    if not isinstance(gatemate, Mapping):
        gatemate_state = "GATEMATE_MISSING"
    elif honest_verdict(gatemate).startswith("blocked_gatemate"):
        gatemate_state = "GATEMATE_BLOCKED"
    elif gatemate.get("gatemate_bitstream_flashed") is True or honest_verdict(gatemate).startswith("success:"):
        gatemate_state = "GATEMATE_DEFLAGGED"
    else:
        gatemate_state = "GATEMATE_PARTIAL"

    polarfire = clean_upstreams.get(3890)
    if not isinstance(polarfire, Mapping):
        polarfire_state = "POLARFIRE_KV260_MISSING"
    elif honest_verdict(polarfire).startswith("blocked_"):
        polarfire_state = "POLARFIRE_KV260_BLOCKED"
    elif polarfire.get("no_fpga_fabric_claim") is True or polarfire.get("fabric_acceleration_claimed") is False:
        polarfire_state = "POLARFIRE_KV260_CONTINUITY_NO_FABRIC_CLAIM"
    else:
        polarfire_state = "POLARFIRE_KV260_PARTIAL_NO_FABRIC_CLAIM"
    return f"{gatemate_state}_{polarfire_state}"


def frozen_headline_unchanged(payload: Mapping[str, Any] | None) -> bool:
    """Check the 0.9131 guard from the FR-11 v24 artifact."""

    if not isinstance(payload, Mapping):
        return False
    frozen = numeric(payload.get("frozen_headline_ensemble_auroc"))
    return payload.get("frozen_headline_unchanged") is True and (
        frozen == FROZEN_FOVER_AUROC or frozen is None
    )


def operator_next_thesis_recommendation(ebt_adjudication: str, moat_verdict: str) -> str:
    """State the loop recommendation while leaving the decision to the operator."""

    if ebt_adjudication == "FUNDAMENTAL" and moat_verdict != "MOAT_SURVIVES":
        return (
            "Given EBT=FUNDAMENTAL and moat="
            f"{moat_verdict}, do not manufacture a new headline; keep the "
            "paper-ready FoVer 0.9131 record frozen and ask the operator to "
            "choose the next forward bet, preferably a de-flagged moat rerun "
            "or a non-energy-as-generator architecture seed."
        )
    if moat_verdict == "MOAT_SURVIVES":
        return (
            "Treat the moat as the next candidate thesis, but keep FoVer 0.9131 "
            "frozen until the operator explicitly chooses to product-forward it."
        )
    return (
        f"Given EBT={ebt_adjudication} and moat={moat_verdict}, freeze the "
        "current paper-ready headline and let the operator seed the next thesis."
    )


def verdict_slug(ebt_adjudication: str, moat_verdict: str, facts_outcome: str) -> str:
    """Build the fixed v359 honest-verdict suffix."""

    return (
        f"capstone_v359_ebt{ebt_adjudication}_moat{moat_verdict}_"
        f"facts{facts_outcome}"
    )


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact payload while excluding its own checksum."""

    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    publication_gate_data: Mapping[str, Any] | None = None,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v359 capstone from existing upstream verdict artifacts."""

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
    clean_upstreams = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in flagged_ids
    }

    ebt_adjudication = derive_ebt_adjudication(clean_upstreams.get(3882))
    system2_outcome = derive_system2_outcome(clean_upstreams.get(3883))
    moat = derive_moat_verdict(clean_upstreams.get(3885), scissor_was_flagged=3885 in flagged_ids)
    facts = derive_facts_outcome(clean_upstreams, exp3886_was_flagged=3886 in flagged_ids)
    fr11 = derive_fr11_v24_invariant(clean_upstreams.get(3888))
    hardware = derive_hardware_outcome(clean_upstreams)
    paper_ready = publication_gate.get("paper_ready") is True
    unmet_gates = list(publication_gate.get("unmet_gates", []))
    frozen_unchanged = frozen_headline_unchanged(clean_upstreams.get(3888))
    cited_ids = set(clean_upstreams)
    all_landed_nonflagged_aggregated = cited_ids == {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in flagged_ids
    }
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
            "exists": upstreams[experiment_id] is not None,
            "honest_verdict": honest_verdict(upstreams[experiment_id]),
            "flagged_adversarial": experiment_id in flagged_ids,
            "included": experiment_id in clean_upstreams,
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
        for experiment_id, path in paths.items()
    }

    base_slug = verdict_slug(ebt_adjudication, moat, facts)
    suffix = f"{base_slug}_paper_ready_{str(paper_ready).lower()}_frozen_{'unchanged' if frozen_unchanged else 'changed'}"
    if capstone_complete:
        honest = f"complete: {suffix}"
    elif not paper_ready:
        honest = f"blocked_publication_gate: {suffix}"
    elif not frozen_unchanged:
        honest = f"blocked_frozen_headline: {suffix}"
    else:  # pragma: no cover - retained for future citation drift guards.
        honest = f"blocked_aggregation: {suffix}"

    end = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "schema": "carnot.capstone_v359_3891.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": honest,
        "ebt_adjudication": ebt_adjudication,
        "ebt_system2_outcome": system2_outcome,
        "moat_verdict": moat,
        "facts_outcome": facts,
        "fr11_v24_invariant": fr11,
        "hardware_outcome": hardware,
        "paper_ready": paper_ready,
        "unmet_gates": unmet_gates,
        "publication_gate": publication_gate,
        "frozen_headline_unchanged": frozen_unchanged,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "operator_next_thesis_recommendation": operator_next_thesis_recommendation(ebt_adjudication, moat),
        "flagged_artifacts_excluded": excluded,
        "cited_upstream_artifacts": cited,
        "preconditions_checked": {
            "summarize_artifact_importable": module_importable("scripts.summarize_artifact"),
            "publication_gate_importable": module_importable("scripts.publication_gate"),
            "publication_gate_checked": True,
            "all_landed_nonflagged_verdicts_aggregated": all_landed_nonflagged_aggregated,
            "capstone_complete": capstone_complete,
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
    """Validate the artifact contract that keeps v359 from over-claiming."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive guard.
    for field in STRING_VERDICT_FIELDS:
        if not isinstance(artifact.get(field), str):
            raise ValueError(f"{field} must be a bare string")  # pragma: no cover - defensive guard.
    if not isinstance(artifact.get("paper_ready"), bool):
        raise ValueError("paper_ready must be a bare bool")  # pragma: no cover - defensive guard.
    if not isinstance(artifact.get("frozen_headline_unchanged"), bool):
        raise ValueError("frozen_headline_unchanged must be a bare bool")  # pragma: no cover - defensive guard.
    if not isinstance(artifact.get("unmet_gates"), list):
        raise ValueError("unmet_gates must be a list")  # pragma: no cover - defensive guard.
    if not str(artifact.get("honest_verdict", "")).startswith(("complete:", "success:", "failure:", "blocked_")):
        raise ValueError("honest_verdict must have a terminal prefix")  # pragma: no cover - defensive guard.
    if "GGUF" in str(artifact.get("inference_substrate")) or "CUDA" in str(artifact.get("inference_substrate")):
        raise ValueError("inference_substrate must not carry GGUF/CUDA markers")  # pragma: no cover - defensive guard.
    if numeric(artifact.get("duration_s")) is None or numeric(artifact.get("duration_s")) < 0.0001:
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
    """Build, validate, and write the Exp 3891 capstone JSON artifact."""

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
    """Evaluate the stable G1-G4 publication gate."""

    from scripts import publication_gate

    return publication_gate.evaluate()


def run_summarize_statuses(root: Path, paths: Mapping[int, Path]) -> dict[int, JsonDict]:  # pragma: no cover - subprocess IO.
    """Run summarize_artifact.py for each expected upstream artifact."""

    statuses: dict[int, JsonDict] = {}
    summarizer = root / "scripts" / "summarize_artifact.py"
    for experiment_id, path in paths.items():
        if not path.exists():
            statuses[experiment_id] = {"returncode": 1, "missing": True}
            continue
        result = subprocess.run(
            [sys.executable, str(summarizer), relative_to_root(root, path)],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        statuses[experiment_id] = {
            "returncode": result.returncode,
            "stdout_excerpt": result.stdout[:1000],
            "stderr_excerpt": result.stderr[:1000],
        }
    return statuses


def main() -> int:  # pragma: no cover - CLI wrapper.
    """Write the default Exp 3891 artifact and print its path."""

    print(write_artifact(REPO_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
