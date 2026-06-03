"""Build the Exp 3764 v344 convergence capstone artifact.

Spec refs: REQ-REPORT-3764, SCENARIO-REPORT-3764,
SCENARIO-REPORT-3764-GATED, SCENARIO-REPORT-3764-FLAGGED.

The capstone is a reconciliation step, not a new experiment. It reads the
checked-in upstream JSON artifacts, rechecks the stable publication gate, and
stores a narrow product-banking summary. That separation is important because
the milestone closes two research routes as bounded while preserving the
positive verifier product claim; this module must not make a new live-inference
or existential energy-generation claim while summarizing that record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct import guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_3764_capstone_v344.json")
RANDOM_SEED = 3764
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
UPSTREAM_IDS = tuple(range(3754, 3764))

DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3754: Path("results/experiment_3754_archive_v343_activate_v344.json"),
    3755: Path("results/experiment_3755_thesis_a_definitive_reconciliation.json"),
    3756: Path("results/experiment_3756_g2_local_reproducer.json"),
    3757: Path("results/experiment_3757_g3_narrowing_lint.json"),
    3758: Path("results/experiment_3758_package_cli_mcp_e2e_smoke.json"),
    3759: Path("results/experiment_3759_distribution_mirror_operator_publish_checklist.json"),
    3760: Path("results/experiment_3760_certified_abstention_operating_point.json"),
    3761: Path("results/experiment_3761_fr11_v17_live_verifier_precision_tracker.json"),
    3762: Path("results/experiment_3762_kv260_opportunistic_continuity_audit.json"),
    3763: Path("results/experiment_3763_next_phase3_thesis_decision_menu.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "thesis_a_definitively_closed",
    "both_energy_routes_bounded",
    "gates_mechanized",
    "verifier_banked_for_ship",
    "certified_abstention_point_status",
    "paper_ready_preserved",
    "frozen_headline_unchanged",
    "next_thesis_handed_to_operator",
    "flagged_artifacts_excluded",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the milestone's one-line outcome.",
    "inference_substrate": (
        "A capstone reads upstream JSON, runs no live model, and must not add "
        "live-inference provenance to an aggregation artifact."
    ),
    "thesis_a_definitively_closed": (
        "Records the milestone-defining fact: energy-as-generator is bounded "
        "(discriminative-not-generative), reconciled into the record (exp3755)."
    ),
    "both_energy_routes_bounded": (
        "BARE bool -- both selection AND generation are now bounded; the reason "
        ".344 is product-banking not re-grind."
    ),
    "gates_mechanized": (
        "Records G2 reproducer + G3 narrowing lint shipped -- the publication-gate "
        "hardening deliverable."
    ),
    "verifier_banked_for_ship": (
        "Records the Phase-1 software-ship evidence (E2E surfaces passed + mirror "
        "checklist) -- the product deliverable."
    ),
    "certified_abstention_point_status": (
        "The deployable abstention operating point outcome (shipped / gate-skipped "
        "if headline did not reproduce)."
    ),
    "paper_ready_preserved": (
        "G1-G4 stay met; the milestone must not regress the banked verifier product."
    ),
    "frozen_headline_unchanged": (
        "Frozen FoVer 0.9131 stays frozen; .344 reproduces but never moves the headline."
    ),
    "next_thesis_handed_to_operator": (
        "Records that the next-Phase-3 decision is now an operator-seeding surface "
        "(exp3763) -- the loop does not self-commit."
    ),
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation "
        "(fabrication gate)."
    ),
    "cited_upstream_artifacts": "Provenance trail from the capstone numbers to the real artifacts.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

SUMMARY_FIELDS: Mapping[int, tuple[str, ...]] = {
    3754: (
        "honest_verdict",
        "paper_ready_preserved",
        "g1",
        "g2",
        "g3",
        "g4",
        "frozen_fover_auroc",
    ),
    3755: (
        "honest_verdict",
        "thesis_a_definitively_closed",
        "part_a_pass_discriminative",
        "part_b_bounded_at_scale_not_generative",
        "in_loop_chain_superseded",
        "both_energy_routes_bounded",
        "energy_as_selector_status",
        "energy_as_generator_status",
    ),
    3756: ("honest_verdict", "g2_local_reproducer_shipped", "auroc_in_ci95", "reproduced_auroc"),
    3757: (
        "honest_verdict",
        "g3_narrowing_lint_shipped",
        "current_paper_lint_clean",
        "energy_as_generator_forbidden_phrase_guard",
    ),
    3758: (
        "honest_verdict",
        "package_e2e_smoke_passed",
        "cli_e2e_smoke_passed",
        "mcp_e2e_smoke_passed",
        "package_cli_mcp_e2e_smoke_passed",
    ),
    3759: (
        "honest_verdict",
        "distribution_mirror_ready",
        "operator_publish_checklist_ready",
        "agent_published_nothing",
    ),
    3760: (
        "honest_verdict",
        "certified_abstention_point_status",
        "deployable_operating_point_selected",
        "exp3756_auroc_in_ci95",
    ),
    3761: (
        "honest_verdict",
        "fr11_v17_pivoted_to_live_verifier",
        "memory_contribution_preserved",
        "memory_contribution_delta",
    ),
    3762: (
        "honest_verdict",
        "terminal_state_holds",
        "kv260_ssh_reachable",
        "kv260_overlay_loadable",
        "speedup_claim_made",
    ),
    3763: (
        "honest_verdict",
        "loop_will_not_self_seed",
        "supersedes_340_menu",
        "ranked_thesis_menu",
    ),
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    gate_data: Mapping[str, Any] | None = None,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    capstone_adversarial_verify_clean: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v344 capstone from upstream JSON and gate evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = {experiment_id: resolve_upstream_path(root_path, experiment_id) for experiment_id in UPSTREAM_IDS}
    upstreams = {
        experiment_id: read_json_object(path) if path.exists() else None
        for experiment_id, path in paths.items()
    }
    reports = (
        {experiment_id: dict(report) for experiment_id, report in adversarial_reports.items()}
        if adversarial_reports is not None
        else verify_upstreams(paths)
    )
    flagged_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True
    }
    clean_upstreams = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if payload is not None and experiment_id not in flagged_ids
    }
    gate = dict(gate_data) if gate_data is not None else load_publication_gate(root_path)
    summaries = compact_summary_records(
        summary_records if summary_records is not None else run_summarize_artifacts(root_path, paths)
    )

    exp3755 = clean_upstreams.get(3755, {})
    exp3756 = clean_upstreams.get(3756, {})
    exp3757 = clean_upstreams.get(3757, {})
    exp3758 = clean_upstreams.get(3758, {})
    exp3759 = clean_upstreams.get(3759, {})
    exp3760 = clean_upstreams.get(3760, {})
    exp3761 = clean_upstreams.get(3761, {})
    exp3762 = clean_upstreams.get(3762, {})
    exp3763 = clean_upstreams.get(3763, {})

    thesis_closed = thesis_a_closed(exp3755)
    both_bounded = both_energy_routes_bounded(exp3755)
    g2_reproduced = g2_local_reproduced(exp3756)
    gates_mech = g2_reproduced and g3_narrowing_lint_shipped(exp3757)
    verifier_banked = package_cli_mcp_passed(exp3758) and distribution_ready(exp3759)
    abstention_status = certified_abstention_status(exp3756, exp3760)
    paper_ready = paper_ready_from_gate(gate)
    frozen = frozen_headline_unchanged(gate, clean_upstreams.get(3754, {}))
    fr11_pivoted = fr11_live_verifier_pivoted(exp3761)
    next_thesis = next_thesis_handed_to_operator(exp3763)
    kv260_confirmed = kv260_terminal_confirmed(exp3762)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v344_thesis_a_closed_3764.v1",
        "experiment_id": "exp3764",
        "honest_verdict": terminal_verdict(
            gates_mechanized=gates_mech,
            verifier_banked=verifier_banked,
            abstention_status=abstention_status,
            fr11_pivoted=fr11_pivoted,
            next_thesis=next_thesis,
            paper_ready=paper_ready,
            frozen=frozen,
            thesis_closed=thesis_closed,
            both_energy_bounded=both_bounded,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "thesis_a_definitively_closed": thesis_closed,
        "both_energy_routes_bounded": both_bounded,
        "gates_mechanized": gates_mech,
        "verifier_banked_for_ship": verifier_banked,
        "certified_abstention_point_status": abstention_status,
        "paper_ready_preserved": paper_ready,
        "frozen_headline_unchanged": frozen,
        "next_thesis_handed_to_operator": next_thesis,
        "milestone_outcome_plain": milestone_outcome(thesis_closed, both_bounded, verifier_banked),
        "thesis_a_closure_reconciled": {
            "part_a": "PASS_discriminative" if thesis_closed else "not_cited_missing_or_flagged_exp3755",
            "part_b": (
                "BOUNDED_at_scale_not_generative"
                if both_bounded
                else "not_cited_missing_or_flagged_exp3755"
            ),
            "in_loop_chain_superseded": truthy(exp3755.get("in_loop_chain_superseded")),
        },
        "energy_as_selector_status": status_value(
            exp3755.get("energy_as_selector_status"),
            "honest-negative-bounded" if both_bounded else "not_cited_missing_or_flagged_exp3755",
        ),
        "energy_as_generator_status": status_value(
            exp3755.get("energy_as_generator_status"),
            "bounded-at-scale-discriminative-not-generative"
            if both_bounded
            else "not_cited_missing_or_flagged_exp3755",
        ),
        "no_new_existential_claim": True,
        "publication_gate": gate_details(gate),
        "g2_local_reproducer_auroc_in_ci95": g2_reproduced,
        "g3_narrowing_lint_shipped": g3_narrowing_lint_shipped(exp3757),
        "phase1_ship_surfaces_passed": package_cli_mcp_passed(exp3758),
        "distribution_mirror_checklist_ready": distribution_ready(exp3759),
        "agent_published_nothing": truthy(exp3759.get("agent_published_nothing")),
        "fr11_v17_pivoted_to_live_verifier": fr11_pivoted,
        "memory_contribution_preserved": truthy(exp3761.get("memory_contribution_preserved")),
        "kv260_terminal_confirmed": kv260_confirmed,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "headline_aggregation_experiment_ids": sorted(clean_upstreams),
        "missing_upstream_artifacts": missing_upstream_artifacts(paths, upstreams),
        "flagged_artifacts_excluded": flagged_artifacts(paths, flagged_ids),
        "cited_upstream_artifacts": cited_upstream_artifacts(root_path, paths, clean_upstreams),
        "summarized_upstream_artifacts": summaries,
        "adversarial_verify_clean": capstone_adversarial_verify_clean,
        "adversarial_verify_report": {"flags": []},
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    gate_data: Mapping[str, Any] | None = None,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write, adversarial-verify, and rewrite the stable Exp 3764 artifact."""

    root_path = Path(root)
    out_path = root_path / OUTPUT_REL_PATH
    artifact = build_artifact(
        root_path,
        gate_data=gate_data,
        summary_records=summary_records,
        adversarial_reports=adversarial_reports,
        capstone_adversarial_verify_clean=True,
        started_s=started_s,
        now_s=now_s,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = adversarial_verify.verify_artifact(out_path)
    artifact["adversarial_verify_report"] = report
    artifact["adversarial_verify_clean"] = report_is_clean(report)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and honesty errors for the Exp 3764 capstone."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required artifact fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete: capstone_v344_"):
        errors.append("honest_verdict must be a terminal Exp 3764 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the v344 aggregation-only substrate")
    for field in (
        "thesis_a_definitively_closed",
        "both_energy_routes_bounded",
        "gates_mechanized",
        "verifier_banked_for_ship",
        "next_thesis_handed_to_operator",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    if artifact.get("certified_abstention_point_status") not in {"shipped", "skipped"}:
        errors.append("certified_abstention_point_status must be shipped or skipped")
    if artifact.get("paper_ready_preserved") is not True:
        errors.append("paper_ready_preserved must be true")
    if artifact.get("frozen_headline_unchanged") is not True:
        errors.append("frozen_headline_unchanged must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    validate_citations(artifact.get("cited_upstream_artifacts"), errors)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3764")
    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, int | float) or isinstance(duration_s, bool) or float(duration_s) < 0.0001:
        errors.append("duration_s must be numeric with the aggregation plausibility floor")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must cover all required artifact fields")
    if has_live_model_markers(artifact):
        errors.append("artifact must not copy live-model substrate markers")
    if not report_is_clean(artifact.get("adversarial_verify_report", {"flags": []})):
        errors.append("adversarial verifier must report no critical flag")
    checksum = artifact.get("reproducibility_checksum")
    if not is_sha256(checksum):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum does not match artifact content")
    return errors


def terminal_verdict(
    *,
    gates_mechanized: bool,
    verifier_banked: bool,
    abstention_status: str,
    fr11_pivoted: bool,
    next_thesis: bool,
    paper_ready: bool,
    frozen: bool,
    thesis_closed: bool = True,
    both_energy_bounded: bool = True,
) -> str:
    """Return the terminal verdict string using only classified states."""

    thesis_token = "thesis_a_closed" if thesis_closed else "thesis_a_not_closed"
    energy_token = "both_energy_routes_bounded" if both_energy_bounded else "both_energy_routes_not_fully_cited"
    return (
        f"complete: capstone_v344_{thesis_token}_{energy_token}_"
        f"{'gates_mechanized' if gates_mechanized else 'gates_not_mechanized'}_"
        f"{'verifier_banked' if verifier_banked else 'verifier_not_banked'}_"
        f"abstention_point_{abstention_status}_"
        f"{'fr11_pivoted' if fr11_pivoted else 'fr11_not_pivoted'}_"
        f"{'next_thesis_to_operator' if next_thesis else 'next_thesis_not_handed'}_"
        f"paper_ready_{str(paper_ready).lower()}_"
        f"{'frozen_headline_unchanged' if frozen else 'frozen_headline_changed'}"
    )


def resolve_upstream_path(root: Path, experiment_id: int) -> Path:
    """Return the default path or the first same-ID artifact in results."""

    default = root / DEFAULT_UPSTREAM_PATHS[experiment_id]
    if default.exists():
        return default
    matches = sorted((root / "results").glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else default


def read_json_object(path: Path) -> JsonDict:
    """Read an upstream JSON object; array artifacts are invalid provenance."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_publication_gate(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary
    completed = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_summarize_artifacts(root: Path, paths: Mapping[int, Path]) -> list[JsonDict]:  # pragma: no cover - subprocess boundary
    records: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        arg = str(path) if path.exists() else str(experiment_id)
        completed = subprocess.run(
            [sys.executable, "scripts/summarize_artifact.py", arg],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        records.append(
            {
                "experiment_id": experiment_id,
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            }
        )
    return records


def verify_upstreams(paths: Mapping[int, Path]) -> dict[int, JsonDict]:  # pragma: no cover - subprocess boundary
    return {
        experiment_id: adversarial_verify.verify_artifact(path) if path.exists() else {"flags": []}
        for experiment_id, path in paths.items()
    }


def compact_summary_records(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep only deterministic metadata proving the summarizer was run."""

    return [
        {
            "experiment_id": record.get("experiment_id", record.get("exp")),
            "returncode": record.get("returncode"),
            "stdout_sha256": record.get("stdout_sha256"),
            "stderr_sha256": record.get("stderr_sha256"),
        }
        for record in records
    ]


def thesis_a_closed(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    return bool(
        payload
        and (
            truthy(payload.get("thesis_a_definitively_closed"))
            or (
                truthy(payload.get("part_a_pass_discriminative"))
                and truthy(payload.get("part_b_bounded_at_scale_not_generative"))
                and truthy(payload.get("in_loop_chain_superseded"))
            )
            or ("thesis_a_definitively_closed" in verdict and "bounded" in verdict)
        )
    )


def both_energy_routes_bounded(payload: Mapping[str, Any]) -> bool:
    selector = str(payload.get("energy_as_selector_status") or "").lower()
    generator = str(payload.get("energy_as_generator_status") or "").lower()
    return bool(
        payload
        and (
            truthy(payload.get("both_energy_routes_bounded"))
            or ("bounded" in selector and "bounded" in generator)
        )
    )


def g2_local_reproduced(payload: Mapping[str, Any]) -> bool:
    return bool(payload and truthy(payload.get("g2_local_reproducer_shipped")) and truthy(payload.get("auroc_in_ci95")))


def g3_narrowing_lint_shipped(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("g3_narrowing_lint_shipped"))
        and truthy(payload.get("current_paper_lint_clean"))
        and truthy(payload.get("energy_as_generator_forbidden_phrase_guard"))
    )


def package_cli_mcp_passed(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and (
            truthy(payload.get("package_cli_mcp_e2e_smoke_passed"))
            or (
                truthy(payload.get("package_e2e_smoke_passed"))
                and truthy(payload.get("cli_e2e_smoke_passed"))
                and truthy(payload.get("mcp_e2e_smoke_passed"))
            )
        )
    )


def distribution_ready(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("distribution_mirror_ready"))
        and truthy(payload.get("operator_publish_checklist_ready"))
        and truthy(payload.get("agent_published_nothing"))
    )


def certified_abstention_status(exp3756: Mapping[str, Any], exp3760: Mapping[str, Any]) -> str:
    if not g2_local_reproduced(exp3756):
        return "skipped"
    if not exp3760:
        return "skipped"
    status = str(exp3760.get("certified_abstention_point_status") or "").lower()
    if status == "shipped" or truthy(exp3760.get("deployable_operating_point_selected")):
        return "shipped"
    return "skipped"


def fr11_live_verifier_pivoted(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("fr11_v17_pivoted_to_live_verifier"))
        and truthy(payload.get("memory_contribution_preserved"))
    )


def kv260_terminal_confirmed(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("terminal_state_holds"))
        and truthy(payload.get("kv260_ssh_reachable"))
        and truthy(payload.get("kv260_overlay_loadable"))
        and payload.get("speedup_claim_made") is not True
    )


def next_thesis_handed_to_operator(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "")
    return bool(
        payload
        and truthy(payload.get("loop_will_not_self_seed"))
        and truthy(payload.get("supersedes_340_menu"))
        and "operator_seeding" in verdict
    )


def paper_ready_from_gate(gate: Mapping[str, Any]) -> bool:
    gates = gate.get("gates")
    return bool(
        gate.get("paper_ready") is True
        and isinstance(gates, Mapping)
        and all(isinstance(gates.get(name), Mapping) and gates[name].get("pass") is True for name in ("G1", "G2", "G3", "G4"))
    )


def frozen_headline_unchanged(gate: Mapping[str, Any], exp3754: Mapping[str, Any]) -> bool:
    if exp3754 and numeric(exp3754.get("frozen_fover_auroc")) == FROZEN_FOVER_AUROC:
        return True
    return paper_ready_from_gate(gate)


def gate_details(gate: Mapping[str, Any]) -> JsonDict:
    gates = gate.get("gates") if isinstance(gate.get("gates"), Mapping) else {}
    return {
        "paper_ready": gate.get("paper_ready") is True,
        "g1": gate_pass(gates, "G1"),
        "g2": gate_pass(gates, "G2"),
        "g3": gate_pass(gates, "G3"),
        "g4": gate_pass(gates, "G4"),
        "unmet_gates": list(gate.get("unmet_gates") or []),
    }


def gate_pass(gates: Mapping[str, Any], name: str) -> bool:
    gate = gates.get(name)
    return isinstance(gate, Mapping) and gate.get("pass") is True


def milestone_outcome(thesis_closed: bool, both_bounded: bool, verifier_banked: bool) -> str:
    if thesis_closed and both_bounded and verifier_banked:
        return (
            ".344 is a convergence milestone: Thesis-A part-(a) passes as a "
            "discriminative result, part-(b) is bounded at scale / not-generative, "
            "the frozen verifier product is banked, and the next research decision "
            "is handed to the operator. It makes no new existential claim."
        )
    return (
        ".344 capstone ran as aggregation, but one or more upstream artifacts needed "
        "for the clean milestone claim were missing or excluded; no absent result is inferred."
    )


def missing_upstream_artifacts(
    paths: Mapping[int, Path],
    upstreams: Mapping[int, Mapping[str, Any] | None],
) -> list[JsonDict]:
    return [
        {"experiment_id": experiment_id, "path": str(paths[experiment_id]), "reason": "artifact_missing"}
        for experiment_id in UPSTREAM_IDS
        if upstreams.get(experiment_id) is None
    ]


def flagged_artifacts(paths: Mapping[int, Path], flagged_ids: set[int]) -> list[JsonDict]:
    return [
        {"experiment_id": experiment_id, "path": str(paths[experiment_id]), "reason": "flagged_adversarial=true"}
        for experiment_id in sorted(flagged_ids)
    ]


def cited_upstream_artifacts(
    root: Path,
    paths: Mapping[int, Path],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    citations: list[JsonDict] = []
    for experiment_id in sorted(clean_upstreams):
        path = paths[experiment_id]
        payload = clean_upstreams[experiment_id]
        citations.append(
            {
                "experiment_id": experiment_id,
                "path": str(path),
                "fields_imported": [
                    field for field in SUMMARY_FIELDS[experiment_id] if nested_get(payload, field) is not None
                ],
                "honest_verdict": payload.get("honest_verdict"),
                "sha256": sha256_file(path if path.is_absolute() else root / path),
            }
        )
    return citations


def validate_citations(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("cited_upstream_artifacts must be a list")
        return
    for item in value:
        if not isinstance(item, Mapping):
            errors.append("each citation must be an object")
            continue
        if not isinstance(item.get("fields_imported"), list):
            errors.append("each citation must include fields_imported")
        if not is_sha256(item.get("sha256")):
            errors.append("each citation must include a sha256 hex string")


def has_live_model_markers(payload: Mapping[str, Any]) -> bool:
    blob = json.dumps(payload, sort_keys=True).lower()
    return any(marker in blob for marker in ("live_llm_inference", "target_model", "model_specs"))


def report_is_clean(report: Any) -> bool:
    if not isinstance(report, Mapping):
        return True
    for flag in report.get("flags") or []:
        if not isinstance(flag, Mapping):
            continue
        if str(flag.get("severity") or "").lower() == "critical":
            return False
    return True


def nested_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in {"true", "pass", "passed", "shipped"})


def status_value(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(float(value), 4)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = dict(payload)
    filtered["reproducibility_checksum"] = ""
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - CLI wrapper
    out_path = run(REPO_ROOT)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
