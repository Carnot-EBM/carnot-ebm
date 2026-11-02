"""REQ-REPORT-8328: distinguish an authenticated empty read from missing evidence.

Cold replay checks both primitive semantics and their byte hashes. Current model
work stays zero even when an imported receipt records historical generation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from carnot.reporting import arc_supervisor_frontier_8328 as m
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.v718_replay_history import POLICY

Json = dict[str, Any]
ROOT = m.ROOT
NAME = "experiment_8328_v718_arc_supervisor_frontier"
CLI = "scripts/experiments/" + NAME + ".py"
OUTPUT = m.ROOT / "results" / (NAME + ".json")
TEST = "tests/python/test_arc_supervisor_frontier_8328.py"
OWNED = [
    "python/carnot/reporting/arc_supervisor_" + x + "_8328.py"
    for x in ["frontier", "artifact", "execution"]
] + [CLI]
MODEL_SPECS: list[Json] = []
BUDGET = dict(
    minimum_overlapping_games=3,
    minimum_shared_arms=2,
    minimum_firings_per_game_arm=5,
    independent_unit="game",
    current_game_runs=0,
    current_model_runs=0,
)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness follows owned validation; scientific support is a separate quantity."""
    primitive = raw / "measurement.json"
    atomic_json(primitive, work)
    owned = all(r["passed"] for r in receipts)
    blocked = bool(work["failures"])
    delta = work["delta"]
    fixture = bool(work.get("fixture_claim_scope"))
    ready = owned and not blocked and not fixture
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if blocked
        else "null"
        if not delta["new_outcome_count"]
        else "positive"
    )
    honest = (
        "complete_null_no_supervisor_outcomes"
        if verdict == "null"
        else "complete_"
        + verdict
        + (
            "_"
            + re.sub(
                r"[^A-Za-z0-9_]+",
                "_",
                Path(work["failures"][0].get("path", "external_operand")).stem,
            )
            if blocked and owned
            else "_supervisor_frontier"
        )
    )
    value = dict(
        delta,
        experiment_id=8328,
        task_id="exp8328-arc-supervisor-frontier",
        milestone="2026.10.718",
        run_date="20261009",
        honest_verdict=honest,
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_game_execution_count=0,
        current_model_invocation_count=0,
        historical_model_provenance=work["historical_model_provenance"],
        sample_size_budget=BUDGET,
        verifier_is_oracle=False,
        exposure_scope="exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned and not blocked,
        flagged_adversarial=any(not a["passed"] for a in work.get("audits", [])),
        acceptance_gates=dict(owned_checks=owned, external_operands_authenticated=not blocked),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("audits", []),
        finding_consumer_policy=POLICY,
        preconditions_checked=dict(
            private_scratch=work["private_scratch"],
            failures=work["failures"],
            tools_checked_before_measurement=True,
        ),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=0,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes={p: sha256_file(m.ROOT / p) for p in OWNED},
        raw_shard_hashes=work["snapshots"],
        coverage_statement_counts=work.get("coverage_statement_counts", {}),
        work_reference=dict(path=str(primitive), sha256=sha256_file(primitive)),
        cited_upstream_artifacts=[
            dict(
                path=str(m.FRONTIER),
                sha256=m.PIN,
                fields_imported=[
                    "arc_reader_ready_score",
                    "current_frontier",
                    "finished_at",
                    "qualified_authority_signature",
                ],
            )
        ],
        arc_reader_ready_score=int(ready),
        arc_outcome_support_score=int(ready and bool(delta.get("new_outcome_count"))),
        frontier_before=work["prior_frontier"],
        frontier_after=delta["current_frontier"],
        frontier_cutoff_finished_at=work["frontier_finished_at"],
        solve_provenance="live_agent_self_discovery",
        solve_claims=[],
        selection_recommendations=m.recommend(delta),
        fixture_claim_scope=fixture,
        finished_at=datetime.now(UTC).isoformat(),
    )
    value["field_principles"] = {
        k: "Bind actual execution, authenticated source evidence and observational limits; absence differs from zero."
        for k in value
    }
    value["field_principles"].update(
        arc_reader_ready_score="Qualified mechanics do not require scientific success.",
        arc_outcome_support_score="Count only authenticated new supervisor outcomes; comparative recommendations require the separate cell floor.",
        selection_recommendations="Compare game progress and action cost within curated arms; unknown propensity permits descriptive claims only.",
        frontier_after="Advance only with new accepted receipt identities; no reread counts as progress.",
    )
    value["reproducibility_checksum"] = canonical_hash(work)
    value["field_principles"]["reproducibility_checksum"] = (
        "Seal primitive execution separately from terminal sidecars."
    )
    return value


def replay(value: Json) -> bool:
    """Rehashed invented primitives still fail fresh native joins and semantic reduction."""
    try:
        ref = value["work_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        work = m.json_document(Path(ref["path"]))
        if canonical_hash(work) != value["reproducibility_checksum"]:
            return False
        for label, digest in {**work["source_artifact_hashes"], **work["snapshots"]}.items():
            if sha256_file(Path(label)) != digest:
                return False
        if not work["failures"]:
            native = m.inspect(
                Path(ref["path"]).parent / "authority_locator.v1.json",
                m.authenticate(m.FRONTIER),
                Path(ref["path"]).parent / "adapter",
            )
            if native != work["delta"]:
                return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(ref["path"]).parent,
            Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json"),
        )
        validate_primary(value, OUTPUT)
        return all(value.get(k) == v for k, v in rebuilt.items() if k != "finished_at")
    except (OSError, ValueError, KeyError, TypeError):
        return False
