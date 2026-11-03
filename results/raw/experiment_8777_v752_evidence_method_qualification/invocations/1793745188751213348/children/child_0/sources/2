"""REQ-REPORT-8777 / REQ-VERIFY-8777: reduce and replay sealed qualification operands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coverage import Coverage

from carnot.reporting import evidence_qualification_8777 as q
from carnot.reporting.current_work_receipt import canonical_hash, ZERO_INVOCATION_COUNTS
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting import v746_retained_evidence as evidence

Json = dict[str, Any]


def audit_validation(validation: Json) -> bool:
    """Rebuild node outcomes and statement coverage from retained JUnit and actual SQLite."""
    try:
        union, coverage = validation["union"], validation["coverage"]
        children = union.get("retained_child_rows", [])
        if not children:
            return not coverage["complete"]
        if evidence.aggregate(children, union["node_ids"]) != union:
            return False
        original = validation["original_node_union"]
        if evidence.aggregate(original["retained_child_rows"], original["node_ids"]) != original:
            return False
        ref = coverage["combined_reference"]
        if reference(Path(ref["path"])) != ref:
            return False
        cov = Coverage(data_file=ref["path"], config_file=False)
        cov.load()
        files: list[Json] = []
        for row in coverage["files"]:
            _, statements, excluded, missing, _ = cov.analysis2(str(q.ROOT / row["path"]))
            files.append(dict(row, statements=len(statements), excluded=excluded, missing=missing))
        total = sum(r["statements"] for r in files)
        missing_count = sum(len(r["missing"]) for r in files)
        return bool(
            files == coverage["files"]
            and coverage["percent"] == 100 * (total - missing_count) / total
            and coverage["complete"]
            == (total > 0 and missing_count == 0 and not any(r["excluded"] for r in files))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def build(work: Json, validation: Json, output: Path) -> Json:
    """Readiness means execution qualified; the independent oracle still limits scientific scope."""
    inputs, labels, decisions = [
        load(work[k])["rows"]
        for k in ["corpus_reference", "oracle_reference", "decisions_reference"]
    ]
    rows = q.score(inputs, labels, decisions)
    summary = q.summarize(rows)
    original_union = validation.get("original_node_union", dict(passed=False, node_ids=[]))
    original_nodes = [
        work["original_tests"][0] + "::" + n.split("::", 1)[1]
        if n.split("::", 1)[0].endswith("/" + Path(work["original_tests"][0]).name)
        else n
        for n in original_union["node_ids"]
    ]
    checks = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["union"]["passed"]
        and validation["coverage"]["complete"]
    )
    gates = dict(
        preconditions=work["preconditions_checked"]["passed"],
        original_failure_publication=work["failure_control"]["passed"],
        original_method_authenticated=work["original_method_authenticated"],
        original_node_inventory=original_union["passed"]
        and sorted(original_nodes) == sorted(work["expected_original_nodes"]),
        graph_disjointness=load(work["manifest_reference"])["disjoint"],
        complete_rows=len(rows) == 15552,
        decision_audit=q.audit(inputs, decisions),
        truth_controls=summary["truth_controls_passed"],
        scoped_validation=bool(checks),
    )
    ready = all(gates.values())
    verdict = (
        "blocked"
        if not gates["preconditions"]
        else "disqualified"
        if not ready
        else "circular_positive"
        if summary["scientific_promotion"]
        else "null"
    )
    failures = [
        dict(
            upstream=q.TASK,
            **work["frozen_input_reference"],
            check=k,
            field="preconditions_checked.passed" if k == "preconditions" else k,
            operator="==",
            expected=True,
            observed=v,
            passed=False,
        )
        for k, v in gates.items()
        if not v
    ]
    value: Json = dict(
        experiment_id=8777,
        task_id=q.TASK,
        milestone="2026.11.752",
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v752_evidence_method_qualification",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        rows=rows,
        intended_count=15552,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=120,
        independent_count_scope="120 new evaluation worlds; 60 development,120 exposed replication,12 canary,120 retention separate; surfaces never multiply N",
        verifier_is_oracle=False,
        oracle_dependent_benchmark=True,
        exposure_scope="All V751 evaluation graphs are exposed replication; all generators and new test graphs inspected by the implementation agent. No blindness claimed.",
        independent_generalization_score=0,
        acceptance_gates=gates,
        required_checks_passed=bool(checks),
        flagged_adversarial=verdict == "disqualified",
        validation_receipts=validation["receipts"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=q.SEED,
        source_artifact_hashes=work["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["shards"] + validation.get("raw_references", []),
        cited_upstream_artifacts=work["references"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope=dict(
            owned=q.OWNED,
            original_owned=work["original_owned"],
            tests=q.TESTS,
            consumers=work["consumers"],
            named_e2es=["E2E-018 private CLI", "E2E-021 private CLI"],
            full_suite_pass_claimed=False,
        ),
        evidence_methods_ready_score=int(ready),
        frozen_method_reference=work["method_reference"],
        new_world_manifest=load(work["manifest_reference"]),
        original_node_union=dict(
            original_union,
            archived_node_ids=original_union["node_ids"],
            node_ids=original_nodes,
            original_assertions_unchanged=work["original_method_authenticated"],
        ),
        owned_statement_coverage=validation["coverage"],
        oracle_access_rows=work["oracle_access_rows"],
        original_failure_publication_control=work["failure_control"],
        original_primary_preserved=work["original_primary_preserved"],
        equal_call_ceiling=5,
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        frozen_input_reference=work["frozen_input_reference"],
        publication_output=str(output.absolute()),
        methodology_note="Constructed signed relational records with independent Boolean-completion labels. The unchanged original method and semantic assertions remain frozen. V751 is exposed replication and its historical disqualification remains. Five calls per arm/unit; longer local queries escalate after at most five premise checks. Six conditions and two surfaces; only120 new evaluation graph classes define the primary estimand. The frozen paired-world bootstrap uses10000 draws and Bonferroni simultaneous97.5 percent intervals. Retention and canaries are separate symbolic controls. Oracle-dependent comparisons establish no natural extraction, factuality or hidden-game generalization. No LLM or training executes.",
        **{k: v for k, v in summary.items() if k != "independent_count"},
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind exact task, invocation and sealed operands; execution readiness is separate from natural usefulness.",
    )
    value["field_principles"].update(
        evidence_methods_ready_score="One only after original failure publication, independent truth, complete rows and every owned check qualify.",
        independent_count="Only120 new canonical evaluation worlds define N; surfaces and exposed replication do not enlarge it.",
        gate_check_summary="Retain exact upstream, path/hash and comparison operands for every failed gate.",
        repository_health_reference="The bounded full-suite diagnostic remains separate from scoped acceptance.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Authenticate independent seals before rebuilding truth, decisions and terminal fields."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if seal != {
            k: value[k]
            for k in ["primitive_reference", "validation_reference", "frozen_input_reference"]
        }:
            return False
        work, validation = load(value["primitive_reference"]), load(value["validation_reference"])
        if not audit_validation(validation):
            raise ValueError("validation_evidence_drift")
        frozen = load(value["frozen_input_reference"])
        if any(work[k] != v for k, v in frozen.items()):
            return False
        for ref in (
            work["sources"]
            + work["references"]
            + work["shards"]
            + validation.get("raw_references", [])
        ):
            if reference(Path(ref["path"])) != ref:
                return False
        archive_refs = load(
            next(r for r in work["shards"] if Path(r["path"]).name == "archive_seal.json")
        )["references"]
        for ref in archive_refs + [
            r
            for r in work["failure_control"].values()
            if isinstance(r, dict) and set(r) == {"path", "sha256"}
        ]:
            if reference(Path(ref["path"])) != ref:
                return False
        inputs, labels, manifest = q.freeze()
        if (
            inputs != load(work["corpus_reference"])["rows"]
            or labels != load(work["oracle_reference"])["rows"]
            or manifest != load(work["manifest_reference"])
        ):
            return False
        if not q.audit(inputs, load(work["decisions_reference"])["rows"]):
            return False
        return bool(value == build(work, validation, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False
