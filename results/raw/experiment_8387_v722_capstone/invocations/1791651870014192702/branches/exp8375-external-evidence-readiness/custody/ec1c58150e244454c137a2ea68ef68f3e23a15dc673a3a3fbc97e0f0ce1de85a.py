"""REQ-REPORT-8375: count released structure without awarding semantic benefit.

The manifest carries bytes from author releases. Private controls use the same
parser, but their invented labels never enter the published feasibility sample.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8375_v722_external_evidence_readiness"
TASK = "exp8375-external-evidence-readiness"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_external_evidence_8375.py"
LANES = ("grounded_qa", "multi_hop_qa", "executable_code")
MODEL_SPECS: list[Json] = []
SEED = 7228375


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so the operator can see unfinished work."""
    print(f"[exp8375] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Exact bytes bind the observation to its local operand."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read_reference(ref: Json) -> bytes:
    """Reject changed bytes before their fields can affect readiness."""
    path = Path(ref["path"])
    data = path.read_bytes()
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("input_hash")
    if (
        ref.get("git_blob")
        and hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
        ).hexdigest()
        != ref["git_blob"]
    ):
        raise ValueError("release_blob")
    return data


def inspect(unit: Json) -> Json:
    """Presence measures availability; it cannot decide whether an answer is true."""
    required = ("context", "output", "label", "authority_reference", "cluster", "claims")
    available = {field: bool(unit.get(field)) for field in required}
    missing = [field for field, present in available.items() if not present]
    independent = unit.get("label_authority") in {"independent_human", "executable_tests"}
    complete = all(available.values())
    eligible = bool(
        complete
        and independent
        and unit.get("overlap") == "disjoint"
        and unit.get("license_usable") is True
        and not unit.get("oracle_only")
    )
    pair = None
    if unit.get("single_evidence") is not None and unit.get("complete_evidence") is not None:
        pair = dict(
            single_available=bool(unit["single_evidence"]),
            complete_available=bool(unit["complete_evidence"]),
            semantic_truth=None,
        )
    return dict(
        id=unit["id"],
        lane=unit["lane"],
        cluster=unit.get("cluster"),
        availability=available,
        label_authority=unit.get("label_authority", "absent"),
        weak_label=unit.get("label_authority") == "model_judge",
        structurally_complete=complete,
        eligible=eligible,
        paired_comparison=pair,
        missing_reason=",".join(missing) or (None if eligible else "authority_or_independence"),
        status="completed",
        arm="released_structure",
        absolute_metric=int(eligible),
        raw_numerator=sum(available.values()),
        raw_denominator=len(required),
        censored=False,
        excluded=not eligible,
        label=unit.get("label"),
        oracle_only=bool(unit.get("oracle_only")),
        overlap=unit.get("overlap", "unknown"),
    )


def reduce_manifest(manifest: Json) -> Json:
    """Keep 96 intended slots while the whole metadata inventory owns support."""
    units = manifest["units"]
    if any(unit.get("lane") not in LANES for unit in units):
        raise ValueError("lane")
    rows: list[Json] = []
    coverage: Json = {}
    independent: dict[str, set[str]] = {}
    for lane in LANES:
        ordered = sorted(
            (u for u in units if u["lane"] == lane),
            key=lambda u: canonical_hash([SEED, lane, u["id"]]),
        )
        chosen = ordered[:32]
        for i in range(32):
            if i < len(chosen):
                rows.append(inspect(chosen[i]))
            else:
                rows.append(
                    dict(
                        id=f"{lane}:unavailable:{i}",
                        lane=lane,
                        cluster=None,
                        status="censored",
                        arm="released_structure",
                        absolute_metric=0,
                        raw_numerator=0,
                        raw_denominator=6,
                        censored=True,
                        excluded=False,
                        eligible=False,
                        availability={
                            field: False
                            for field in (
                                "context",
                                "output",
                                "label",
                                "authority_reference",
                                "cluster",
                                "claims",
                            )
                        },
                        label_authority="absent",
                        paired_comparison=None,
                        missing_reason="released_example_absent",
                    )
                )
        panel = rows[-32:]
        coverage[lane] = dict(
            intended=32,
            released=len(chosen),
            unavailable=32 - len(chosen),
            structural_complete=sum(r.get("structurally_complete", False) for r in panel),
            independently_usable=sum(r["eligible"] for r in panel),
            paired_released=sum(r["paired_comparison"] is not None for r in panel),
        )
        progress("lane_" + lane, len(chosen), 32 - len(chosen))
        for u in ordered:
            if inspect(u)["eligible"]:
                independent.setdefault(u["cluster"], set()).add(u["label"])
    clusters = {c: next(iter(labels)) for c, labels in independent.items() if len(labels) == 1}
    classes = dict(sorted(Counter(clusters.values()).items()))
    ready = len(clusters) >= 80 and len(classes) >= 2 and min(classes.values(), default=0) >= 8
    return dict(
        rows=rows,
        intended_count=96,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=len(clusters),
        coverage_by_lane=coverage,
        label_authority_counts=dict(sorted(Counter(r["label_authority"] for r in rows).items())),
        class_support=classes,
        external_corpus_ready_score=int(ready),
    )


def hydrate(manifest: Json) -> Json:
    """Read only selected lines; unrelated evaluation targets stay unopened."""
    for ref in manifest.get("operands", []):
        read_reference(ref)
    if "units" in manifest:
        return manifest
    units = []
    source = read_reference(manifest["split_reference"]).splitlines()
    traces = {
        ref["offset"]: read_reference(ref).splitlines() for ref in manifest["trace_references"]
    }
    for ordinal in manifest["selection_ordinals"]:
        rec = json.loads(source[ordinal])
        trace = json.loads(traces[ordinal // 50 * 50][ordinal % 50])
        if rec["qid"] != trace["qid"]:
            raise ValueError("source_identity")
        units.append(
            dict(
                id=rec["qid"],
                lane="multi_hop_qa",
                cluster=canonical_hash(sorted(p["title"] for p in rec["paragraphs"])),
                context=[p["text"] for p in rec["paragraphs"]],
                output=trace["pred"],
                label="exact_match" if trace["em"] else "non_match",
                label_authority="exact_match_proxy",
                authority_reference="HotpotQA gold answer",
                overlap="unknown",
                license_usable=True,
                claims=[],
                oracle_only=False,
                single_evidence=None,
                complete_evidence=[p["text"] for p in rec["paragraphs"] if p["is_gold"]],
            )
        )
    return dict(manifest, units=units)


def gate(path: Path, field: str, expected: Any, observed: Any, op: str = "==") -> Json:
    """A blocked operand must name what was expected and what was actually seen."""
    return dict(
        check=field,
        upstream=TASK,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        operator=op,
        expected_value=expected,
        observed_value=observed,
        passed=False,
    )


def build(work: Json, receipts: list[Json]) -> Json:
    """Execution success cannot turn incomplete labels into an independent corpus."""
    manifest = json.loads(read_reference(work["manifest_reference"]))
    reduction = reduce_manifest(hydrate(manifest))
    owned = [r for r in receipts if r.get("scope") != "global"]
    passed = bool(owned) and all(r["passed"] for r in owned)
    blocked = list(work["gate_check_summary"])
    path = Path(work["manifest_reference"]["path"])
    if not reduction["external_corpus_ready_score"]:
        blocked.extend(
            [
                gate(path, "independent_count", 80, reduction["independent_count"], ">="),
                gate(path, "class_support_each", 8, reduction["class_support"], ">="),
            ]
        )
    for lane, coverage in reduction["coverage_by_lane"].items():
        if coverage["released"] < 32:
            blocked.append(gate(path, lane + ".released_examples", "released", None))
    if reduction["excluded_count"]:
        blocked.append(gate(path, "independent_label_context_claims_overlap", True, False))
    kind = "disqualified" if not passed else "blocked" if blocked else "null"
    value = dict(
        reduction,
        experiment_id=8375,
        task_id=TASK,
        milestone="2026.10.722",
        run_date="20261010",
        honest_verdict="complete_" + kind + "_external_evidence_readiness",
        verdict_class=kind,
        gate_check_summary=blocked,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=work["model_invocation_counts"],
        historical_model_provenance=[
            dict(
                producer="Verification Without Sufficiency",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                fields=["pred", "gold", "em", "f1"],
                current_calls=0,
                references=manifest.get("trace_references", []),
            )
        ],
        sample_size_budget=dict(
            lanes=3,
            per_lane=32,
            intended_slots=96,
            support_floor=80,
            per_class_floor=8,
            powered_benchmark=False,
        ),
        verifier_is_oracle=False,
        exposure_scope="released feasibility inspection; no detector fit",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=dict(
            external_ready=bool(
                reduction["external_corpus_ready_score"] and passed and not blocked
            ),
            owned_validation=passed,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=work["terminal_validation_sidecar_path"],
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        reproducibility_checksum=canonical_hash(
            [
                work["manifest_reference"],
                work["source_artifact_hashes"],
                work["code_config_hashes"],
                SEED,
            ]
        ),
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[work["manifest_reference"], work["primitive_reference"]],
        cited_upstream_artifacts=[
            dict(r, imported_fields="release metadata or historical context")
            for r in work["source_artifact_hashes"]
        ],
        release_manifest=manifest.get("release_manifest", []),
        license_manifest=manifest.get("license_manifest", []),
        source_overlap_rows=manifest.get("source_overlap_rows", []),
        future_acquisition_budget=manifest.get("future_acquisition_budget", {}),
        adoption_decisions=manifest.get("adoption_decisions", []),
        lane_inventory=manifest.get("lane_inventory", []),
        future_sampling_protocol=manifest.get("sampling_protocol", {}),
        manifest_reference=work["manifest_reference"],
        primitive_reference=work["primitive_reference"],
        work_reference=reference(Path(work["work_path"])),
        invocation_id=work["invocation_id"],
        authority=work["authority"],
        retrieval_receipt=manifest.get("retrieval_receipt", {}),
        repository_health=[r for r in receipts if r.get("scope") == "global"],
        methodology_note="Structural availability from pinned author bytes only. Exact-match proxy is not human response truth. Gold supporting evidence is oracle-only. Unknown source overlap blocks. At most32 examples per lane; no detector quality or semantic benefit measured. Closed V717/V721 utility outcomes remain unchanged. No current model load or paid annotation; future costs are assumptions.",
    )
    value["external_corpus_ready_score"] = int(value["acceptance_gates"]["external_ready"])
    principles = {
        "identity": (
            ["experiment_id", "task_id", "milestone", "run_date", "invocation_id", "authority"],
            "Bind exact task, invocation and full authority objects; missing independent authority stays blocked.",
        ),
        "verdict": (
            [
                "honest_verdict",
                "verdict_class",
                "gate_check_summary",
                "acceptance_gates",
                "external_corpus_ready_score",
            ],
            "Separate owned validation from released independently usable evidence; external absence blocks.",
        ),
        "substrate": (
            [
                "inference_substrate",
                "inference_substrate_class",
                "MODEL_SPECS",
                "model_invocation_counts",
                "historical_model_provenance",
            ],
            "Zero current LLM calls; imported Qwen2.5 observations are historical provenance.",
        ),
        "sample": (
            [
                "rows",
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
                "sample_size_budget",
                "class_support",
                "coverage_by_lane",
                "label_authority_counts",
            ],
            "Keep96 slots and explicit absence; field presence is structural.80 disjoint clusters and8 per class are required.",
        ),
        "science": (
            [
                "verifier_is_oracle",
                "exposure_scope",
                "independent_generalization_score",
                "generalized_learning_benefit_score",
                "methodology_note",
            ],
            "Feasibility is not detector quality; gold evidence is oracle-only and both generalization scores stay zero.",
        ),
        "validation": (
            [
                "required_checks_passed",
                "flagged_adversarial",
                "validation_receipts",
                "terminal_validation_sidecar_path",
                "adversarial_findings",
                "repository_health",
            ],
            "Retain commands, exits, log hashes and findings; global health cannot qualify or disqualify owned code.",
        ),
        "custody": (
            [
                "preconditions_checked",
                "duration_s",
                "phase_spans",
                "random_seed",
                "reproducibility_checksum",
                "source_artifact_hashes",
                "code_config_hashes",
                "raw_shard_hashes",
                "cited_upstream_artifacts",
                "manifest_reference",
                "primitive_reference",
                "work_reference",
                "retrieval_receipt",
            ],
            "Bind reproducible primitive bytes, author git blobs, input snapshots, bounded resources and measured acquisition.",
        ),
        "release": (
            [
                "release_manifest",
                "license_manifest",
                "source_overlap_rows",
                "lane_inventory",
                "future_acquisition_budget",
                "adoption_decisions",
                "future_sampling_protocol",
            ],
            "Release availability, permission, overlap and target authority are distinct. Future acquisition is costed planning only.",
        ),
    }
    value["field_principles"] = {
        field: principle for fields, principle in principles.values() for field in fields
    }
    value["field_principles"]["field_principles"] = (
        "Explain each result field without changing its plain typed value."
    )
    return value


def replay(path: Path) -> bool:
    """Recompute meaning from sealed release bytes, even after aggregate hashes change."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(read_reference(value["work_reference"]))
        for ref in work["source_artifact_hashes"] + work["code_config_hashes"]:
            read_reference(ref)
        primitive = json.loads(read_reference(work["primitive_reference"]))
        reduction = reduce_manifest(hydrate(json.loads(read_reference(work["manifest_reference"]))))
        if primitive != reduction:
            raise ValueError("primitive_reduction_drift")
        expected = build(work, value["validation_receipts"])
        if value != expected:
            raise ValueError("reduction_drift")
        from carnot.reporting.primary_publication import validate_primary

        validate_primary(value, ROOT / "results" / (NAME + ".json"))
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        progress("replay_rejected_" + type(error).__name__)
        return False
