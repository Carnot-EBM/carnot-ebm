"""REQ-REPORT-8403: recover question custody without claiming learned benefit."""

from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from typing import Any

from carnot.reporting import human_label_custody_8389 as old

__all__ = ["old"]
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.human_label_custody_8389 import (
    reference as reference,
    read_reference as read_reference,
)

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8403_v724_released_question_identity"
TASK = "exp8403-released-question-identity"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_released_question_identity_8403.py"
OWNED = [
    "python/carnot/reporting/released_question_identity_8403.py",
    "python/carnot/reporting/released_question_identity_runner_8403.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
ARMS = ("p1", "p1s", "p2t", "p2")
PINS = {
    "8389": (
        "experiment_8389_v723_human_label_custody",
        "sha256:3cc8b7659d145f5ace20e850b4dd0abda2074b8e9c88d61091431722ec618e0e",
    ),
    "8305": (
        "experiment_8305_v717_cached_sentence_custody",
        "sha256:84564c8702db3a3665c84e9db9f62d6a2bffe00729cc19a19bb4c3a6c2ea13ba",
    ),
    "8375": (
        "experiment_8375_v722_external_evidence_readiness",
        "sha256:376a80778efb4be78693f0b854f0d9da25621ad265de563a42f99809b6cd69d3",
    ),
}
RAG_PIN = "sha256:0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so a supervisor can distinguish work from a stall."""
    print(f"[exp8403] phase={phase} completed={completed} pending={pending}", flush=True)


def question_hash(text: str) -> str:
    """Formatting aliases share a cluster; complete evidence bytes have a separate hash."""
    return str(old.text_hash(old.normalized(text)))


def input_manifest(root: Path) -> Json:
    """Independent primary pins prevent a later manifest from authorizing changed inputs."""
    return dict(
        target_mapping=old.TARGET,
        primaries={
            key: dict(path=str(root / "results" / (name + ".json")), sha256=pin)
            for key, (name, pin) in PINS.items()
        },
    )


def recover_sources(slots: list[Json], sources: list[Json]) -> list[Json]:
    """Check each source serialization before opening its question; keep non-QA units."""
    index = {str(r["source_id"]): r for r in sources}
    if len(index) != len(sources):
        raise ValueError("duplicate_source")
    rows = []
    for ordinal, slot in enumerate(slots):
        if ordinal % 50 == 0:
            progress("prior_source_recovery", ordinal, len(slots) - ordinal)
        original_id = slot.get("source_id")
        source = index.get(str(original_id))
        data = bytes.fromhex(slot["source_bytes"])
        info = source["source_info"] if source else None
        expected = (
            info.encode()
            if isinstance(info, str)
            else json.dumps(
                info, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        if source and data != expected:
            raise ValueError("source_bytes_identity")
        question = info.get("question") if isinstance(info, dict) else None
        rows.append(
            dict(
                source_slot=ordinal,
                dataset=old.normalized(source["source"]) if source else None,
                original_id=original_id,
                role=slot["role"],
                source_bytes_sha256=old.text_hash(data.decode()),
                source_bytes=slot["source_bytes"],
                question_hash=question_hash(question) if question else None,
                identity_status="recovered" if question else "missing",
                missing_reason=None
                if question
                else "source_identity_absent"
                if not source
                else "source_has_no_question",
            )
        )
    progress("prior_source_recovery_complete", len(rows), 0)
    return rows


def join_panel(
    master: list[Json], audit: list[Json], prior: list[Json], obligations: list[str]
) -> Json:
    """Audit one released answer per generator; exact text cannot prove training disjointness."""
    peers = {(r["dataset"], old.normalized(r["qid"]), r["model"]): r for r in audit}
    unique = len(peers) == len(audit) == len(master) == 900
    seen: set[tuple[str, str, str]] = set()
    items: set[str] = set()
    identities: dict[tuple[str, str], set[str]] = defaultdict(set)
    clusters: dict[str, set[tuple[str, str]]] = defaultdict(set)
    support: dict[str, set[str]] = {"0": set(), "1": set()}
    prior_ids = {
        (r["dataset"], old.normalized(str(r["original_id"])))
        for r in prior
        if r.get("dataset") and r.get("original_id")
    }
    prior_hashes = {r["question_hash"] for r in prior if r.get("question_hash")}
    known = not obligations
    rows = []
    for ordinal, row in enumerate(master):
        if ordinal % 100 == 0:
            progress("factorial_join", ordinal, 900 - ordinal)
        key = (row["dataset"], old.normalized(row["qid"]), row["model"])
        cluster = question_hash(row["question"])
        peer = peers.get(key)
        joined = bool(
            peer
            and all(
                peer.get(k) == row[k]
                for k in ("item_id", "question", "answer_text", "annotator1_label")
            )
        )
        outputs = {}
        complete = joined
        for arm in ARMS:
            try:
                label = int(peer[f"label_{arm}"]) if peer else -1
                confidence = float(peer[f"conf_{arm}"]) if peer else -1
                parsed = json.loads(peer[f"raw_{arm}"]) if peer else {}
                valid = (
                    label in (0, 1)
                    and 0 <= confidence <= 1
                    and parsed == dict(label=label, confidence=confidence)
                    and peer is not None
                    and not peer[f"error_{arm}"]
                )
            except (ValueError, KeyError, TypeError):
                label, confidence, valid = -1, None, False
            outputs[arm] = dict(
                label=label,
                confidence=confidence,
                valid=bool(valid),
                raw_sha256=old.text_hash(peer.get(f"raw_{arm}", "")) if peer else None,
            )
            complete = complete and bool(valid)
        target, sensitivity = int(row["annotator1_label"]), int(row["annotator2_label"])
        if target not in (-1, 0, 1) or sensitivity not in (-1, 0, 1):
            raise ValueError("human_label_encoding")
        qualified = (
            complete
            and target >= 0
            and bool(row["qid"].strip())
            and bool(row["question"].strip())
            and key not in seen
            and row["item_id"] not in items
            and row["model"] in old.GENERATORS
        )
        unique = unique and qualified
        seen.add(key)
        items.add(row["item_id"])
        identities[key[:2]].add(cluster)
        clusters[cluster].add(key[:2])
        overlap = (
            "overlap"
            if key[:2] in prior_ids or cluster in prior_hashes
            else "disjoint"
            if known
            else "unknown"
        )
        if qualified and overlap == "disjoint":
            support[str(target)].add(cluster)
        rows.append(
            dict(
                unit_id=row["item_id"],
                dataset=key[0],
                original_id=row["qid"],
                generator=key[2],
                question_hash=cluster,
                response_hash=old.text_hash(row["answer_text"]),
                join_hash=canonical_hash(
                    [*key, row["item_id"], cluster, old.text_hash(row["answer_text"])]
                ),
                human_primary=target if target >= 0 else None,
                human_sensitivity=sensitivity if sensitivity >= 0 else None,
                factorial_outputs=outputs,
                audit_aligned=joined,
                qualified=bool(qualified),
                overlap=overlap,
                arm="release_custody",
                family=cluster,
                seed=None,
                status="completed",
                absolute_metric=int(qualified),
                raw_numerator=int(qualified),
                raw_denominator=1,
                censored=False,
                excluded=not qualified,
                missing_reason=None if qualified else "release_join_or_factorial_incomplete",
            )
        )
    for ordinal in range(len(rows), 900):
        rows.append(
            dict(
                unit_id=f"missing:{ordinal}",
                arm="release_custody",
                family=None,
                seed=None,
                status="censored",
                absolute_metric=0,
                raw_numerator=0,
                raw_denominator=1,
                censored=True,
                excluded=False,
                missing_reason="released_response_absent",
            )
        )
    identity_rows = [
        dict(
            dataset=d,
            original_id=q,
            original_id_scope="author_sampled_question_pool",
            question_hash=next(iter(h)) if len(h) == 1 else None,
            question_hashes=sorted(h),
            overlap=next(
                r["overlap"]
                for r in rows
                if r.get("dataset") == d and old.normalized(r["original_id"]) == q
            ),
        )
        for (d, q), h in sorted(identities.items())
    ]
    disjoint = {
        r["question_hash"]
        for r in identity_rows
        if r["overlap"] == "disjoint" and r["question_hash"]
    }
    release_ready = (
        unique
        and len(identities) == len(clusters) == 300
        and all(len(h) == 1 for h in identities.values())
    )
    return dict(
        rows=rows,
        intended_count=900,
        completed_count=len(master),
        failed_count=0,
        censored_count=max(0, 900 - len(master)),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=len(disjoint),
        question_cluster_count=len(clusters),
        response_count=len(master),
        question_identity_rows=identity_rows,
        factorial_join_rows=[r for r in rows if not r["censored"]],
        release_criterion_ready_score=int(release_ready),
        disjoint_eval_ready_score=int(
            release_ready
            and known
            and len(disjoint) >= 80
            and all(len(v) >= 8 for v in support.values())
        ),
        disjoint_cluster_count=len(disjoint),
        class_support={k: len(v) for k, v in support.items()},
        exact_overlap_count=sum(r["overlap"] == "overlap" for r in identity_rows),
        overlap_unknown_count=sum(r["overlap"] == "unknown" for r in identity_rows),
        cross_dataset_duplicates=[
            dict(question_hash=h, dataset_ids=sorted(ids))
            for h, ids in sorted(clusters.items())
            if len({d for d, _ in ids}) > 1
        ],
        human_targets_previously_opened=True,
    )


def reduce_manifest(manifest: Json) -> Json:
    """Traverse declared reference types only; missing upstream bytes remain external gates."""
    if manifest["target_mapping"] != old.TARGET:
        raise ValueError("target_mapping")
    primaries = {}
    gates = []
    for key, (_, pin) in PINS.items():
        ref = manifest["primaries"][key]
        if ref["sha256"] != pin:
            raise ValueError("primary_pin")
        try:
            primaries[key] = json.loads(old.read_reference(ref))
        except OSError:
            gates.append(
                old.gate(
                    ref["path"], "pinned_primary_available", pin, None, pin, upstream="exp" + key
                )
            )
    master, audit, release_files = [], [], {}
    licensed = False
    try:
        if "8389" in primaries:
            release = json.loads(old.read_reference(primaries["8389"]["manifest_reference"]))
            if release["target_mapping"] != old.TARGET or release["release_commit"] != old.COMMIT:
                raise ValueError("target_mapping_or_commit")
            release_files = release["release_files"]
            operands = {
                name: old.authenticate(name, release_files[name]) for name in old.RELEASE_FILES
            }
            master = list(csv.DictReader(io.StringIO(operands[old.MASTER].decode("utf-8-sig"))))
            audit = list(csv.DictReader(io.StringIO(operands[old.AUDIT].decode("utf-8-sig"))))
            licensed = release["license_permitted"] is True
    except OSError as error:
        gates.append(
            old.gate(str(error.filename), "release_bytes_available", True, None, upstream="exp8389")
        )
    recovered: list[Json] = []
    obligations: list[str] = []
    source_ref: Json = {}
    try:
        if "8305" in primaries and "8375" in primaries:
            prior = primaries["8375"]["source_overlap_rows"]
            candidates = [
                r
                for lane in prior
                if lane["lane"] == "grounded_qa"
                for r in lane["local_corpus_references"]
                if Path(r["path"]).name == "source_info.jsonl"
            ]
            if len(candidates) != 1 or candidates[0]["sha256"] != RAG_PIN:
                raise ValueError("independent_source_pin")
            source_ref = candidates[0]
            sources = [json.loads(line) for line in old.read_reference(source_ref).splitlines()]
            measurement = json.loads(old.read_reference(primaries["8305"]["measurement_reference"]))
            recovered = recover_sources(measurement["bundle"]["slots"], sources)
            obligations.extend(
                "exp8375:" + lane["lane"] + ":exposure_unknown"
                for lane in prior
                if lane["overlap"] != "disjoint"
            )
            if any(r["question_hash"] is None for r in recovered):
                obligations.append("exp8305:non_question_or_missing_identity")
        else:
            obligations.append("prior_primary_absent")
    except OSError as error:
        obligations.append("prior_typed_reference_absent")
        gates.append(
            old.gate(
                str(error.filename),
                "prior_source_bytes_available",
                True,
                None,
                upstream="exp8305/exp8375",
            )
        )
    value = join_panel(master, audit, recovered, obligations)
    value["release_criterion_ready_score"] &= int(licensed)
    value["disjoint_eval_ready_score"] &= value["release_criterion_ready_score"]
    acceptance = dict(
        permitted_pinned_release=licensed,
        complete_factorial_panel=bool(value["release_criterion_ready_score"]),
        all_old_overlap_obligations_resolved=not obligations,
        disjoint_question_floor=value["disjoint_cluster_count"] >= 80,
        each_class_question_floor=all(v >= 8 for v in value["class_support"].values()),
    )
    for field, passed in acceptance.items():
        if not passed:
            gates.append(
                old.gate(
                    manifest["primaries"]["8389"]["path"],
                    field,
                    80
                    if field == "disjoint_question_floor"
                    else 8
                    if field == "each_class_question_floor"
                    else True,
                    value["disjoint_cluster_count"]
                    if field == "disjoint_question_floor"
                    else value["class_support"]
                    if field == "each_class_question_floor"
                    else obligations
                    if field == "all_old_overlap_obligations_resolved"
                    else passed,
                    PINS["8389"][1],
                    ">=" if field.endswith("floor") else "==",
                    upstream=TASK,
                )
            )
    value.update(
        gate_check_summary=gates,
        acceptance_gates=acceptance,
        target_mapping=old.TARGET,
        release_commit=old.COMMIT,
        release_blob_hashes=dict(old.RELEASE_FILES),
        release_file_hashes=release_files,
        released_label_panel_ready_score=primaries.get("8389", {}).get(
            "released_label_panel_ready_score", 0
        ),
        prior_source_custody=dict(
            rows=recovered,
            source_reference=source_ref,
            question_identities_recovered=sum(r["question_hash"] is not None for r in recovered),
            missing_identity_count=sum(r["question_hash"] is None for r in recovered),
        ),
        exposure_ledger=dict(
            unresolved_obligations=obligations,
            exact_match_scope="dataset-qualified sampled IDs or normalized question text only",
            semantic_contamination="unknown",
            training_contamination="unknown",
            old_independent_gate=dict(
                primary=manifest["primaries"]["8389"],
                gate_check_summary=primaries.get("8389", {}).get("gate_check_summary", []),
            ),
            targets_opened_by=manifest["primaries"]["8389"],
        ),
    )
    return value


def build(work: Json, receipts: list[Json]) -> Json:
    """Validation qualifies a retrospective panel while independent scientific gates stay closed."""
    value = reduce_manifest(json.loads(old.read_reference(work["manifest_reference"])))
    gates = value["gate_check_summary"] + work["gate_check_summary"]
    passed = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    kind = "disqualified" if not passed else "blocked" if gates else "positive"
    value.update(
        experiment_id=8403,
        task_id=TASK,
        milestone="2026.10.724",
        run_date="20261011",
        honest_verdict=f"complete_{kind}_released_question_identity",
        verdict_class=kind,
        gate_check_summary=gates,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=work["model_invocation_counts"],
        historical_model_provenance=[
            dict(generator=m, kind="cached_third_party_answers", current_calls=0)
            for m in old.GENERATORS
        ]
        + [dict(judge="GPT-5-mini", kind="cached_factorial_outputs", current_calls=0)],
        sample_size_budget=dict(
            intended_responses=900,
            intended_questions=300,
            minimum_disjoint_questions=80,
            each_class_floor=8,
            repeats_are_independent=False,
        ),
        verifier_is_oracle=False,
        exposure_scope="released previously opened human targets; retrospective within-release audit only; no fitted model",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=work["terminal_validation_sidecar_path"],
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7248403,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[work["manifest_reference"], work["primitive_reference"]],
        manifest_reference=work["manifest_reference"],
        primitive_reference=work["primitive_reference"],
        work_reference=old.reference(Path(work["work_path"])),
        invocation_id=work["invocation_id"],
        authority=work["authority"],
        repository_health=work["repository_health"],
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields="pinned release, bundle.slots, source_overlap_rows or invocation authority",
            )
            for r in work["source_artifact_hashes"]
        ],
        reproducibility_checksum=canonical_hash(
            [
                work["manifest_reference"],
                work["source_artifact_hashes"],
                work["code_config_hashes"],
                7248403,
            ]
        ),
        methodology_note="Byte custody and exact overlap are engineering observations. H1=-0.00390625 and H2=0 remain closed exposed-development nulls. No semantic, training-disjointness or learned benefit is measured.",
    )
    value["release_criterion_ready_score"] &= int(passed and not work["gate_check_summary"])
    value["disjoint_eval_ready_score"] &= int(passed and not gates)
    value["acceptance_gates"]["owned_validation"] = passed
    value["field_principles"] = {
        k: "Bind actual invocation, independently pinned operands, preserved units, or qualified retrospective scope; never infer scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        release_criterion_ready_score="Permit only the fully joined released factorial audit after owned checks.",
        disjoint_eval_ready_score="Also require resolved old obligations and independent cluster and class support; opened targets stay exposed.",
        field_principles="Explain each field without replacing typed evidence.",
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild primitives in a new process so rehashed derived reports cannot authorize themselves."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(old.read_reference(value["work_reference"]))
        for ref in work["source_artifact_hashes"] + work["code_config_hashes"]:
            old.read_reference(ref)
        manifest = json.loads(old.read_reference(work["manifest_reference"]))
        if json.loads(old.read_reference(work["primitive_reference"])) != reduce_manifest(manifest):
            raise ValueError("primitive_reduction_drift")
        if value != build(work, value["validation_receipts"]):
            raise ValueError("reduction_drift")
        validate_primary(value, ROOT / "results" / (NAME + ".json"))
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        progress("replay_rejected_" + type(error).__name__)
        return False
