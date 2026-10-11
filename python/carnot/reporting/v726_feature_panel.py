"""REQ-REPORT-8430: frozen public inputs measure reference agreement, not truth.

Original release bytes and independently pinned methods determine every row.
Targets are kept outside feature rows so the later trainer cannot receive test
labels by accidentally loading the panel. Public exposure still limits science.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v726_contract_methods as methods
from carnot.reporting import released_question_identity_8403 as identity
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.human_label_custody_8389 import authenticate, gate, text_hash
from carnot.verify.evidence_features_7980 import FEATURES, extract
from carnot.verify import source_alignment as alignment

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8430_v726_criterion_feature_panel", "exp8430-criterion-feature-panel"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v726_feature_panel_8430.py"
OWNED = [
    "python/carnot/reporting/v726_feature_panel.py",
    "python/carnot/reporting/v726_feature_panel_runner.py",
    CLI,
]
METHODS_RECEIPT = "results/experiment_8429_v726_contract_methods.json"
RECEIPT_PIN = "sha256:dfad99775207ce8073b35c7d5efbc0cbefa70ad1ddadadafa584157add3a3095"
MODEL_SPECS: list[Json] = []
reference, checksum, read = methods.reference, methods.checksum, methods.read
failure = methods.failure


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so a supervisor can see remaining bounded work."""
    print(f"[exp8430] phase={phase} completed={completed} pending={pending}", flush=True)


def operands() -> tuple[list[Json], list[Json], Json, Json]:
    """Reviewed author git blobs prevent a rehashed manifest from inventing a corpus."""
    protocol = methods.protocol()
    inherited = read(protocol["release_primitive_reference"])
    files = inherited["release_file_hashes"]
    rows = [
        list(csv.DictReader(io.StringIO(authenticate(n, files[n]).decode("utf-8-sig"))))
        for n in [identity.old.MASTER, identity.old.AUDIT]
    ]
    return rows[0], rows[1], inherited, protocol


def probability(judgment: Json) -> float | None:
    """Only the released p2 confidence determines q; invalid observations stay absent."""
    label, confidence = judgment.get("label"), judgment.get("confidence")
    if not judgment.get("valid") or label not in (0, 1) or not isinstance(confidence, (int, float)):
        return None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    return float(min(0.9999, max(0.0001, confidence if label == 1 else 1 - confidence)))


def panel(
    master: list[Json], audit: list[Json], inherited: Json, protocol: Json
) -> tuple[Json, list[Json]]:
    """Join IDs before extraction and keep whole sibling groups in their inherited role.

    Source text is the author's ref_str, not a label-derived explanation. Test
    targets are opened here only in evaluator custody for aggregate support;
    the saved trainer view contains fit and tune targets alone.
    """
    roles = protocol["partition"]["roles"]
    if roles != methods.protocol()["partition"]["roles"]:
        raise ValueError("partition_drift")
    role_index = {q: role for role, questions in roles.items() for q in questions}
    original = {r["unit_id"]: r for r in inherited["factorial_join_rows"]}
    if len(master) != 900 or len({r["item_id"] for r in master}) != 900:
        raise ValueError("intended_identity")
    joined = identity.join_panel(master, audit, [], ["old_overlap_unknown"])["rows"]
    peers = {r["item_id"]: r for r in audit}
    rows, public, trainer, evaluator, joins, strata = [], [], [], [], [], []
    groups: dict[str, list[Json]] = defaultdict(list)
    for index, (rec, custody) in enumerate(zip(master, joined, strict=True)):
        if index % 100 == 0:
            progress("feature_extraction", index, 900 - index)
        old = original.get(rec["item_id"])
        if old is None or any(
            custody[k] != old[k] for k in ["join_hash", "response_hash", "question_hash"]
        ):
            raise ValueError("authenticated_identity")
        qid, uid = custody["question_hash"], rec["item_id"]
        role = role_index[qid]
        peer = peers.get(uid, {})
        source_valid = bool(rec["ref_str"].strip()) and rec["ref_str"] == peer.get("ref_str")
        descriptor = dict(
            family_id=uid,
            source_bytes=(rec["ref_str"] if source_valid else "").encode().hex(),
            answer_bytes=rec["answer_text"].encode().hex(),
        )
        features = extract(descriptor)
        q = probability(custody["factorial_outputs"]["p2"])
        target = custody["human_primary"]
        reason = (
            "source_missing_or_mismatched"
            if not source_valid
            else features["abstention"]
            or ("p2_missing" if q is None else "target_missing" if target is None else None)
        )
        x = [q, *features["values"]] if q is not None and features["values"] is not None else None
        row = dict(
            unit_id=uid,
            family=qid,
            question_hash=qid,
            role=role,
            arm="frozen_feature_panel",
            seed=None,
            x=x,
            q=q,
            source_features=features["values"],
            status="completed",
            absolute_metric=int(reason is None),
            raw_numerator=int(reason is None),
            raw_denominator=1,
            censored=False,
            excluded=reason is not None,
            missing_reason=reason,
            factorial_outputs=custody["factorial_outputs"],
            feature_hash=features["feature_hash"],
        )
        rows.append(row)
        public.append(descriptor)
        groups[qid].append(dict(row, y=target))
        target_row = dict(
            unit_id=uid,
            question_hash=qid,
            role=role,
            y=target,
            y_sensitivity=custody["human_sensitivity"],
        )
        if role == "test":
            evaluator.append(target_row)
        else:
            trainer.append(dict(target_row, x=x))
        joins.append(
            dict(
                unit_id=uid,
                question_hash=qid,
                role=role,
                source_sha256=text_hash(rec["ref_str"]),
                answer_sha256=custody["response_hash"],
                join_hash=custody["join_hash"],
                source_usable=source_valid,
            )
        )
        strata.append(
            dict(
                unit_id=uid,
                question_hash=qid,
                role=role,
                dataset=rec["dataset"],
                generator=rec["model"],
            )
        )
    support: Json = {}
    exclusions: list[Json] = []
    for role, questions in roles.items():
        usable = {
            q
            for q in questions
            if len(groups[q]) == 3 and all(not r["excluded"] for r in groups[q])
        }
        classes = {str(y): sum(any(r["y"] == y for r in groups[q]) for q in usable) for y in (0, 1)}
        floor = protocol["partition"]["usable_support_floors"][role]
        support[role] = dict(
            intended_questions=len(questions),
            usable_questions=len(usable),
            class_questions=classes,
            usable_floor=floor,
            passed=len(usable) >= floor and (role == "fit" or min(classes.values()) >= 8),
        )
        exclusions.extend(
            dict(
                question_hash=q,
                role=role,
                excluded=q not in usable,
                reasons=sorted({r["missing_reason"] for r in groups[q] if r["missing_reason"]}),
            )
            for q in questions
        )
    progress("feature_panel_complete", 900, 0)
    return dict(
        rows=rows,
        trainer_rows=trainer,
        evaluator_rows=evaluator,
        source_join_rows=joins,
        stratum_roster=strata,
        class_support=support,
        question_exclusions=exclusions,
        partition_hashes={k: canonical_hash(v) for k, v in roles.items()},
        exposure_ledger=[
            dict(question_hash=q, role=role, overlap="unknown", public_labels_exposed=True)
            for role, questions in roles.items()
            for q in questions
        ],
        support_passed=all(s["passed"] for s in support.values()),
    ), public


def independent_sample(public: list[Json], rows: list[Json]) -> bool:
    """A fixed sample recomputes lexical reductions without calling the extractor.

    This catches feature ordering and probability mistakes with independent
    arithmetic, while using the same shipped public text preparation rules.
    """
    for index in range(0, len(public), 113):
        descriptor, row = public[index], rows[index]
        source, answer = (bytes.fromhex(descriptor[k]) for k in ("source_bytes", "answer_bytes"))
        view = alignment.prepare(source, answer)
        windows = list(dict.fromkeys(w.strip() for w in view["windows"]))
        if view["abstention"] or not source.strip():
            expected = None
        else:
            pairs = [alignment.pair_features(w, view["answer_units"])[-4:] for w in windows]
            expected = [
                v
                for i in range(4)
                for v in (sum(p[i] for p in pairs) / len(pairs), max(p[i] for p in pairs))
            ]
        judgment = row["factorial_outputs"]["p2"]
        q = probability(judgment)
        independent_q = (
            max(
                0.0001,
                min(
                    0.9999,
                    judgment["confidence"]
                    if judgment["label"] == 1
                    else 1.0 - judgment["confidence"],
                ),
            )
            if q is not None
            else None
        )
        expected_x = (
            [independent_q, *expected]
            if expected is not None and independent_q is not None
            else None
        )
        if row["x"] != expected_x:
            return False
    return True


def measure(root: Path, raw: Path) -> Json:
    """Pin actual authority and inputs; optional hardware never enters CPU gates."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 900)
    resources = dict(
        methods.legacy.resources(raw),
        validation_reserve_s=900,
        measurement_cap_s=3900,
        free_disk_bytes=shutil.disk_usage(raw).free,
        tool_paths={
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
    )
    contract = methods.authority(root, raw / "authority")
    failures = list(contract["gate_check_summary"])
    sources = []
    methods_ready = False
    receipt_path = root / METHODS_RECEIPT
    try:
        upstream = read(dict(path=str(receipt_path), sha256=RECEIPT_PIN))
        if (
            upstream["methods_ready_score"] != 1
            or upstream["methods_protocol_sha256"] != methods.PROTOCOL_PIN
        ):
            raise ValueError("methods_not_ready")
        sources.append(reference(receipt_path))
        methods_ready = True
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                str(receipt_path),
                "methods_ready_score",
                1,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream=methods.TASK,
            )
        )
    if (
        not resources["private_disk_backed_scratch"]
        or resources["available_memory_bytes"] < resources["minimum_memory_bytes"]
        or resources["free_disk_bytes"] < 104857600
        or not all(resources["tool_paths"].values())
    ):
        failures.append(
            gate(str(raw), "private_disk_memory_deadline", True, resources, upstream=TASK)
        )
    try:
        master, audit, inherited, protocol = operands()
        result, public = panel(master, audit, inherited, protocol)
        sources.extend(
            [
                dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN),
                protocol["release_primitive_reference"],
                *[
                    inherited["release_file_hashes"][n]
                    for n in [identity.old.MASTER, identity.old.AUDIT]
                ],
            ]
        )
        if not independent_sample(public, result["rows"]):
            raise ValueError("independent_sample")
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            gate(
                str(ROOT / methods.PROTOCOL),
                "authenticated_release_operands",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream=identity.TASK,
            )
        )
        protocol = methods.protocol()
        inherited = read(protocol["release_primitive_reference"])
        role_index = {
            q: role for role, questions in protocol["partition"]["roles"].items() for q in questions
        }
        result = dict(
            rows=[
                dict(
                    unit_id=r["unit_id"],
                    family=r["question_hash"],
                    role=role_index[r["question_hash"]],
                    arm="frozen_feature_panel",
                    seed=None,
                    x=None,
                    status="censored",
                    censored=True,
                    excluded=False,
                    absolute_metric=0,
                    raw_numerator=0,
                    raw_denominator=1,
                    missing_reason="required_release_unavailable",
                )
                for r in inherited["rows"]
            ],
            support_passed=False,
            class_support={},
            trainer_rows=[],
            evaluator_rows=[],
            source_join_rows=[],
            stratum_roster=[],
            exposure_ledger=[],
            partition_hashes={},
            question_exclusions=[],
        )
        public = []
    if not result["support_passed"]:
        failures.append(
            gate(
                str(raw / "feature_panel.json"),
                "registered_question_class_support",
                protocol["partition"]["usable_support_floors"],
                result["class_support"],
                operator=">=",
                upstream=TASK,
            )
        )
    files = {
        "feature_panel": dict(rows=result["rows"], feature_schema=protocol["features"]),
        "public_rows": dict(request_rows=public),
        "trainer_view": dict(rows=result["trainer_rows"]),
        "evaluator_custody": dict(rows=result["evaluator_rows"]),
    }
    for name, value in files.items():
        atomic_json(raw / (name + ".json"), value)
        (raw / (name + ".json")).chmod(0o600)
    work = dict(
        authority=contract,
        methods_ready=methods_ready,
        methods_receipt_reference=dict(path=str(receipt_path), sha256=RECEIPT_PIN),
        panel=result,
        public_rows=public,
        failures=failures,
        preconditions=resources,
        source_refs=sources,
        code_refs=[reference(ROOT / path) for path in OWNED],
        primitive_files=[reference(raw / (n + ".json")) for n in files],
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[
            dict(name="reconstruct_panel", duration_s=(time.monotonic_ns() - start) / 1e9)
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Input qualification and scientific benefit remain separate, even when ready."""
    qualified = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and not work.get("adversarial_findings")
    )
    p = work["panel"]
    ready = (
        qualified
        and not work["failures"]
        and p["support_passed"]
        and work["authority"]["activated"]
        and work["methods_ready"]
    )
    verdict = "disqualified" if not qualified else "circular_positive" if ready else "blocked"
    feature_ref = reference(raw / "feature_panel.json")
    value = dict(
        experiment_id=8430,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261011",
        honest_verdict=f"complete_{verdict}_v726_criterion_feature_panel",
        verdict_class=verdict,
        gate_check_summary=[
            *work["failures"],
            *[
                gate(
                    str(r.get("stdout_path", raw)),
                    "required_owned_check",
                    True,
                    False,
                    upstream=TASK,
                )
                for r in receipts
                if not r["passed"]
            ],
        ],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(methods.legacy.ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[
            dict(
                producer=identity.TASK,
                models=["GPT-5-mini", *identity.old.GENERATORS],
                scope="released cached judgments and generator answers",
            )
        ],
        rows=p["rows"],
        intended_count=900,
        completed_count=sum(r["status"] == "completed" for r in p["rows"]),
        failed_count=0,
        censored_count=sum(r["censored"] for r in p["rows"]),
        excluded_count=sum(r["excluded"] for r in p["rows"]),
        independent_count=300 if ready else 0,
        sample_size_budget=dict(
            intended_answers=900,
            question_clusters=300,
            independent_generalization_questions=0,
            sibling_answers_are_not_independent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="public human labels exposed; old question overlap unknown; retrospective only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            owned_validation=qualified, frozen_inputs=bool(ready), scientific_generalization=False
        ),
        validation_receipts=receipts,
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=725,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[
            reference(raw / "measurement.json"),
            *work["primitive_files"],
            *work.get("coverage_shards", []),
        ],
        cited_upstream_artifacts=[
            dict(ref, imported_fields=["methods_ready_score", "partition", "release_file_hashes"])
            for ref in work["source_refs"]
        ],
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        feature_panel_ready_score=int(ready),
        feature_panel_path=feature_ref["path"],
        feature_panel_sha256=feature_ref["sha256"],
        feature_schema=dict(
            names=["p2_hallucination_probability", *FEATURES],
            predictor_count=9,
            generator_and_dataset_are_predictors=False,
            truth_claim=False,
        ),
        trainer_view=reference(raw / "trainer_view.json"),
        evaluator_custody=reference(raw / "evaluator_custody.json"),
        numeric_learning_substrate="verifier_ensemble_against_cached_candidates",
        methodology_note="Frozen released p2 plus eight public reference features; independent sample and cold reconstruction; no model calls or independent factuality claim.",
    )
    value.update(
        {
            k: p[k]
            for k in [
                "partition_hashes",
                "class_support",
                "source_join_rows",
                "exposure_ledger",
                "stratum_roster",
                "question_exclusions",
            ]
        }
    )
    value["field_principles"] = {
        k: "Bound to original released operands and immutable question roles; retrospective input qualification only."
        for k in value
    }
    value["field_principles"].update(
        feature_panel_ready_score="Qualified reproducible nine-input rows and fixed whole-question support; never independent truth.",
        independent_count="At most 300 statistical question clusters; sibling answers and conditions are dependent. Independent generalization remains zero.",
        trainer_view="Only fit/tune targets; test targets are absent from every trainer row.",
        evaluator_custody="Separately sealed test targets, previously public; custody cannot undo exposure.",
        exposure_ledger="Old overlap stays unknown; public labels cannot establish independent factuality.",
        stratum_roster="Dataset and generator identity frozen for reporting and never included in numeric predictor vectors.",
        acceptance_gates="Required owned checks and immutable input support qualify fitting; scientific generalization remains closed.",
        question_exclusions="All intended questions remain present, with every observed missing operand and no reseeding.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh replay rebuilds derived rows from independent release pins, not local hashes."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        with TemporaryDirectory(prefix="exp8430-cold-authority-", dir="/var/tmp") as directory:
            root = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                snapshot = work["authority"]["authority_snapshots"][role]
                if "snapshot_path" in snapshot:
                    source = Path(snapshot["snapshot_path"])
                    if sha256_file(source) != snapshot["sha256"]:
                        raise ValueError("authority_snapshot_drift")
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            cold = methods.authority(root, root / "reduction")
            if cold["activated"] != work["authority"]["activated"]:
                raise ValueError("authority_reduction_drift")
        try:
            upstream = read(work["methods_receipt_reference"])
            methods_ready = (
                upstream["methods_ready_score"] == 1
                and upstream["methods_protocol_sha256"] == methods.PROTOCOL_PIN
            )
        except (OSError, ValueError, KeyError):
            methods_ready = False
        if methods_ready != work["methods_ready"]:
            raise ValueError("methods_gate_drift")
        for ref in [*work["source_refs"], *work["code_refs"], *work["primitive_files"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("operand_drift")
        if work["public_rows"]:
            expected_panel, public = panel(*operands())
            if expected_panel != work["panel"] or public != work["public_rows"]:
                raise ValueError("primitive_drift")
        elif any(r["x"] is not None for r in work["panel"]["rows"]):
            raise ValueError("missing_operand_value")
        primitive_views = dict(
            feature_panel=dict(
                rows=work["panel"]["rows"], feature_schema=methods.protocol()["features"]
            ),
            public_rows=dict(request_rows=work["public_rows"]),
            trainer_view=dict(rows=work["panel"]["trainer_rows"]),
            evaluator_custody=dict(rows=work["panel"]["evaluator_rows"]),
        )
        for ref in work["primitive_files"]:
            if read(ref) != primitive_views[Path(ref["path"]).stem]:
                raise ValueError("custody_or_feature_file_drift")
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    raise ValueError("receipt_drift")
        raw = Path(value["primitive_reference"]["path"]).parent
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
