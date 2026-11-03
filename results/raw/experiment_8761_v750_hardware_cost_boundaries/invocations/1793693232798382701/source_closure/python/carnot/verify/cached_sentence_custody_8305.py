"""REQ-VERIFY-8305: reconstruct predictions while keeping human truth separate.

Cached development rows have already informed earlier experiments. Byte custody
permits reuse of these observations, but cannot turn them into a fresh holdout.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import fit_sentence_capture_8182 as fit
from carnot.verify import development_methods_8098 as human
from carnot.verify import reserved_sentence_capture_8184 as reserved
from carnot.verify import sentence_evidence_8166 as sentence
from carnot.verify import sentence_labels_7942 as labels
from carnot.verify import sentence_transport_8179 as transport

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8305_v717_cached_sentence_custody"
TASK = "exp8305-cached-sentence-custody"
MODULE = "python/carnot/verify/cached_sentence_custody_8305.py"
RUNNER = "python/carnot/reporting/cached_sentence_execution_8305.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_cached_sentence_custody_8305.py"
OWNED = [MODULE, RUNNER, CLI]
MODEL_SPECS: list[str] = []
FEATURES = [
    "holistic_logit",
    *reserved.historical.fit.lexical.FEATURES,
    "span_logit",
    "quote_validity",
    "quote_source_byte_ratio",
    "local_unsupported_mean",
    "local_unsupported_max",
    "local_contradicted_fraction",
    "local_baseless_fraction",
]
PINS = {
    "experiment_8304_v717_contract_methods": "53c4286349b038c34b994dd339e871bc8baca8477c7fb819aac1ab18ebe7a292",
    "experiment_8182_v707_fit_sentence_capture": "6f0f6f3dd5c1b067ab50040a3ce9bc3358a48dee2e71180cf91b3c335bdbcf19",
    "experiment_8184_v707_reserved_sentence_capture": "e70bc2604c31780fb61a7a13aafb11e97ad422383523a88d8ec21e416e7fbf82",
    "experiment_8153_v705_fit_evidence_capture": "c4d1be0937d6fe6d5e538e009d8ed82242973d8616569c8369cc0ea7172103ec",
    "experiment_8155_v705_reserved_evidence_capture": "6a4311de73e3fca6e8fbc756f4e80824eabba4ab962936e08767359b93713673",
    "experiment_8185_v707_sentence_decision_audit": "85e67f7b39e7d89e621326a0bc8572468c33b43e7caee781628c69ff67cf1083",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counters so the conductor can distinguish progress from a stall."""
    print(f"[exp8305] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind evidence to exact bytes so later readers cannot silently change it."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def check_predictor(row: Json) -> None:
    """An allowlist prevents nested targets from becoming prediction inputs."""
    if set(row) != {
        "unit_id",
        "source_cluster_id",
        "role",
        "slot",
        "source_sha256",
        "answer_sha256",
        "x",
        "status",
        "exclusion_reason",
    }:
        raise ValueError("predictor_fields")
    x = row["x"]
    if x is not None and (
        len(x) != 16
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in x)
        or any(not 0 <= v <= 1 for v in x[12:16])
    ):
        raise ValueError("finite_features")


def authenticate(root: Path, raw: Path) -> Json:
    """Copy byte-bound upstream operands before reduction; absent operands block."""
    plan: Json = dict(checks=[], refs=[], bundle={}, historical=[], origins={}, anchors={})

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        plan["checks"].append(
            dict(
                upstream=path.stem,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )
        if expected != observed:
            raise ValueError(field)

    def bind(ref: Json) -> Json:
        p = Path(ref["path"])
        require(p, "input_sha256", ref["sha256"], sha256_file(p) if p.is_file() else None)
        dest = raw / "inputs" / (ref["sha256"][7:] + "-" + p.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        dest.chmod(0o600)
        plan["refs"].append(reference(dest))
        plan["origins"][str(p)] = reference(dest)
        return dict(json.loads(dest.read_text()))

    try:
        raw.mkdir(parents=True, exist_ok=True)
        raw.chmod(0o700)
        require(
            raw,
            "private_scratch_writable_and_protected",
            True,
            os.access(raw, os.W_OK) and raw.stat().st_mode & 0o077 == 0,
        )
        for name in ("python", "pytest", "coverage", "ruff", "mypy"):
            require(
                ROOT / ".venv/bin" / name,
                "tool_executable",
                True,
                os.access(ROOT / ".venv/bin" / name, os.X_OK),
            )
        values = {}
        for name, pin in PINS.items():
            p = root / "results" / (name + ".json")
            v = bind(dict(path=str(p), sha256="sha256:" + pin))
            plan["anchors"][name] = plan["refs"][-1]
            for field, expected in [
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                require(p, field, expected, v.get(field))
            terminal = bind(reference(Path(v["terminal_validation_sidecar_path"])))
            side = Path(terminal["publication"]["sidecar_path"])
            attestation = read_bound_sidecar(p, side)
            require(side, "terminal_passed", True, attestation["report"]["passed"])
            require(
                p,
                "publication_primary_sha256",
                "sha256:" + pin,
                terminal["publication"]["primary_sha256"],
            )
            bind(reference(side))
            values[int(name.split("_")[1])] = v
            progress("authenticated_primary", len(values), len(PINS) - len(values))
        authority = values[8304]
        require(
            root / "results" / (next(iter(PINS)) + ".json"),
            "protocol_ready_score",
            1,
            authority.get("protocol_ready_score"),
        )
        protocol = bind(dict(path=authority["protocol_path"], sha256=authority["protocol_sha256"]))
        fw, rw = [bind(values[n]["measurement_reference"]) for n in (8182, 8184)]
        fc, rc = [bind(values[n]["raw_shard_hashes"][0])["rows"] for n in (8182, 8184)]
        require(
            Path(values[8182]["measurement_reference"]["path"]),
            "primitive_calls",
            canonical_hash(fc),
            canonical_hash(fw["calls"]),
        )
        require(
            Path(values[8184]["measurement_reference"]["path"]),
            "primitive_calls",
            canonical_hash(rc),
            canonical_hash(rw["calls"]),
        )
        base_calls = [bind(values[n]["raw_shard_hashes"][1])["rows"] for n in (8153, 8155)]
        targets = bind(values[8153]["raw_shard_hashes"][2])
        evaluation = bind(values[8185]["label_provenance"]["original_manifest"])
        originals = {str(r["id"]): r for r in evaluation["original_response_records"]}
        for target, slot in zip(evaluation["rows"], rw["slots"], strict=True):
            original = originals[slot["response_id"]]
            expected, _ = human.target(original, bytes.fromhex(slot["answer_bytes"]))
            require(
                Path(values[8185]["label_provenance"]["original_manifest"]["path"]),
                "original_human_target",
                True,
                [slot["unit_id"], slot["source_cluster_id"], expected["y"]]
                == [target["unit_id"], target["source_cluster_id"], target["y"]],
            )
            targets[target["unit_id"]] = target["y"]
        identity = values[8182]["model_receipt"]["capture_identity"]
        require(
            Path(values[8184]["measurement_reference"]["path"]),
            "historical_model_identity",
            identity,
            values[8184]["model_receipt"]["capture_identity"],
        )
        plan["bundle"] = dict(
            slots=fw["slots"] + rw["slots"],
            calls=fc + rc,
            base_calls=base_calls,
            labels=targets,
            protocol=protocol,
            identity=identity,
            reply_hashes=[canonical_hash(c) for c in fc + rc],
        )
        for n in (8182, 8184, 8153, 8155):
            v = values[n]
            receipt = v["model_receipt"]
            gpu = receipt["resident_gpu_receipt"]
            require(
                Path(gpu["log_path"]),
                "historical_gpu_log_sha256",
                gpu["log_sha256"],
                sha256_file(Path(gpu["log_path"])),
            )
            plan["refs"].append(reference(Path(gpu["log_path"])))
            plan["historical"].append(
                dict(
                    experiment_id=n,
                    primary=reference(
                        root
                        / "results"
                        / (next(k for k in PINS if k.startswith(f"experiment_{n}_")) + ".json")
                    ),
                    MODEL_SPECS=v["MODEL_SPECS"],
                    imported_counts=v["model_invocation_counts"],
                    model_receipt=receipt,
                    scope="historical_only_zero_current_calls",
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in plan["checks"]):
            plan["checks"].append(
                dict(
                    upstream="cached_operands",
                    path=str(root),
                    hash=None,
                    artifact_field="structure",
                    op="==",
                    expected="authenticated_operands",
                    observed=str(error),
                    passed=False,
                )
            )
        plan["bundle"] = {}
    return plan


@lru_cache(maxsize=2)
def public_baseline(serialized: str) -> Json:
    """Cache pure public computation by all primitive bytes within one process.

    A fresh replay process always recomputes the lexical signals. Labels never
    enter this cache key, and changed replies cannot reuse an earlier result.
    """
    return {r["unit_id"]: r for group in json.loads(serialized) for r in reserved.baseline(group)}


def reconstruct(bundle: Json) -> Json:
    """Recover all original slots without borrowing labels to fill missing rows."""
    if not bundle:
        return dict(
            rows=[],
            predictors={},
            evaluators={},
            class_support_by_role={},
            historical_capture_counts={},
            missing_source_rows=[],
            fit_support_ready_score=0,
        )
    slots, calls = bundle["slots"], bundle["calls"]
    expected_roles = [
        (r, i + 1) for r, n in [("fit", 128), ("tune", 64), ("evaluation", 128)] for i in range(n)
    ]
    if [(s["role"], s["slot"]) for s in slots] != expected_roles or len(
        {s["source_cluster_id"] for s in slots}
    ) != 320:
        raise ValueError("source_role_collision_or_slots")
    if [canonical_hash(c) for c in calls] != bundle["reply_hashes"]:
        raise ValueError("reply_bytes")
    progress("before_cached_public_feature_reconstruction")
    baseline = public_baseline(json.dumps(bundle["base_calls"], sort_keys=True))
    progress("after_cached_public_feature_reconstruction")
    predictors: Json = {r: [] for r in ("fit", "calibration", "comparator_selection", "reserved")}
    evaluators: Json = {r: [] for r in predictors}
    rows, transport_counts, source_hashes = [], Counter(), set()
    for number, slot in enumerate(slots):
        role, index, unit = slot["role"], slot["slot"], slot["unit_id"]
        if any(slot.get(k) is not None for k in ("y", "human_target", "entailment_label")):
            raise ValueError("target_injection")
        original = bundle["protocol"]["original_roles"][role][index - 1]
        source_hash = labels.digest(bytes.fromhex(slot["source_bytes"]))
        if (
            original["unit_id"] != unit
            or original["source_cluster_id"] != slot["source_cluster_id"]
            or original["original_source_sha256"] != source_hash
            or source_hash in source_hashes
        ):
            raise ValueError("public_source_or_roster")
        source_hashes.add(source_hash)
        counts = {r["payload"]: r["input_tokens"] for r in slot["requests"]}
        prepared = transport.requests(slot, lambda text: counts.get(text, 0))
        if prepared["requests"] != slot["requests"]:
            raise ValueError("public_answer_requests")
        group = [
            c
            for c in calls
            if c["unit_id"] == unit and c.get("condition", "original") == "original"
        ]
        for call, request in zip(group, slot["requests"]):
            if (
                call["identity"] != bundle["identity"]
                or call["arm"] != "grammar"
                or call["request"] != request
                or call["source_cluster_id"] != slot["source_cluster_id"]
                or call["cache_key"] != fit.cache_key(slot, request, bundle["identity"])
            ):
                raise ValueError("cached_call_identity")
        predicted = [p for c in group for p in fit.accepted(slot, c)]
        complete = (
            bool(slot["requests"])
            and len(group) == len(slot["requests"])
            and [p["sentence_index"] for p in predicted] == list(range(len(slot["sentences"])))
        )
        transport_counts[role] += int(complete)
        old = baseline[unit]
        if any(
            old[k] != slot[k] for k in ("source_bytes", "answer_bytes", "source_cluster_id", "role")
        ):
            raise ValueError("baseline_source_identity")
        local = sentence.local_features(
            [
                dict(
                    p,
                    relation={"E": "entailed", "C": "contradicted", "B": "baseless"}[p["relation"]],
                )
                for p in predicted
            ],
            len(slot["sentences"]),
        )
        x = [*old["x"], *local] if complete and old["x"] is not None and local else None
        status = (
            "completed"
            if x is not None
            else "excluded"
            if not slot["requests"]
            else (
                "censored" if group and all(c["status"] == "censored" for c in group) else "failed"
            )
        )
        new_role = (
            "reserved"
            if role == "evaluation"
            else "fit"
            if role == "fit"
            else ("calibration" if index <= 32 else "comparator_selection")
        )
        row = dict(
            unit_id=unit,
            source_cluster_id=slot["source_cluster_id"],
            role=new_role,
            slot=index,
            source_sha256=source_hash,
            answer_sha256=labels.digest(bytes.fromhex(slot["answer_bytes"])),
            x=x,
            status=status,
            exclusion_reason=None
            if x is not None
            else (
                slot["exclusion_reason"] or old["exclusion_reason"] or "unavailable_paired_features"
            ),
        )
        check_predictor(row)
        predictors[new_role].append(row)
        y = bundle["labels"].get(unit)
        if y is not None and (type(y) is not int or y not in (0, 1)):
            raise ValueError("binary_human_target")
        evaluators[new_role].append(
            dict(
                unit_id=unit,
                source_cluster_id=row["source_cluster_id"],
                role=new_role,
                slot=index,
                y=y,
                target_authority="original_response_level_human",
            )
        )
        rows.append(
            {k: v for k, v in row.items() if k != "x"}
            | dict(
                condition="original",
                arm="cached_features",
                metric="finite_source_features",
                numerator=int(x is not None),
                denominator=1,
            )
        )
        if (number + 1) % 32 == 0:
            progress("source_reconstruction", number + 1, 320 - number - 1)
    support = {}
    for role in [*predictors, "stream", "later", "retention"]:
        origin = role if role in predictors else "reserved"
        pairs = [
            (p, t)
            for p, t in zip(predictors[origin], evaluators[origin], strict=True)
            if role in predictors
            or (
                p["slot"] <= 96
                if role == "stream"
                else 9 <= p["slot"] <= 96
                if role == "later"
                else p["slot"] >= 97
            )
        ]
        classes = Counter(t["y"] for p, t in pairs if p["x"] is not None and t["y"] in (0, 1))
        support[role] = dict(
            intended=len(pairs),
            feature_rows=sum(p["x"] is not None for p, _ in pairs),
            usable=sum(classes.values()),
            supported=classes[0],
            unsupported=classes[1],
            missing_labels=sum(p["x"] is not None and t["y"] is None for p, t in pairs),
        )
    enough = all(
        support[r]["usable"] >= n and min(support[r]["supported"], support[r]["unsupported"]) >= c
        for r, n, c in [("fit", 96, 12), ("calibration", 24, 4), ("comparator_selection", 24, 4)]
    )
    return dict(
        rows=rows,
        predictors=predictors,
        evaluators=evaluators,
        class_support_by_role=support,
        historical_capture_counts=dict(
            fit_tune_transport_completed=transport_counts["fit"] + transport_counts["tune"],
            fit_tune_transport_intended=192,
            reserved_transport_completed=transport_counts["evaluation"],
            fit_tune_feature_rows=sum(support[r]["feature_rows"] for r in list(predictors)[:3]),
            fit_tune_labeled_rows=sum(support[r]["usable"] for r in list(predictors)[:3]),
            reserved_feature_rows=support["reserved"]["feature_rows"],
        ),
        missing_source_rows=[r for r in rows if r["status"] != "completed"],
        fit_support_ready_score=int(enough),
    )


def measure(root: Path, raw: Path) -> Json:
    """Seal public predictors and protected evaluator operands before validation."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    progress("before_input_authentication")
    work = authenticate(root, raw)
    progress("after_input_authentication")
    result = reconstruct(work["bundle"])
    shards: Json = dict(predictor_shards={}, evaluator_shards={})
    for kind, field in [("predictors", "predictor_shards"), ("evaluators", "evaluator_shards")]:
        for role, rows in result[kind].items():
            path = raw / kind / (role + ".json")
            atomic_json(path, dict(rows=rows))
            path.chmod(0o400 if kind == "predictors" else 0o600)
            shards[field][role] = reference(path)
        progress(kind + "_sealed", len(result[kind]), 0)
    work.update(
        shards,
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_reconstruct_seal",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "python/carnot/verify/sentence_transport_8179.py",
                "python/carnot/verify/sentence_labels_7942.py",
                "python/carnot/verify/sentence_evidence_8166.py",
                reserved.fit.MODULE,
                reserved.MODULE,
                reserved.historical.fit.NUMERIC,
                "python/carnot/verify/evidence_features_7980.py",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    (raw / "measurement.json").chmod(0o600)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Custody readiness requires owned checks; scientific benefit stays unclaimed."""
    r = reconstruct(work["bundle"])
    failed = [c for c in work["checks"] if not c["passed"]]
    owned = bool(receipts) and all(c["passed"] for c in receipts)
    fixture = fixture or work.get("fixture", False)
    ready = int(owned and not failed and len(r["rows"]) == 320 and not fixture)
    support = int(ready and r["fit_support_ready_score"])
    klass = (
        "disqualified"
        if not owned
        else "blocked"
        if failed
        else "circular_positive"
        if fixture
        else "blocked"
        if not support
        else "null"
    )
    reason = (
        "owned_validation"
        if not owned
        else failed[0]["artifact_field"]
        if failed
        else ("fit_support" if not support else "cached_sentence_custody")
    )
    counts = Counter(row["status"] for row in r["rows"])
    schema = dict(
        names=FEATURES,
        indices={n: i for i, n in enumerate(FEATURES)},
        dimension=16,
        holistic_input="x[0]",
        local_indices=[12, 13, 14, 15],
        local_bounds=[0, 1],
        holistic_slope="static_fit_trainable_then_frozen_online",
        identity_features=False,
        gold_join_prohibited=True,
        target="binary_response_unsupported_y1",
    )
    value = dict(
        experiment_id=8305,
        task_id=TASK,
        milestone="2026.10.717",
        run_date="20261008",
        honest_verdict="complete_" + klass + "_" + reason,
        verdict_class=klass,
        cached_cohort_ready_score=ready,
        fit_support_ready_score=support,
        required_checks_passed=owned,
        flagged_adversarial=False,
        verifier_is_oracle=fixture,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        MODEL_SPECS=MODEL_SPECS,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        model_invocation_counts=dict(
            model_loads_attempted=0,
            model_loads_completed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
            generate=0,
        ),
        historical_model_provenance=work["historical"],
        gate_check_summary=work["checks"],
        preconditions_checked=work["checks"],
        acceptance_gates=dict(
            custody="all320 original slots authenticated",
            fit="usable>=96 and each label>=12",
            tune_halves="each usable>=24 and each label>=4",
            owned="100 percent new statements and required private CLI checks",
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178305,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[
            reference(raw / "measurement.json"),
            *[work["predictor_shards"][k] for k in sorted(work["predictor_shards"])],
            *[work["evaluator_shards"][k] for k in sorted(work["evaluator_shards"])],
        ],
        measurement_reference=reference(raw / "measurement.json"),
        predictor_shards=work["predictor_shards"],
        evaluator_shards=work["evaluator_shards"],
        feature_schema=schema,
        source_role_manifest=[
            dict(
                unit_id=row["unit_id"],
                source_cluster_id=row["source_cluster_id"],
                role=row["role"],
                slot=row["slot"],
                source_sha256=row["source_sha256"],
                answer_sha256=row["answer_sha256"],
            )
            for row in r["rows"]
        ],
        rows=r["rows"],
        intended_count=320,
        completed_count=counts["completed"],
        failed_count=counts["failed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        independent_count=len(r["rows"]),
        sample_size_budget=dict(
            fit=128,
            tune=64,
            reserved=128,
            stream=96,
            retention=32,
            calibration=32,
            comparator_selection=32,
            unit="original_source_cluster",
        ),
        class_support_by_role=r["class_support_by_role"],
        historical_capture_counts=r["historical_capture_counts"],
        missing_source_rows=r["missing_source_rows"],
        exposure_ledger=dict(
            scope="previously_exposed_development",
            original_capture_experiments=[8153, 8155, 8182, 8184],
            fresh_holdout=False,
            predictor_hashes_frozen_before_downstream=True,
            labels="separate protected evaluator shards",
        ),
        cited_upstream_artifacts=[
            dict(
                name=n,
                sha256="sha256:" + p,
                fields_imported=[
                    "primitive_calls",
                    "source_roster",
                    "original_human_targets",
                    "model_receipt",
                    "terminal_attestation",
                ],
            )
            for n, p in PINS.items()
        ],
        methodology_note="Reparse bound cached grammar replies and twelve public evidence signals; append four local prediction aggregates. Binary human labels stay separate. No current inference or independent generalization.",
    )
    value["owned_coverage_reference"] = work.get("owned_coverage_reference")
    value["invocation_argv"] = work.get("invocation_argv", [])
    support_gates = [
        dict(
            upstream=TASK,
            path=str(raw / "measurement.json"),
            hash=sha256_file(raw / "measurement.json"),
            artifact_field=f"class_support_by_role.{role}.{field}",
            op=">=",
            expected=minimum,
            observed=r["class_support_by_role"][role][field],
            passed=r["class_support_by_role"][role][field] >= minimum,
        )
        for role, size, per_class in [
            ("fit", 96, 12),
            ("calibration", 24, 4),
            ("comparator_selection", 24, 4),
        ]
        for field, minimum in [
            ("usable", size),
            ("supported", per_class),
            ("unsupported", per_class),
        ]
        if role in r["class_support_by_role"]
    ]
    value["gate_check_summary"] = [*work["checks"], *support_gates]
    value["field_principles"] = {
        k: "Bind measured invocation, byte custody, original slots and exposed-development scope."
        for k in value
    }
    for group, purpose in [
        (
            ["predictor_shards", "evaluator_shards", "source_role_manifest", "feature_schema"],
            "Preserve public feature custody, exact indices and protected binary targets; forbid target and identity features.",
        ),
        (
            [
                "cached_cohort_ready_score",
                "fit_support_ready_score",
                "class_support_by_role",
                "historical_capture_counts",
                "missing_source_rows",
            ],
            "Separate exact cached reconstruction from usable label support and preserve missing original slots.",
        ),
        (
            [
                "historical_model_provenance",
                "MODEL_SPECS",
                "model_invocation_counts",
                "inference_substrate",
                "inference_substrate_class",
            ],
            "Keep imported Qwen bytes and original GPU work separate from zero current model execution.",
        ),
        (
            [
                "validation_receipts",
                "terminal_validation_sidecar_path",
                "required_checks_passed",
                "owned_coverage_reference",
            ],
            "Require byte-bound normal validation, complete new statement coverage and unchanged terminal auditors.",
        ),
        (
            [
                "invocation_argv",
                "gate_check_summary",
                "exposure_ledger",
                "independent_generalization_score",
                "generalized_learning_benefit_score",
            ],
            "Identify actual invocation and exact gate operands; exposed development earns no generalization credit.",
        ),
    ]:
        value["field_principles"].update(dict.fromkeys(group, purpose))
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def bound_bundle(work: Json) -> Json:
    """Recover operands through pinned primaries so rehashing cannot invent data."""
    if not work["bundle"]:
        return {}
    values = {}
    for name, pin in PINS.items():
        ref = work["anchors"][name]
        if ref["sha256"] != "sha256:" + pin:
            raise ValueError("primary_anchor")
        values[int(name.split("_")[1])] = json.loads(Path(ref["path"]).read_text())

    def read(ref: Json) -> Json:
        snapshot = work["origins"][ref["path"]]
        if (
            snapshot["sha256"] != ref["sha256"]
            or sha256_file(Path(snapshot["path"])) != ref["sha256"]
        ):
            raise ValueError("primitive_anchor")
        return dict(json.loads(Path(snapshot["path"]).read_text()))

    fw, rw = [read(values[n]["measurement_reference"]) for n in (8182, 8184)]
    calls = [c for n in (8182, 8184) for c in read(values[n]["raw_shard_hashes"][0])["rows"]]
    targets = read(values[8153]["raw_shard_hashes"][2])
    evaluation = read(values[8185]["label_provenance"]["original_manifest"])
    targets.update({r["unit_id"]: r["y"] for r in evaluation["rows"]})
    return dict(
        slots=fw["slots"] + rw["slots"],
        calls=calls,
        base_calls=[read(values[n]["raw_shard_hashes"][1])["rows"] for n in (8153, 8155)],
        labels=targets,
        protocol=read(
            dict(path=values[8304]["protocol_path"], sha256=values[8304]["protocol_sha256"])
        ),
        identity=values[8182]["model_receipt"]["capture_identity"],
        reply_hashes=[canonical_hash(c) for c in calls],
    )


def replay(path: Path) -> bool:
    """A fresh process rebuilds primitives and shards instead of trusting summaries."""
    try:
        v = json.loads(path.read_text())
        for ref in [*v["raw_shard_hashes"], *v["source_artifact_hashes"], *v["code_config_hashes"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in v["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        w = json.loads(Path(v["measurement_reference"]["path"]).read_text())
        if bound_bundle(w) != w["bundle"]:
            return False
        rebuilt = reconstruct(w["bundle"])
        for kind, field in [("predictors", "predictor_shards"), ("evaluators", "evaluator_shards")]:
            for role, ref in w[field].items():
                if json.loads(Path(ref["path"]).read_text()) != dict(rows=rebuilt[kind][role]):
                    return False
        return bool(
            v == build(w, Path(v["measurement_reference"]["path"]).parent, v["validation_receipts"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Keep the public entry small so execution and evidence have separate duties."""
    from carnot.reporting.cached_sentence_execution_8305 import main as execute

    return execute(argv)
