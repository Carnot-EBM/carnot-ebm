"""REQ-REPORT-8553: preserve release evidence without treating construction as truth."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8553_v735_hallmark_release_admissibility"
TASK = "exp8553-hallmark-release-admissibility"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
COMMIT = "f774fa40675daa83eca6201637a94c4536b7bb3e"
OWNED = [
    "python/carnot/reporting/hallmark_release_8553.py",
    "python/carnot/reporting/hallmark_release_runner_8553.py",
    CLI,
]


def reference(path: Path) -> Json:
    """Bind exact bytes because a repaired summary cannot replace source evidence."""
    return dict(path=str(path.resolve()), sha256=sha256_file(path))


def read(ref: Json) -> Any:
    """Reject changed operands before interpreting their fields."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("source_hash")
    return json.loads(path.read_bytes())


def gate(field: str, observed: Any, expected: Any = True, path: str = "release") -> Json:
    """Missing values stay distinct from measured zero in every blocked condition."""
    return dict(
        check=field,
        upstream="HALLMARK",
        path=path,
        sha256=None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def origin(row: Json) -> str:
    """Human-coded supplements are distinct from pipeline labels and later author review."""
    method = row.get("generation_method")
    if row.get("source") == "walters_wilder_2023":
        return "explicit_human_adjudication"
    if method == "scraped" or row.get("label") == "VALID":
        return "database_cross_check"
    return {
        "perturbation": "deterministic_perturbation",
        "canary": "deterministic_perturbation",
        "adversarial": "deterministic_perturbation",
        "llm_generated": "unknown",
    }.get(str(method), "unknown")


def identities(row: Json) -> list[str]:
    """Unchanged DOI or title can link siblings; corrupted fields leave unresolved bases."""
    fields = row.get("fields", {})
    title = fields.get("title", "")
    original = re.search(r"original: '([^']+)'", row.get("explanation") or "")
    title = original.group(1) if original else title

    def norm(s: str) -> str:
        return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", str(s)).casefold())

    operands = [
        "title:" + norm(title),
        "doi:" + str(fields.get("doi", "")).casefold().removeprefix("https://doi.org/"),
    ]
    return sorted(canonical_hash(x) for x in operands if x.split(":", 1)[1])


def reduce(data: Json) -> Json:
    """Compute inventory and rejection from primitive rows, retaining every denominator."""
    failures = [
        gate("resource_available", None, True, str(p)) for p in data.get("missing_resources", [])
    ]
    rows, inventories, samples, bases = [], [], [], {}
    strata: dict[str, list[Json]] = defaultdict(list)
    splits = data["splits"]
    if not splits:
        failures.append(gate("public_splits_available", None))
    for split, entries in sorted(splits.items()):
        ids = [r.get("bibtex_key") for r in entries]
        missing = sum(r.get("label") not in ("VALID", "HALLUCINATED") for r in entries)
        expected = data["declared_counts"].get(split)
        if len(ids) != len(set(ids)) or None in ids or missing or expected != len(entries):
            failures.append(
                gate(
                    "complete_split:" + split,
                    dict(count=len(entries), missing_labels=missing, unique=len(set(ids))),
                    expected,
                )
            )
        inventories.append(
            dict(
                split=split,
                count=len(entries),
                labels=dict(Counter(r.get("label") for r in entries)),
                missing_labels=missing,
            )
        )
        bases[split] = {r["bibtex_key"]: set(identities(r)) for r in entries}
        for r in sorted(entries, key=lambda r: str(r.get("bibtex_key"))):
            o = origin(r)
            item = dict(
                split=split,
                stable_id=r.get("bibtex_key"),
                label=r.get("label"),
                origin=o,
                generation_method=r.get("generation_method"),
                source=r.get("source"),
                base_identity_hashes=identities(r),
                record_hash=canonical_hash(r),
                independent_human_truth=False,
                target_compatible=False,
                author_manual_review_documented=r.get("generation_method")
                in ("real_world", "adversarial"),
            )
            rows.append(item)
            strata[o + ":" + str(r.get("generation_method"))].append(
                dict(
                    item,
                    fields=r.get("fields", {}),
                    explanation=r.get("explanation"),
                    subtests=r.get("subtests"),
                )
            )
    for stratum, members in sorted(strata.items()):
        samples.append(
            dict(
                stratum=stratum,
                inventory_count=len(members),
                inspected_count=min(20, len(members)),
                records=sorted(members, key=lambda r: (r["stable_id"], r["split"]))[:20],
            )
        )
    overlaps = []
    for a, b in itertools.combinations(sorted(splits), 2):
        left = set().union(*bases[a].values()) if bases[a] else set()
        right = set().union(*bases[b].values()) if bases[b] else set()
        shared = left & right
        groups: list[set[str]] = []
        for aliases in [*bases[a].values(), *bases[b].values()]:
            if not aliases & shared:
                continue
            merged = set(aliases)
            separate = []
            for group in groups:
                if merged & group:
                    merged |= group
                else:
                    separate.append(group)
            groups = [*separate, merged]
        exact_a = {canonical_hash(r.get("fields", {})) for r in splits[a]}
        exact_b = {canonical_hash(r.get("fields", {})) for r in splits[b]}
        overlaps.append(
            dict(
                left_split=a,
                right_split=b,
                shared_base_identity_count=len(groups),
                shared_identity_alias_count=len(shared),
                shared_work_groups=[
                    dict(
                        base_alias_hashes=sorted(group),
                        members=[
                            dict(
                                split=name,
                                stable_id=r["bibtex_key"],
                                label=r["label"],
                                hallucination_type=r.get("hallucination_type"),
                            )
                            for name in (a, b)
                            for r in splits[name]
                            if bases[name][r["bibtex_key"]] & group
                        ],
                    )
                    for group in groups
                ],
                hallucinated_left_matching_valid_right=sum(
                    r["label"] == "HALLUCINATED"
                    and any(
                        v["label"] == "VALID"
                        and bases[a][r["bibtex_key"]] & bases[b][v["bibtex_key"]]
                        for v in splits[b]
                    )
                    for r in splits[a]
                ),
                hallucinated_right_matching_valid_left=sum(
                    r["label"] == "HALLUCINATED"
                    and any(
                        v["label"] == "VALID"
                        and bases[b][r["bibtex_key"]] & bases[a][v["bibtex_key"]]
                        for v in splits[a]
                    )
                    for r in splits[b]
                ),
                exact_citation_overlap_count=len(exact_a & exact_b),
                shared_hashes=sorted(left & right),
                matching_left_rows=sum(bool(v & right) for v in bases[a].values()),
                matching_right_rows=sum(bool(v & left) for v in bases[b].values()),
            )
        )
    predictions = []
    for p in data.get("predictions", []):
        truth = {r["bibtex_key"]: r["label"] for r in splits.get(p["split"], [])}
        predicted = {r["bibtex_key"]: r["label"] for r in p["rows"]}
        if (
            not p.get("version")
            or not truth
            or set(truth) != set(predicted)
            or len(predicted) != len(p["rows"])
            or any(v not in ("VALID", "HALLUCINATED", "UNCERTAIN") for v in predicted.values())
        ):
            failures.append(
                gate(
                    "prediction_denominator_and_version",
                    dict(name=p["name"], count=len(predicted)),
                    len(truth),
                )
            )
            continue
        counts = []
        for mode in ["conservative", "selective", "aggressive"]:
            cm = dict(TP=0, FP=0, TN=0, FN=0, denominator=0)
            for key, t in truth.items():
                label = predicted[key]
                if mode == "selective" and label == "UNCERTAIN":
                    continue
                positive = label == "HALLUCINATED" or (
                    mode == "aggressive" and label == "UNCERTAIN"
                )
                cell = (
                    ("TP" if positive else "FN")
                    if t == "HALLUCINATED"
                    else ("FP" if positive else "TN")
                )
                cm[cell] += 1
                cm["denominator"] += 1
            counts.append(cm)
        predictions.append(
            dict(
                name=p["name"],
                version=p["version"],
                committed=counts[0],
                conservative=counts[0],
                selective=counts[1],
                aggressive=counts[2],
                coverage_numerator=counts[1]["denominator"],
                coverage_denominator=len(truth),
                convention="conservative: UNCERTAIN to VALID; aggressive: UNCERTAIN to flag; selective: exclude UNCERTAIN. committed is the conservative full-split alias.",
                abstention_aware=counts[1],
                abstained_count=sum(v == "UNCERTAIN" for v in predicted.values()),
                rows=[
                    dict(stable_id=k, truth=truth[k], prediction=v)
                    for k, v in sorted(predicted.items())
                ],
                source_availability="historical_snapshot_only; not re-queried",
            )
        )
    criteria = [
        dict(criterion=k, passed=False, reason=v)
        for k, v in {
            "disjoint_independent_adjudication": "Author review and pipeline checks do not establish disjoint independent human truth.",
            "target_compatibility": "Bibliographic existence and metadata validity do not label Carnot sentence truth.",
            "codesign_independence": "Paper sections 4.2 and G.2 identify taxonomy/tool co-design; reference scores are excluded from independent ranking.",
        }.items()
    ]
    total = len(rows)
    return dict(
        rows=rows,
        intended_count=total,
        completed_count=total,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            public_inventory=total, per_stratum_inspection=20, independent_labels=0
        ),
        release_audit_ready_score=int(not failures),
        independent_label_ready_score=0,
        gate_check_summary=failures,
        split_inventory=inventories,
        criterion_rows=criteria,
        base_work_overlap_rows=overlaps,
        label_provenance_rows=[
            dict(origin=k, count=Counter(r["origin"] for r in rows)[k])
            for k in [
                "database_cross_check",
                "deterministic_perturbation",
                "explicit_human_adjudication",
                "model_judgment",
                "unknown",
            ]
        ],
        provenance_samples=samples,
        prediction_reductions=predictions,
        released_prediction_summaries=data.get("prediction_summaries", {}),
        prediction_limitation=None
        if predictions
        else "No eligible versioned complete per-entry predictions; no verifier run.",
        overlap_limit="DOI/title alias hashes measure observable overlap; fully corrupted or fabricated bases remain unresolved. Hidden splits were not fetched.",
        admission_decision=dict(
            decision="reject_for_carnot_independent_labels",
            future_use="Citation metadata diagnostics only, after base-work grouping, human review and source-availability checks; no nine-input head ingestion.",
            independent_adjudication_required=True,
            target_compatible=False,
        ),
        count_reconciliation=dict(
            paper_core=2526,
            repository_description=2525,
            observed_public_core=sum(
                len(splits.get(k, [])) for k in ("dev_public", "test_public", "stress_test")
            ),
            hidden_declared_only=454,
            explanation="1119 + 831 + 122 + 454 = 2526. The stress split includes one VALID contamination canary; 2525 omits that record. Hidden count is metadata, not acquired rows.",
        ),
    )


def fixture() -> Json:
    """A tiny explicit control checks rejection without borrowing release accuracy."""

    def row(key: str, method: str, title: str) -> Json:
        return dict(
            bibtex_key=key, label="HALLUCINATED", generation_method=method, fields=dict(title=title)
        )

    return dict(
        splits={
            "a": [row("a0", "perturbation", "shared"), row("a1", "llm_generated", "unique")],
            "b": [
                row("b0", "scraped", "shared"),
                row("b1", "real_world", "other"),
                row("b2", "adversarial", "last"),
            ],
        },
        declared_counts={"a": 2, "b": 3},
        missing_resources=[],
        release_manifest=[],
        predictions=[
            dict(
                name="rejection_control",
                split="a",
                version="hand_checked_fixture_v1",
                rows=[
                    dict(bibtex_key="a0", label="HALLUCINATED"),
                    dict(bibtex_key="a1", label="UNCERTAIN"),
                ],
            )
        ],
    )


def build(data: Json, source: Json, receipts: list[Json], output: Path) -> Json:
    """Administrative readiness cannot clear old scientific flags or owned failures."""
    result = reduce(data)
    passed = all(r["passed"] for r in receipts)
    verdict = (
        "disqualified" if not passed else ("blocked" if result["gate_check_summary"] else "null")
    )
    if not passed:
        result["release_audit_ready_score"] = 0
    value = dict(
        result,
        experiment_id=8553,
        task_id=TASK,
        milestone="2026.10.735",
        run_date="20261015",
        honest_verdict="complete_" + verdict + "_hallmark_independent_label_rejection",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(llm_calls=0, model_loads=0, generations=0, verifier_calls=0),
        historical_model_provenance=data.get("historical_model_provenance", []),
        verifier_is_oracle=False,
        exposure_scope="public_release_and_exposed_history",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=result["criterion_rows"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=[],
        preconditions_checked=data.get("preconditions_checked", {}),
        duration_s=data.get("duration_s", 0.001),
        phase_spans=data.get("phase_spans", []),
        random_seed=8042,
        source_reference=source,
        source_artifact_hashes=data.get("source_artifact_hashes", []),
        code_config_hashes=data.get("code_config_hashes", []),
        raw_shard_hashes=[source],
        release_manifest=data.get("release_manifest", []),
        release_file_inventory=data.get("release_file_inventory", []),
        construction_constraints=data.get("construction_constraints", {}),
        preregistration_reference=data.get("preregistration_reference"),
        authority_receipt=data.get("authority_receipt"),
        cited_upstream_artifacts=data.get("cited_upstream_artifacts", []),
        preserved_quarantines=data.get("preserved_quarantines", {}),
        field_principles={
            "inventory": "Rows are released records, never independent human judgments.",
            "readiness": "Custody and a justified rejection can qualify an audit; scientific readiness stays zero.",
            "prediction_reductions": "Historical release denominators only; no Carnot detector accuracy.",
            "count_reconciliation": "Hidden counts are declared metadata, never an empty acquired corpus.",
            "source_reference": "Cold replay re-reads exact primitive bytes and authenticated public release operands.",
            "validation": "Owned failures disqualify; no new artifact clears historical flags.",
        },
    )
    value["field_principles"].update(
        {
            key: value["field_principles"].get(
                key,
                "Bind this field to authenticated release bytes, invocation checks and the stated citation-only scope.",
            )
            for key in value
        }
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Cold reduction rejects changed aggregates even after the envelope is rehashed."""
    value = json.loads(path.read_bytes())
    if value["reproducibility_checksum"] != canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    ):
        raise ValueError("checksum")
    data = read(value["source_reference"])
    if "acquisition_reference" in data:
        from carnot.reporting.hallmark_release_inputs_8553 import load

        authenticated = load(Path(data["acquisition_reference"]["path"]))
        if any(
            data[k] != authenticated[k]
            for k in (
                "splits",
                "predictions",
                "declared_counts",
                "release_manifest",
                "missing_resources",
            )
        ):
            raise ValueError("primitive_drift")
    fresh = reduce(data)
    for key, expected in fresh.items():
        if key == "release_audit_ready_score" and not value["required_checks_passed"]:
            expected = 0
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if value["required_checks_passed"] != all(r["passed"] for r in value["validation_receipts"]):
        raise ValueError("owned_check_drift")
    sidecar = Path(value["terminal_validation_sidecar_path"])
    primary = sidecar.parents[2] / (sidecar.parent.name + ".json")
    expected_value = build(data, value["source_reference"], value["validation_receipts"], primary)
    for key, expected in expected_value.items():
        if (
            key not in ("reproducibility_checksum", "adversarial_findings")
            and value.get(key) != expected
        ):
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, primitive_rows=len(fresh["rows"]), independent_label_ready_score=0)
