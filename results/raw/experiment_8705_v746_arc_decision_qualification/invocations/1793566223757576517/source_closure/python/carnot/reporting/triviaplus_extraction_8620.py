"""REQ-REPORT-8620: complete lexical reach is distinct from factual correctness."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import triviaplus_release_8617 as release
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify.evidence_features_7980 import extract
from carnot.verify import source_alignment as alignment

Json = dict[str, Any]
ROOT = release.ROOT
NAME, TASK = "experiment_8620_v740_triviaplus_extraction", "exp8620-triviaplus-extraction"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_triviaplus_extraction_8620.py"
OWNED = [
    "python/carnot/reporting/triviaplus_extraction_8620.py",
    "python/carnot/reporting/triviaplus_extraction_runner_8620.py",
    CLI,
]
RELEASE = "results/experiment_8617_v740_triviaplus_release.json"
RELEASE_HASH = "sha256:7c45c28d09837ede0b976100cbdd356f7bc778da717fc3876c1c9bd80ca138cd"
MODEL_SPECS: list[Json] = []
SCORES = [
    "numeric_mismatch maximum",
    "negation_mismatch maximum",
    "uncovered_fraction mean",
    "1-overlap mean",
]
SEED = 7398607
reference, read = release.reference, release.read


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so a supervisor can distinguish work from a stall."""
    print(f"[exp8620] phase={phase} completed={completed} pending={pending}", flush=True)


def predict(row: Json) -> Json:
    """Preserve full bytes and the shipped budgets; lexical conflict is not truth."""
    if (
        set(row) != set(release.ALLOWLIST)
        or not isinstance(row["opaque_id"], str)
        or not row["opaque_id"]
    ):
        raise ValueError("predictor_fields")
    public = dict(
        family_id=row["opaque_id"],
        source_bytes=row["source_bytes"],
        answer_bytes=row["answer_bytes"],
    )
    features = extract(public)
    source, answer = (bytes.fromhex(row[k]) for k in ["source_bytes", "answer_bytes"])
    view = alignment.prepare(source, answer)
    values = features["values"]
    scores = (
        dict(zip(SCORES, [values[5], values[3], values[6], 1 - values[0]], strict=True))
        if values is not None
        else None
    )
    return dict(
        opaque_id=row["opaque_id"],
        source_length_bytes=len(source),
        answer_length_bytes=len(answer),
        source_length_characters=len(source.decode()),
        answer_length_characters=len(answer.decode()),
        source_sentence_count=len(view["source_sentences"]),
        source_window_count=len(view["windows"]),
        unique_window_count=features["window_count"],
        answer_unit_count=len(view["answer_units"]),
        values=values,
        abstention=features["abstention"],
        fixed_scores=scores,
        source_hash=release.text_hash(source.decode()),
        answer_hash=release.text_hash(answer.decode()),
    )


def extract_file(public: Path, output: Path) -> None:
    """The child opens only the content descriptor, never a human-label operand."""
    rows = [json.loads(line) for line in public.read_text().splitlines()]
    if len({r["opaque_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate_opaque_id")
    result = []
    for i, row in enumerate(rows):
        result.append(predict(row))
        if i % 8 == 0:
            progress("predict", i + 1, len(rows) - i - 1)
    atomic_json(output, dict(rows=result, predictor_reference=reference(public)))
    progress("scores_sealed", len(rows), 0)


def join_labels(features: list[Json], original: list[Json], questions: Json) -> list[Json]:
    """Keep raw votes distinct while recomputing the release's strict tie priority."""
    rows = []
    for feature in features:
        uid = feature["opaque_id"]
        raw = original[int(uid)]
        a = release.aggregate(raw)
        if not a["matches_release"]:
            raise ValueError("human_aggregation")
        rows.append(
            dict(
                feature,
                question_hash=questions[uid],
                binary_target=a["binary_target"],
                strict_aggregation=a,
                raw_votes=raw["sentence-level-word-labels"],
                annotation_disagreement=any(s["disagreement_count"] for s in a["sentences"]),
                unsupported_count=sum(s["label"] == "not-mentioned" for s in a["sentences"]),
                contradicted_count=sum(s["label"] == "contradicts" for s in a["sentences"]),
            )
        )
    return rows


def intervals(rows: list[Json]) -> list[Json]:
    """Resample questions with all usable siblings; never optimize a score on labels."""
    usable = [r for r in rows if r["fixed_scores"] is not None and r["binary_target"] in (0, 1)]
    clusters = sorted({r["question_hash"] for r in usable})
    support = [len({r["question_hash"] for r in usable if r["binary_target"] == c}) for c in (0, 1)]
    enough = len(clusters) >= 32 and min(support) >= 8
    # Each draw assigns one multiplicity to a whole question, retaining its siblings.
    weights = (
        np.random.default_rng(SEED).multinomial(
            len(clusters), np.full(len(clusters), 1 / len(clusters)), size=10000
        )
        if enough
        else None
    )
    reports = []
    for score in SCORES:
        values = sorted({r["fixed_scores"][score] for r in usable})
        positive = np.zeros((len(clusters), len(values)))
        negative = positive.copy()
        for row in usable:
            array = positive if row["binary_target"] else negative
            array[
                clusters.index(row["question_hash"]), values.index(row["fixed_scores"][score])
            ] += 1
        p, n = positive.sum(axis=0), negative.sum(axis=0)
        denominator = p.sum() * n.sum()
        auc = float(np.sum(p * (np.cumsum(n) - n / 2)) / denominator) if denominator else None
        interval, valid = None, 0
        if weights is not None:
            ps, ns = weights @ positive, weights @ negative
            ds = ps.sum(axis=1) * ns.sum(axis=1)
            good = ds > 0
            draws = (ps * (np.cumsum(ns, axis=1) - ns / 2)).sum(axis=1)[good] / ds[good]
            valid = len(draws)
            interval = np.quantile(draws, [0.025, 0.975]).tolist()
        reports.append(
            dict(
                score=score,
                auroc=auc,
                interval95=interval,
                status="descriptive_exploratory" if enough else "insufficient_support",
                usable_rows=len(usable),
                usable_clusters=len(clusters),
                class_clusters=support,
                bootstrap_draws=10000 if enough else 0,
                valid_draws=valid,
                random_seed=SEED,
                selected_winner=False,
            )
        )
    return reports


def describe(rows: list[Json]) -> Json:
    """Account for every row, including unsupported long contexts and disputed votes."""
    usable = [r for r in rows if r["fixed_scores"] is not None and r["binary_target"] in (0, 1)]
    strata = []
    for windows in ["1..32", "33..128", ">128"]:
        for units in ["<=16", ">16"]:
            selected = [
                r
                for r in rows
                if (
                    "1..32"
                    if r["source_window_count"] <= 32
                    else "33..128"
                    if r["source_window_count"] <= 128
                    else ">128"
                )
                == windows
                and ("<=16" if r["answer_unit_count"] <= 16 else ">16") == units
            ]
            valid = [r for r in selected if r in usable]
            strata.append(
                dict(
                    source_windows=windows,
                    answer_units=units,
                    intended_count=len(selected),
                    usable_count=len(valid),
                    abstained_count=sum(r["abstention"] is not None for r in selected),
                    usable_coverage=len(valid) / len(selected) if selected else None,
                    empty_source_count=sum(r["source_window_count"] == 0 for r in selected),
                )
            )
    annotations = []
    for disputed in (False, True):
        for label, field in [
            ("unsupported", "unsupported_count"),
            ("contradicted", "contradicted_count"),
            ("all", None),
        ]:
            selected = [
                r
                for r in rows
                if r["annotation_disagreement"] == disputed and (field is None or r[field] > 0)
            ]
            annotations.append(
                dict(
                    stratum="disputed" if disputed else "unanimous",
                    label=label,
                    row_count=len(selected),
                    usable_count=sum(r in usable for r in selected),
                    intervals=intervals(selected),
                )
            )
    return dict(
        extraction_rows=rows,
        fixed_score_rows=[
            dict(
                opaque_id=r["opaque_id"],
                question_hash=r["question_hash"],
                scores=r["fixed_scores"],
                missing_reason=r["abstention"],
            )
            for r in rows
        ],
        source_length_strata=strata,
        abstention_reason_counts=dict(Counter(r["abstention"] for r in rows if r["abstention"])),
        usable_question_support=dict(
            intended_clusters=len({r["question_hash"] for r in rows}),
            usable_clusters=len({r["question_hash"] for r in usable}),
            usable_rows=len(usable),
            class_clusters=[
                len({r["question_hash"] for r in usable if r["binary_target"] == c}) for c in (0, 1)
            ],
            selective_coverage=len(usable) / len(rows) if rows else None,
        ),
        descriptive_auroc_intervals=intervals(rows),
        annotation_disagreement_rows=annotations,
    )


def operands(root: Path) -> tuple[Json, Json, Json]:
    """Independent primary and author-byte pins defeat repaired derived manifests."""
    method = read(dict(path=str(root / release.METHOD), sha256=release.METHOD_HASH))
    upstream = read(dict(path=str(root / RELEASE), sha256=RELEASE_HASH))
    read(dict(path=str(root / release.PROTOCOL), sha256=release.PROTOCOL_HASH))
    if method["methods_ready_score"] != 1 or not method["required_checks_passed"]:
        raise ValueError("methods_qualification")
    if (
        upstream["triviaplus_coverage_inputs_ready_score"] != 1
        or not upstream["required_checks_passed"]
    ):
        raise ValueError("release_qualification")
    primitive = read(upstream["primitive_reference"])
    for name, ref in upstream["release_file_hashes"].items():
        if release.authenticate(Path(ref["path"]), name) != ref:
            raise ValueError("release_reference")
    descriptor = primitive["private_predictor_reference"]
    if sha256_file(Path(descriptor["path"])) != descriptor["sha256"]:
        raise ValueError("private_descriptor_hash")
    return method, upstream, primitive


def content(root: Path) -> tuple[Json, Json, list[Json], Json]:
    """Read content-only parquet columns and freeze joins before opening labels."""
    import pyarrow.parquet as pq

    method, upstream, primitive = operands(root)
    cache = Path(primitive["cache"])
    data = pq.read_table(
        cache / release.PARQUET, columns=["question", "split", "article", "answer"]
    ).to_pylist()
    roster = release.seal_roster(data)
    if (
        roster != upstream["label_blind_roster"]
        or len(roster["question_hashes"]) != method["triviaplus_coverage_plan"]["question_clusters"]
    ):
        raise ValueError("roster_authority")
    public = [release.predictor_row(data[int(uid)], uid) for uid in roster["all_sibling_ids"]]
    original = [
        json.loads(line)
        for line in Path(primitive["private_predictor_reference"]["path"]).read_text().splitlines()
    ]
    if public != original:
        raise ValueError("descriptor_custody")
    questions = {uid: g["question_hash"] for g in roster["groups"] for uid in g["sibling_ids"]}
    return method, upstream, public, questions


def replay(path: Path) -> bool:
    """Fresh replay derives scores and labels again from pinned original release bytes."""
    try:
        from carnot.reporting.triviaplus_extraction_runner_8620 import build

        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in work["code_refs"] + work["source_refs"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if not work["gate_check_summary"]:
            _, upstream, public, questions = content(Path(work["root"]))
            features = [predict(r) for r in public]
            sealed = read(work["features_reference"])
            if sealed["rows"] != features or canonical_hash(features) != work["feature_seal"]:
                return False
            if (
                reference(Path(sealed["predictor_reference"]["path"]))
                != sealed["predictor_reference"]
            ):
                return False
            import pyarrow.parquet as pq

            original = pq.read_table(
                Path(upstream["private_cache_reference"]["path"]) / release.PARQUET
            ).to_pylist()
            if describe(join_labels(features, original, questions)) != work["report"]:
                return False
        return (
            build(work, read(value["validation_reference"]), Path(value["publication_output"]))
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
