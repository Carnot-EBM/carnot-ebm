"""REQ-REPORT-8617: private human-label custody is not detector performance."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import unicodedata
import urllib.request

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8617_v740_triviaplus_release"
TASK = "exp8617-triviaplus-release"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_triviaplus_release_8617.py"
OWNED = [
    "python/carnot/reporting/triviaplus_release_8617.py",
    "python/carnot/reporting/triviaplus_release_runner_8617.py",
    CLI,
]
REVISION = "9416ae0c45ec4c4815c4b28820baa4016741deb7"
REPO = "amazon-science/hallucination-benchmark-trivialplus"
PARQUET = "triviaplus_dataset.parquet"
BLOBS = {
    "README.md": "cb069d2f6c8989e2d39e35be5e7460d79ecf6fdc",
    "DATA_DETAILS.md": "5e0107a7277ce87206f6f01efc3dc28af942cb8d",
    "LICENSE": "9badf7abf51e412ff7bbe1ce7b66c5b05145138c",
    "verify_label_consistency.py": "c318e24333134c632922817f823a47f2626cc211",
    PARQUET: "bb2e584b8101175421e984d35492c640556430ba",
}
PRIORITY = ["supplementary", "supports", "not-mentioned", "contradicts"]
UNFAITHFUL = {"not-mentioned", "contradicts"}
ALLOWLIST = ["opaque_id", "source_bytes", "answer_bytes"]
MODEL_SPECS: list[Json] = []
METHOD = "results/experiment_8615_v740_contract_methods.json"
PROTOCOL = "openspec/change-proposals/v740-methods-protocol.json"
PROTOCOL_HASH = "sha256:e498f2c621b7c85bdc1262181a734ab94db73082ad860df0ed36e76f8bf415de"
METHOD_HASH = "sha256:6f7ea729a61997543357bc926f558510d0dbfdb95cabcd42aecbce936267209e"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries and actual counts so supervisors can detect unfinished work."""
    print(f"[exp8617] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Keep the original operand path beside its exact byte identity."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read(ref: Json) -> Json:
    """A repaired derived checksum cannot authorize changed referenced bytes."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("reference_hash")
    return dict(json.loads(path.read_bytes()))


def question_hash(text: str) -> str:
    """Formatting aliases share one question rather than inflating independence."""
    return text_hash(" ".join(unicodedata.normalize("NFKC", text).casefold().split()))


def text_hash(text: str) -> str:
    """Hashes permit public joins without redistributing the private text."""
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def authenticate(path: Path, name: str) -> Json:
    """Reviewed Git blobs remain independent of local manifests and repaired hashes."""
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != BLOBS[name]:
        raise ValueError("git_blob:" + name)
    return dict(reference(path), git_blob=blob, bytes=len(data))


def acquire(cache: Path) -> tuple[Json, list[Json]]:
    """Retrieve only pinned bytes within the request, elapsed and byte budgets."""
    cache.mkdir(parents=True, mode=0o700, exist_ok=True)
    refs: Json = {}
    gates = []
    started = time.monotonic()
    total = 0
    for index, name in enumerate(BLOBS):
        progress("before_acquire_" + name, index, len(BLOBS) - index)
        path = cache / name
        try:
            if not path.is_file():
                remaining = 300 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("retrieval_deadline")
                url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=min(30, remaining)) as stream:
                    data = stream.read(100 * 1024 * 1024 - total + 1)
                total += len(data)
                if total > 100 * 1024 * 1024:
                    raise ValueError("retrieval_byte_cap")
                path.write_bytes(data)
                path.chmod(0o600)
            refs[name] = authenticate(path, name)
        except (OSError, ValueError) as error:
            gates.append(gate(path, "authenticated_release_file", BLOBS[name], str(error)))
        progress("after_acquire_" + name, index + 1, len(BLOBS) - index - 1)
    return refs, gates


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Missing evidence has an explicit operand rather than a fabricated zero."""
    return dict(
        check=field,
        upstream=TASK,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def aggregate(row: Json) -> Json:
    """Recompute human votes without allowing noisy training targets into authority."""
    annotations = row["sentence-level-word-labels"]
    sentences = []
    invalid = 0
    for i in range(len(row["answer_sentence_list"])):
        votes = [a[i] if i < len(a) else None for a in annotations]
        counts = Counter(v for v in votes if v in PRIORITY)
        invalid += sum(v not in (None, "n/a", "") and v not in PRIORITY for v in votes)
        label = max(counts, key=lambda v: (counts[v], PRIORITY.index(v))) if counts else None
        sentences.append(
            dict(
                index=i,
                label=label,
                votes=dict(counts),
                missing_count=len(votes) - sum(counts.values()),
                disagreement_count=sum(counts.values()) - counts.get(label, 0),
            )
        )
    labels = [s["label"] for s in sentences]
    target = int(any(v in UNFAITHFUL for v in labels)) if labels and None not in labels else None
    return dict(
        sentences=sentences,
        binary_target=target,
        invalid_count=invalid,
        worker_count=len(row["workerId"]),
        annotation_array_count=len(annotations),
        surplus_annotation_positions=sum(max(0, len(a) - len(sentences)) for a in annotations),
        matches_release=labels == row["sentence_level_majority_vote"]
        and target == row["response_level_label_binary"]
        and not invalid,
    )


def seal_roster(rows: list[Json]) -> Json:
    """Select questions using only content identities before targets are opened."""
    groups: dict[str, list[str]] = defaultdict(list)
    eligible = set()
    for index, row in enumerate(rows):
        qhash = question_hash(row["question"])
        groups[qhash].append(str(index))
        if row["split"] == "test":
            eligible.add(qhash)
    ordered = sorted(
        eligible, key=lambda q: (hashlib.sha256(b"seed7398607" + q.encode("ascii")).hexdigest(), q)
    )[:128]
    return dict(
        question_hashes=ordered,
        census=len(eligible) < 128,
        eligible_question_count=len(eligible),
        seed="seed7398607",
        selection="SHA256(UTF8(seed7398607)||ASCII(question_hash)); tie question_hash",
        all_sibling_ids=[i for q in ordered for i in groups[q]],
        groups=[dict(question_hash=q, sibling_ids=groups[q]) for q in ordered],
    )


def predictor_row(row: Json, opaque_id: str) -> Json:
    """Explicit projection keeps targets, workers, generators and splits out of predictors."""
    return dict(
        opaque_id=opaque_id,
        source_bytes=row["article"].encode().hex(),
        answer_bytes=row["answer"].encode().hex(),
    )


def reduce_rows(rows: list[Json]) -> Json:
    """Emit descriptive support and custody, without inferring any detector benefit."""
    public = []
    aggregation = []
    groups: dict[str, list[str]] = defaultdict(list)
    source_groups: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        if index % 250 == 0:
            progress("aggregate", index, len(rows) - index)
        a = aggregate(row)
        qhash = question_hash(row["question"])
        source_hash = question_hash(row["article"])
        groups[qhash].append(str(index))
        source_groups[source_hash].append(str(index))
        aggregation.append(dict(opaque_id=str(index), **a))
        public.append(
            dict(
                opaque_id=str(index),
                question_hash=qhash,
                source_hash=source_hash,
                source_exact_hash=text_hash(row["article"]),
                answer_hash=text_hash(row["answer"]),
                split=row["split"],
                source_length_bytes=len(row["article"].encode()),
                answer_length_bytes=len(row["answer"].encode()),
                sentence_count=len(row["answer_sentence_list"]),
                supported_count=sum(s["label"] == "supports" for s in a["sentences"]),
                unsupported_count=sum(s["label"] == "not-mentioned" for s in a["sentences"]),
                contradicted_count=sum(s["label"] == "contradicts" for s in a["sentences"]),
                supplementary_count=sum(s["label"] == "supplementary" for s in a["sentences"]),
                metric=None,
                missing_reason="descriptive_custody_no_detector_evaluation",
            )
        )
    cross_split = [
        dict(
            question_hash=q, sibling_ids=ids, splits=sorted({public[int(i)]["split"] for i in ids})
        )
        for q, ids in sorted(groups.items())
        if len({public[int(i)]["split"] for i in ids}) > 1
    ]
    return dict(
        rows=public,
        label_aggregation_rows=aggregation,
        label_blind_roster=seal_roster(rows),
        sibling_groups=dict(groups),
        source_sibling_groups=dict(source_groups),
        annotation_authority_score=int(
            bool(rows) and all(a["matches_release"] for a in aggregation)
        ),
        complete_custody=bool(rows) and all(r["article"] and r["answer"] for r in rows),
        question_disjoint_score=0,
        cross_split_rows=cross_split,
        historical_model_provenance=sorted({r["model"] for r in rows}),
    )


def overlap_inventory(rows: list[Json], paths: list[Path]) -> list[Json]:
    """Inspect actual historical identities; opaque manifests leave exposure unknown."""
    current = {question_hash(r["question"]) for r in rows}
    result = []
    for path in paths:
        identities: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("question"), str):
                    identities.add(question_hash(value["question"]))
                if isinstance(value.get("question_hash"), str):
                    identities.add(value["question_hash"])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        if path.is_file():
            visit(json.loads(path.read_bytes()))
        result.append(
            dict(
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                comparable_question_count=len(identities),
                overlap_question_count=len(current & identities) if identities else None,
                status="comparable_manifest" if identities else "unknown_historical_overlap",
            )
        )
    return result


def release_rows(cache: Path, *, persist: bool = True) -> list[Json]:
    """Freeze content-only selection before loading raw annotation columns."""
    import pyarrow.parquet as pq

    path = cache / PARQUET
    content = pq.read_table(path, columns=["question", "split"]).to_pylist()
    roster = seal_roster(content)
    sealed = cache / "label_blind_roster.json"
    if persist:
        atomic_json(sealed, roster)
        sealed.chmod(0o600)
    elif json.loads(sealed.read_bytes()) != roster:
        raise ValueError("sealed_roster_drift")
    rows: list[Json] = pq.read_table(path).to_pylist()
    if seal_roster(rows) != roster:
        raise ValueError("roster_drift")
    predictor = cache / "predictor.jsonl"
    content_bytes = "".join(
        json.dumps(predictor_row(rows[int(index)], index)) + "\n"
        for index in roster["all_sibling_ids"]
    ).encode()
    if persist:
        predictor.write_bytes(content_bytes)
        predictor.chmod(0o600)
    elif predictor.read_bytes() != content_bytes:
        raise ValueError("predictor_custody_drift")
    return rows


def replay(path: Path) -> bool:
    """Rebuild from pinned original release bytes, not self-authorizing derived hashes."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        cache = Path(work["cache"])
        refs = {name: authenticate(cache / name, name) for name in work["release_file_hashes"]}
        if refs != work["release_file_hashes"]:
            return False
        for ref in work["source_refs"] + work["code_refs"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if not work["gate_check_summary"]:
            method = read(dict(path=str(ROOT / METHOD), sha256=METHOD_HASH))
            read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_HASH))
            if method["triviaplus_coverage_plan"] != work["plan"]:
                return False
        rows = release_rows(cache, persist=False) if len(refs) == len(BLOBS) else []
        rebuilt = reduce_rows(rows)
        if any(value[k] != v for k, v in rebuilt.items()):
            return False
        if (
            overlap_inventory(rows, [Path(r["path"]) for r in work["overlap_rows"]])
            != value["overlap_rows"]
        ):
            return False
        from carnot.reporting.triviaplus_release_runner_8617 import build

        return (
            build(
                work,
                read(value["validation_reference"]),
                Path(work["raw"]),
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
