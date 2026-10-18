"""REQ-REPORT-8553: authenticate versioned author bytes before reducing public rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hallmark_release_8553 as e
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
RAW = e.ROOT / "results/raw" / e.NAME
MANIFEST = RAW / "acquisition_manifest.json"
MANIFEST_PIN = "sha256:6b9ead46199fc8520afefe1e04b3fe165d9daae4f0d6fa3867072931b57b8436"
INPUT_PIN = "sha256:5a371939ceebad7a7bb0abdc00e473dd7ed7bb1ae0ffca1cf8e2b52d43fb7407"
TREE_PIN = "sha256:0a89a9bfa124f588033e28a55b5e8df857970efa9ba89ec36b462962a6285ea5"
SPLITS = {
    "dev_public": "data/v1.2/dev_public.jsonl",
    "test_public": "data/v1.2/test_public.jsonl",
    "stress_test": "data/v1.2/stress_test.jsonl",
    "test_crossdomain": "data/v1.2/test_crossdomain.jsonl",
    "crossdomain_matched": "data/v1.1_crossdomain_matched/test_crossdomain_matched.jsonl",
    "chatgpt_supplement": "data/v1.2/supplement_chatgpt_citations.jsonl",
    "temporal_448": "results/temporal_supplement/temporal_supplement_448_canonical.jsonl",
}
PREDICTIONS = [
    (
        "gpt51_supplement",
        "chatgpt_supplement",
        "results/walters_wilder_llm/gpt51_predictions.jsonl",
    ),
    ("gpt54_cascade", "test_public", "results/cascade_gpt54/gpt54_test_public_preds.jsonl"),
]
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "ops/exclusion_manifest.yaml",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "research-references.md",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/v734_contract_methods.py",
    "results/experiment_8389_v723_human_label_custody.json",
    "research-roadmap.yaml",
]


def load(manifest: Path) -> Json:
    """A code-pinned inventory and author git blobs prevent self-rehashed data substitution."""
    entries = e.read(dict(path=str(manifest), sha256=MANIFEST_PIN))
    tree = e.read(dict(path=str(RAW / "tree.json"), sha256=TREE_PIN))
    blobs = {r["path"]: r["sha"] for r in tree["tree"]}
    content, missing = {}, []
    for ref in entries:
        if ref.get("missing"):
            missing.append(ref["name"])
            continue
        path = Path(ref["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
        ).hexdigest()
        if (
            ref["commit"] != e.COMMIT
            or sha256_file(path) != ref["sha256"]
            or blob != blobs[ref["name"]]
        ):
            raise ValueError("author_blob:" + ref["name"])
        content[ref["name"]] = data
    splits = {
        name: [json.loads(line) for line in content[path].splitlines() if line.strip()]
        for name, path in SPLITS.items()
        if path in content
    }
    missing.extend(path for path in SPLITS.values() if path not in content)
    predictions = [
        dict(
            name=name,
            split=split,
            version=e.COMMIT + ":v1.2.3",
            rows=[json.loads(line) for line in content[path].splitlines() if line.strip()],
        )
        for name, split, path in PREDICTIONS
        if path in content
    ]
    metadata = json.loads(content.get("data/v1.2/metadata.json", b"{}"))
    counts = {
        k: metadata.get("splits", {}).get(k, {}).get("total")
        for k in ("dev_public", "test_public", "stress_test", "test_crossdomain")
    }
    counts.update(
        test_crossdomain=500, crossdomain_matched=452, chatgpt_supplement=341, temporal_448=448
    )
    sources = e.read(dict(path=str(RAW / "input_manifest.json"), sha256=INPUT_PIN))
    for source in sources:
        if source.get("missing"):
            missing.append(source["original_path"])
        elif sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("original_input_hash")
    upstream = []
    for pattern in [
        "experiment_8541_*.json",
        "experiment_8545_*.json",
        "experiment_8548_*.json",
        "experiment_8551_*.json",
        "experiment_8529_*.json",
    ]:
        for path in sorted((e.ROOT / "results").glob(pattern)):
            old = json.loads(path.read_bytes())
            upstream.append(
                dict(
                    e.reference(path),
                    experiment_id=old.get("experiment_id"),
                    original_run_date=old.get("run_date"),
                    honest_verdict=old.get("honest_verdict"),
                    flagged_adversarial=old.get("flagged_adversarial"),
                    role="historical_quarantine_only",
                )
            )
    licenses = {
        ref["name"]: (
            "MIT; hf/README.md explicitly covers synthetic data; bibliographic metadata stated public domain"
            if ref["name"] != "paper.html"
            else "CC BY 4.0"
        )
        for ref in entries
    }
    return dict(
        splits=splits,
        declared_counts=counts,
        predictions=predictions,
        missing_resources=missing,
        acquisition_reference=e.reference(manifest),
        prediction_summaries={
            name: json.loads(content[path])
            for name, path in [
                ("gpt51_supplement", "results/walters_wilder_llm/gpt51_metrics.json"),
                ("gpt54_cascade", "results/cascade_gpt54/gpt54_test_public.json"),
            ]
            if path in content
        },
        release_manifest=[
            dict(ref, license=licenses[ref["name"]], release_id=381198153, release_tag="v1.2.3")
            for ref in entries
        ],
        source_artifact_hashes=sources,
        cited_upstream_artifacts=upstream,
        historical_model_provenance=[
            dict(model="gpt-5.1", scope="released supplement predictions"),
            dict(model="gpt-5.4", scope="released cascade predictions"),
        ],
        preserved_quarantines=dict(
            exp8541="152/158 original nodes passed; family4 killed at480s",
            exp8545="numerical and complete-request parity; preliminary cold replay failed",
            exp8548="stored replay134s; four multiline conditional anchors failed exact receipts",
            exp8529="qualified exposed source-information null",
            learning="unmeasured",
            native_speed="unmeasured",
            capability_mask_benefit="unmeasured",
        ),
        release_file_inventory=[
            dict(
                name=name,
                row_count=len(blob.splitlines()),
                missing_labels=sum(
                    json.loads(line).get("label") not in ("VALID", "HALLUCINATED", "UNCERTAIN")
                    for line in blob.splitlines()
                    if line.strip()
                ),
                role="prediction"
                if name.startswith("results/") and "predictions" in name or "preds" in name
                else "released_split_or_provenance_pool",
            )
            for name, blob in content.items()
            if name.endswith(".jsonl")
        ],
        preregistration_reference=e.reference(RAW / "preregistration.json"),
        authority_receipt=json.loads((RAW / "authority_receipt.json").read_bytes()),
        construction_constraints=dict(
            label_origin="Pipeline labels; LLM generations database-filtered; real-world/adversarial author manual review.",
            valid_inclusion="DOI resolves where present, title in at least two databases, author/venue consistency; database-confirmable positives.",
            scope="Citation metadata validity; neither source entails sentence nor nine-input features.",
            historical_dates="Original paper/release/producer dates retained; no historical receipt regenerated.",
        ),
    )
