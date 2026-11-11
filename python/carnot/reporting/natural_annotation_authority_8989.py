"""REQ-REPORT-8989: inspect released annotations without inventing independent truth."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.natural_label_eligibility_8975 import ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8989_v768_natural_annotation_authority"
TASK = "exp8989-natural-annotation-authority"
CLI = "scripts/experiments/" + NAME + ".py"
REVISIONS = dict(
    repository="1d3e8fdd6de638dc2d06265829a4d9ced387e0ba",
    dataset="aab54518c2a7c0d25fff8bffbf5337d0321de142",
)
PINNED = {
    "dataset/README.md": "sha256:d007ca16c19a9c8fa5ed8e2154cb002de5e160e58ac9e1ae09f0e40629f582f4",
    "test": "sha256:8fbfe21df70a07926e0983fd03b77ae53ce0543471b92386a5d7156b71bf3a8c",
    "train": "sha256:6ad56f84a06863b4dea28e86014cc16e37a64fefe8965c53902b370f51839a44",
    "github_metadata.json": "sha256:b7ef28b4cd8799b09072a64d66cdbcf2e59c4663c806955164da054c19266c39",
    "github_tree.json": "sha256:289bce279e9844625f1e280874f120cf4db94b2526f6208fa3ca0ccf39667636",
    "hf_metadata.json": "sha256:29a9390dc0ae2264d69925731c996ce17f3551d2d2ad04657ea2928a41935636",
    "hf_tree.json": "sha256:1127a7a24693f9b6b75deed066972afe1d06ef534e8b79194af9b0e705b6073e",
    "repo/LICENSE": "sha256:5b046e571c6af4d584f37def1d520c794e1a2255ea79c9bc67612695a8308cf6",
    "repo/README.md": "sha256:a71861ae1586301cc57d0228f4e8b556c7fea6e6fba1cfb6e6a5931243e98371",
    "repo/ragognize/00_wikipedia_scraper.py": "sha256:68ce19f6a10cd97b30669616e91b37110c7fae197b90434b20936893e182c23f",
    "repo/ragognize/01_extract_suitable_articles.py": "sha256:ee658deffb3726c2248eda89f4856bdf7a91588c44cb1cef33bcdd1d06987b4e",
    "repo/ragognize/02_qa_generation/filter_user_prompts.py": "sha256:0193851361bacc79926ddb4bb917af0a4b8d29c0d23d0ada218801ca77b8a7ff",
    "repo/ragognize/02_qa_generation/generate_user_prompts.py": "sha256:fb6382ed505dbe7b19e6f1227e43d3f6f1aebbca326300ffc07bd29af9966d29",
    "repo/ragognize/02_qa_generation/user_prompt_generator.py": "sha256:52881a8c3f93000f5d07c170620c8197aeabf9b0c9ac16429a05e3271bbc7d72",
    "repo/ragognize/03_prompt_generation/generate_rag_prompts.py": "sha256:5cd3fb52948c53bd8cc5c9626882c1c97e99d356ed59a7ccf331d4cad1966623",
    "repo/ragognize/03_prompt_generation/prompt_template_randomizer.py": "sha256:a9ef1ec044aeda71dc5cb94cb2fe33007bebcf4a9198b7b66fc56b0ebc0371e4",
    "repo/ragognize/04_llm_responses/add_samples_to_ragonize.py": "sha256:f36a02b95a2afd5d9ebb65827837216d2f4c448f94a8479e10ed62a0ca60d683",
    "repo/ragognize/04_llm_responses/annotate_outputs.py": "sha256:f8bb3859abdc72dcd4ad673c523ef5041747d9b2ff3788119d5be3a9c3271ca5",
    "repo/ragognize/04_llm_responses/generate_outputs.py": "sha256:4aa58ee58d80c82429d40b926e3c5da30d8481edd80cc78c949c3542c9759e85",
    "repo/ragognize/04_llm_responses/merge.py": "sha256:c78163d02e1e34254ded8931241769925b7002ee4ef6dda119a8f58cb8be095d",
    "repo/ragognize/install.sh": "sha256:d36f20f59baeaa48b1c8efbf41a80be980f88d4f2ce5cf6326551502eb8d174a",
    "repo/ragognize/requirements.txt": "sha256:a38fae6c85da140aa527f8221a9820d9a0f07e5a126fc3ce2d817e94169648d2",
    "repo/ragognize/run.sh": "sha256:bbf4320faaabfe25ac14fa1f2acfc6ce3d703ebc223caa12b7f3463df72de787",
    "repo/ragognize/utils/__init__.py": "sha256:7899fed109925978281734c96e1541d73740bc0b2781a4263b66eba9b5ec61e2",
    "repo/ragognize/utils/llama.py": "sha256:877a63c50a561d05e2ffbab0a3e28914c10b4a0c4f16089abef9414dfad3c655",
    "repo/ragognize/utils/llm.py": "sha256:57d67fd269b1a8090ec4d352f37408385e9e9f01dfe4dbb8b30a2662a9c56299",
    "repo/ragognize/utils/mistral.py": "sha256:1786753f00f416cbaba113f79cc26d17e3c6dfd4856f2366f2e926b9d2db9e8d",
    "repo/ragognize/utils/rate_limiter.py": "sha256:3294e3cd65b5555108d92e81b405533a69da6dc30a72091af79ceed207426206",
    "repo/ragognize/utils/shd_SimpleChatTokenLevel.py": "sha256:6d2c9beb809b3c7e00bc32bd303e65e1b6079c633e7a0ee66c94edcbc5fbdc3c",
    "repo/ragognize/utils/structured_output_llm.py": "sha256:aa136ffec7f3b5ae8aad92de4344d1b86a080ba335ced0ee40e2a81a70377e8a",
    "repo/ragognize/utils/supervised_hallucination_detector.py": "sha256:e2a673c97e88db60bfaa516843c1634b0b567ff57481eb939ed46cc9f4f8f749",
    "repo/ragognizer/ragognizer/benchmarks/RAGognize.py": "sha256:75abf2b483dc45e2e5d192cb9060d3c0ee8ee6a34cf4d775f620dfd8ae9a5937",
}
OWNED = [
    "python/carnot/reporting/natural_annotation_authority_8989.py",
    "python/carnot/reporting/natural_annotation_runner_8989.py",
    CLI,
]
CHECKS = {"owned_pytest", "owned_coverage", "ruff", "format", "mypy", "spec", "e2e018_021"}
EXCLUDED_FEATURES = [
    "responses.golden_answer",
    "responses.*.hallucinations",
    "responses.*.details.annotations",
    "answerable",
    "tags",
    "details.user_prompt.golden_answer",
    "details.user_prompt.cluelessness_answer",
    "details.user_prompt.details.answer_quote_from_passage",
    "details.answerable_chunks",
    "details.unanswerable_chunks",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed units so read-only processing remains observable."""
    print(f"[exp8989] phase={phase} completed={completed} pending={pending}", flush=True)


def alignment(response: Json) -> Json:
    """Author spans use Python characters; byte offsets are derived from exact UTF8."""
    text = response["text"]
    raw = text.encode("utf-8")
    spans = []
    for span in response["hallucinations"]:
        start, end = span["start"], span["end"]
        bounded = type(start) is int and type(end) is int and 0 <= start < end <= len(text)
        match = bounded and text[start:end] == span["text"]
        a, b = (len(text[:start].encode()), len(text[:end].encode())) if bounded else (None, None)
        spans.append(
            dict(
                **span,
                character_match=match,
                byte_start=a,
                byte_end=b,
                utf8_match=bool(match and raw[a:b] == span["text"].encode()),
            )
        )
    starts = response["details"]["token_starts"]
    starts_valid = (
        bool(starts)
        and all(type(i) is int and 0 <= i <= len(text) for i in starts)
        and starts == sorted(starts)
    )
    duplicate_output_matches = response["details"]["output"] == text
    valid = (
        bool(text)
        and starts_valid
        and duplicate_output_matches
        and all(s["character_match"] and s["valid"] is True for s in spans)
    )
    return dict(
        response_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        utf8_bytes=len(raw),
        characters=len(text),
        spans=spans,
        token_start_count=len(starts),
        token_starts_valid=starts_valid,
        duplicate_output_matches=duplicate_output_matches,
        alignment_valid=valid,
        token_semantics="character lengths of decoded prefixes with special tokens; response skips special tokens; no tokenizer loaded",
    )


def audit(records: list[tuple[str, int, Json]], references: Json) -> Json:
    """Count source groups separately from dependent responses and human authority."""
    rows, sources, tokens = [], [], []
    clusters: dict[tuple[str, str], list[str]] = defaultdict(list)
    split_by_unit, fields = {}, set()
    group_labels: dict[str, set[int]] = defaultdict(set)
    for index, (split, offset, item) in enumerate(records):
        source = item["details"]["user_prompt"]
        article = source["details"]["suitable_article"]
        group = canonical_hash([article["url"], source["article_title"], item["user_prompt"]])
        unit = f"{split}:{offset}"
        split_by_unit[unit] = split
        contexts = [d["text"] for d in item["documents"]]
        sources.append(
            dict(
                unit_id=unit,
                group_id=group,
                split=split,
                row_offset=offset,
                release_reference=references.get(split),
                original_article_url=article["url"],
                original_article_revision=article["revision_id"],
                question_id=item["user_prompt_index"],
                question_sha256=canonical_hash(item["user_prompt"]),
                context_sha256=canonical_hash(contexts),
                usable_original_text=bool(article["content"] and contexts and all(contexts)),
                answerable=item["answerable"],
                source_reference_sha256=canonical_hash(article.get("passage_data", [])),
            )
        )
        for kind, identity in [
            ("article", article["url"]),
            ("question", group),
            ("context", canonical_hash(contexts)),
            ("reference", canonical_hash(article.get("passage_data", []))),
        ]:
            clusters[kind, identity].append(unit)
        for model, response in item["responses"].items():
            if model == "golden_answer":
                continue
            ident = unit + ":" + model
            checked = alignment(response)
            tokens.append(dict(unit_id=ident, group_id=group, **checked))
            label = int(bool(response["hallucinations"]))
            group_labels[group].add(label)
            rows.append(
                dict(
                    unit_id=ident,
                    source_unit_id=unit,
                    group_id=group,
                    arm=model,
                    condition="answerable" if item["answerable"] else "unanswerable",
                    metric="released_machine_hallucination",
                    value=label,
                    numerator=label,
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                    split=split,
                    row_offset=offset,
                    response_sha256=checked["response_sha256"],
                    release_reference=references.get(split),
                )
            )
            clusters["response", checked["response_sha256"]].append(unit)
            for section, values in response["details"]["annotations"].items():
                fields.update("details.annotations." + section + "." + key for key in values)
        if index % 100 == 0:
            progress("audit_release_rows", index + 1, len(records) - index - 1)
    overlaps = [
        dict(
            kind=k,
            identity=h,
            source_units=sorted(set(units)),
            occurrence_count=len(units),
            cross_split=len({split_by_unit[u] for u in units}) > 1,
        )
        for (k, h), units in sorted(clusters.items())
        if len(units) > 1
    ]
    origins = [
        dict(
            field=f,
            origin="machine_annotation_or_deterministic_projection",
            model="gemini-2.5-flash",
            pipeline="ragognize/04_llm_responses/annotate_outputs.py; ragognize/utils/shd_SimpleChatTokenLevel.py; ragognize/04_llm_responses/merge.py",
            independent_human_decision=False,
        )
        for f in sorted(
            fields
            | {
                "hallucinations.*.start",
                "hallucinations.*.end",
                "hallucinations.*.text",
                "hallucinations.*.valid",
            }
        )
    ]
    origins.append(
        dict(
            field="responses.golden_answer; details.user_prompt.golden_answer; cluelessness_answer",
            origin="machine_generated_reference",
            model="gemini-2.5-pro",
            pipeline="ragognize/02_qa_generation/generate_user_prompts.py; user_prompt_generator.py",
            independent_human_decision=False,
        )
    )
    return dict(
        rows=rows,
        source_rows=sources,
        token_alignment_rows=tokens,
        group_overlap_rows=overlaps,
        annotation_origin_rows=origins,
        independent_label_gate_operands=dict(
            documented_independent_human_decisions=0,
            required_independent_question_groups=200,
            independent_groups_with_both_labels=0,
            authorized_future_model_use="not_established",
            usable_original_text=bool(sources) and all(s["usable_original_text"] for s in sources),
        ),
        intended_count=len(rows),
        completed_count=len(rows),
        independent_count=len(group_labels),
        independent_human_label_group_count=0,
        excluded_count=0,
        censored_count=0,
        failed_count=0,
        source_access_ready_score=int(
            bool(sources) and all(s["usable_original_text"] for s in sources)
        ),
        annotation_audit_ready_score=int(bool(rows)),
        independent_label_ready_score=0,
        independent_generalization_score=0,
        label_counts=dict(Counter(str(row["value"]) for row in rows)),
        machine_both_label_group_count=sum(v == {0, 1} for v in group_labels.values()),
        alignment_failure_count=sum(not t["alignment_valid"] for t in tokens),
        audited_population=dict(
            scope="full_release",
            split_rows=dict(Counter(s[0] for s in records)),
            response_variants=len(rows),
            question_groups=len(group_labels),
        ),
        human_review_evidence="not_documented_in_inspected_release",
        adjudication_evidence="no_independent_human_adjudication_documented",
        source_reference_dependence="Questions and references share Wikipedia source; annotation prompt sees RAG context, not golden_answer; generated reference is not independent truth.",
        excluded_feature_keys=EXCLUDED_FEATURES,
        eligible_use_matrix=[
            dict(use=u, eligible=b)
            for u, b in [
                ("release and offset audit", True),
                ("machine-supervised development under separately verified terms", True),
                ("independent human factuality benchmark", False),
                ("oracle-distinct verifier headline", False),
            ]
        ],
        missing_evidence=[
            "Item-level independent human decisions, reviewers, votes and adjudication",
            "At least200 independent question groups supporting both labels under human decisions",
            "Independent factual evidence beyond shared source/reference",
            "Response-model and original-source terms for any future training or redistribution",
        ],
        gate_check_summary=[],
    )


def inspect(manifest: Path) -> Json:
    """Authenticate retained release bytes before interpreting any public label."""
    if not manifest.is_file():
        raise FileNotFoundError("missing_file:" + str(manifest))
    value = json.loads(manifest.read_bytes())
    if value["release_revision"] != REVISIONS:
        raise ValueError("release_revision")
    files = value["files"]
    authenticated = dict(files)
    if value.get("retrieval_receipt"):
        authenticated["retrieval_receipt"] = value["retrieval_receipt"]
    for key, reference in authenticated.items():
        path = Path(reference["path"])
        if not path.is_file():
            raise FileNotFoundError("missing_file:" + str(path))
        if ref(path) != reference:
            raise ValueError("changed_bytes:" + str(path))
    public = value["scope"] == "author_release"
    if public and ({key: r["sha256"] for key, r in files.items()} != PINNED):
        raise ValueError("pinned_release")
    records = []
    for split in ("train", "test"):
        if split in files:
            progress("before_parquet_" + split)
            offset = 0
            parquet = pq.ParquetFile(files[split]["path"])
            for batch in parquet.iter_batches(batch_size=100):
                items = batch.to_pylist()
                records.extend((split, offset + i, row) for i, row in enumerate(items))
                offset += len(items)
                progress("parquet_" + split, offset, parquet.metadata.num_rows - offset)
            progress("after_parquet_" + split, offset, 0)
    reduced = audit(records, files)
    if not public:
        reduced.update(source_access_ready_score=0, annotation_audit_ready_score=0)
    reduced.update(
        input_scope=value["scope"],
        release_revision=REVISIONS,
        license_references=value["license_references"],
        source_urls=value["source_urls"],
        source_artifact_hashes=[ref(manifest), *files.values()],
        retrieval_receipt=value.get("retrieval_receipt"),
    )
    return reduced
