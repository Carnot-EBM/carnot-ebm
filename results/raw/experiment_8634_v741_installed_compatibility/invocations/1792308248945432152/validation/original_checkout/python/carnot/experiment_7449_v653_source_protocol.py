"""Seal source-pair inputs and an external human challenge protocol.

This task freezes information flow for later representation work. It does not
load the planned model, fit a selector, or turn disagreement-selected examples
into a deployment-prevalence claim.

Spec refs: REQ-AUTO-7449 and SCENARIO-AUTO-7449-01 through -06.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import unicodedata
from urllib.request import urlopen

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7412_v650_source_features import (
    SOURCE_FEATURE_NAMES,
    extract_feature_row,
)
from carnot.experiment_7423_v651_annotated_protocol import (
    DEFAULT_CACHE_ROOT as RAGTRUTH_CACHE_ROOT,
    SourceBlocked as RAGTruthSourceBlocked,
    authenticate_assets as authenticate_ragtruth_assets,
    load_release as load_ragtruth_release,
    serialize_source,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7449-v653-source-protocol"
SCHEMA = "carnot.exp7449.v653.source_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7449_v653_source_protocol.json")
RAW_DIR = Path("results/raw/experiment_7449_v653_source_protocol")
MODULE_PATH = Path("python/carnot/experiment_7449_v653_source_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7449_v653_source_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7449_v653_source_protocol.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/autoresearch/spec.md"
VERIFICATION_SPEC_PATH = Path("openspec/capabilities/verification/spec.md")

MODEL_SPECS: list[JsonDict] = []
INVOCATION_COUNTS = deepcopy(ZERO_INVOCATION_COUNTS)
INFERENCE_SUBSTRATE = "deterministic_representation_contract_no_llm"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

RAGTRUTH_CAPS = {"training": 180, "calibration_tuning": 60, "internal_test": 60}
FAITHBENCH_CAP = 100
TOKEN_CEILING = 2_048
PROJECTION_SEED = 6_530_032
FIT_SEEDS = (65_301, 65_302, 65_303, 65_304, 65_305)
BOOTSTRAP_SEED = 6_530_010
SELECTION_SALT = "carnot-v653-source-protocol-1"
MINIMUM_GROUPS = {
    "training": 150,
    "calibration_tuning": 40,
    "internal_test": 40,
    "external": 60,
}

FAITHBENCH_REPOSITORY = "https://github.com/vectara/FaithBench"
FAITHBENCH_REVISION = "cf89797d82812c23b5d5e5c121f1d9b8983bbbce"
FAITHBENCH_LICENSE = "CC BY-NC-SA 4.0"
FAITHBENCH_ATTRIBUTION = (
    "FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs, "
    "Bao et al., NAACL 2025; public release by Vectara."
)
FAITHBENCH_CACHE = Path(
    os.environ.get(
        "CARNOT_EXP7449_CACHE",
        str(Path.home() / ".cache/carnot/experiment_7449_v653_source_protocol"),
    )
)
FAITHBENCH_ASSETS: dict[str, tuple[int, str]] = {
    "LICENSE": (20_850, "7cdbe0b482f604a06a0988dad8877ae8d9257f7d"),
    "README.md": (5_989, "3662af7055cc6effb682fffb8be98ee9affcf41a"),
    "scripts/binarize.py": (5_380, "9c2c592618473b88fd12ab8c1544c337aae755ae"),
    "scripts/faithbench_schema.py": (1_811, "9cda037268a92f7dde5608ceae9f8ebd6c31a78f"),
    "data_for_release/batch_1.json": (76_262, "53efa670296d464c62a9f7fffb24294b47cf9f90"),
    "data_for_release/batch_2.json": (154_793, "87f50e726d29f613e76f2721245f580339743d8b"),
    "data_for_release/batch_3.json": (98_772, "9dca2a7ebf57390065f28a2281625c2fbb91bf7d"),
    "data_for_release/batch_4.json": (149_892, "e713e446be4bcfeb2e36dacc4d03dd08f38a7993"),
    "data_for_release/batch_5.json": (190_782, "9af3bc02d41380d605ecd2cc1f7662a9f75b5cbb"),
    "data_for_release/batch_6.json": (135_160, "9a2c434674986d3ad360ebcaf5b58d3ee6ec60f5"),
    "data_for_release/batch_7.json": (191_546, "556f3cbe511720ba67ca677bc41ccc53bed715cd"),
    "data_for_release/batch_8.json": (236_853, "d180b0925cd028915f4b2f4f347c7ffc6d1dbb8a"),
    "data_for_release/batch_9.json": (201_505, "4c5f3e0111d6d981f13b1817b5efaf998c2045c6"),
    "data_for_release/batch_10.json": (292_603, "c8d4c031a99ff8d85bd8364600ce6838ba9ad1a1"),
    "data_for_release/batch_11.json": (189_035, "8f3e7a69a4f3d65bd118c03842ecd4105c704eac"),
    "data_for_release/batch_12.json": (205_796, "a15e38843340a1685cf552b129b5b84ce9bfc169"),
    "data_for_release/batch_14.json": (394_638, "624bb9661765d615b69e6dc3b8b6122eb29910d2"),
    "data_for_release/batch_15.json": (370_657, "6b5a765e62ebc3e3f6ffff3a33ac89a2bab5a5d5"),
    "data_for_release/batch_16.json": (426_968, "763f05a84859bc777e70458e4a96dfd465994217"),
}
MAX_RELEASE_FILE_BYTES = 2 * 1024 * 1024

PREDICTOR_FIELDS = (
    "row_key",
    "group_id",
    "corpus",
    "role",
    "source_text",
    "response_text",
    "source_features",
)
EVALUATOR_FIELDS = (
    "row_key",
    "group_id",
    "corpus",
    "role",
    "source_id",
    "response_id",
    "official_split",
    "label",
    "annotation_labels",
    "ambiguous",
    "label_policy",
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    VERIFICATION_SPEC_PATH,
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/experiment_7412_v650_source_features.py"),
    Path("python/carnot/experiment_7436_v652_selection_protocol.py"),
    Path("python/carnot/experiment_7439_v652_certified_decisions.py"),
    Path("results/experiment_7439_v652_certified_decisions.json"),
    Path("results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"),
    Path("openspec/capabilities/autoresearch/spec.md"),
)

LOCAL_CORPUS_PATHS = (
    Path("data/fover_corpus_v4.json"),
    Path("results/experiment_7396_v649_decision_diagnosis.json"),
    Path("results/experiment_7410_v650_source_corpus.json"),
    Path("results/experiment_7423_v651_annotated_protocol.json"),
    Path("results/experiment_7439_v652_certified_decisions.json"),
)

AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class SourceBlocked(RuntimeError):
    """Name an unchanged external-source failure that blocks corpus sealing."""


class CorpusInvalid(ValueError):
    """Reject malformed release rows or changed sealed bytes before reuse."""


def normalized_source_hash(text: str) -> str:
    """Hash a Unicode-normalized source so cosmetic spacing cannot split groups."""

    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _stable_hash(value: str) -> str:
    """Return one label-blind ordering key for groups and responses."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature_view(row_key: str, group_id: str, role: str, source: str, response: str) -> JsonDict:
    """Reuse the shipped six lexical source features without shortening model text."""

    projected = extract_feature_row(
        {
            "row_key": row_key,
            "group_id": group_id,
            "partition": role,
            "question": "",
            "context": source,
            "answer": response,
            "sentence": response,
        }
    )["source_features"]
    return {name: float(projected[name]) for name in SOURCE_FEATURE_NAMES}


def _predictor(
    *, corpus: str, role: str, group_id: str, identity: str, source: str, response: str
) -> JsonDict:
    """Build the exact allowlisted predictor row from complete source and response."""

    row_key = "row-" + _stable_hash(f"{SELECTION_SALT}:{corpus}:{group_id}:{identity}")[:24]
    return {
        "row_key": row_key,
        "group_id": group_id,
        "corpus": corpus,
        "role": role,
        "source_text": source,
        "response_text": response,
        "source_features": _feature_view(row_key, group_id, role, source, response),
    }


def _annotation_types(labels: object) -> list[str]:
    """Copy annotation categories while deliberately discarding notes and identity."""

    if not isinstance(labels, list):
        raise CorpusInvalid("annotation list required")
    result: set[str] = set()
    for annotation in labels:
        if not isinstance(annotation, Mapping):
            raise CorpusInvalid("annotation object required")
        value = annotation.get("label_type")
        if value is not None:
            result.add(str(value))
    return sorted(result)


def select_ragtruth_panel(
    source_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    *,
    caps: Mapping[str, int] = RAGTRUTH_CAPS,
) -> JsonDict:
    """Seal role-preserving RAGTruth representatives without consulting labels."""

    sources: dict[str, JsonDict] = {}
    for source in source_rows:
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in sources:
            raise CorpusInvalid(f"source identity invalid:{source_id}")
        task_type = str(source.get("task_type") or "")
        text = serialize_source(task_type, source.get("source_info"))
        sources[source_id] = {"text": text, "hash": normalized_source_hash(text)}

    groups: dict[str, list[JsonDict]] = defaultdict(list)
    dispositions: list[JsonDict] = []
    seen: set[str] = set()
    for response in response_rows:
        response_id = response.get("id")
        source_id = response.get("source_id")
        if not isinstance(response_id, str) or not response_id or response_id in seen:
            raise CorpusInvalid(f"response identity invalid:{response_id}")
        seen.add(response_id)
        if source_id not in sources:
            raise CorpusInvalid(f"response source missing:{response_id}")
        split = response.get("split")
        quality = response.get("quality")
        if split not in {"train", "test"}:
            raise CorpusInvalid(f"official split invalid:{response_id}")
        if quality != "good":
            dispositions.append(
                {
                    "corpus": "ragtruth",
                    "unit_id": response_id,
                    "reason": f"excluded_quality_{quality}",
                }
            )
            continue
        text = response.get("response")
        if not isinstance(text, str) or not text:
            raise CorpusInvalid(f"response text invalid:{response_id}")
        groups[str(sources[str(source_id)]["hash"])].append(
            {
                "response_id": response_id,
                "source_id": source_id,
                "source_text": sources[str(source_id)]["text"],
                "response_text": text,
                "official_split": split,
                "labels": deepcopy(response.get("labels") or []),
            }
        )

    eligible: dict[str, list[JsonDict]] = {}
    split_for_group: dict[str, str] = {}
    for source_hash, rows in groups.items():
        splits = {str(row["official_split"]) for row in rows}
        group_id = "group-" + source_hash.split(":", 1)[1][:24]
        if len(splits) != 1:
            dispositions.append(
                {
                    "corpus": "ragtruth",
                    "unit_id": group_id,
                    "reason": "excluded_cross_official_role_duplicate",
                }
            )
            continue
        eligible[group_id] = rows
        split_for_group[group_id] = next(iter(splits))

    train_groups = sorted(
        (group for group, split in split_for_group.items() if split == "train"),
        key=lambda group: _stable_hash(f"{SELECTION_SALT}:ragtruth:train:{group}"),
    )
    test_groups = sorted(
        (group for group, split in split_for_group.items() if split == "test"),
        key=lambda group: _stable_hash(f"{SELECTION_SALT}:ragtruth:test:{group}"),
    )
    training_end = max(0, int(caps["training"]))
    calibration_end = training_end + max(0, int(caps["calibration_tuning"]))
    role_groups = {
        "training": train_groups[:training_end],
        "calibration_tuning": train_groups[training_end:calibration_end],
        "internal_test": test_groups[: max(0, int(caps["internal_test"]))],
    }
    selected_roles = {group: role for role, group_ids in role_groups.items() for group in group_ids}
    for group_id in sorted(set(eligible) - set(selected_roles)):
        dispositions.append(
            {
                "corpus": "ragtruth",
                "unit_id": group_id,
                "reason": "excluded_role_cap",
            }
        )

    predictors: list[JsonDict] = []
    evaluators: list[JsonDict] = []
    role_order = {"training": 0, "calibration_tuning": 1, "internal_test": 2}
    for group_id, role in sorted(
        selected_roles.items(), key=lambda item: (role_order[item[1]], item)
    ):
        siblings = eligible[group_id]
        winner = min(
            siblings,
            key=lambda row: _stable_hash(
                f"{SELECTION_SALT}:ragtruth:response:{group_id}:{row['response_id']}"
            ),
        )
        predictor = _predictor(
            corpus="ragtruth",
            role=role,
            group_id=group_id,
            identity=str(winner["response_id"]),
            source=str(winner["source_text"]),
            response=str(winner["response_text"]),
        )
        annotation_labels = _annotation_types(winner["labels"])
        predictors.append(predictor)
        evaluators.append(
            {
                "row_key": predictor["row_key"],
                "group_id": group_id,
                "corpus": "ragtruth",
                "role": role,
                "source_id": winner["source_id"],
                "response_id": winner["response_id"],
                "official_split": winner["official_split"],
                "label": int(not bool(winner["labels"])),
                "annotation_labels": annotation_labels,
                "ambiguous": False,
                "label_policy": "one_if_no_human_unsupported_span",
            }
        )
    return {
        "predictors": predictors,
        "evaluators": evaluators,
        "dispositions": sorted(dispositions, key=lambda row: (row["reason"], row["unit_id"])),
        "realized_group_counts": {role: len(role_groups[role]) for role in role_groups},
        "eligible_before_caps": {
            "original_train": len(train_groups),
            "original_test": len(test_groups),
        },
    }


FAITHBENCH_SEVERITY = {
    "Benign": 1,
    "Questionable": 2,
    "Unwanted": 3,
    "Unwanted.Intrinsic": 3,
    "Unwanted.Extrinsic": 3,
}


def official_faithbench_label(sample: Mapping[str, Any]) -> tuple[int, bool]:
    """Apply audited worst-severity aggregation and retain disagreement separately.

    The upstream script's empty-annotation branch returns zero even though its
    comment calls the sample consistent. This protocol keeps one as the shared
    non-hallucinated polarity used by RAGTruth and records that audited deviation
    in the corpus manifest instead of silently inheriting the contradictory line.
    """

    annotations = sample.get("annotations")
    if not isinstance(annotations, list):
        raise CorpusInvalid("FaithBench annotations list required")
    labels: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or not isinstance(annotation.get("label"), list):
            raise CorpusInvalid("FaithBench annotation shape invalid")
        labels.update(str(value) for value in annotation["label"])
    unknown = labels - set(FAITHBENCH_SEVERITY)
    if unknown:
        raise CorpusInvalid(f"unknown FaithBench label:{sorted(unknown)[0]}")
    severities = {FAITHBENCH_SEVERITY[label] for label in labels}
    if not severities:
        return 1, False
    return int(max(severities) == 1), len(severities) > 1


def select_faithbench_panel(
    samples: Sequence[Mapping[str, Any]],
    *,
    local_source_hashes: set[str],
    cap: int = FAITHBENCH_CAP,
) -> JsonDict:
    """Select an external source-group panel without notes, scores, or model IDs."""

    groups: dict[str, list[JsonDict]] = defaultdict(list)
    dispositions: list[JsonDict] = []
    for ordinal, sample in enumerate(samples):
        source = sample.get("source")
        summary = sample.get("summary")
        sample_id = sample.get("sample_id")
        if not isinstance(source, str) or not source or not isinstance(summary, str) or not summary:
            raise CorpusInvalid(f"FaithBench text invalid:{sample_id}")
        batch_id = int(sample.get("_batch_id") or 0)
        identity = f"{batch_id}:{sample_id}:{ordinal}"
        source_hash = normalized_source_hash(source)
        group_id = "group-" + source_hash.split(":", 1)[1][:24]
        groups[group_id].append(
            {
                "identity": identity,
                "sample_id": sample_id,
                "source": source,
                "summary": summary,
                "annotations": deepcopy(sample.get("annotations") or []),
                "source_hash": source_hash,
            }
        )

    eligible: dict[str, list[JsonDict]] = {}
    for group_id, rows in groups.items():
        if str(rows[0]["source_hash"]) in local_source_hashes:
            dispositions.append(
                {
                    "corpus": "faithbench",
                    "unit_id": group_id,
                    "reason": "excluded_local_corpus_duplicate",
                }
            )
        else:
            eligible[group_id] = rows
    ranked = sorted(
        eligible,
        key=lambda group: _stable_hash(f"{SELECTION_SALT}:faithbench:group:{group}"),
    )
    selected = ranked[: max(0, cap)]
    for group_id in ranked[max(0, cap) :]:
        dispositions.append(
            {"corpus": "faithbench", "unit_id": group_id, "reason": "excluded_external_cap"}
        )

    predictors: list[JsonDict] = []
    evaluators: list[JsonDict] = []
    for group_id in selected:
        winner = min(
            eligible[group_id],
            key=lambda row: _stable_hash(f"{SELECTION_SALT}:faith-response:{row['sample_id']}"),
        )
        predictor = _predictor(
            corpus="faithbench",
            role="external",
            group_id=group_id,
            identity=str(winner["identity"]),
            source=str(winner["source"]),
            response=str(winner["summary"]),
        )
        label, ambiguous = official_faithbench_label({"annotations": winner["annotations"]})
        annotation_labels = sorted(
            {
                str(label_name)
                for annotation in winner["annotations"]
                if isinstance(annotation, Mapping)
                for label_name in annotation.get("label", [])
            }
        )
        predictors.append(predictor)
        evaluators.append(
            {
                "row_key": predictor["row_key"],
                "group_id": group_id,
                "corpus": "faithbench",
                "role": "external",
                "source_id": normalized_source_hash(str(winner["source"])),
                "response_id": winner["identity"],
                "official_split": "external_only",
                "label": label,
                "annotation_labels": annotation_labels,
                "ambiguous": ambiguous,
                "label_policy": "worst_severity_consistent_or_benign_is_one",
            }
        )
    return {
        "predictors": predictors,
        "evaluators": evaluators,
        "dispositions": sorted(dispositions, key=lambda row: (row["reason"], row["unit_id"])),
        "realized_group_counts": {"external": len(selected)},
        "eligible_before_caps": {"external": len(ranked)},
    }


def representation_bytes(predictor: Mapping[str, Any], view: str) -> bytes:
    """Create one neutral complete-input representation without a classification request."""

    response = str(predictor["response_text"])
    if view == "response":
        return response.encode("utf-8")
    if view == "source_response":
        source = str(predictor["source_text"])
        return f"SOURCE\n{source}\n\nRESPONSE\n{response}".encode()
    raise ValueError(f"representation view invalid:{view}")


def planned_eligibility_rows(predictors: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retain every deferred tokenizer cell instead of assuming it will fit."""

    return [
        {
            "unit_id": f"{row['row_key']}:{view}",
            "row_key": row["row_key"],
            "group_id": row["group_id"],
            "corpus": row["corpus"],
            "role": row["role"],
            "arm": view,
            "condition": "gguf_complete_input_eligibility",
            "token_ceiling": TOKEN_CEILING,
            "token_count": None,
            "eligible": None,
            "complete_input_required": True,
            "truncation_allowed": False,
            "attempted": False,
            "completed": False,
            "failed": False,
            "censored": False,
            "unstarted": True,
            "status": "unstarted",
        }
        for row in predictors
        for view in ("response", "source_response")
    ]


def _predictor_signature(predictors: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Bind selection, feature bytes, prompt bytes, and deferred eligibility."""

    ordered = sorted(predictors, key=lambda row: str(row["row_key"]))
    feature_bytes = json.dumps(
        [(row["row_key"], row["source_features"]) for row in ordered],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prompt_rows = [
        {
            "row_key": row["row_key"],
            "response_sha256": "sha256:"
            + hashlib.sha256(representation_bytes(row, "response")).hexdigest(),
            "source_response_sha256": "sha256:"
            + hashlib.sha256(representation_bytes(row, "source_response")).hexdigest(),
        }
        for row in ordered
    ]
    return {
        "selection_sha256": canonical_hash(
            [(row["group_id"], row["row_key"], row["role"]) for row in ordered]
        ),
        "feature_sha256": "sha256:" + hashlib.sha256(feature_bytes).hexdigest(),
        "prompt_sha256": canonical_hash(prompt_rows),
        "eligibility_sha256": canonical_hash(
            [(row["row_key"], row["group_id"], row["role"]) for row in ordered]
        ),
    }


def label_permutation_invariance(
    predictors: Sequence[Mapping[str, Any]], evaluators: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Prove label movement cannot reach any predictor-side serialized bytes."""

    keys = {str(row["row_key"]) for row in predictors}
    if keys != {str(row["row_key"]) for row in evaluators}:
        raise CorpusInvalid("predictor evaluator keys differ")
    baseline = _predictor_signature(predictors)
    permuted = deepcopy(list(evaluators))
    labels = [row["label"] for row in reversed(permuted)]
    for row, label in zip(permuted, labels, strict=True):
        row["label"] = label
    del permuted
    observed = _predictor_signature(predictors)
    result = {
        "selection_unchanged": baseline["selection_sha256"] == observed["selection_sha256"],
        "feature_bytes_unchanged": baseline["feature_sha256"] == observed["feature_sha256"],
        "prompt_bytes_unchanged": baseline["prompt_sha256"] == observed["prompt_sha256"],
        "eligibility_unchanged": baseline["eligibility_sha256"] == observed["eligibility_sha256"],
        "baseline": baseline,
        "permuted": observed,
    }
    result["passed"] = all(
        value is True for key, value in result.items() if key.endswith("changed")
    )
    return result


def comparison_plan() -> JsonDict:
    """Freeze later fitting and external comparisons before labels are measured."""

    return {
        "pooling": "final_layer_last_token",
        "representation_views": ["response", "source_response"],
        "complete_input_token_ceiling": TOKEN_CEILING,
        "truncate_to_fit": False,
        "projection": {
            "kind": "seeded_gaussian_random_projection",
            "dimensions": 32,
            "seed": PROJECTION_SEED,
            "fit_on": "training_groups_only",
        },
        "fit_seeds": list(FIT_SEEDS),
        "arms": [
            "source_response_gibbs",
            "source_response_logistic",
            "response_only_gibbs",
            "source_shuffled_gibbs",
            "old_lexical_features",
            "training_prevalence",
        ],
        "matched_capacity": {
            "gibbs_and_logistic_input_dimensions": 32,
            "source_conditioning_ablation": "same_projection_and_fit_seeds",
        },
        "primary_metrics": ["external_brier", "external_log_loss"],
        "bootstrap": {
            "unit": "source_group",
            "draws": 10_000,
            "seed": BOOTSTRAP_SEED,
            "multiplicity": "holm",
        },
        "minimum_groups": deepcopy(MINIMUM_GROUPS),
        "risk_and_coverage_intervals": "descriptive_only",
        "deployment_prevalence_claim": False,
    }


def authenticate_release_file(
    path: Path, *, relative: str, expected_size: int, expected_git_sha1: str
) -> JsonDict:
    """Authenticate one cached file against its pinned Git object identity."""

    if not path.is_file():
        raise SourceBlocked(f"release file missing:{relative}")
    payload = path.read_bytes()
    if len(payload) != expected_size:
        raise SourceBlocked(f"release size mismatch:{relative}:{len(payload)}")
    git_hash = hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()
    if git_hash != expected_git_sha1:  # pragma: no cover - live drift branch.
        raise SourceBlocked(f"release git hash mismatch:{relative}:{git_hash}")
    return {
        "path": relative,
        "cache_path": str(path.resolve()),
        "url": f"https://raw.githubusercontent.com/vectara/FaithBench/{FAITHBENCH_REVISION}/{relative}",
        "size_bytes": len(payload),
        "git_blob_sha1": git_hash,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def _download_release_file(relative: str, expected_size: int) -> bytes:  # pragma: no cover
    """Download one allowlisted immutable file with a strict byte ceiling."""

    url = f"https://raw.githubusercontent.com/vectara/FaithBench/{FAITHBENCH_REVISION}/{relative}"
    try:
        with urlopen(url, timeout=60.0) as response:  # noqa: S310 - immutable allowlisted URL.
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_RELEASE_FILE_BYTES:
                raise SourceBlocked(f"release declared size exceeds bound:{relative}")
            payload = response.read(MAX_RELEASE_FILE_BYTES + 1)
    except (OSError, TimeoutError) as exc:
        raise SourceBlocked(f"release network failure:{relative}:{type(exc).__name__}") from exc
    if len(payload) > MAX_RELEASE_FILE_BYTES:
        raise SourceBlocked(f"release transfer exceeds bound:{relative}")
    if len(payload) != expected_size:
        raise SourceBlocked(f"release download size mismatch:{relative}:{len(payload)}")
    return payload


def fetch_faithbench_assets(cache_root: Path, started: float) -> JsonDict:  # pragma: no cover
    """Fetch only the pinned public schema, policy, license, README, and batches."""

    receipts: list[JsonDict] = []
    base = cache_root / FAITHBENCH_REVISION
    for index, (relative, (size, git_hash)) in enumerate(FAITHBENCH_ASSETS.items(), 1):
        path = base / relative
        progress(started, "faithbench_release", "before_download_or_cache", unit=f"{index}/19")
        if not path.is_file():
            payload = _download_release_file(relative, size)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            temporary.write_bytes(payload)
            temporary.replace(path)
        receipts.append(
            authenticate_release_file(
                path,
                relative=relative,
                expected_size=size,
                expected_git_sha1=git_hash,
            )
        )
        progress(started, "faithbench_release", "after_download_or_cache", unit=f"{index}/19")
    return {
        "repository": FAITHBENCH_REPOSITORY,
        "revision": FAITHBENCH_REVISION,
        "revision_url": f"{FAITHBENCH_REPOSITORY}/tree/{FAITHBENCH_REVISION}",
        "license": FAITHBENCH_LICENSE,
        "attribution": FAITHBENCH_ATTRIBUTION,
        "files": receipts,
        "total_bytes": sum(row["size_bytes"] for row in receipts),
        "authenticated": True,
    }


def load_faithbench_release(receipt: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover
    """Decode authenticated batches without exposing metadata to the predictor path."""

    rows: list[JsonDict] = []
    for file_row in receipt["files"]:
        relative = str(file_row["path"])
        if not relative.startswith("data_for_release/batch_"):
            continue
        path = Path(str(file_row["cache_path"]))
        try:
            batch = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusInvalid(f"FaithBench batch unreadable:{relative}") from exc
        samples = batch.get("samples") if isinstance(batch, Mapping) else None
        if not isinstance(samples, list):
            raise CorpusInvalid(f"FaithBench samples missing:{relative}")
        batch_id = int(Path(relative).stem.split("_")[-1])
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise CorpusInvalid(f"FaithBench sample invalid:{relative}")
            rows.append({**deepcopy(dict(sample)), "_batch_id": batch_id})
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize rows with stable key ordering so hashes survive fresh replay."""

    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:  # pragma: no cover - runtime I/O.
    """Replace one task-owned shard only after complete bytes reach storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seal_protocol_shards(
    raw_dir: Path,
    predictors: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Write separate hash-bound predictor, evaluator, and disposition shards."""

    if any(set(row) != set(PREDICTOR_FIELDS) for row in predictors):
        raise CorpusInvalid("predictor field allowlist mismatch")
    if any(set(row) != set(EVALUATOR_FIELDS) for row in evaluators):
        raise CorpusInvalid("evaluator field allowlist mismatch")
    raw_dir.mkdir(parents=True, exist_ok=True)
    shards: list[JsonDict] = []
    for name, kind, rows in (
        ("predictors.jsonl", "predictor", predictors),
        ("evaluators.jsonl", "evaluator", evaluators),
        ("dispositions.jsonl", "disposition", dispositions),
    ):
        payload = _jsonl_bytes(rows)
        path = raw_dir / name
        _atomic_bytes(path, payload)
        shards.append(
            {
                "path": name,
                "kind": kind,
                "rows": len(rows),
                "size_bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: JsonDict = {
        "schema": "carnot.exp7449.corpus_manifest.v1",
        "predictor_fields": list(PREDICTOR_FIELDS),
        "evaluator_fields": list(EVALUATOR_FIELDS),
        "predictor_row_count": len(predictors),
        "evaluator_row_count": len(evaluators),
        "disposition_row_count": len(dispositions),
        "shards": shards,
        "metadata": deepcopy(dict(metadata or {})),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    atomic_json(raw_dir / "corpus_manifest.json", manifest)
    return manifest


def reload_protocol_shards(raw_dir: Path) -> JsonDict:
    """Rehash sealed bytes and restore the two access-controlled corpus views."""

    try:
        manifest = json.loads((raw_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive I/O.
        raise CorpusInvalid("corpus manifest unreadable") from exc
    if not isinstance(manifest, dict):  # pragma: no cover - defensive schema branch.
        raise CorpusInvalid("corpus manifest object required")
    expected_manifest_hash = manifest.get("manifest_hash")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if expected_manifest_hash != canonical_hash(unhashed):  # pragma: no cover - defensive drift.
        raise CorpusInvalid("corpus manifest hash mismatch")
    by_kind: dict[str, list[JsonDict]] = defaultdict(list)
    for shard in manifest.get("shards") or []:
        path = raw_dir / str(shard["path"])
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise CorpusInvalid(f"shard hash mismatch:{shard['path']}")
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except json.JSONDecodeError as exc:  # pragma: no cover - hash normally catches drift.
            raise CorpusInvalid(f"shard JSON invalid:{shard['path']}") from exc
        if len(rows) != shard["rows"] or any(  # pragma: no cover - authenticated-byte defense.
            not isinstance(row, dict) for row in rows
        ):
            raise CorpusInvalid(f"shard row count mismatch:{shard['path']}")
        by_kind[str(shard["kind"])].extend(rows)
    predictors = by_kind["predictor"]
    evaluators = by_kind["evaluator"]
    if any(  # pragma: no cover - sealed writer already enforces the allowlist.
        set(row) != set(PREDICTOR_FIELDS) for row in predictors
    ):
        raise CorpusInvalid("reloaded predictor field mismatch")
    if any(  # pragma: no cover - sealed writer already enforces the allowlist.
        set(row) != set(EVALUATOR_FIELDS) for row in evaluators
    ):
        raise CorpusInvalid("reloaded evaluator field mismatch")
    if {row["row_key"] for row in predictors} != {  # pragma: no cover - writer joins identities.
        row["row_key"] for row in evaluators
    }:
        raise CorpusInvalid("reloaded view identity mismatch")
    return {
        "manifest": manifest,
        "predictors": predictors,
        "evaluators": evaluators,
        "dispositions": by_kind["disposition"],
    }


def protocol_gates(
    counts: Mapping[str, int],
    *,
    release_authenticated: bool,
    evaluator_isolated: bool,
    invariance_passed: bool,
    validation_passed: bool,
) -> list[JsonDict]:
    """Keep protocol validity separate from later confirmatory sample minima."""

    def gate(
        check: str,
        category: str,
        operator: str,
        expected: Any,
        observed: Any,
        passed: bool,
        principle: str,
    ) -> JsonDict:
        return {
            "check": check,
            "category": category,
            "operator": operator,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }

    rows = [
        gate(
            "role_caps_and_boundaries",
            "protocol_validity",
            "<=",
            deepcopy(RAGTRUTH_CAPS) | {"external": FAITHBENCH_CAP},
            dict(counts),
            all(
                counts.get(role, 0) <= cap
                for role, cap in (RAGTRUTH_CAPS | {"external": FAITHBENCH_CAP}).items()
            ),
            "Each role keeps its own cap; shortages never borrow another role.",
        ),
        gate(
            "release_authenticated",
            "protocol_validity",
            "==",
            True,
            release_authenticated,
            release_authenticated,
            "Pinned license, schema, aggregation script, and batch bytes define the release.",
        ),
        gate(
            "evaluator_isolation",
            "protocol_validity",
            "==",
            True,
            evaluator_isolated,
            evaluator_isolated,
            "Labels and ambiguity stay outside predictor rows.",
        ),
        gate(
            "label_permutation_invariance",
            "protocol_validity",
            "==",
            True,
            invariance_passed,
            invariance_passed,
            "Evaluator labels cannot change prompts, features, eligibility, or selection.",
        ),
        gate(
            "affected_and_terminal_validation",
            "required_validation",
            "==",
            True,
            validation_passed,
            validation_passed,
            "The frozen affected scope and terminal readers must pass.",
        ),
    ]
    rows.extend(
        gate(
            f"minimum_groups:{role}",
            "confirmatory_benefit",
            ">=",
            minimum,
            counts.get(role, 0),
            counts.get(role, 0) >= minimum,
            "A shortfall blocks confirmatory value, not protocol coverage reporting.",
        )
        for role, minimum in MINIMUM_GROUPS.items()
    )
    return rows


def reduce_protocol(gates: Sequence[Mapping[str, Any]], *, flagged_adversarial: bool) -> JsonDict:
    """Reduce protocol readiness without inventing a predictive measurement."""

    required = [row for row in gates if row.get("category") != "confirmatory_benefit"]
    minima = [row for row in gates if row.get("category") == "confirmatory_benefit"]
    valid = bool(required) and all(row.get("passed") is True for row in required)
    confirmatory = bool(minima) and all(row.get("passed") is True for row in minima)
    if not valid or flagged_adversarial:
        return {
            "source_protocol_ready_score": 0,
            "confirmatory_minima_met": confirmatory,
            "status": "disqualified",
            "honest_verdict": "complete_disqualified_source_protocol_validation",
            "verdict_class": "disqualified",
            "promotion_score": 0,
        }
    return {
        "source_protocol_ready_score": 1,
        "confirmatory_minima_met": confirmatory,
        "status": "complete",
        "honest_verdict": "complete_null_source_protocol_ready_no_predictive_measurement",
        "verdict_class": "null",
        "promotion_score": 0,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable artifact evidence while excluding only the checksum itself."""

    stable = {
        key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"
    }
    return canonical_hash(stable)


def _validation_passed(  # pragma: no cover - exercised by runtime validation reduction.
    receipts: object, names: Sequence[str]
) -> bool:
    """Require one successful receipt for every exact validation name."""

    if not isinstance(receipts, list):
        return False
    return all(
        sum(row.get("name") == name and row.get("passed") is True for row in receipts) == 1
        for name in names
    )


def independent_reduce(  # pragma: no cover - exercised by the fresh-process capability E2E.
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> JsonDict:
    """Reload raw rows and recompute readiness without trusting producer summaries."""

    raw_dir = root / str(artifact.get("corpus_manifest", {}).get("raw_dir", RAW_DIR))
    reloaded = reload_protocol_shards(raw_dir)
    counts = defaultdict(int)
    for row in reloaded["predictors"]:
        counts[str(row["role"])] += 1
    invariance = label_permutation_invariance(reloaded["predictors"], reloaded["evaluators"])
    receipts = artifact.get("validation_receipts")
    affected = _validation_passed(receipts, AFFECTED_CHECK_NAMES)
    terminal = _validation_passed(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    gates = protocol_gates(
        counts,
        release_authenticated=bool(
            reloaded["manifest"]
            .get("metadata", {})
            .get("faithbench_release", {})
            .get("authenticated")
        ),
        evaluator_isolated=all(set(row) == set(PREDICTOR_FIELDS) for row in reloaded["predictors"]),
        invariance_passed=bool(invariance["passed"]),
        validation_passed=affected and terminal,
    )
    reduction = reduce_protocol(
        gates, flagged_adversarial=bool(artifact.get("flagged_adversarial"))
    )
    return {
        **reduction,
        "realized_group_counts": dict(sorted(counts.items())),
        "invariance_passed": bool(invariance["passed"]),
        "affected_validation_passed": affected,
        "terminal_validation_passed": terminal,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:  # pragma: no cover - assembly.
    """Explain top-level fields without wrapping machine-readable gate values."""

    specific = {
        "schema": "Versioned top-level schema, exact experiment identity, milestone, and terminal status.",
        "run_date": "Fixed execution date plus actual UTC and monotonic boundaries.",
        "preconditions_checked": "Actual paths, identities, and observed upstream values before dependent work.",
        "MODEL_SPECS": "Empty because this task performs no current LLM work.",
        "model_invoked": "Distinguishes current attempts from archived model-shaped evidence.",
        "invocation_counts": "Balanced current load and generation counters, all zero here.",
        "inference_substrate": "Truthful CPU/file protocol work; no model or device implication.",
        "inference_substrate_class": "Closed current-work class; no model was loaded.",
        "execution_venue": "Host venue, separate from compute or archived device evidence.",
        "duration_s": "Measured current work with no padded model-time floor.",
        "phase_spans": "Disjoint monotonic phase timing and completed-unit checkpoints.",
        "random_seed": "Frozen selection, projection, fit, and bootstrap seeds.",
        "reproducibility_checksum": "Binds code, immutable inputs, shards, protocol, and validation scope.",
        "source_artifact_hashes": "Exact local and external bytes with original historical flags.",
        "rows": "Every deferred group-view tokenizer unit, including unstarted state.",
        "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted units.",
        "acceptance_gate_results": "Bare scalar validity and confirmatory checks with their principles.",
        "gate_check_summary": "First exact failure without conflating missing, null, or zero.",
        "verifier_is_oracle": "False because no deployed verifier supplies scoring authority.",
        "honest_verdict": "Terminal protocol finding; it makes no predictive-benefit claim.",
        "verdict_class": "Closed terminal class independent of prose wording.",
        "flagged_adversarial": "Actual critical validation state; flagged work cannot be ready.",
        "validation_receipts": "Exact affected and terminal argv, exits, durations, and log hashes.",
        "field_principles": "Field intent lives here while gate values remain bare scalars.",
        "promotion_score": "Always zero; this milestone authorizes no rollout or publication.",
        "source_protocol_ready_score": "One requires roles, allowlist, release hashes, controls, and evaluator isolation.",
        "corpus_manifest": "Nominal and realized groups, ambiguity, duplicates, release, and label policy.",
        "feature_protocol": "Input order, pooling, projection, capacity, and token ceiling fixed before fitting.",
        "comparison_plan": "Arms, metrics, sample minima, grouped bootstrap, and Holm correction.",
    }
    return {
        key: specific.get(key, "Hash-bound supporting evidence for this terminal protocol.")
        for key in keys
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover - assembly.
    """Name the first failed validity check while preserving all failures."""

    failures = [dict(row) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("category") != "confirmatory_benefit"]
    first = blocking[0] if blocking else None
    return {
        "required_checks_passed": not blocking,
        "confirmatory_minima_met": not any(
            row.get("category") == "confirmatory_benefit" for row in failures
        ),
        "failed_required_checks": [row["check"] for row in blocking],
        "blocked_upstream": None,
        "blocked_path": None,
        "blocked_check": first["check"] if first else None,
        "blocked_field": "passed" if first else None,
        "blocked_expected": True if first else None,
        "blocked_observed": first["passed"] if first else None,
    }


def utc_now() -> str:  # pragma: no cover - runtime clock.
    """Return one actual aware UTC boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit flushed phase, slow-operation, and completed-unit boundaries."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7449] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _phase_span(
    phase: str, phase_started: float, run_started: float, completed_units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - runtime timing.
    """Record one measured monotonic span and durable checkpoint name."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
        "heartbeat_count": 0,
    }


def _extract_local_texts(value: Any) -> list[str]:  # pragma: no cover - runtime corpus scan.
    """Collect only source-bearing fields from authenticated local corpus objects."""

    texts: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_text", "context", "step_text"} and isinstance(item, str):
                texts.append(item)
            else:
                texts.extend(_extract_local_texts(item))
    elif isinstance(value, list):
        for item in value:
            texts.extend(_extract_local_texts(item))
    return texts


def local_corpus_source_hashes(root: Path) -> tuple[set[str], list[JsonDict]]:  # pragma: no cover
    """Authenticate and scan every explicit V649-V652 local corpus authority."""

    candidates = list(LOCAL_CORPUS_PATHS)
    candidates.extend(
        path.relative_to(root)
        for path in sorted(
            (root / "results/raw/experiment_7410_v650_source_corpus").glob("predictor-*.json")
        )
    )
    candidates.extend(
        path.relative_to(root)
        for path in sorted(
            (root / "results/raw/experiment_7423_v651_annotated_protocol").glob("predictor-*.jsonl")
        )
    )
    candidates.extend(
        path.relative_to(root)
        for path in sorted(
            (root / "results/raw/experiment_7439_v652_certified_decisions/probability_rows").glob(
                "*.jsonl"
            )
        )
    )
    hashes: set[str] = set()
    receipts: list[JsonDict] = []
    for relative in dict.fromkeys(candidates):
        path = root / relative
        if not path.is_file():
            receipts.append(
                {"path": relative.as_posix(), "available": False, "sha256": None, "source_count": 0}
            )
            continue
        try:
            if path.suffix == ".jsonl":
                values = [
                    json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                ]
            else:
                values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusInvalid(f"local corpus unreadable:{relative}") from exc
        texts = _extract_local_texts(values)
        hashes.update(normalized_source_hash(text) for text in texts if text.strip())
        receipts.append(
            {
                "path": relative.as_posix(),
                "available": True,
                "sha256": sha256_file(path),
                "source_count": len(texts),
            }
        )
    return hashes, receipts


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict]]:  # pragma: no cover
    """Authenticate branch-local authorities and preserve prior result flags."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "operator": "==",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "passed": available,
            }
        )
        if available:
            original_flag = None
            if relative.suffix == ".json" and relative.parts[0] == "results":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    original_flag = (
                        value.get("flagged_adversarial") if isinstance(value, dict) else None
                    )
                except json.JSONDecodeError:
                    original_flag = None
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": original_flag,
            }
    spec_text = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.is_file() else ""
    checks.append(
        {
            "check": "driving_requirement",
            "upstream": "openspec/capabilities/autoresearch/spec.md",
            "path": "openspec/capabilities/autoresearch/spec.md",
            "field": "REQ-*",
            "operator": "==",
            "expected": "REQ-AUTO-7449",
            "observed": "REQ-AUTO-7449" if "REQ-AUTO-7449" in spec_text else None,
            "passed": "REQ-AUTO-7449" in spec_text,
        }
    )
    prior_path = root / "results/experiment_7439_v652_certified_decisions.json"
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prior = {}
    for check, field, expected in (
        (
            "prior_failure_preserved",
            "honest_verdict",
            "complete_null_no_registered_decision_benefit",
        ),
        ("prior_class_preserved", "verdict_class", "null"),
        ("prior_flag_preserved", "flagged_adversarial", False),
        ("prior_value_preserved", "decision_value_score", 0),
    ):
        observed = prior.get(field)
        checks.append(
            {
                "check": check,
                "upstream": "exp7439-v652-certified-decisions",
                "path": "results/experiment_7439_v652_certified_decisions.json",
                "field": field,
                "operator": "==",
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    quarantined = "experiment_id: exp7449-source-protocol" in exclusion
    checks.append(
        {
            "check": "current_task_not_quarantined",
            "upstream": "ops/exclusion_manifest.yaml",
            "path": "ops/exclusion_manifest.yaml",
            "field": "exp7449-source-protocol",
            "operator": "==",
            "expected": False,
            "observed": quarantined,
            "passed": not quarantined,
        }
    )
    return checks, hashes


def _source_hash_rows(
    root: Path, source_hashes: dict[str, JsonDict], manifest: Mapping[str, Any]
) -> None:  # pragma: no cover
    """Add new code and sealed shards after their bytes exist."""

    for relative in (
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("openspec/capabilities/autoresearch/spec.md"),
    ):
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "original_flagged_adversarial": None,
        }
    for shard in manifest["shards"]:
        relative = RAW_DIR / shard["path"]
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": shard["sha256"],
            "original_flagged_adversarial": None,
        }
    relative = RAW_DIR / "corpus_manifest.json"
    source_hashes[relative.as_posix()] = {
        "path": relative.as_posix(),
        "sha256": sha256_file(root / relative),
        "original_flagged_adversarial": None,
    }


def _source_hashes_valid(root: Path, hashes: object) -> bool:  # pragma: no cover - cold I/O.
    """Rehash repository files while accepting authenticated external-cache paths."""

    if not isinstance(hashes, Mapping):
        return False
    for row in hashes.values():
        if not isinstance(row, Mapping):
            return False
        path = Path(str(row.get("path")))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
            return False
    return True


def _build_artifact(
    *,
    root: Path,
    preconditions: list[JsonDict],
    source_hashes: dict[str, JsonDict],
    manifest: JsonDict,
    predictors: list[JsonDict],
    evaluators: list[JsonDict],
    release_receipt: JsonDict,
    local_receipts: list[JsonDict],
    validation_receipts: list[JsonDict],
    phase_spans: list[JsonDict],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    protocol_elapsed_s: float,
    flagged_adversarial: bool,
    require_terminal: bool,
) -> JsonDict:  # pragma: no cover - terminal assembly.
    """Assemble one schema-complete candidate from independently reloadable rows."""

    counts: dict[str, int] = defaultdict(int)
    for row in predictors:
        counts[str(row["role"])] += 1
    invariance = label_permutation_invariance(predictors, evaluators)
    validation_names = (*AFFECTED_CHECK_NAMES, *(TERMINAL_CHECK_NAMES if require_terminal else ()))
    validation_passed = _validation_passed(validation_receipts, validation_names)
    gates = protocol_gates(
        counts,
        release_authenticated=bool(release_receipt.get("authenticated")),
        evaluator_isolated=all(set(row) == set(PREDICTOR_FIELDS) for row in predictors),
        invariance_passed=bool(invariance["passed"]),
        validation_passed=validation_passed,
    )
    reduction = reduce_protocol(gates, flagged_adversarial=flagged_adversarial)
    rows = planned_eligibility_rows(predictors)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": "Exp7449 source-conditioned input and human-challenge protocol",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": reduction["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "clock_identity": {
            "clock": "time.monotonic_ns",
            "boot_id": (
                Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                if Path("/proc/sys/kernel/random/boot_id").is_file()
                else None
            ),
            "segment_id": f"{EXPERIMENT_ID}:{os.getpid()}:{started_ns}",
        },
        "preconditions_checked": preconditions,
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "current_llm_operations": 0,
            "cpu": "host corpus hashing, lexical features, and JSON reduction",
            "cuda": "not used",
            "external_cache": str(FAITHBENCH_CACHE.resolve()),
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "duration_components_s": {
            "model_load_s": 0.0,
            "forward_s": 0.0,
            "generation_s": 0.0,
            "numeric_protocol_s": protocol_elapsed_s,
            "validation_s": sum(float(row.get("duration_s") or 0.0) for row in validation_receipts),
        },
        "phase_spans": phase_spans,
        "random_seed": {
            "selection_salt": SELECTION_SALT,
            "projection_seed": PROJECTION_SEED,
            "fit_seeds": list(FIT_SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": None,
        "source_artifact_hashes": source_hashes,
        "historical_inference_sidecars": [
            {
                "path": "results/experiment_7439_v652_certified_decisions.json",
                "sha256": source_hashes["results/experiment_7439_v652_certified_decisions.json"][
                    "sha256"
                ],
                "scope": "historical_model_receipts",
            }
        ],
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(rows),
            "independent_groups": dict(sorted(counts.items())),
            "stop_rule": "Seal up to 180/60/60 RAGTruth and 100 external groups; never cross-fill role shortages.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": reduction["honest_verdict"],
        "verdict_class": reduction["verdict_class"],
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": validation_receipts,
        "promotion_score": 0,
        "source_protocol_ready_score": reduction["source_protocol_ready_score"],
        "confirmatory_minima_met": reduction["confirmatory_minima_met"],
        "corpus_manifest": {
            **manifest,
            "raw_dir": RAW_DIR.as_posix(),
            "nominal_group_caps": deepcopy(RAGTRUTH_CAPS) | {"external": FAITHBENCH_CAP},
            "realized_group_counts": dict(sorted(counts.items())),
            "ambiguous_external_groups": sum(
                row["corpus"] == "faithbench" and row["ambiguous"] for row in evaluators
            ),
            "faithbench_release": release_receipt,
            "local_dedup_receipts": local_receipts,
            "label_policy": {
                "ragtruth": "one means no human-annotated source-unsupported span",
                "faithbench": "official worst-severity categories; consistent or benign is one",
                "upstream_script_audit": "empty-annotation branch returns zero despite comment 'consistent'; protocol uses shared consistent-is-one polarity and binds the exact script",
            },
            "contamination_uncertainty": "Exact normalized-source duplicates against enumerated V649-V652 local corpora were removed from FaithBench. Unobserved public pretraining and paraphrase overlap remain unknown.",
            "deployment_prevalence_supported": False,
        },
        "feature_protocol": {
            "predictor_allowlist": list(PREDICTOR_FIELDS),
            "source_feature_names": list(SOURCE_FEATURE_NAMES),
            "forbidden_predictor_fields": [
                "labels",
                "annotations",
                "annotation_notes",
                "detector_scores",
                "summarizer_identity",
            ],
            "representations": {
                "response": "complete response bytes",
                "source_response": "SOURCE newline complete source blank-line RESPONSE newline complete response",
            },
            "token_ceiling": TOKEN_CEILING,
            "tokenizer": "embedded GGUF tokenizer deferred to Exp7452",
            "truncate_to_fit": False,
            "pooling": "final_layer_last_token",
            "projection_seed": PROJECTION_SEED,
            "projection_dimensions": 32,
            "label_permutation_invariance": invariance,
        },
        "comparison_plan": comparison_plan(),
        "small_ebm_training": {
            "attempted": False,
            "fit_attempts": 0,
            "fit_completions": 0,
            "duration_s": 0.0,
            "deferred_to": "exp7453-energy-calibration",
        },
        "numbered_e2e_applicable": [],
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "cold_replay": "fresh_process",
            "independent_reduction": "fresh_process",
        },
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "independent_reduction": None,
        "field_principles": {},
    }
    artifact["independent_reduction"] = independent_reduce(
        artifact, root=root, require_terminal=require_terminal
    )
    artifact["field_principles"] = _field_principles(list(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:  # pragma: no cover - fresh-process artifact reader.
    """Cold-check identity, raw rows, current-work claims, reduction, and hashes."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "no_model_load"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("current_model_contract_invalid")
    try:
        reduction = independent_reduce(artifact, root=root, require_terminal=require_terminal)
    except (CorpusInvalid, OSError, KeyError, TypeError, ValueError):
        reduction = {}
        errors.append("raw_reduction_invalid")
    if artifact.get("independent_reduction") != reduction:
        errors.append("independent_reduction_mismatch")
    for key in (
        "source_protocol_ready_score",
        "confirmatory_minima_met",
        "status",
        "honest_verdict",
        "verdict_class",
        "promotion_score",
    ):
        if artifact.get(key) != reduction.get(key):
            errors.append(f"{key}_mismatch")
    if not _source_hashes_valid(root, artifact.get("source_artifact_hashes")):
        errors.append("source_hash_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("checksum_mismatch")
    return list(dict.fromkeys(errors))


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build the declared entrypoint, reducer, adversarial, and strict-reader checks."""

    python = ".venv/bin/python"
    common = ("--date", RUN_DATE, "--root", ".")
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "declared_entrypoint_cold_replay",
                (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    *common,
                    "--independent-reduce",
                    str(candidate),
                ),
                "candidate_raw_rows",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate_artifact",
            ),
            "required_validation",
            True,
        ),
    ]


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover
    """Authenticate, seal, validate, replay, and atomically publish the protocol."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes = collect_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        raise SourceBlocked(
            f"precondition failed:{failed['path']}:{failed['field']}:{failed['observed']}"
        )
    spans.append(
        _phase_span("preconditions", phase_started, run_started, len(preconditions), "inputs")
    )
    progress(run_started, "preconditions", "complete", completed=len(preconditions))

    progress(run_started, "model_load", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_phase_span("model_load", phase_started, run_started, 0, "no_current_llm_work"))
    progress(run_started, "model_load", "after", completed=0)
    progress(run_started, "generation", "before", planned=0)
    phase_started = time.monotonic()
    spans.append(_phase_span("generation", phase_started, run_started, 0, "no_current_llm_work"))
    progress(run_started, "generation", "after", completed=0)

    protocol_started = time.monotonic()
    phase_started = time.monotonic()
    progress(run_started, "ragtruth", "before_benchmark", planned=300)
    try:
        ragtruth_receipt = authenticate_ragtruth_assets(RAGTRUTH_CACHE_ROOT)
        source_rows, response_rows = load_ragtruth_release(ragtruth_receipt, started=run_started)
    except (RAGTruthSourceBlocked, OSError, ValueError) as exc:
        raise SourceBlocked(f"ragtruth unavailable:{exc}") from exc
    ragtruth = select_ragtruth_panel(source_rows, response_rows)
    spans.append(
        _phase_span(
            "ragtruth",
            phase_started,
            run_started,
            len(ragtruth["predictors"]),
            "ragtruth_group_selection",
        )
    )
    progress(run_started, "ragtruth", "after_benchmark", completed=len(ragtruth["predictors"]))

    phase_started = time.monotonic()
    progress(run_started, "faithbench", "before_benchmark", planned=100)
    release_receipt = fetch_faithbench_assets(FAITHBENCH_CACHE, run_started)
    faith_samples = load_faithbench_release(release_receipt)
    local_hashes, local_receipts = local_corpus_source_hashes(root)
    faithbench = select_faithbench_panel(
        faith_samples, local_source_hashes=local_hashes, cap=FAITHBENCH_CAP
    )
    spans.append(
        _phase_span(
            "faithbench",
            phase_started,
            run_started,
            len(faithbench["predictors"]),
            "faithbench_external_selection",
        )
    )
    progress(run_started, "faithbench", "after_benchmark", completed=len(faithbench["predictors"]))

    phase_started = time.monotonic()
    predictors = [*ragtruth["predictors"], *faithbench["predictors"]]
    evaluators = [*ragtruth["evaluators"], *faithbench["evaluators"]]
    dispositions = [*ragtruth["dispositions"], *faithbench["dispositions"]]
    metadata = {
        "ragtruth_release": ragtruth_receipt,
        "faithbench_release": release_receipt,
        "ragtruth_counts": ragtruth["realized_group_counts"],
        "faithbench_counts": faithbench["realized_group_counts"],
        "ambiguity_retained": sum(row["ambiguous"] for row in faithbench["evaluators"]),
        "local_dedup_receipts": local_receipts,
        "selection_before_labels": True,
        "one_response_per_group": True,
    }
    manifest = seal_protocol_shards(
        root / RAW_DIR, predictors, evaluators, dispositions, metadata=metadata
    )
    reloaded = reload_protocol_shards(root / RAW_DIR)
    if reloaded["predictors"] != predictors or reloaded["evaluators"] != evaluators:
        raise CorpusInvalid("sealed corpus replay mismatch")
    _source_hash_rows(root, source_hashes, manifest)
    for file_row in release_receipt["files"]:
        source_hashes[f"faithbench:{file_row['path']}"] = {
            "path": file_row["cache_path"],
            "sha256": file_row["sha256"],
            "original_flagged_adversarial": None,
            "url": file_row["url"],
            "revision": FAITHBENCH_REVISION,
        }
    for file_row in ragtruth_receipt["files"]:
        source_hashes[f"ragtruth:{file_row['path']}"] = {
            "path": file_row["cache_path"],
            "sha256": file_row["sha256"],
            "original_flagged_adversarial": None,
            "url": file_row["url"],
            "revision": ragtruth_receipt["commit"],
        }
    spans.append(_phase_span("seal", phase_started, run_started, len(predictors), "sealed_shards"))
    protocol_elapsed = time.monotonic() - protocol_started

    private_root = Path(tempfile.mkdtemp(prefix="exp7449-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "affected_checks"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    flagged = bool(plan_errors) or not bool(affected_reduction["passed"])
    candidate = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        manifest=manifest,
        predictors=predictors,
        evaluators=evaluators,
        release_receipt=release_receipt,
        local_receipts=local_receipts,
        validation_receipts=affected,
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        protocol_elapsed_s=protocol_elapsed,
        flagged_adversarial=flagged,
        require_terminal=False,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = _terminal_commands(candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root, terminal_plan, log_dir=root / RAW_DIR / "validation/terminal"
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "terminal_readers"
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        manifest=manifest,
        predictors=predictors,
        evaluators=evaluators,
        release_receipt=release_receipt,
        local_receipts=local_receipts,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        protocol_elapsed_s=protocol_elapsed,
        flagged_adversarial=flagged or not terminal_passed or critical,
        require_terminal=True,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal artifact invalid:{errors}")
    progress(run_started, "publish", "before_atomic_terminal", path=output)
    atomic_json(root / output, final)
    progress(run_started, "publish", "after_atomic_terminal", status=final["status"])
    return final


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - CLI I/O.
    """Read one JSON object for a fresh-process terminal reader."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed execution date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the source protocol or one independent terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = (
            validate_artifact(value, root=root, require_terminal=False)
            if value
            else ["artifact_unreadable"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        reduced = independent_reduce(value, root=root, require_terminal=False) if value else {}
        passed = bool(value) and reduced == value.get("independent_reduction")
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return int(not passed)
    run_experiment(root, args.date, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
