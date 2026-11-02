"""Freeze sentence labels on exposed source families (REQ-REPORT-7740)."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

from carnot.experiment_7727_v673_development_corpus import COUNTS, validate_public
from carnot.experiment_7730_v673_set_energy_fit import FEATURE_DICTIONARY, _advisory
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import source_alignment as alignment
from carnot.verify import source_set_energy as energy

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7740_v674_sentence_label_protocol"
ROLES = tuple(COUNTS)
FEATURE_KEYS = frozenset(
    {
        "family_id",
        "role",
        "source_sha256",
        "response_sha256",
        "abstention",
        "sentence_count",
        "window_count",
        "group_ids",
        "pair_features",
        "local_pair_features",
        "pooled_features",
        "advisory_features",
    }
)
PRINCIPLE = "Completion and scientific benefit are different facts."


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Show real work so a long corpus pass never looks stalled."""
    print(
        f"[exp7740] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def digest(value: bytes) -> str:
    """Bind exact bytes instead of paths that can later change."""
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write complete rows with a stable byte encoding for cold replay."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256_file(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a role shard only after its manifest hash has been checked."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _check(name: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep both operands so an upstream block can be diagnosed exactly."""
    return {
        "check": name,
        "upstream_id": "exp7727",
        "artifact_path": str(path),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def authenticate(
    manifest_path: Path, expected_counts: dict[str, int]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Authenticate all shards without opening held-out target values."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "carnot.exp7727.development_manifest.v1"
        or manifest.get("counts") != expected_counts
        or manifest.get("initially_open_evaluator_roles") != ["fit"]
    ):
        raise ValueError("development_manifest_contract")
    rows: list[dict[str, Any]] = []
    checks = [
        _check(
            "development_manifest",
            manifest_path,
            "schema",
            "carnot.exp7727.development_manifest.v1",
            manifest["schema"],
        )
    ]
    families: set[str] = set()
    sources: set[str] = set()
    responses: set[str] = set()
    for role, count in expected_counts.items():
        meta = manifest["roles"][role]
        if meta["count"] != count or len(meta["families"]) != count:
            raise ValueError("role_count_or_roster")
        for kind in ("public", "evaluator"):
            path = manifest_path.parent / meta[f"{kind}_path"]
            observed = sha256_file(path) if path.is_file() else None
            expected = meta[f"{kind}_sha256"]
            checks.append(_check("role_shard", path, f"{kind}_sha256", expected, observed))
            if observed != expected:
                raise ValueError(f"{kind}_sha256")
        public = _read_jsonl(manifest_path.parent / meta["public_path"])
        if len(public) != count or [row["family_id"] for row in public] != meta["families"]:
            raise ValueError("family_roster")
        for row in public:
            validate_public(row, role)
            if (
                row["family_id"] in families
                or row["source_sha256"] in sources
                or row["response_sha256"] in responses
            ):
                raise ValueError("duplicate_family_or_source")
            families.add(row["family_id"])
            sources.add(row["source_sha256"])
            responses.add(row["response_sha256"])
            rows.append(row)
    return rows, manifest, checks


def map_targets(answer: bytes, annotations: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Map half-open Unicode offsets to the unchanged complete UTF-8 sentences."""
    text = answer.decode("utf-8", "strict")
    units = alignment.sentence_spans(answer)
    offsets = []
    position = 0
    for unit in units:
        end = position + len(unit.decode("utf-8", "strict"))
        offsets.append([position, end])
        position = end
    if position != len(text):
        raise ValueError("sentence_partition")
    if annotations is None:
        return {
            "targets": [None] * len(units),
            "reason": "missing_annotation_custody",
            "char_offsets": offsets,
        }
    if not units:
        return {"targets": [], "reason": "empty_answer", "char_offsets": offsets}
    targets = [0] * len(units)
    unmappable = False
    for item in annotations:
        start, end = item.get("start"), item.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
            or end > len(text)
            or text[start:end] != item.get("text")
        ):
            raise ValueError("annotation_offset_or_text")
        if item.get("implicit_true"):
            continue
        matched = False
        for index, (left, right) in enumerate(offsets):
            if start < right and end > left:
                targets[index] = 1
                matched = True
        if not matched:
            unmappable = True
    if unmappable:
        return {
            "targets": [None] * len(units),
            "reason": "unmappable_annotation",
            "char_offsets": offsets,
        }
    return {"targets": targets, "reason": "mapped", "char_offsets": offsets}


def _feature_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build the same public source and answer input for every paired arm."""
    view = energy.prepare(row["complete_source"].encode(), row["complete_response"].encode())
    local = (
        [
            [alignment.pair_features(window, [unit]) for window in [*view["windows"], b""]]
            for unit in view["answer_units"]
        ]
        if not view["abstention"]
        else []
    )
    return {
        "family_id": row["family_id"],
        "role": row["role"],
        "source_sha256": row["source_sha256"],
        "response_sha256": row["response_sha256"],
        "abstention": view["abstention"],
        "sentence_count": len(view["answer_units"]),
        "window_count": len(view["windows"]),
        "group_ids": view["group_ids"],
        "pair_features": view["pair_features"],
        "local_pair_features": local,
        "pooled_features": alignment.pooled_features(view) if not view["abstention"] else None,
        "advisory_features": _advisory(view),
    }


def _protocol(
    manifest_path: Path, manifest: dict[str, Any], scope: dict[str, Any]
) -> dict[str, Any]:
    """Freeze analysis before any target shard is decoded."""
    return {
        "schema": "carnot.exp7740.sentence_protocol.v1",
        "development_manifest_path": str(manifest_path),
        "development_manifest_sha256": sha256_file(manifest_path),
        "role_counts": manifest["counts"],
        "role_hashes": {
            role: {key: value for key, value in meta.items() if key.endswith("_sha256")}
            for role, meta in manifest["roles"].items()
        },
        "initial_fit_roles": ["fit"],
        "initially_closed_roles": [role for role in manifest["counts"] if role != "fit"],
        "response_label": "1 iff any annotation has implicit_true=false; 0 otherwise",
        "sentence_label": "1 iff eligible unsupported half-open character span overlaps the complete sentence; 0 means no eligible overlap, not truth; unknown on missing custody or unmappable span",
        "offset_semantics": "Unicode code points, half-open [start,end), exact text equality",
        "source_window_selection": "all Exp7728 source windows in original order; source-only duplicate-group prior; no annotation-based selection",
        "input_caps": {
            "source_windows": alignment.MAX_WINDOWS,
            "answer_units": alignment.MAX_ANSWER_UNITS,
        },
        "feature_dim": alignment.FEATURE_DIM,
        "advisory_dictionary": list(FEATURE_DICTIONARY),
        "arms": [
            {
                "name": "local_set_energy",
                "family": "set_energy",
                "loss": "response_nll_plus_known_sentence_mean_nll",
            },
            {"name": "response_set_energy", "family": "set_energy", "loss": "response_nll"},
            {
                "name": "local_pooled_mlp",
                "family": "pooled_mlp",
                "loss": "response_nll_plus_known_sentence_mean_nll",
            },
            {"name": "response_pooled_mlp", "family": "pooled_mlp", "loss": "response_nll"},
            {"name": "pooled_logistic", "family": "pooled_logistic", "loss": "response_nll"},
            {"name": "source_erased", "family": "set_energy", "loss": "response_nll"},
            {"name": "complete_static", "family": "dictionary", "loss": "response_nll"},
        ],
        "paired_budget": {
            "fit_families": manifest["counts"]["fit"],
            "family_weight": 1,
            "head_parameter_cap": energy.MAX_PARAMETERS,
            "set_energy_parameters": 266,
            "pooled_mlp_parameters": energy.MLP_PARAMETERS,
            "epochs": 12,
            "seeds": [67401, 67402, 67403, 67404, 67405],
            "learning_rates": [0.01, 0.05],
            "ridges": [0.0, 0.001],
        },
        "pooled_mlp_sentence_head": "Both pooled MLP arms use the same shared 132-input/16-hidden/two-output head on each sentence's prior-weighted source-window features; response unsupported is one minus the product of sentence support. Only the auxiliary loss differs.",
        "auxiliary_loss": {
            "formula": "response_NLL + 1 * mean(NLL(known_sentence_targets))",
            "coefficient": 1,
            "unknown_sentence_contribution": 0,
            "primary_denominator": "all non-abstaining responses",
            "local_denominator": "known sentence targets",
        },
        "abstention_action": "escalate",
        "claim_scope": "development_only",
        "fresh_generalization_eligible": False,
        "affected_scope": scope,
    }


def capture(
    manifest_path: Path,
    raw: Path,
    public: list[dict[str, Any]],
    manifest: dict[str, Any],
    start: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seal features first, then map evaluator annotations into private roles."""
    features = []
    last = time.monotonic()
    for index, row in enumerate(public, 1):
        features.append(_feature_row(row))
        if time.monotonic() - last >= 30:
            progress(start, "features", "heartbeat", index)
            last = time.monotonic()
    features_hash = _jsonl(raw / "features.jsonl", features)
    if any(set(row) != FEATURE_KEYS for row in features):
        raise ValueError("label_bearing_feature_row")
    progress(start, "features", "complete", len(features))
    features_by_family = {row["family_id"]: row for row in features}
    by_role = {role: [row for row in public if row["role"] == role] for role in manifest["counts"]}
    coverage = []
    target_hashes = {}
    for role in manifest["counts"]:
        path = manifest_path.parent / manifest["roles"][role]["evaluator_path"]
        evaluator = _read_jsonl(path)
        if len(evaluator) != len(by_role[role]):
            raise ValueError("evaluator_count")
        targets = []
        for row, label in zip(by_role[role], evaluator, strict=True):
            if (
                set(label) != {"family_id", "response_id", "annotations", "label"}
                or label["family_id"] != row["family_id"]
                or label["response_id"] != row["response_id"]
            ):
                raise ValueError("evaluator_join")
            mapped = map_targets(row["complete_response"].encode(), label["annotations"])
            expected = int(any(not bool(a.get("implicit_true")) for a in label["annotations"]))
            if label["label"] != expected:
                raise ValueError("response_label_mapping")
            targets.append(
                {
                    "family_id": row["family_id"],
                    "role": role,
                    "response_label": expected,
                    "sentence_targets": mapped["targets"],
                    "char_offsets": mapped["char_offsets"],
                    "reason": mapped["reason"],
                    "response_sha256": row["response_sha256"],
                }
            )
            coverage.append(
                {
                    "family_id": row["family_id"],
                    "role": role,
                    "mapping_eligible": mapped["reason"] == "mapped",
                    "reason": mapped["reason"],
                    "sentence_count": len(mapped["targets"]),
                    "known_sentence_count": sum(value is not None for value in mapped["targets"]),
                    "abstention": features_by_family[row["family_id"]]["abstention"],
                }
            )
        target_path = raw / f"{role}_targets.jsonl"
        target_hashes[role] = _jsonl(target_path, targets)
        os.chmod(target_path, 0o600)
        progress(start, "targets", role, len(coverage))
    coverage_hash = _jsonl(raw / "annotation_coverage.jsonl", coverage)
    evidence = {
        "features_sha256": features_hash,
        "target_sha256": target_hashes,
        "annotation_coverage_sha256": coverage_hash,
        "input_manifest_sha256": sha256_file(manifest_path),
        "role_counts": manifest["counts"],
    }
    atomic_json(raw / "evidence_manifest.json", evidence)
    return coverage, evidence


def cold_reduce(raw: Path, candidate_path: Path) -> dict[str, Any]:
    """Reopen exact inputs and recompute every evaluator target in a fresh process."""
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    protocol_path = raw / "sentence_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256_file(protocol_path) != candidate["sentence_protocol_manifest_path"]["sha256"]:
        raise ValueError("protocol_sha256")
    manifest_path = Path(protocol["development_manifest_path"])
    public, manifest, _ = authenticate(manifest_path, protocol["role_counts"])
    evidence = json.loads((raw / "evidence_manifest.json").read_text(encoding="utf-8"))
    if (
        sha256_file(manifest_path) != evidence["input_manifest_sha256"]
        or sha256_file(raw / "features.jsonl") != evidence["features_sha256"]
    ):
        raise ValueError("features_sha256_or_input_manifest")
    if sha256_file(raw / "annotation_coverage.jsonl") != evidence["annotation_coverage_sha256"]:
        raise ValueError("coverage_hash_or_count")
    features = _read_jsonl(raw / "features.jsonl")
    coverage = _read_jsonl(raw / "annotation_coverage.jsonl")
    if len(features) != len(public) or len(coverage) != len(public):
        raise ValueError("coverage_hash_or_count")
    feature_by_family = {row["family_id"]: row for row in features}
    coverage_by_family = {row["family_id"]: row for row in coverage}
    if len(feature_by_family) != len(public) or len(coverage_by_family) != len(public):
        raise ValueError("duplicate_feature_or_coverage")
    for row in public:
        feature = feature_by_family[row["family_id"]]
        if (
            set(feature) != FEATURE_KEYS
            or feature["source_sha256"] != row["source_sha256"]
            or feature["response_sha256"] != row["response_sha256"]
            or feature["role"] != row["role"]
        ):
            raise ValueError("label_bearing_feature_or_hash")
    for role in manifest["counts"]:
        target_path = raw / f"{role}_targets.jsonl"
        if sha256_file(target_path) != evidence["target_sha256"][role]:
            raise ValueError("target_sha256")
        targets = _read_jsonl(target_path)
        evaluators = _read_jsonl(manifest_path.parent / manifest["roles"][role]["evaluator_path"])
        members = [row for row in public if row["role"] == role]
        if len(targets) != len(members):
            raise ValueError("target_count")
        for row, label, target in zip(members, evaluators, targets, strict=True):
            mapped = map_targets(row["complete_response"].encode(), label["annotations"])
            response_label = int(
                any(not bool(a.get("implicit_true")) for a in label["annotations"])
            )
            if target != {
                "family_id": row["family_id"],
                "role": role,
                "response_label": response_label,
                "sentence_targets": mapped["targets"],
                "char_offsets": mapped["char_offsets"],
                "reason": mapped["reason"],
                "response_sha256": row["response_sha256"],
            }:
                raise ValueError("target_mapping_mismatch")
            report = coverage_by_family[row["family_id"]]
            if (
                report["role"] != role
                or report["reason"] != mapped["reason"]
                or report["known_sentence_count"]
                != sum(value is not None for value in mapped["targets"])
            ):
                raise ValueError("coverage_summary_mismatch")
    if (
        candidate["annotation_coverage_rows"] != coverage
        or candidate["sample_size_budget"]["completed"] != len(public)
        or candidate["source_artifact_hashes"]["pre_gate_receipts"]["evidence_manifest_sha256"]
        != sha256_file(raw / "evidence_manifest.json")
    ):
        raise ValueError("candidate_summary_mismatch")
    return {
        "families": len(public),
        "role_counts": dict(Counter(row["role"] for row in public)),
        "targets_recomputed": True,
    }


def _span(
    name: str, start: float, end: float, origin: float, date: str, units: int, checkpoint: Path
) -> dict[str, Any]:
    """Record disjoint monotonic work and its sealed checkpoint."""
    return {
        "phase": name,
        "start_s": start - origin,
        "end_s": end - origin,
        "duration_s": end - start,
        "run_date": date,
        "heartbeat_times_s": [end - origin],
        "completed_units": units,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
    }


def _artifact(
    date: str,
    raw: Path,
    manifest_path: Path,
    public: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    evidence: dict[str, Any],
    scope: dict[str, Any],
    spans: list[dict[str, Any]],
    fixture: bool,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep protocol completion separate from unmeasured scientific benefit."""
    passed = all(item["passed"] for item in receipts)
    verdict = "circular_positive" if fixture else "null" if passed else "disqualified"
    coverage_by_family = {row["family_id"]: row for row in coverage}
    rows = [
        {
            "family_id": row["family_id"],
            "role": row["role"],
            "source_sha256": row["source_sha256"],
            "response_sha256": row["response_sha256"],
            "raw_metrics": {
                "sentence_count": report["sentence_count"],
                "known_sentence_count": report["known_sentence_count"],
            },
            "denominator": 1,
            "excluded": False,
            "censored": report["abstention"] is not None,
            "exclusion_reason": None,
            "censor_reason": report["abstention"],
        }
        for row in public
        for report in [coverage_by_family[row["family_id"]]]
    ]
    hashes = {
        "eligible_producers": {
            str(manifest_path): sha256_file(manifest_path),
            **{
                str(manifest_path.parent / meta[f"{kind}_path"]): meta[f"{kind}_sha256"]
                for meta in json.loads(manifest_path.read_text())["roles"].values()
                for kind in ("public", "evaluator")
            },
        },
        "historical_disqualified_sources": {
            "results/experiment_7730_v673_set_energy_fit.json": sha256_file(
                ROOT / "results/experiment_7730_v673_set_energy_fit.json"
            )
            if (ROOT / "results/experiment_7730_v673_set_energy_fit.json").is_file()
            else None
        },
        "missing_inputs": [],
        "pre_gate_receipts": {
            "evidence_manifest_sha256": sha256_file(raw / "evidence_manifest.json"),
            "protocol_sha256": sha256_file(raw / "sentence_protocol_manifest.json"),
        },
    }
    value: dict[str, Any] = {
        "experiment_id": "exp7740-sentence-label-protocol",
        "milestone": "2026.09.674",
        "run_date": date,
        "honest_verdict": f"complete_{verdict}_sentence_label_protocol",
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": [],
        "acceptance_gate_results": {
            "validity": passed,
            "readiness": None,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "rows": rows,
        "annotation_coverage_rows": coverage,
        "sample_size_budget": {
            "intended": len(public),
            "started": len(public),
            "completed": len(public),
            "eligible": sum(not row["censored"] for row in rows),
            "excluded": 0,
            "censored": sum(row["censored"] for row in rows),
            "effective_independent_N": len(public),
            "role_counts": dict(Counter(row["role"] for row in public)),
            "sentences_do_not_increase_N": True,
        },
        "claim_scope": {
            "value": "fixture_only" if fixture else "development_only",
            "fresh_generalization_eligible": False,
            "adapter_withheld_public": None,
        },
        "inference_substrate": "cpu_jax_finite_energy_protocol_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "model_invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {"host": platform.node(), "pid": os.getpid(), "gpu_uuid": None},
        "phase_spans": spans,
        "random_seed": {
            "cohort_selection": "Exp7727 frozen salt",
            "planned_fit_initialization": [67401, 67402, 67403, 67404, 67405],
        },
        "reproducibility_checksum": digest(
            json.dumps(
                {
                    "inputs": hashes["eligible_producers"],
                    "protocol": hashes["pre_gate_receipts"]["protocol_sha256"],
                    "reducer": sha256_file(Path(__file__)),
                },
                sort_keys=True,
            ).encode()
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks
        + [
            {
                "check": "host_resources",
                "repo_root": str(ROOT),
                "cpu_count": os.cpu_count(),
                "effective_coding_backend": os.environ.get("CODEX_MODEL", "gpt-6 (session)"),
            }
        ],
        "validation_receipts": {
            "affected_scope": scope,
            "commands": receipts,
            "cold_replay": None,
            "terminal_checks_path": str(raw / "terminal_checks.json"),
            "global_suite_debt": json.loads((raw / "global_suite_debt.json").read_text())
            if (raw / "global_suite_debt.json").is_file()
            else None,
        },
        "verifier_is_oracle": fixture,
        "sentence_protocol_ready_score": int(passed),
        "sentence_protocol_manifest_path": {
            "path": str(raw / "sentence_protocol_manifest.json"),
            "sha256": sha256_file(raw / "sentence_protocol_manifest.json"),
        },
        "field_principles": {},
    }
    value["field_principles"] = {key: PRINCIPLE for key in value if key != "field_principles"}
    value["field_principles"].update(
        {f"acceptance_gate_results.{key}": PRINCIPLE for key in value["acceptance_gate_results"]}
    )
    return value


def _blocked_missing_manifest(
    manifest_path: Path, raw: Path, output: Path, date: str, origin: float
) -> dict[str, Any]:
    """Record an absent external corpus as a terminal block with exact operands."""
    check = _check("development_manifest_presence", manifest_path, "is_file", True, False)
    scope = {
        "test_paths": [f"tests/python/test_{NAME}.py"],
        "changed_modules": [f"python/carnot/{NAME}.py"],
        "specs": ["REQ-REPORT-7740", "REQ-VERIFY-7740"],
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    result: dict[str, Any] = {
        "experiment_id": "exp7740-sentence-label-protocol",
        "milestone": "2026.09.674",
        "run_date": date,
        "honest_verdict": "complete_blocked_missing_development_manifest",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": [check],
        "acceptance_gate_results": {
            "validity": False,
            "readiness": None,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "rows": [],
        "annotation_coverage_rows": [],
        "sample_size_budget": {
            "intended": sum(COUNTS.values()),
            "started": 0,
            "completed": 0,
            "eligible": 0,
            "excluded": 0,
            "censored": 0,
            "effective_independent_N": 0,
        },
        "claim_scope": {
            "value": "development_only",
            "fresh_generalization_eligible": False,
            "adapter_withheld_public": None,
        },
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "model_invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {"host": platform.node(), "pid": os.getpid(), "gpu_uuid": None},
        "phase_spans": [
            _span(
                "preconditions",
                origin,
                time.monotonic(),
                origin,
                date,
                0,
                raw / "frozen_affected_scope.json",
            )
        ],
        "random_seed": {
            "cohort_selection": "Exp7727 frozen salt",
            "planned_fit_initialization": [67401, 67402, 67403, 67404, 67405],
        },
        "reproducibility_checksum": digest(
            json.dumps(
                {"missing": str(manifest_path), "reducer": sha256_file(Path(__file__))},
                sort_keys=True,
            ).encode()
        ),
        "source_artifact_hashes": {
            "eligible_producers": {},
            "historical_disqualified_sources": {},
            "missing_inputs": [str(manifest_path)],
            "pre_gate_receipts": {},
        },
        "preconditions_checked": [
            check,
            {
                "check": "host_resources",
                "repo_root": str(ROOT),
                "cpu_count": os.cpu_count(),
                "effective_coding_backend": os.environ.get("CODEX_MODEL", "gpt-6 (session)"),
            },
        ],
        "validation_receipts": {
            "affected_scope": scope,
            "commands": [],
            "cold_replay": None,
            "global_suite_debt": None,
        },
        "verifier_is_oracle": False,
        "sentence_protocol_ready_score": 0,
        "sentence_protocol_manifest_path": None,
        "field_principles": {},
    }
    result["field_principles"] = {key: PRINCIPLE for key in result if key != "field_principles"}
    result["field_principles"].update(
        {f"acceptance_gate_results.{key}": PRINCIPLE for key in result["acceptance_gate_results"]}
    )
    atomic_json(raw / "terminal_candidate.json", result)
    atomic_json(output, result)
    return result


def run_experiment(
    manifest_path: Path,
    raw: Path,
    output: Path,
    date: str,
    *,
    fixture: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Capture frozen protocol evidence and publish only validated candidate bytes."""
    from carnot.reporting.experiment_7303_validation_scope import (
        CommandSpec,
        build_scoped_commands,
        run_commands,
    )

    origin = time.monotonic()
    progress(origin, "preconditions", "begin")
    if date != "20260927":
        raise ValueError("run_date")
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        return _blocked_missing_manifest(manifest_path, raw, output, date, origin)
    expected = {role: 1 for role in ROLES} if fixture else COUNTS
    public, manifest, checks = authenticate(manifest_path, expected)
    if (
        not fixture
        and manifest_path
        != ROOT / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
    ):
        raise ValueError("production_manifest_path")
    scope = {
        "test_paths": [
            "tests/python/test_experiment_7740_v674_sentence_label_protocol.py",
            "tests/python/test_experiment_7727_v673_development_corpus.py",
            "tests/python/test_experiment_7730_v673_set_energy_fit.py",
        ],
        "changed_modules": [f"python/carnot/{NAME}.py"],
        "static_paths": [f"scripts/experiments/{NAME}.py"],
        "specs": ["REQ-REPORT-7740", "REQ-VERIFY-7740"],
    }
    atomic_json(raw / "frozen_affected_scope.json", scope)
    protocol = _protocol(manifest_path, manifest, scope)
    atomic_json(raw / "sentence_protocol_manifest.json", protocol)
    spans = [
        _span(
            "preconditions",
            origin,
            time.monotonic(),
            origin,
            date,
            len(checks),
            raw / "sentence_protocol_manifest.json",
        )
    ]
    progress(origin, "preconditions", "complete", len(checks))
    phase = time.monotonic()
    progress(origin, "features", "begin")
    coverage, evidence = capture(manifest_path, raw, public, manifest, origin)
    spans.append(
        _span(
            "capture",
            phase,
            time.monotonic(),
            origin,
            date,
            len(public),
            raw / "evidence_manifest.json",
        )
    )
    receipts: list[dict[str, Any]] = []
    if validate:
        phase = time.monotonic()
        progress(origin, "validation", "before_subprocess", 0)
        private = Path("/tmp") / f"carnot-7740-{os.getpid()}"
        (private / "basetemp").mkdir(parents=True, exist_ok=True)
        commands = build_scoped_commands(
            ROOT,
            scope["test_paths"],
            scope["changed_modules"],
            static_paths=scope["static_paths"],
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage",
        )
        receipts = run_commands(
            ROOT,
            commands,
            log_dir=raw / "validation_logs",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        spans.append(
            _span(
                "validation",
                phase,
                time.monotonic(),
                origin,
                date,
                len(receipts),
                raw / "frozen_affected_scope.json",
            )
        )
        progress(origin, "validation", "after_subprocess", len(receipts))
    artifact = _artifact(
        date,
        raw,
        manifest_path,
        public,
        checks,
        coverage,
        evidence,
        scope,
        spans,
        fixture,
        receipts,
    )
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    if validate:
        terminal = [
            CommandSpec(
                "cold_replay",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "-m",
                    f"carnot.{NAME}",
                    "--cold-reduce",
                    str(raw),
                    "--candidate",
                    str(candidate),
                ),
                "exact_candidate",
                600,
            ),
            CommandSpec(
                "adversarial_verify",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ),
                "exact_candidate",
                600,
            ),
            CommandSpec(
                "verdict_row_consistency_strict",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "exact_candidate",
                600,
            ),
        ]
        progress(origin, "terminal", "before_subprocess", 0)
        terminal_receipts = run_commands(
            ROOT,
            terminal,
            log_dir=raw / "terminal_logs",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        atomic_json(
            raw / "terminal_checks.json",
            {"candidate_sha256": sha256_file(candidate), "commands": terminal_receipts},
        )
        progress(origin, "terminal", "after_subprocess", len(terminal_receipts))
        if not all(item["passed"] for item in terminal_receipts):
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
            artifact["sentence_protocol_ready_score"] = 0
            artifact["flagged_adversarial"] = not terminal_receipts[1]["passed"]
            artifact["gate_check_summary"] = [
                {
                    **_check(item["name"], candidate, "exit_code", 0, item["exit_code"]),
                    "upstream_id": NAME,
                }
                for item in terminal_receipts
                if not item["passed"]
            ]
            atomic_json(candidate, artifact)
    else:
        cold_reduce(raw, candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, output)
    progress(origin, "publish", "complete", len(public))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Offer one dated producer and one fresh-process reduction entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce is not None:
        if args.candidate is None:
            parser.error("--candidate is required for cold reduction")
        cold_reduce(args.cold_reduce, args.candidate)
        print("cold reduction passed", flush=True)
        return 0
    fixture = args.fixture_manifest is not None
    manifest_path = (
        args.fixture_manifest
        or ROOT / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
    )
    raw = args.raw or ROOT / "results/raw" / NAME
    output = args.output or ROOT / "results" / f"{NAME}.json"
    artifact = run_experiment(
        manifest_path, raw, output, args.date, fixture=fixture, validate=not fixture
    )
    return 0 if artifact["verdict_class"] in {"null", "circular_positive", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
