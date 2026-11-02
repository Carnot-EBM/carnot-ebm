"""Cold-read current V680 science inputs (REQ-REPORT-7835)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.experiment_7821_v679_independent_evidence_audit import (
    private_mutations as prior_mutations,
)

BASE = "results/raw/experiment_7835_v680_independent_evidence_audit"
MANIFEST = f"{BASE}/validation_command_manifest.json"
PLAN = (
    (7824, "results/experiment_7824_v680_source_feature_isolation.json"),
    (7826, "results/experiment_7826_v680_view_energy_fit.json"),
    (7827, "results/experiment_7827_v680_decision_measurement.json"),
    (7829, "results/experiment_7829_v680_qwen_counter_evidence.json"),
    (7830, "results/experiment_7830_v680_continuous_acquisition.json"),
    (7832, "results/experiment_7832_v680_selective_abstention.json"),
)
PRE_GATE = {
    7826: "results/experiment_7826_view_energy_fit.json",
    7829: "results/experiment_7829_qwen_counter_evidence.json",
}
PRIVATE = ("confidence", "gold", "label", "sentinel", "target")
FEATURE_FIELDS = {
    "family_id",
    "feature_dim",
    "view_a_tensor_rows",
    "view_a_tensor_sha256",
    "view_b_tensor_rows",
    "view_b_tensor_sha256",
}


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _fail(
    number: int, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> dict[str, Any]:
    return {
        "upstream_id": f"Exp{number}",
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def inspect_sources(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Name each producer and every failed upstream operand before compute."""
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, relative in PLAN:
        path = root / relative
        receipt = root / PRE_GATE[number] if number in PRE_GATE else None
        state = "missing"
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_bytes())
                state = "eligible" if isinstance(data, dict) else "disqualified"
            except (ValueError, UnicodeError):
                state = "disqualified"
        elif receipt and receipt.is_file():
            state = "conductor_only"
        source = {
            "upstream_id": f"Exp{number}",
            "path": relative,
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": data.get("run_date"),
            "role": "science_producer",
            "state": state,
            "eligibility": False,
            "pre_gate_receipt": {
                "path": str(receipt),
                "sha256": sha256_file(receipt),
                "role": "explanation_only",
            }
            if receipt and receipt.is_file()
            else None,
        }
        if state != "eligible":
            failures.append(
                _fail(
                    number, path, "science_producer", "existing qualified current artifact", state
                )
            )
        else:
            for field, expected in (
                ("experiment_id", number),
                ("milestone", "2026.09.680"),
                ("run_date", "20260928"),
                ("flagged_adversarial", False),
            ):
                if data.get(field) != expected:
                    failures.append(_fail(number, path, field, expected, data.get(field)))
            if data.get("verdict_class") not in ("positive", "circular_positive", "null"):
                failures.append(
                    _fail(
                        number,
                        path,
                        "verdict_class",
                        ["positive", "circular_positive", "null"],
                        data.get("verdict_class"),
                        "in",
                    )
                )
            score = {
                7824: "source_isolation_ready_score",
                7826: "view_energy_fit_ready_score",
                7827: "decision_measurement_ready_score",
                7829: "qwen_counter_evidence_ready_score",
                7830: "continuous_learning_ready_score",
                7832: "selective_abstention_ready_score",
            }[number]
            if data.get(score) != 1:
                failures.append(_fail(number, path, score, 1, data.get(score)))
            if any(f["upstream_id"] == f"Exp{number}" for f in failures):
                source["state"] = "disqualified"
            else:
                source["eligibility"] = True
        if receipt and receipt.is_file():
            gate = json.loads(receipt.read_bytes())
            for check in gate.get("gates_evaluated", []):
                if check.get("passed") is False:
                    upstream = Path(check["artifact_path"])
                    failures.append(
                        _fail(
                            number,
                            upstream,
                            check["artifact_field"],
                            check.get("expected"),
                            check.get("actual"),
                            check.get("op", "=="),
                        )
                    )
        sources.append(source)
    return sources, failures


def check_public_feature(row: dict[str, Any]) -> list[str]:
    """Reject both explicitly private names and unexpected public columns."""
    names = set(row)
    errors = []
    if names != FEATURE_FIELDS:
        errors.append("public_schema")
    if any(any(word in name.lower() for word in PRIVATE) for name in names):
        errors.append("private_feature")
    return errors


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_isolation(root: Path) -> dict[str, Any]:
    """Join original bytes, public tensors and evaluator labels by family."""
    base = root / "results/raw/experiment_7824_v680_source_feature_isolation"
    public_manifest = json.loads((base / "public_feature_manifest.json").read_bytes())
    side_manifest = json.loads((base / "label_sidecar_manifest.json").read_bytes())
    source_manifest = json.loads(Path(public_manifest["source_view_manifest_path"]).read_bytes())
    for manifest, pairs in (
        (public_manifest, (("feature_path", "feature_sha256"), ("public_path", "public_sha256"))),
        (side_manifest, (("sidecar_path", "sidecar_sha256"),)),
    ):
        for path_key, hash_key in pairs:
            if sha256_file(Path(manifest[path_key])) != manifest[hash_key]:
                raise ValueError(f"changed {path_key}")
    raw_path = Path(source_manifest["rows_path"])
    original = {r["family_id"]: r for r in _jsonl(raw_path)}
    features = {r["family_id"]: r for r in _jsonl(Path(public_manifest["feature_path"]))}
    records = {r["family_id"]: r for r in _jsonl(Path(public_manifest["public_path"]))}
    labels = {r["family_id"]: r for r in _jsonl(Path(side_manifest["sidecar_path"]))}
    errors: set[str] = set()
    ids = set(original)
    if any(set(table) != ids or len(table) != 640 for table in (features, records, labels)):
        errors.add("family_join")
    rows = []
    for family in sorted(ids):
        row, feature, public, label = (
            original[family],
            features.get(family),
            records.get(family),
            labels.get(family),
        )
        if feature is None or public is None or label is None:
            continue
        errors.update(check_public_feature(feature))
        if set(public) != set(public_manifest["allowlist"]):
            errors.add("public_allowlist")
        if label["role"] != row["role"] or label["source_role"] != row["role"]:
            errors.add("role_join")
        view = public["view_a"]
        source_hash = _digest(bytes.fromhex(view["source_bytes"]))
        answer_hash = _digest(bytes.fromhex(view["answer_bytes"]))
        if source_hash != row["source_sha256"] or answer_hash != row["response_sha256"]:
            errors.add("source_answer_bytes")
        if answer_hash != label["response_sha256"]:
            errors.add("label_join")
        rows.append(
            {
                "upstream_id": "Exp7824",
                "family_id": family,
                "seed": None,
                "arm": "source_isolation",
                "role": row["role"],
                "label": label["response_label"],
                "metrics": None,
                "raw_provenance": {
                    "path": str(raw_path),
                    "source_sha256": source_hash,
                    "answer_sha256": answer_hash,
                },
                "excluded": True,
                "censored": False,
            }
        )
    return {
        "rows": rows,
        "failed_checks": sorted(errors),
        "independent_n": len(ids),
        "raw_paths": [
            str(raw_path),
            public_manifest["feature_path"],
            public_manifest["public_path"],
            side_manifest["sidecar_path"],
        ],
    }


def random_matched_ids(families: list[str], count: int, seed: int) -> list[str]:
    """Select a matched count without accepting labels as an input."""
    if count < 0 or count > len(set(families)):
        raise ValueError("invalid matched count")
    return sorted(random.Random(seed).sample(sorted(set(families)), count))


def check_policy_rows(rows: list[dict[str, Any]], threshold: float) -> list[str]:
    """Reapply a frozen policy to predictions without consulting labels."""
    errors: set[str] = set()
    for row in rows:
        if row.get("selection_role") != "fit":
            errors.add("post_evaluation_tuning")
        probability = row.get("probability")
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            errors.add("invalid_probability")
        elif row.get("action") != ("escalate" if probability >= threshold else "accept"):
            errors.add("label_selected_action")
    return sorted(errors)


def private_mutations() -> list[dict[str, Any]]:
    """Challenge public custody and a frozen policy in addition to prior raw checks."""
    outcomes = prior_mutations()
    additions = (
        (
            "unknown_label_sentinel",
            check_public_feature({"unknown_label_sentinel": -1}),
            "private_feature",
        ),
        (
            "post_evaluation_threshold_tuning",
            check_policy_rows(
                [{"selection_role": "evaluation", "probability": 0.8, "action": "escalate"}], 0.7
            ),
            "post_evaluation_tuning",
        ),
        (
            "label_selected_random_abstention",
            check_policy_rows(
                [{"selection_role": "fit", "probability": 0.8, "action": "accept"}], 0.7
            ),
            "label_selected_action",
        ),
    )
    outcomes.extend(
        {
            "mutation": name,
            "expected_check": expected,
            "observed_checks": observed,
            "rejected": expected in observed,
        }
        for name, observed, expected in additions
    )
    return outcomes


def load_manifest(root: Path) -> dict[str, Any]:
    """Read the prospective command list, never a caller supplied command."""
    return json.loads((root / MANIFEST).read_bytes())


def assert_declared(
    manifest: dict[str, Any], name: str, argv: list[str], classification: str
) -> bool:
    """Reject a changed name, argument vector, or required class."""
    matches = [row for row in manifest["commands"] if row["name"] == name]
    if (
        len(matches) != 1
        or matches[0]["argv"] != argv
        or matches[0]["classification"] != classification
    ):
        raise ValueError("undeclared child command")
    return True


def seal_log(directory: Path, name: str, content: bytes) -> dict[str, str]:
    """Write closed child output once beneath a unique attempt directory."""
    import uuid

    attempt = directory / f"attempt-{uuid.uuid4().hex}"
    attempt.mkdir(parents=True)
    digest = _digest(content)
    path = attempt / f"{name}_{digest[7:]}.log"
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    return {"log_path": str(path), "log_sha256": digest}


def check_log(receipt: dict[str, Any]) -> list[str]:
    """Reject changed or missing sealed bytes on later cold reads."""
    path = Path(receipt["log_path"])
    return (
        []
        if path.is_file() and sha256_file(path) == receipt["log_sha256"]
        else ["validation_log_changed"]
    )


def build_artifact(
    root: Path,
    date: str,
    sources: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    isolation: dict[str, Any],
) -> dict[str, Any]:
    """Keep every branch and raw unit while missing science closes readiness."""
    blocked = any(row["state"] in ("missing", "conductor_only") for row in sources)
    disqualified = bool(isolation["failed_checks"]) or any(
        row["state"] == "disqualified" for row in sources
    )
    ready = int(not blocked and not disqualified and not failures)
    rows = [
        {
            "upstream_id": row["upstream_id"],
            "family_id": None,
            "seed": None,
            "arm": None,
            "role": "science_producer",
            "metrics": None,
            "raw_provenance": {"path": row["path"], "sha256": row["sha256"]},
            "excluded": row["state"] != "eligible",
            "censored": False,
            "state": row["state"],
        }
        for row in sources
    ]
    rows.extend(isolation["rows"])
    raw_sources = [
        {
            "path": path,
            "sha256": sha256_file(Path(path)),
            "role": "raw_development_custody",
            "date": date,
            "eligibility": "exposed_development",
        }
        for path in isolation["raw_paths"]
    ]
    verdict = (
        "complete_blocked_required_v680_science"
        if blocked
        else "complete_disqualified_source_custody"
        if disqualified
        else "complete_null_exposed_development"
    )
    result: dict[str, Any] = {
        "schema": "independent_evidence_audit_v3",
        "experiment_id": 7835,
        "milestone": "2026.09.680",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": "blocked" if blocked else "disqualified" if disqualified else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "discrepancy_rows": failures,
        "branch_dispositions": [
            {
                "upstream_id": row["upstream_id"],
                "state": row["state"],
                "reason": "current_science_missing"
                if row["state"] == "missing"
                else "conductor_receipt_only"
                if row["state"] == "conductor_only"
                else "producer_disqualified"
                if row["state"] == "disqualified"
                else "cold_recomputed",
            }
            for row in sources
        ],
        "mutation_results": private_mutations(),
        "acceptance_gate_results": {
            "validity": not bool(isolation["failed_checks"]),
            "readiness": ready,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "independent_evidence_ready_score": ready,
        "sample_size_budget": {
            "intended": {"Exp7824": 640, "Exp7827": 64, "Exp7829": 48, "Exp7830": 64},
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "excluded": isolation["independent_n"],
            "censored": 0,
            "independent_n": 0,
            "development_custody_n": isolation["independent_n"],
        },
        "source_artifact_hashes": [*sources, *raw_sources],
        "preconditions_checked": {
            "root": str(root.resolve()),
            "declared_paths": [p for _, p in PLAN],
            "source_count": len(sources),
            "resource_check": "CPU and local files",
            "isolation_raw_checks": isolation["failed_checks"],
        },
        "claim_scope": {
            "all_640_source_families_exposed": True,
            "natural_annotations": "exposed_development_only",
            "fresh_generalization_eligible": False,
            "fixtures": "circular_positive_only",
        },
        "verifier_is_oracle": False,
        "oracle_distinct_gate_eligible": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
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
        "random_seed": {"audit": 68035, "abstention": 68001},
        "duration_s": 0.0,
        "phase_spans": [],
        "validation_receipts": [],
        "validation_command_manifest_path": MANIFEST,
        "validation_command_manifest_sha256": sha256_file(root / MANIFEST),
        "observed_child_commands": [],
        "repository_health": {"status": "historical_failures_open"},
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "code": sha256_file(Path(__file__)),
            "sources": result["source_artifact_hashes"],
            "manifest": result["validation_command_manifest_sha256"],
            "seed": result["random_seed"],
        }
    )
    result["field_principles"] = {
        key: "Preserve exact current evidence and its claim scope." for key in result
    }
    return result


def cold_replay(candidate: Path) -> list[str]:
    """Reopen source and sealed validation bytes in a fresh reader."""
    value = json.loads(candidate.read_bytes())
    root = Path(value["preconditions_checked"]["root"])
    sources, failures = inspect_sources(root)
    try:
        isolation = audit_isolation(root)
    except (OSError, ValueError, KeyError, TypeError):
        return ["raw_custody_changed"]
    expected = build_artifact(root, value["run_date"], sources, failures, isolation)
    fields = (
        "source_artifact_hashes",
        "rows",
        "mutation_results",
        "branch_dispositions",
        "sample_size_budget",
        "reproducibility_checksum",
        "validation_command_manifest_sha256",
    )
    errors = [f"{field}_changed" for field in fields if value.get(field) != expected[field]]
    external = [r for r in value.get("gate_check_summary", []) if r.get("upstream_id") != "Exp7835"]
    if external != expected["gate_check_summary"]:
        errors.append("gate_check_summary_changed")
    for receipt in value.get("observed_child_commands", []):
        errors.extend(check_log(receipt))
    return sorted(set(errors))


def run_declared(root: Path, manifest: dict[str, Any], name: str, log_dir: Path) -> dict[str, Any]:
    """Run exactly one frozen child and seal output only after it exits."""
    import os
    import subprocess
    import time

    matches = [row for row in manifest["commands"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError("undeclared child command")
    spec = matches[0]
    assert_declared(manifest, name, spec["argv"], spec["classification"])
    started = time.monotonic()
    print(f"exp7835 before_subprocess name={name} elapsed_s=0 completed_units=0", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "python:."
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(  # noqa: S603 - argv is frozen in the task manifest.
        spec["argv"], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output = b""
    timed_out = False
    while True:
        elapsed = time.monotonic() - started
        remaining = spec["timeout_s"] - elapsed
        if remaining <= 0:
            timed_out = True
            process.terminate()
            try:
                chunk, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                chunk, _ = process.communicate()
            output += chunk
            break
        try:
            chunk, _ = process.communicate(timeout=min(60, remaining))
            output += chunk
            break
        except subprocess.TimeoutExpired:
            print(
                f"exp7835 child_outstanding name={name} elapsed_s={time.monotonic() - started:.1f} completed_units=0",
                flush=True,
            )
    sealed = seal_log(log_dir, name, output)
    receipt = {
        "name": name,
        "command_argv": spec["argv"],
        "classification": spec["classification"],
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        **sealed,
        "output_tail": output.decode(errors="replace")[-2000:],
    }
    print(
        f"exp7835 after_subprocess name={name} exit={process.returncode} "
        f"elapsed_s={receipt['duration_s']:.1f} completed_units=1",
        flush=True,
    )
    return receipt
