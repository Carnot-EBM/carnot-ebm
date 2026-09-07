"""Build the V624 forward evidence-ingress and historical census receipt.

The historical scan only classifies existing references. The shared ingress
helper is the enforcement path for future consumers. No historical result is
opened for writing or given a synthetic date.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot.reporting.evidence_ingress import ingest_evidence
from scripts import adversarial_verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = Path("results/experiment_7110_v624_evidence_ingress_quarantine.json")
HELPER_PATH = Path("python/carnot/reporting/evidence_ingress.py")
MODULE_PATH = Path("python/carnot/experiment_7110_v624_evidence_ingress_quarantine.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7110_v624_evidence_ingress_quarantine.py")
TEST_PATHS = (
    Path("tests/python/test_evidence_ingress_7110.py"),
    Path("tests/python/test_experiment_7110_v624_evidence_ingress_quarantine.py"),
)
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HARNESS_SPEC_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
KNOWN_ISSUES_PATH = Path("ops/known-issues.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
EXP7108_PATH = Path("results/experiment_7108_v623_capstone.json")
EXP7099_PATH = Path("results/experiment_7099_v623_adapter_withheld_preflight.json")

INFERENCE_SUBSTRATE = "deterministic evidence-ingress and historical census replay"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
RANDOM_SEED = 711020260907
EXPECTED_HISTORICAL_CENSUS_COUNT = 55

# Freeze the exact cohort behind the recorded 55-row finding. Later capstones
# must use the helper, but they must not change what this historical number means.
HISTORICAL_CENSUS_MIN_EXPERIMENT_ID = 4000
HISTORICAL_CENSUS_MAX_EXPERIMENT_ID = 7108
HISTORICAL_REFERENCE_CLASSES = frozenset(
    {"exclusion_recorded", "consumed_passing_row", "reference_only", "mixed_reference"}
)
EXPECTED_FIXTURES = frozenset(
    {
        "accepted_clean_input",
        "artifact_level_flag",
        "verifier_critical_flag",
        "missing_run_date",
        "malformed_run_date",
        "absent_artifact",
        "duplicate_artifact",
        "hash_drift",
        "unknown_flag_status",
        "empty_clean_set",
    }
)

POSITIVE_VERDICT = "complete_positive_v624_evidence_ingress_quarantine_ready"
BLOCKED_VERDICT = "complete_blocked_v624_evidence_ingress_precondition_unavailable"
DISQUALIFIED_VERDICT = "complete_disqualified_v624_evidence_ingress_validation_failed"

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "run_date",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        "fixture_rows",
        "capstone_reference_rows",
        "accepted_upstreams",
        "rejected_upstream_rows",
        "excluded_flagged_upstreams",
        "missing_run_date_rows",
        "malformed_run_date_rows",
        "exp7108_exp7099_regression_row",
        "historical_artifacts_rewritten",
        "evidence_ingress_helper_path",
        "evidence_ingress_quarantine_ready_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)

FIELD_PRINCIPLES = {
    "field_principles": "Every result field states why it is needed for the scientific record.",
    "preconditions_checked": "Unavailable evidence must block the replay before it can look clean.",
    "run_date": "The run's own date places new evidence on the forward-only contract timeline.",
    "inference_substrate": "The substrate states that this task replays bytes and does not produce new science.",
    "inference_substrate_class": "The compute class separates completed aggregation from a blocked no-run state.",
    "execution_venue": "The venue states where the deterministic replay executed.",
    "duration_s": "Measured wall time makes the bounded replay auditable.",
    "source_artifact_hashes": "Hashes bind the receipt to the exact code, contracts, ledgers, and evidence.",
    "rows": "Gate rows let an independent reader recompute readiness from small facts.",
    "fixture_rows": "Positive and negative fixtures prove that the boundary fails closed.",
    "capstone_reference_rows": "One row per historical capstone classifies references without rewriting history.",
    "accepted_upstreams": "Only eligible inputs appear on the measurement-bearing side of the boundary.",
    "rejected_upstream_rows": "Rejected inputs retain exact reasons without exposing their payload for aggregation.",
    "excluded_flagged_upstreams": "Flag exclusions stay visible to every downstream consumer.",
    "missing_run_date_rows": "Missing dates remain visible instead of receiving an unreliable mtime substitute.",
    "malformed_run_date_rows": "Invalid calendar claims remain visible and cannot enter current aggregation.",
    "exp7108_exp7099_regression_row": "The founding failure must stay reproducible against its exact artifacts.",
    "historical_artifacts_rewritten": "False confirms that the forward rule did not alter historical evidence.",
    "evidence_ingress_helper_path": "A stable import path lets later audit and capstone tasks share one boundary.",
    "evidence_ingress_quarantine_ready_score": "Readiness requires every fixture, census row, and regression gate.",
    "random_seed": "A fixed identity keeps the deterministic receipt comparable across runs.",
    "reproducibility_checksum": "A canonical digest detects changed evidence or derived fields.",
    "gate_check_summary": "The first exact failure distinguishes blocked, disqualified, and complete outcomes.",
    "verifier_is_oracle": "False prevents a hygiene check from becoming scientific correctness evidence.",
    "verdict_class": "A closed class keeps infrastructure readiness separate from scientific value.",
    "honest_verdict": "The terminal prefix gives automation one unambiguous outcome.",
}

SOURCE_PATHS = (
    HELPER_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    *TEST_PATHS,
    SPEC_PATH,
    HARNESS_SPEC_PATH,
    KNOWN_ISSUES_PATH,
    EXCLUSION_PATH,
    ADVERSARIAL_PATH,
    EXP7108_PATH,
    EXP7099_PATH,
)


def _sha256(path: Path) -> str:
    """Return the byte identity used before and after every historical read."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _experiment_id(path: Path, payload: Mapping[str, Any]) -> int | None:
    """Read a declared identity first, then use the standard result filename."""

    value = payload.get("experiment_id", payload.get("experiment"))
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match is not None:
            return int(match.group())
    match = re.match(r"experiment_(\d+)", path.name)
    return int(match.group(1)) if match is not None else None


def _load_object(path: Path) -> dict[str, Any] | None:
    """Return a JSON object, or None when a corpus file is not usable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _stream_search(path: Path, pattern: re.Pattern[bytes]) -> re.Match[bytes] | None:
    """Search a potentially huge artifact without retaining its JSON payload."""

    carry = b""
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                window = carry + chunk
                if match := pattern.search(window):
                    return match
                carry = window[-512:]
    except OSError:
        return None
    return None


def _is_stamped_flagged(path: Path) -> bool:
    """Recognize only the two-space-indented top-level adversarial stamp."""

    pattern = re.compile(rb'(?m)^ {2}"flagged_adversarial"\s*:\s*true\b')
    return _stream_search(path, pattern) is not None


def _streamed_run_date(path: Path) -> object:
    """Read a top-level historical run date without materializing large arrays."""

    pattern = re.compile(rb'(?m)^ {2}"run_date"\s*:\s*(null|"(?:[^"\\]|\\.)*")')
    match = _stream_search(path, pattern)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except (UnicodeError, ValueError):
        return None


def _flagged_upstreams(results_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Index stamped non-capstones without parsing multi-gigabyte payloads."""

    index: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(results_dir.glob("*.json")):
        if "capstone" in path.name.lower():
            continue
        if not _is_stamped_flagged(path):
            continue
        experiment_id = _experiment_id(path, {})
        if experiment_id is None:
            continue
        index.setdefault(experiment_id, []).append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "run_date": _streamed_run_date(path),
            }
        )
    return index


def _reference_state(path: tuple[str, ...], row: Mapping[str, Any]) -> str:
    """Classify one direct flagged identity without guessing metric meaning."""

    path_text = ".".join(path).lower()
    row_text = json.dumps(row, sort_keys=True, default=str).lower()
    exclusion_words = ("flagged", "quarant", "exclud", "skip", "adversarial")
    if any(word in path_text for word in exclusion_words) or any(
        word in row_text for word in ("excluded_flagged", "flagged_adversarial", "quarantined")
    ):
        return "exclusion_recorded"
    if (
        row.get("passed") is True
        or row.get("included") is True
        or row.get("value_promoted") is True
    ):
        return "consumed_passing_row"
    return "reference_only"


def _flagged_reference_rows(
    value: object,
    flagged_ids: frozenset[int],
    path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Find direct experiment-id references and retain their JSON locations."""

    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        direct = value.get("experiment_id")
        experiment_id: int | None = None
        if isinstance(direct, int) and not isinstance(direct, bool):
            experiment_id = direct
        elif isinstance(direct, str):
            match = re.search(r"\d+", direct)
            experiment_id = int(match.group()) if match is not None else None
        if experiment_id in flagged_ids:
            found.append(
                {
                    "upstream_experiment_id": experiment_id,
                    "json_path": ".".join(path + ("experiment_id",)),
                    "reference_state": _reference_state(path, value),
                }
            )
        for key, item in value.items():
            found.extend(_flagged_reference_rows(item, flagged_ids, path + (str(key),)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_flagged_reference_rows(item, flagged_ids, path + (str(index),)))
    return found


def _capstone_classification(reference_rows: Sequence[Mapping[str, Any]]) -> str:
    """Reduce direct reference states without calling an ambiguous row safe."""

    states = {str(row.get("reference_state")) for row in reference_rows}
    if "consumed_passing_row" in states:
        return "consumed_passing_row"
    if states == {"exclusion_recorded"}:
        return "exclusion_recorded"
    if states == {"reference_only"}:
        return "reference_only"
    return "mixed_reference"


def historical_capstone_reference_census(results_dir: Path) -> list[dict[str, Any]]:
    """Replay the frozen 55-capstone cohort without changing any source bytes."""

    flagged = _flagged_upstreams(results_dir)
    flagged_ids = frozenset(flagged)
    census: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*capstone*.json")):
        payload = _load_object(path)
        if payload is None:
            continue
        capstone_id = _experiment_id(path, payload)
        if capstone_id is None or not (
            HISTORICAL_CENSUS_MIN_EXPERIMENT_ID
            <= capstone_id
            <= HISTORICAL_CENSUS_MAX_EXPERIMENT_ID
        ):
            continue
        before_hash = _sha256(path)
        references = _flagged_reference_rows(payload, flagged_ids)
        if not references:
            continue
        after_hash = _sha256(path)
        upstream_ids = sorted({row["upstream_experiment_id"] for row in references})
        census.append(
            {
                "capstone_experiment_id": capstone_id,
                "capstone_path": str(path),
                "capstone_sha256": before_hash,
                "capstone_run_date": payload.get("run_date"),
                "flagged_upstream_experiment_ids": upstream_ids,
                "flagged_upstream_artifacts": [
                    artifact for upstream_id in upstream_ids for artifact in flagged[upstream_id]
                ],
                "reference_paths": references,
                "reference_count": len(references),
                "classification": _capstone_classification(references),
                "byte_hash_unchanged": before_hash == after_hash,
            }
        )
    return census


def _path_writable(path: Path) -> bool:
    """Check an existing file or the nearest existing parent for write access."""

    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.exists() and os.access(candidate, os.W_OK)


def _precondition_row(check: str, observed: bool) -> dict[str, Any]:
    """Use one common blocked-diagnostic shape for every required resource."""

    return {
        "check": check,
        "expected_value": True,
        "observed_value": observed,
        "available": observed,
    }


def _load_adversarial_report(path: Path) -> Mapping[str, Any]:
    """Run the repository verifier so a stale sidecar is not the only authority."""

    return adversarial_verify.verify_artifact(path)


def _preconditions(
    root: Path, output_path: Path
) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None]:
    """Check all evidence and write locations before the replay starts."""

    results_dir = root / "results"
    checks = [
        _precondition_row(
            "results_readable", results_dir.is_dir() and os.access(results_dir, os.R_OK)
        ),
        _precondition_row("known_issues_readable", (root / KNOWN_ISSUES_PATH).is_file()),
        _precondition_row("exclusion_manifest_readable", (root / EXCLUSION_PATH).is_file()),
        _precondition_row("adversarial_verifier_readable", (root / ADVERSARIAL_PATH).is_file()),
        _precondition_row("exp7108_readable", (root / EXP7108_PATH).is_file()),
        _precondition_row("exp7099_readable", (root / EXP7099_PATH).is_file()),
        _precondition_row("helper_source_writable", _path_writable(root / HELPER_PATH)),
        _precondition_row("module_source_writable", _path_writable(root / MODULE_PATH)),
        _precondition_row("test_paths_writable", all(_path_writable(root / p) for p in TEST_PATHS)),
        _precondition_row("spec_path_writable", _path_writable(root / SPEC_PATH)),
        _precondition_row("artifact_path_writable", _path_writable(output_path)),
    ]
    report: Mapping[str, Any] | None = None
    replay_available = False
    if all(row["available"] for row in checks[:6]):
        try:
            report = _load_adversarial_report(root / EXP7099_PATH)
            replay_available = report.get("loaded") is True
        except Exception:  # noqa: BLE001 - this becomes an exact blocked precondition.
            replay_available = False
    checks.append(_precondition_row("adversarial_replay_available", replay_available))
    return checks, report


def _public_fixture_row(row: Mapping[str, Any], fixture: str) -> dict[str, Any]:
    """Remove temporary payloads and paths before the durable receipt is built."""

    public = {key: value for key, value in row.items() if key not in {"payload", "read_error"}}
    public["path"] = f"fixture://{fixture}/{Path(str(row.get('path'))).name}"
    public["fixture"] = fixture
    return public


def _fixture_replay() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Exercise every named boundary with small temporary JSON inputs."""

    fixture_rows: list[dict[str, Any]] = []
    aggregate = {
        "accepted_upstreams": [],
        "rejected_upstream_rows": [],
        "excluded_flagged_upstreams": [],
        "missing_run_date_rows": [],
        "malformed_run_date_rows": [],
    }
    with tempfile.TemporaryDirectory(prefix="carnot-exp7110-") as directory:
        temp = Path(directory)

        def write(name: str, payload: Mapping[str, Any]) -> Path:
            path = temp / name
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            return path

        clean_payload = {
            "run_date": "20260907",
            "flagged_adversarial": False,
            "honest_verdict": "complete_null_fixture",
            "verdict_class": "null",
        }
        clean = write("clean.json", clean_payload)
        stamped = write("stamped.json", {**clean_payload, "flagged_adversarial": True})
        verifier_flag = write("verifier.json", {**clean_payload, "flagged_adversarial": None})
        missing_date = write(
            "missing-date.json",
            {key: value for key, value in clean_payload.items() if key != "run_date"},
        )
        malformed_date = write("malformed-date.json", {**clean_payload, "run_date": "20260230"})
        duplicate = write("duplicate.json", clean_payload)
        drift = write("drift.json", clean_payload)
        unknown = write("unknown.json", {**clean_payload, "flagged_adversarial": None})
        absent = temp / "absent.json"
        critical_report = {
            "loaded": True,
            "flags": [{"kind": "FIXTURE", "severity": "critical", "detail": "fixture"}],
        }
        cases: list[tuple[str, list[object], Any, int, list[str]]] = [
            (
                "accepted_clean_input",
                [{"path": clean, "expected_sha256": _sha256(clean), "consumer": "exp7115"}],
                None,
                1,
                [],
            ),
            ("artifact_level_flag", [{"path": stamped}], None, 0, ["artifact_flagged_adversarial"]),
            (
                "verifier_critical_flag",
                [{"path": verifier_flag, "consumer": "exp7119"}],
                lambda _path: critical_report,
                0,
                ["verifier_flagged_adversarial"],
            ),
            ("missing_run_date", [{"path": missing_date}], None, 0, ["run_date_missing"]),
            ("malformed_run_date", [{"path": malformed_date}], None, 0, ["run_date_malformed"]),
            ("absent_artifact", [{"path": absent}], None, 0, ["artifact_not_found"]),
            (
                "duplicate_artifact",
                [{"path": duplicate}, {"path": duplicate}],
                None,
                0,
                ["duplicate_expected_artifact"],
            ),
            (
                "hash_drift",
                [{"path": drift, "expected_sha256": "sha256:" + "0" * 64}],
                None,
                0,
                ["hash_drift"],
            ),
            ("unknown_flag_status", [{"path": unknown}], None, 0, ["flag_status_unknown"]),
            ("empty_clean_set", [], None, 0, []),
        ]
        for fixture, expectations, verifier, expected_accepted, expected_reasons in cases:
            result = ingest_evidence(expectations, verifier=verifier)
            accepted = [_public_fixture_row(row, fixture) for row in result["accepted_upstreams"]]
            rejected = [
                _public_fixture_row(row, fixture) for row in result["rejected_upstream_rows"]
            ]
            excluded = [
                _public_fixture_row(row, fixture) for row in result["excluded_flagged_upstreams"]
            ]
            missing = [_public_fixture_row(row, fixture) for row in result["missing_run_date_rows"]]
            malformed = [
                _public_fixture_row(row, fixture) for row in result["malformed_run_date_rows"]
            ]
            observed_reasons = sorted(
                {reason for row in rejected for reason in row["reason_codes"]}
            )
            fixture_rows.append(
                {
                    "fixture": fixture,
                    "expected_accepted_count": expected_accepted,
                    "observed_accepted_count": len(accepted),
                    "expected_reason_codes": sorted(expected_reasons),
                    "observed_reason_codes": observed_reasons,
                    "passed": len(accepted) == expected_accepted
                    and observed_reasons == sorted(expected_reasons),
                }
            )
            aggregate["accepted_upstreams"].extend(accepted)
            aggregate["rejected_upstream_rows"].extend(rejected)
            aggregate["excluded_flagged_upstreams"].extend(excluded)
            aggregate["missing_run_date_rows"].extend(missing)
            aggregate["malformed_run_date_rows"].extend(malformed)
    return fixture_rows, aggregate


def _date_state(value: object) -> str:
    """Classify historical dates without providing any replacement value."""

    if value is None:
        return "missing"
    if not isinstance(value, str):
        return "malformed"
    for shape in ("%Y%m%d", "%Y-%m-%d"):
        try:
            datetime.strptime(value, shape)
            return "valid"
        except ValueError:
            pass
    return "malformed"


def _historical_date_gaps(
    census: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain missing and malformed dates as immutable historical observations."""

    rows: dict[str, dict[str, Any]] = {}
    for capstone in census:
        rows[str(capstone["capstone_path"])] = {
            "path": capstone["capstone_path"],
            "artifact_role": "historical_capstone",
            "run_date": capstone.get("capstone_run_date"),
            "historical_audit_only": True,
            "rewritten": False,
        }
        for artifact in capstone["flagged_upstream_artifacts"]:
            rows[str(artifact["path"])] = {
                "path": artifact["path"],
                "artifact_role": "historical_flagged_upstream",
                "run_date": artifact.get("run_date"),
                "historical_audit_only": True,
                "rewritten": False,
            }
    missing = [row for row in rows.values() if _date_state(row["run_date"]) == "missing"]
    malformed = [row for row in rows.values() if _date_state(row["run_date"]) == "malformed"]
    return sorted(missing, key=lambda row: str(row["path"])), sorted(
        malformed, key=lambda row: str(row["path"])
    )


def _source_hashes(root: Path) -> list[dict[str, str]]:
    """Hash every available source that determines the artifact."""

    rows = []
    for relative in SOURCE_PATHS:
        path = root / relative
        if path.is_file():
            rows.append({"path": str(relative), "sha256": _sha256(path)})
    return rows


def _without_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only metadata on the durable side of an ingress receipt."""

    return [{key: value for key, value in row.items() if key != "payload"} for row in rows]


def _base_artifact(
    root: Path,
    run_date: str,
    output_path: Path,
    started: float,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """Create a schema-complete base before any precondition can stop work."""

    preconditions, verifier_report = _preconditions(root, output_path)
    artifact: dict[str, Any] = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": round(time.perf_counter() - started, 6),
        "source_artifact_hashes": _source_hashes(root),
        "rows": [],
        "fixture_rows": [],
        "capstone_reference_rows": [],
        "accepted_upstreams": [],
        "rejected_upstream_rows": [],
        "excluded_flagged_upstreams": [],
        "missing_run_date_rows": [],
        "malformed_run_date_rows": [],
        "exp7108_exp7099_regression_row": {},
        "historical_artifacts_rewritten": False,
        "evidence_ingress_helper_path": str(HELPER_PATH),
        "evidence_ingress_quarantine_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    return artifact, verifier_report


def _regression_row(
    root: Path, verifier_report: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply forward ingress to the exact V623 consumer failure."""

    path = root / EXP7099_PATH
    result = ingest_evidence(
        [
            {
                "path": path,
                "expected_sha256": _sha256(path),
                "consumer_experiment_id": 7108,
                "upstream_experiment_id": 7099,
            }
        ],
        verifier=lambda _path: verifier_report,
    )
    rejected = result["rejected_upstream_rows"][0]
    public = _without_payload([rejected])[0]
    public.update(
        {
            "consumer_experiment_id": 7108,
            "upstream_experiment_id": 7099,
            "scientific_honest_verdict": rejected.get("honest_verdict"),
            "scientific_verdict_class": rejected.get("verdict_class"),
        }
    )
    return public, result


def _gate_rows(
    fixture_rows: Sequence[Mapping[str, Any]],
    census: Sequence[Mapping[str, Any]],
    regression: Mapping[str, Any],
    historical_rewritten: bool,
) -> list[dict[str, Any]]:
    """Reduce the large replay to four exact readiness checks."""

    return [
        {
            "check": "all_fixtures_pass",
            "expected_value": True,
            "observed_value": bool(fixture_rows)
            and all(row.get("passed") is True for row in fixture_rows)
            and {row.get("fixture") for row in fixture_rows} == EXPECTED_FIXTURES,
        },
        {
            "check": "historical_capstone_census_count",
            "expected_value": EXPECTED_HISTORICAL_CENSUS_COUNT,
            "observed_value": len(census),
        },
        {
            "check": "historical_artifacts_rewritten",
            "expected_value": False,
            "observed_value": historical_rewritten,
        },
        {
            "check": "exp7108_rejects_flagged_exp7099",
            "expected_value": True,
            "observed_value": regression.get("ingestion_eligible") is False
            and "artifact_flagged_adversarial" in regression.get("reason_codes", []),
        },
    ]


def _row_passes(row: Mapping[str, Any]) -> bool:
    """Compare an exact expected value without truthiness shortcuts."""

    return row.get("expected_value") == row.get("observed_value")


def _expected_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return the first exact failure, or the completed readiness receipt."""

    preconditions = artifact.get("preconditions_checked")
    if isinstance(preconditions, list):
        for row in preconditions:
            if isinstance(row, Mapping) and row.get("available") is not True:
                return {
                    "failed_check": row.get("check"),
                    "expected_value": row.get("expected_value"),
                    "observed_value": row.get("observed_value"),
                    "passed": False,
                }
    fixture_rows = artifact.get("fixture_rows")
    if isinstance(fixture_rows, list):
        for row in fixture_rows:
            if isinstance(row, Mapping) and row.get("passed") is not True:
                return {
                    "failed_check": row.get("fixture"),
                    "expected_value": True,
                    "observed_value": row.get("passed"),
                    "passed": False,
                }
    rows = artifact.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and not _row_passes(row):
                return {
                    "failed_check": row.get("check"),
                    "expected_value": row.get("expected_value"),
                    "observed_value": row.get("observed_value"),
                    "passed": False,
                }
    return {"failed_check": None, "expected_value": 1, "observed_value": 1, "passed": True}


def _ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute readiness from fixtures, census, immutability, and regression."""

    fixture_rows = artifact.get("fixture_rows")
    census = artifact.get("capstone_reference_rows")
    regression = artifact.get("exp7108_exp7099_regression_row")
    rows = artifact.get("rows")
    return bool(
        isinstance(fixture_rows, list)
        and {row.get("fixture") for row in fixture_rows if isinstance(row, Mapping)}
        == EXPECTED_FIXTURES
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in fixture_rows)
        and isinstance(census, list)
        and len(census) == EXPECTED_HISTORICAL_CENSUS_COUNT
        and all(
            isinstance(row, Mapping)
            and row.get("classification") in HISTORICAL_REFERENCE_CLASSES
            and row.get("byte_hash_unchanged") is True
            for row in census
        )
        and artifact.get("historical_artifacts_rewritten") is False
        and isinstance(regression, Mapping)
        and regression.get("ingestion_eligible") is False
        and "artifact_flagged_adversarial" in regression.get("reason_codes", [])
        and isinstance(rows, list)
        and rows
        and all(isinstance(row, Mapping) and _row_passes(row) for row in rows)
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding wall time and the digest itself."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_artifact(root: Path, run_date: str, *, output_path: Path) -> dict[str, Any]:
    """Build a positive, blocked, or disqualified deterministic receipt."""

    started = time.perf_counter()
    root = root.resolve()
    output_path = output_path if output_path.is_absolute() else root / output_path
    artifact, verifier_report = _base_artifact(root, run_date, output_path, started)
    unavailable = next(
        (
            row
            for row in artifact["preconditions_checked"]
            if isinstance(row, Mapping) and row.get("available") is not True
        ),
        None,
    )
    if unavailable is None and verifier_report is not None:
        fixture_rows, fixture_evidence = _fixture_replay()
        census = historical_capstone_reference_census(root / "results")
        historical_rewritten = not all(row["byte_hash_unchanged"] for row in census)
        regression, regression_result = _regression_row(root, verifier_report)
        historical_missing, historical_malformed = _historical_date_gaps(census)
        rows = _gate_rows(
            fixture_rows,
            census,
            regression,
            historical_rewritten,
        )
        artifact.update(
            {
                "rows": rows,
                "fixture_rows": fixture_rows,
                "capstone_reference_rows": census,
                "accepted_upstreams": fixture_evidence["accepted_upstreams"],
                "rejected_upstream_rows": fixture_evidence["rejected_upstream_rows"]
                + _without_payload(regression_result["rejected_upstream_rows"]),
                "excluded_flagged_upstreams": fixture_evidence["excluded_flagged_upstreams"]
                + _without_payload(regression_result["excluded_flagged_upstreams"]),
                "missing_run_date_rows": fixture_evidence["missing_run_date_rows"]
                + historical_missing,
                "malformed_run_date_rows": fixture_evidence["malformed_run_date_rows"]
                + historical_malformed,
                "exp7108_exp7099_regression_row": regression,
                "historical_artifacts_rewritten": historical_rewritten,
            }
        )
        ready = _ready_from_artifact(artifact)
        artifact["evidence_ingress_quarantine_ready_score"] = int(ready)
        if ready:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = POSITIVE_VERDICT
        else:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = DISQUALIFIED_VERDICT
    else:
        artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["gate_check_summary"] = _expected_summary(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _valid_run_date(value: object) -> bool:
    """Require the compact date format declared for the new V624 artifact."""

    if not isinstance(value, str) or re.fullmatch(r"\d{8}", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, readiness, terminal class, summary, and checksum."""

    errors: list[str] = []
    if not REQUIRED_ARTIFACT_FIELDS <= artifact.keys():
        errors.append("required_artifact_fields_missing")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not isinstance(principles.get(field), str) or not principles.get(field, "").strip()
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles_incomplete")
    if not _valid_run_date(artifact.get("run_date")):
        errors.append("run_date_invalid")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("historical_artifacts_rewritten") is not False:
        errors.append("historical_artifacts_rewritten_invalid")
    if artifact.get("evidence_ingress_helper_path") != str(HELPER_PATH):
        errors.append("evidence_ingress_helper_path_invalid")

    preconditions = artifact.get("preconditions_checked")
    preconditions_ok = (
        isinstance(preconditions, list)
        and bool(preconditions)
        and all(isinstance(row, Mapping) and row.get("available") is True for row in preconditions)
    )
    ready = _ready_from_artifact(artifact) if preconditions_ok else False
    if artifact.get("evidence_ingress_quarantine_ready_score") != int(ready):
        errors.append("evidence_ingress_quarantine_ready_score_invalid")
    if not preconditions_ok:
        expected_class = "blocked"
        expected_verdict = BLOCKED_VERDICT
        expected_substrate_class = "blocked_no_run"
    elif ready:
        expected_class = "positive"
        expected_verdict = POSITIVE_VERDICT
        expected_substrate_class = INFERENCE_SUBSTRATE_CLASS
    else:
        expected_class = "disqualified"
        expected_verdict = DISQUALIFIED_VERDICT
        expected_substrate_class = INFERENCE_SUBSTRATE_CLASS
    if artifact.get("inference_substrate_class") != expected_substrate_class:
        errors.append("inference_substrate_class_invalid")
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_not_derived")
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict_not_derived")
    if artifact.get("gate_check_summary") != _expected_summary(artifact):
        errors.append("gate_check_summary_not_derived")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def _date_argument(value: str) -> str:  # pragma: no cover - argparse boundary.
    """Reject a malformed execution date before building an artifact."""

    if not _valid_run_date(value):
        raise argparse.ArgumentTypeError("date must use a real YYYYMMDD calendar date")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Replace the destination only after a complete temporary file is ready."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the replay or validate a saved artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validate-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.validate_artifact is not None:
        try:
            value = json.loads(args.validate_artifact.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"invalid artifact: {type(exc).__name__}: {exc}")
            return 1
        errors = validate_artifact(value) if isinstance(value, Mapping) else ["artifact_not_object"]
        if errors:
            print("invalid artifact: " + ", ".join(errors))
            return 1
        print(f"valid artifact: {args.validate_artifact}")
        return 0
    if args.date is None:
        parser.error("--date is required unless --validate-artifact is used")
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(root, args.date, output_path=output)
    errors = validate_artifact(artifact)
    _write_json_atomic(output, artifact)
    if errors:
        print("invalid generated artifact: " + ", ".join(errors))
        return 1
    print(f"{artifact['honest_verdict']}: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
