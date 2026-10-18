"""Runtime-contract E2E harness for Exp 1520.

Spec: REQ-VERIFY-1520, SCENARIO-VERIFY-1520.

The .116 experiments proved four separate runtime-contract pieces.  This
module does not ask a model for new evidence.  It loads the checked-in .116
artifacts, converts each source row into the same closed contract-case shape,
and computes false accepts only where a source row already carries an explicit
deterministic label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
CONTRACT_CASE_SCHEMA_VERSION = "runtime-contract-e2e/v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1520_runtime_contract_e2e_harness.json")
DEFAULT_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
DEFAULT_SAFE_DSL_ARTIFACT_PATH = Path(
    "results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json"
)
DEFAULT_SAFE_DSL_MANIFEST_PATH = Path("results/safe_dsl_verifier_induction_1507.jsonl")
DEFAULT_GRAMMAR_CERTIFICATE_ARTIFACT_PATH = Path(
    "results/experiment_1508_trigger_grammar_certificate_decoder_audit.json"
)
DEFAULT_GRAMMAR_CERTIFICATE_MANIFEST_PATH = Path(
    "results/trigger_grammar_certificate_decoder_1508.jsonl"
)
DEFAULT_GRAMMAR_CERTIFICATE_FALLBACK_MANIFEST_PATHS = (
    Path("results/trigger_grammar_certificates_1508.jsonl"),
)
DEFAULT_MONITOR_ARTIFACT_PATH = Path(
    "results/experiment_1509_executable_monitor_runtime_adapter.json"
)
DEFAULT_MONITOR_EVENT_MANIFEST_PATH = Path("results/executable_monitor_runtime_events_1509.jsonl")
DEFAULT_MONITOR_EVENT_FALLBACK_MANIFEST_PATHS = (
    Path("results/executable_monitor_events_1509.jsonl"),
)
DEFAULT_STRUCTURAL_CONTRACT_ARTIFACT_PATH = Path(
    "results/experiment_1510_plan_graph_structural_contract_gate.json"
)
DEFAULT_STRUCTURAL_CONTRACT_MANIFEST_PATH = Path(
    "results/plan_graph_structural_contracts_1510.jsonl"
)
DEFAULT_PRODUCT_LINE_ARTIFACT_PATH = Path(
    "results/experiment_1511_product_line_solver_oracle_benchmark.json"
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "runtime_contract_e2e_ready",
    "source_artifacts_loaded",
    "contract_cases_total",
    "safe_dsl_cases_linked",
    "grammar_certificate_cases_linked",
    "monitor_events_linked",
    "structural_contract_cases_linked",
    "false_accept_count",
    "false_accept_rate",
    "false_reject_count",
    "runtime_contract_manifest_path",
    "focused_tests_passed",
    "blockers",
    "honest_verdict",
)

REQUIRED_CONTRACT_CASE_FIELDS: tuple[str, ...] = (
    "row_type",
    "contract_schema_version",
    "contract_case_id",
    "prompt_or_case_id",
    "proposed_output",
    "certificate_parse_result",
    "safe_dsl_verifier_result",
    "monitor_event_result",
    "structural_contract_result",
    "expected_label",
    "final_deterministic_accept",
    "final_deterministic_decision",
    "source_family",
    "source_path",
    "source_line",
)


@dataclass(frozen=True)
class LoadedRuntimeContractSources:
    """Loaded terminal JSON artifacts and resolved JSONL rows for Exp 1520."""

    terminal_artifacts: dict[str, JsonDict]
    manifest_rows: dict[str, list[JsonDict]]
    resolved_paths: dict[str, Path]
    blockers: list[str]


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact before any source rows are loaded."""

    payload = _terminal_artifact(
        status="in_progress",
        ready=False,
        source_artifacts_loaded=False,
        manifest_path=Path(manifest_path),
        case_rows=[],
        ledger=_empty_ledger(),
        focused_tests_passed=False,
        blockers=[],
        run_date=run_date,
        honest_verdict="complete: in-progress Exp 1520 runtime-contract bootstrap artifact",
    )
    _write_json(Path(output_path), payload)
    return payload


def load_runtime_contract_sources(
    *,
    safe_dsl_artifact_path: Path | str = DEFAULT_SAFE_DSL_ARTIFACT_PATH,
    safe_dsl_manifest_path: Path | str = DEFAULT_SAFE_DSL_MANIFEST_PATH,
    grammar_certificate_artifact_path: Path | str = DEFAULT_GRAMMAR_CERTIFICATE_ARTIFACT_PATH,
    grammar_certificate_manifest_path: Path | str = DEFAULT_GRAMMAR_CERTIFICATE_MANIFEST_PATH,
    monitor_artifact_path: Path | str = DEFAULT_MONITOR_ARTIFACT_PATH,
    monitor_event_manifest_path: Path | str = DEFAULT_MONITOR_EVENT_MANIFEST_PATH,
    structural_contract_artifact_path: Path | str = DEFAULT_STRUCTURAL_CONTRACT_ARTIFACT_PATH,
    structural_contract_manifest_path: Path | str = DEFAULT_STRUCTURAL_CONTRACT_MANIFEST_PATH,
    product_line_artifact_path: Path | str = DEFAULT_PRODUCT_LINE_ARTIFACT_PATH,
) -> LoadedRuntimeContractSources:
    """Load all mandatory .116 terminal artifacts and source manifests."""

    blockers: list[str] = []
    terminal_artifacts: dict[str, JsonDict] = {}
    for label, path in (
        ("safe_dsl_artifact", safe_dsl_artifact_path),
        ("grammar_certificate_artifact", grammar_certificate_artifact_path),
        ("monitor_artifact", monitor_artifact_path),
        ("structural_contract_artifact", structural_contract_artifact_path),
        ("product_line_artifact", product_line_artifact_path),
    ):
        artifact = _load_json_or_blocker(label, Path(path), blockers)
        if artifact is not None:
            terminal_artifacts[label] = artifact

    resolved_paths: dict[str, Path] = {}
    for label, preferred, artifact_label, artifact_field, fallbacks in (
        (
            "safe_dsl_manifest",
            safe_dsl_manifest_path,
            "safe_dsl_artifact",
            "induction_manifest_path",
            (),
        ),
        (
            "grammar_certificate_manifest",
            grammar_certificate_manifest_path,
            "grammar_certificate_artifact",
            "decoder_manifest_path",
            DEFAULT_GRAMMAR_CERTIFICATE_FALLBACK_MANIFEST_PATHS,
        ),
        (
            "monitor_event_manifest",
            monitor_event_manifest_path,
            "monitor_artifact",
            "monitor_event_manifest_path",
            DEFAULT_MONITOR_EVENT_FALLBACK_MANIFEST_PATHS,
        ),
        (
            "structural_contract_manifest",
            structural_contract_manifest_path,
            "structural_contract_artifact",
            "contract_manifest_path",
            (),
        ),
    ):
        resolved = _resolve_manifest_path(
            label,
            preferred_path=Path(preferred),
            artifact=terminal_artifacts.get(artifact_label),
            artifact_field=artifact_field,
            fallback_paths=fallbacks,
            blockers=blockers,
        )
        if resolved is not None:
            resolved_paths[label] = resolved

    manifest_rows = (
        {
            "safe_dsl": _read_jsonl(resolved_paths["safe_dsl_manifest"]),
            "grammar_certificate": _read_jsonl(resolved_paths["grammar_certificate_manifest"]),
            "monitor_event": _read_jsonl(resolved_paths["monitor_event_manifest"]),
            "structural_contract": _read_jsonl(resolved_paths["structural_contract_manifest"]),
        }
        if not blockers
        else {}
    )
    return LoadedRuntimeContractSources(
        terminal_artifacts=terminal_artifacts,
        manifest_rows=manifest_rows,
        resolved_paths=resolved_paths,
        blockers=blockers,
    )


def normalize_contract_cases(
    *,
    safe_dsl_rows: list[JsonDict],
    grammar_certificate_rows: list[JsonDict],
    monitor_event_rows: list[JsonDict],
    structural_contract_rows: list[JsonDict],
    source_paths: Mapping[str, Path],
) -> list[JsonDict]:
    """Convert .116 source rows into one deterministic contract-case schema."""

    rows: list[JsonDict] = []
    rows.extend(_normalize_safe_dsl_rows(safe_dsl_rows, source_paths["safe_dsl"]))
    rows.extend(
        _normalize_certificate_rows(
            grammar_certificate_rows,
            source_paths["grammar_certificate"],
        )
    )
    rows.extend(_normalize_monitor_rows(monitor_event_rows, source_paths["monitor_event"]))
    rows.extend(
        _normalize_structural_rows(
            structural_contract_rows,
            source_paths["structural_contract"],
        )
    )
    return rows


def compute_false_accept_ledger(rows: Iterable[Mapping[str, Any]]) -> JsonDict:
    """Compute false accepts/rejects only where expected labels are explicit."""

    row_list = list(rows)
    explicit = [row for row in row_list if isinstance(row.get("expected_label"), bool)]
    explicit_rejects = [row for row in explicit if row["expected_label"] is False]
    false_accepts = [
        row for row in explicit_rejects if row.get("final_deterministic_accept") is True
    ]
    false_rejects = [
        row
        for row in explicit
        if row["expected_label"] is True and row.get("final_deterministic_accept") is False
    ]
    return {
        "explicit_label_count": len(explicit),
        "explicit_reject_count": len(explicit_rejects),
        "false_accept_count": len(false_accepts),
        "false_accept_rate": _rate(len(false_accepts), len(explicit_rejects)),
        "false_reject_count": len(false_rejects),
    }


def run_runtime_contract_e2e_harness(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    safe_dsl_artifact_path: Path | str = DEFAULT_SAFE_DSL_ARTIFACT_PATH,
    safe_dsl_manifest_path: Path | str = DEFAULT_SAFE_DSL_MANIFEST_PATH,
    grammar_certificate_artifact_path: Path | str = DEFAULT_GRAMMAR_CERTIFICATE_ARTIFACT_PATH,
    grammar_certificate_manifest_path: Path | str = DEFAULT_GRAMMAR_CERTIFICATE_MANIFEST_PATH,
    monitor_artifact_path: Path | str = DEFAULT_MONITOR_ARTIFACT_PATH,
    monitor_event_manifest_path: Path | str = DEFAULT_MONITOR_EVENT_MANIFEST_PATH,
    structural_contract_artifact_path: Path | str = DEFAULT_STRUCTURAL_CONTRACT_ARTIFACT_PATH,
    structural_contract_manifest_path: Path | str = DEFAULT_STRUCTURAL_CONTRACT_MANIFEST_PATH,
    product_line_artifact_path: Path | str = DEFAULT_PRODUCT_LINE_ARTIFACT_PATH,
    focused_tests_passed: bool = False,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run the deterministic Exp 1520 source load, normalization, and ledger."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, manifest_path=manifest, run_date=run_date)

    loaded = load_runtime_contract_sources(
        safe_dsl_artifact_path=safe_dsl_artifact_path,
        safe_dsl_manifest_path=safe_dsl_manifest_path,
        grammar_certificate_artifact_path=grammar_certificate_artifact_path,
        grammar_certificate_manifest_path=grammar_certificate_manifest_path,
        monitor_artifact_path=monitor_artifact_path,
        monitor_event_manifest_path=monitor_event_manifest_path,
        structural_contract_artifact_path=structural_contract_artifact_path,
        structural_contract_manifest_path=structural_contract_manifest_path,
        product_line_artifact_path=product_line_artifact_path,
    )
    if loaded.blockers:
        _write_jsonl(manifest, [])
        artifact = _terminal_artifact(
            status="blocked",
            ready=False,
            source_artifacts_loaded=False,
            manifest_path=manifest,
            case_rows=[],
            ledger=_empty_ledger(),
            focused_tests_passed=focused_tests_passed,
            blockers=loaded.blockers,
            run_date=run_date,
            honest_verdict="complete: blocked before runtime-contract E2E source loading",
        )
        _write_json(output, artifact)
        return artifact

    case_rows = normalize_contract_cases(
        safe_dsl_rows=loaded.manifest_rows["safe_dsl"],
        grammar_certificate_rows=loaded.manifest_rows["grammar_certificate"],
        monitor_event_rows=loaded.manifest_rows["monitor_event"],
        structural_contract_rows=loaded.manifest_rows["structural_contract"],
        source_paths={
            "safe_dsl": loaded.resolved_paths["safe_dsl_manifest"],
            "grammar_certificate": loaded.resolved_paths["grammar_certificate_manifest"],
            "monitor_event": loaded.resolved_paths["monitor_event_manifest"],
            "structural_contract": loaded.resolved_paths["structural_contract_manifest"],
        },
    )
    ledger = compute_false_accept_ledger(case_rows)
    linked_counts = _linked_counts(case_rows)
    readiness_blockers = _readiness_blockers(linked_counts, ledger, focused_tests_passed)
    summary_row = _summary_manifest_row(case_rows, ledger, not readiness_blockers)
    _write_jsonl(manifest, [*case_rows, summary_row])
    ready = not readiness_blockers and manifest.exists() and ledger["false_accept_rate"] == 0.0
    artifact = _terminal_artifact(
        status="complete" if ready else "blocked",
        ready=ready,
        source_artifacts_loaded=True,
        manifest_path=manifest,
        case_rows=case_rows,
        ledger=ledger,
        focused_tests_passed=focused_tests_passed,
        blockers=readiness_blockers,
        run_date=run_date,
        honest_verdict=(
            "complete: runtime-contract E2E harness linked .116 contract families"
            if ready
            else "complete: blocked before runtime-contract E2E readiness"
        ),
    )
    artifact["source_artifact_paths"] = {
        name: _display_path(path) for name, path in loaded.resolved_paths.items()
    }
    artifact["terminal_artifacts_loaded"] = sorted(loaded.terminal_artifacts)
    _write_json(output, artifact)
    return artifact


def _normalize_safe_dsl_rows(rows: list[JsonDict], source_path: Path) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for line_number, row in enumerate(rows, start=1):
        if row.get("row_type") != "selected_set_summary":
            continue
        false_accept_ids = [str(row_id) for row_id in row.get("false_accept_row_ids", [])]
        false_accept_set = set(false_accept_ids)
        accepted_ids = [
            str(row_id)
            for row_id in row.get("accepted_labeled_row_ids", [])
            if str(row_id) not in false_accept_set
        ]
        for labeled_row_id in accepted_ids:
            normalized.append(
                _contract_case(
                    source_family="safe_dsl",
                    source_path=source_path,
                    source_line=line_number,
                    contract_case_id=f"safe_dsl:{labeled_row_id}",
                    prompt_or_case_id=_case_id_from_labeled_row(labeled_row_id),
                    proposed_output=labeled_row_id,
                    safe_dsl_verifier_result={
                        "linked": True,
                        "accepted_labeled_row_id": labeled_row_id,
                        "candidate_names": row.get("candidate_names", []),
                        "verifier_false_accept_rate": row.get("verifier_false_accept_rate"),
                    },
                    expected_label=True,
                    final_deterministic_accept=True,
                )
            )
        for labeled_row_id in false_accept_ids:
            normalized.append(
                _contract_case(
                    source_family="safe_dsl",
                    source_path=source_path,
                    source_line=line_number,
                    contract_case_id=f"safe_dsl:{labeled_row_id}",
                    prompt_or_case_id=_case_id_from_labeled_row(labeled_row_id),
                    proposed_output=labeled_row_id,
                    safe_dsl_verifier_result={
                        "linked": True,
                        "false_accept_row_id": labeled_row_id,
                        "candidate_names": row.get("candidate_names", []),
                        "verifier_false_accept_rate": row.get("verifier_false_accept_rate"),
                    },
                    expected_label=False,
                    final_deterministic_accept=True,
                )
            )
    return normalized


def _normalize_certificate_rows(rows: list[JsonDict], source_path: Path) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for line_number, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or "")
        decoder_mode = str(row.get("decoder_mode") or "unknown")
        parser_result = (
            row.get("parser_result") if isinstance(row.get("parser_result"), dict) else {}
        )
        verifier_result = (
            row.get("verifier_result") if isinstance(row.get("verifier_result"), dict) else {}
        )
        parsed = bool(parser_result.get("parsed"))
        accepted = bool(verifier_result.get("accepted", row.get("deterministic_validation_passed")))
        final_accept = (
            parsed
            and bool(row.get("deterministic_validation_passed"))
            and accepted
            and not bool(row.get("false_accept_status") or verifier_result.get("false_accept"))
        )
        normalized.append(
            _contract_case(
                source_family="grammar_certificate",
                source_path=source_path,
                source_line=line_number,
                contract_case_id=f"grammar_certificate:{case_id}:{decoder_mode}:{line_number}",
                prompt_or_case_id=case_id,
                proposed_output=row.get("model_output") or row.get("certificate_json"),
                certificate_parse_result={
                    "linked": True,
                    "decoder_mode": decoder_mode,
                    "parsed": parsed,
                    "grammar_backend": row.get("grammar_backend"),
                    "deterministic_validation_passed": bool(
                        row.get("deterministic_validation_passed")
                    ),
                    "verifier_accepted": accepted,
                    "false_accept_status": bool(
                        row.get("false_accept_status") or verifier_result.get("false_accept")
                    ),
                },
                expected_label=_explicit_bool(verifier_result.get("base_valid")),
                final_deterministic_accept=final_accept,
            )
        )
    return normalized


def _normalize_monitor_rows(rows: list[JsonDict], source_path: Path) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for line_number, row in enumerate(rows, start=1):
        event_id = str(row.get("event_id") or f"monitor:{line_number}")
        validation_status = str(row.get("validation_status") or "unknown")
        final_accept = validation_status == "pass" and not bool(row.get("verifier_false_accept"))
        normalized.append(
            _contract_case(
                source_family="monitor_event",
                source_path=source_path,
                source_line=line_number,
                contract_case_id=f"monitor_event:{event_id}",
                prompt_or_case_id=str(row.get("case_id") or ""),
                proposed_output=row.get("event_kind"),
                monitor_event_result={
                    "linked": True,
                    "event_id": event_id,
                    "event_kind": row.get("event_kind"),
                    "validation_status": validation_status,
                    "verifier_false_accept": bool(row.get("verifier_false_accept")),
                },
                expected_label=None,
                final_deterministic_accept=final_accept,
            )
        )
    return normalized


def _normalize_structural_rows(rows: list[JsonDict], source_path: Path) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for line_number, row in enumerate(rows, start=1):
        graph_id = str(row.get("graph_id") or f"structural:{line_number}")
        contract_family = str(row.get("contract_family") or "unknown")
        detected_violation = bool(row.get("detected_violation"))
        expected_violation = _explicit_bool(row.get("expected_violation"))
        normalized.append(
            _contract_case(
                source_family="structural_contract",
                source_path=source_path,
                source_line=line_number,
                contract_case_id=(
                    f"structural_contract:{graph_id}:{contract_family}:{line_number}"
                ),
                prompt_or_case_id=str(row.get("case_id") or graph_id),
                proposed_output=row.get("contract_evidence") or graph_id,
                structural_contract_result={
                    "linked": True,
                    "graph_id": graph_id,
                    "contract_family": contract_family,
                    "expected_violation": expected_violation,
                    "detected_violation": detected_violation,
                    "classifier_outcome": row.get("classifier_outcome"),
                },
                expected_label=(None if expected_violation is None else not expected_violation),
                final_deterministic_accept=not detected_violation,
            )
        )
    return normalized


def _contract_case(
    *,
    source_family: str,
    source_path: Path,
    source_line: int,
    contract_case_id: str,
    prompt_or_case_id: str,
    proposed_output: Any,
    expected_label: bool | None,
    final_deterministic_accept: bool,
    certificate_parse_result: JsonDict | None = None,
    safe_dsl_verifier_result: JsonDict | None = None,
    monitor_event_result: JsonDict | None = None,
    structural_contract_result: JsonDict | None = None,
) -> JsonDict:
    return {
        "row_type": "contract_case",
        "contract_schema_version": CONTRACT_CASE_SCHEMA_VERSION,
        "contract_case_id": contract_case_id,
        "prompt_or_case_id": prompt_or_case_id,
        "proposed_output": proposed_output,
        "certificate_parse_result": certificate_parse_result or {"linked": False},
        "safe_dsl_verifier_result": safe_dsl_verifier_result or {"linked": False},
        "monitor_event_result": monitor_event_result or {"linked": False},
        "structural_contract_result": structural_contract_result or {"linked": False},
        "expected_label": expected_label,
        "final_deterministic_accept": bool(final_deterministic_accept),
        "final_deterministic_decision": ("accept" if final_deterministic_accept else "reject"),
        "source_family": source_family,
        "source_path": _display_path(source_path),
        "source_line": int(source_line),
    }


def _terminal_artifact(
    *,
    status: str,
    ready: bool,
    source_artifacts_loaded: bool,
    manifest_path: Path,
    case_rows: list[JsonDict],
    ledger: JsonDict,
    focused_tests_passed: bool,
    blockers: list[str],
    run_date: str,
    honest_verdict: str,
) -> JsonDict:
    linked_counts = _linked_counts(case_rows)
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "runtime_contract_e2e_ready": bool(ready),
        "source_artifacts_loaded": bool(source_artifacts_loaded),
        "contract_cases_total": len(case_rows),
        "safe_dsl_cases_linked": linked_counts["safe_dsl"],
        "grammar_certificate_cases_linked": linked_counts["grammar_certificate"],
        "monitor_events_linked": linked_counts["monitor_event"],
        "structural_contract_cases_linked": linked_counts["structural_contract"],
        "false_accept_count": ledger["false_accept_count"],
        "false_accept_rate": ledger["false_accept_rate"],
        "false_reject_count": ledger["false_reject_count"],
        "runtime_contract_manifest_path": _display_path(manifest_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "blockers": blockers,
        "honest_verdict": honest_verdict,
        "explicit_label_count": ledger["explicit_label_count"],
        "explicit_reject_count": ledger["explicit_reject_count"],
    }


def _summary_manifest_row(
    case_rows: list[JsonDict],
    ledger: JsonDict,
    source_artifacts_loaded: bool,
) -> JsonDict:
    linked_counts = _linked_counts(case_rows)
    return {
        "row_type": "summary",
        "contract_schema_version": CONTRACT_CASE_SCHEMA_VERSION,
        "source_artifacts_loaded": bool(source_artifacts_loaded),
        "contract_cases_total": len(case_rows),
        "safe_dsl_cases_linked": linked_counts["safe_dsl"],
        "grammar_certificate_cases_linked": linked_counts["grammar_certificate"],
        "monitor_events_linked": linked_counts["monitor_event"],
        "structural_contract_cases_linked": linked_counts["structural_contract"],
        "explicit_label_count": ledger["explicit_label_count"],
        "explicit_reject_count": ledger["explicit_reject_count"],
        "false_accept_count": ledger["false_accept_count"],
        "false_accept_rate": ledger["false_accept_rate"],
        "false_reject_count": ledger["false_reject_count"],
    }


def _readiness_blockers(
    linked_counts: JsonDict,
    ledger: JsonDict,
    focused_tests_passed: bool,
) -> list[str]:
    blockers = [
        f"no_{family}_rows_linked" for family, count in linked_counts.items() if int(count) <= 0
    ]
    blockers.extend([] if ledger["false_accept_rate"] == 0.0 else ["false_accept_rate_nonzero"])
    blockers.extend([] if focused_tests_passed else ["focused_tests_not_passed"])
    return blockers


def _linked_counts(rows: list[JsonDict]) -> JsonDict:
    return {
        "safe_dsl": sum(1 for row in rows if row.get("source_family") == "safe_dsl"),
        "grammar_certificate": sum(
            1 for row in rows if row.get("source_family") == "grammar_certificate"
        ),
        "monitor_event": sum(1 for row in rows if row.get("source_family") == "monitor_event"),
        "structural_contract": sum(
            1 for row in rows if row.get("source_family") == "structural_contract"
        ),
    }


def _load_json_or_blocker(label: str, path: Path, blockers: list[str]) -> JsonDict | None:
    if not path.exists():
        blockers.append(f"missing_{label}:{path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(
    label: str,
    *,
    preferred_path: Path,
    artifact: JsonDict | None,
    artifact_field: str,
    fallback_paths: Iterable[Path],
    blockers: list[str],
) -> Path | None:
    candidates = [preferred_path]
    artifact_path = (artifact or {}).get(artifact_field)
    if artifact_path:
        candidates.append(Path(str(artifact_path)))
    candidates.extend(fallback_paths)
    for path in _unique_paths(candidates):
        if path.exists():
            return path
    blockers.append(f"missing_{label}:{preferred_path}")
    return None


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _empty_ledger() -> JsonDict:
    return {
        "explicit_label_count": 0,
        "explicit_reject_count": 0,
        "false_accept_count": 0,
        "false_accept_rate": None,
        "false_reject_count": 0,
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _case_id_from_labeled_row(labeled_row_id: str) -> str:
    parts = labeled_row_id.split(":")
    return parts[1] if len(parts) > 1 else labeled_row_id


def _explicit_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
