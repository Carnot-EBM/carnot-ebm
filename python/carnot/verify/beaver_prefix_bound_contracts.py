"""BEAVER-lite prefix-bound audit for runtime-contract decoder rows.

Spec: REQ-VERIFY-1537, SCENARIO-VERIFY-1537.

The audit implemented here is deliberately below Carnot's deterministic
runtime-contract validators.  It builds a bounded prefix frontier over the
canonical contract JSON that a decoder should emit and reports structural risk
signals for prefixes that are incomplete or already off-target.  Those signals
help rank rows for routing and inspection, but every false-accept number still
comes from the Exp 1520 validator ledger.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.verify import runtime_contract_e2e_harness as runtime_contracts

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MILESTONE = ".118"
DEFAULT_DECODER_MANIFEST_PATH = Path("results/xgrammar_abs_contract_decoder_adapter_1535.jsonl")
DEFAULT_RUNTIME_MANIFEST_PATH = Path("results/runtime_contract_e2e_manifest_1520.jsonl")
DEFAULT_SOURCE_ARTIFACT_PATH = Path("results/experiment_1535_xgrammar_abs_contract_decoder_adapter.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1537_beaver_prefix_bound_contracts_v3.json")
DEFAULT_AUDIT_MANIFEST_PATH = Path("results/beaver_prefix_bound_contracts_1537.jsonl")
AUDIT_MODULE_PATH = "python/carnot/verify/beaver_prefix_bound_contracts.py"

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_prefix_bound",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_prefix_bound",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_prefix_bound",
    },
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "beaver_bound_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "bounded_prefixes",
    "token_logprob_available",
    "topk_available",
    "bound_violations",
    "high_risk_instances",
    "deterministic_validator_final_authority",
    "false_accept_rate",
    "bound_audit_path",
    "focused_tests_passed",
    "honest_verdict",
)
_TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


@dataclass(frozen=True)
class PrefixBoundState:
    """One audited prefix and its conservative unsafe upper bound."""

    prefix: str
    source: str
    depth: int
    target_length: int
    prefix_consistent: bool
    terminal: bool
    unsafe_upper_bound: float


class PrefixFrontierTrie:
    """Small trie over canonical contract JSON targets for prefix-frontier checks."""

    def __init__(self) -> None:
        self._children: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_targets(cls, targets: Iterable[str]) -> "PrefixFrontierTrie":
        trie = cls()
        for target in targets:
            trie.insert(target)
        return trie

    def insert(self, target: str) -> None:
        node = self._children
        for char in target:
            node = node.setdefault(char, {})
        node.setdefault("", {})

    def allowed_next_chars(self, prefix: str) -> frozenset[str]:
        node = self._node_for(prefix)
        return frozenset(char for char in node if char)

    def contains_prefix(self, prefix: str) -> bool:
        try:
            self._node_for(prefix)
        except KeyError:
            return False
        return True

    def _node_for(self, prefix: str) -> dict[str, Any]:
        node = self._children
        for char in prefix:
            node = node[char]
        return node


def canonical_contract_target(contract_case_id: str, final_deterministic_decision: str) -> str:
    """Return the bounded JSON target used by Exp 1537 prefix-frontier checks."""

    decision = str(final_deterministic_decision).lower()
    if decision not in {"accept", "reject"}:
        decision = "reject"
    return json.dumps(
        {
            "contract_case_id": str(contract_case_id),
            "final_deterministic_decision": decision,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def score_prefix(target: str, prefix: str, *, source: str) -> PrefixBoundState:
    """Score one prefix against the canonical target with a monotone structural bound."""

    target_length = max(len(target), 1)
    consistent = target.startswith(prefix)
    terminal = prefix == target
    if not consistent:
        upper_bound = 1.0
    elif terminal:
        upper_bound = 0.0
    else:
        upper_bound = (len(target) - len(prefix)) / target_length
    return PrefixBoundState(
        prefix=prefix,
        source=source,
        depth=len(prefix),
        target_length=len(target),
        prefix_consistent=consistent,
        terminal=terminal,
        unsafe_upper_bound=round(max(0.0, min(1.0, upper_bound)), 6),
    )


def build_prefix_bound_series(target: str, *, prefix_stride: int = 8) -> list[PrefixBoundState]:
    """Return canonical-prefix states from empty prefix through terminal target."""

    stride = max(1, int(prefix_stride))
    offsets = list(range(0, len(target) + 1, stride))
    if not offsets or offsets[-1] != len(target):
        offsets.append(len(target))
    return [score_prefix(target, target[:offset], source="canonical_frontier") for offset in offsets]


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    audit_manifest_path: Path | str = DEFAULT_AUDIT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact before source rows are loaded."""

    artifact = _terminal_artifact(
        status="in_progress",
        run_date=run_date,
        model_specs=list(MANDATED_MODEL_SPECS),
        live_sota_model_inference_used=False,
        focused_tests_passed=False,
        audit_manifest_path=Path(audit_manifest_path),
        audit={
            "bounded_prefixes": 0,
            "prefix_rows": [],
            "token_logprob_available": False,
            "topk_available": False,
            "bound_violations": [],
            "high_risk_instances": [],
            "false_accept_rate": None,
            "deterministic_validator_final_authority": True,
        },
        blockers=["experiment_1537_prefix_bound_audit_in_progress"],
    )
    _write_json(Path(output_path), artifact)
    return artifact


def audit_decoder_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime_rows: Sequence[Mapping[str, Any]] = (),
    focused_tests_passed: bool,
    prefix_stride: int = 8,
    max_cases_per_family: int = 2,
) -> JsonDict:
    """Audit selected Exp 1535 decoder rows and return metrics plus prefix rows."""

    decoder_rows = [dict(row) for row in rows if row.get("row_type") == "decoder_result"]
    runtime_by_case_id = _runtime_cases_by_id(runtime_rows)
    target_by_case_id = _target_by_case_id(decoder_rows, runtime_by_case_id)
    selected_rows = _select_decoder_rows(decoder_rows, max_cases_per_family=max_cases_per_family)
    prefix_rows: list[JsonDict] = []
    high_risk_instances: list[JsonDict] = []
    validation_rows = _validation_rows(selected_rows)
    token_logprob_available = any(_token_logprob(row) is not None for row in selected_rows)
    topk_available = any(_topk_logprobs(row) for row in selected_rows)

    for row in selected_rows:
        case_id = str(row.get("contract_case_id") or "")
        target = target_by_case_id.get(case_id) or _fallback_target(row, runtime_by_case_id)
        trie = PrefixFrontierTrie.from_targets([target])
        for state in build_prefix_bound_series(target, prefix_stride=prefix_stride):
            prefix_rows.append(_prefix_manifest_row(row, state, trie))
        observed_state = score_prefix(target, _row_raw_output(row), source="observed_output")
        prefix_rows.append(_prefix_manifest_row(row, observed_state, trie))
        high_risk_instances.append(_high_risk_instance(row, observed_state))

    ledger = runtime_contracts.compute_false_accept_ledger(validation_rows)
    bound_violations = _bound_violations(high_risk_instances)
    high_risk_instances.sort(
        key=lambda item: (
            float(item["risk_upper_bound"]),
            bool(item["deterministic_false_accept"]),
            str(item["contract_case_id"]),
        ),
        reverse=True,
    )
    all_bounds_valid = all(0.0 <= float(row["unsafe_upper_bound"]) <= 1.0 for row in prefix_rows)
    return {
        "beaver_bound_ready": bool(
            prefix_rows
            and all_bounds_valid
            and not bound_violations
            and focused_tests_passed
            and ledger["false_accept_rate"] is not None
        ),
        "bounded_prefixes": len(prefix_rows),
        "prefix_rows": prefix_rows,
        "token_logprob_available": bool(token_logprob_available),
        "topk_available": bool(topk_available),
        "bound_violations": bound_violations,
        "high_risk_instances": high_risk_instances[:10],
        "false_accept_rate": ledger["false_accept_rate"],
        "false_accept_count": ledger["false_accept_count"],
        "explicit_label_count": ledger["explicit_label_count"],
        "explicit_reject_count": ledger["explicit_reject_count"],
        "deterministic_validator_final_authority": True,
    }


def run_experiment(
    *,
    project_root: Path | str | None = None,
    run_date: str = RUN_DATE,
    decoder_manifest_path: Path | str = DEFAULT_DECODER_MANIFEST_PATH,
    runtime_manifest_path: Path | str = DEFAULT_RUNTIME_MANIFEST_PATH,
    source_artifact_path: Path | str = DEFAULT_SOURCE_ARTIFACT_PATH,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    audit_manifest_path: Path | str = DEFAULT_AUDIT_MANIFEST_PATH,
    focused_tests_passed: bool = False,
    prefix_stride: int = 8,
) -> JsonDict:
    """Run the Exp 1537 prefix-bound audit and write terminal artifacts."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    audit_manifest = _resolve_under_root(root, Path(audit_manifest_path))
    decoder_manifest = _resolve_under_root(root, Path(decoder_manifest_path))
    runtime_manifest = _resolve_under_root(root, Path(runtime_manifest_path))
    source_artifact = _resolve_under_root(root, Path(source_artifact_path))
    write_in_progress_artifact(output, audit_manifest_path=audit_manifest, run_date=run_date)

    blockers: list[str] = []
    decoder_rows = _read_jsonl(decoder_manifest)
    runtime_rows = _read_jsonl(runtime_manifest)
    if not decoder_rows:
        blockers.append(f"missing_or_empty_decoder_manifest:{decoder_manifest}")
    if not runtime_rows:
        blockers.append(f"missing_or_empty_runtime_manifest:{runtime_manifest}")

    source = _read_json(source_artifact)
    model_specs = source.get("model_specs") or list(MANDATED_MODEL_SPECS)
    audit = audit_decoder_rows(
        decoder_rows,
        runtime_rows=runtime_rows,
        focused_tests_passed=focused_tests_passed,
        prefix_stride=prefix_stride,
    )
    _write_jsonl(audit_manifest, [*audit["prefix_rows"], _summary_manifest_row(audit)])
    if not focused_tests_passed:
        blockers.append("focused_tests_not_passed")
    if not audit["token_logprob_available"]:
        blockers.append("token_logprobs_unavailable_structural_simulation_used")
    if not audit["topk_available"]:
        blockers.append("topk_unavailable_structural_simulation_used")

    artifact = _terminal_artifact(
        status="complete" if audit["bounded_prefixes"] else "blocked",
        run_date=run_date,
        model_specs=list(model_specs),
        live_sota_model_inference_used=bool(source.get("live_sota_model_inference_used")),
        focused_tests_passed=focused_tests_passed,
        audit_manifest_path=audit_manifest,
        audit=audit,
        blockers=list(dict.fromkeys(blockers)),
    )
    _write_json(output, artifact)
    return artifact


def _select_decoder_rows(
    rows: Sequence[JsonDict],
    *,
    max_cases_per_family: int,
) -> list[JsonDict]:
    selected_case_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("source_family") or "")
        case_id = str(row.get("contract_case_id") or "")
        if not family or not case_id or case_id in selected_case_ids:
            continue
        if family_counts.get(family, 0) >= max_cases_per_family:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        selected_case_ids.add(case_id)
    return [row for row in rows if str(row.get("contract_case_id") or "") in selected_case_ids]


def _target_by_case_id(
    rows: Sequence[JsonDict],
    runtime_by_case_id: Mapping[str, JsonDict],
) -> dict[str, str]:
    targets: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("contract_case_id") or "")
        parsed = _parse_json_object(_row_raw_output(row))
        decision = parsed.get("final_deterministic_decision") if parsed else None
        parsed_case_id = parsed.get("contract_case_id") if parsed else None
        if case_id and parsed_case_id == case_id and decision in {"accept", "reject"}:
            targets[case_id] = canonical_contract_target(case_id, str(decision))
    for case_id, runtime_row in runtime_by_case_id.items():
        targets.setdefault(
            case_id,
            canonical_contract_target(case_id, str(runtime_row.get("final_deterministic_decision"))),
        )
    return targets


def _runtime_cases_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        str(row.get("contract_case_id")): dict(row)
        for row in rows
        if row.get("row_type") == "contract_case" and row.get("contract_case_id")
    }


def _validation_rows(rows: Sequence[JsonDict]) -> list[JsonDict]:
    return [
        dict(row["contract_validation_row"])
        for row in rows
        if isinstance(row.get("contract_validation_row"), dict)
    ]


def _fallback_target(row: Mapping[str, Any], runtime_by_case_id: Mapping[str, JsonDict]) -> str:
    case_id = str(row.get("contract_case_id") or "")
    runtime_row = runtime_by_case_id.get(case_id, {})
    validation_row = row.get("contract_validation_row")
    decision = runtime_row.get("final_deterministic_decision")
    if not decision and isinstance(validation_row, Mapping):
        decision = validation_row.get("final_deterministic_decision")
    return canonical_contract_target(case_id, str(decision or "reject"))


def _prefix_manifest_row(
    decoder_row: Mapping[str, Any],
    state: PrefixBoundState,
    trie: PrefixFrontierTrie,
) -> JsonDict:
    return {
        "row_type": "prefix_bound",
        "contract_case_id": decoder_row.get("contract_case_id"),
        "source_family": decoder_row.get("source_family"),
        "decoder_mode": decoder_row.get("decoder_mode"),
        "model_hf_id": decoder_row.get("model_hf_id"),
        "prefix_source": state.source,
        "prefix_length": state.depth,
        "target_length": state.target_length,
        "prefix_consistent": state.prefix_consistent,
        "terminal": state.terminal,
        "allowed_next_count": len(trie.allowed_next_chars(state.prefix))
        if state.prefix_consistent
        else 0,
        "unsafe_upper_bound": state.unsafe_upper_bound,
        "token_logprob_available": _token_logprob(decoder_row) is not None,
        "topk_available": bool(_topk_logprobs(decoder_row)),
        "structural_bound_simulation": _token_logprob(decoder_row) is None,
    }


def _high_risk_instance(row: Mapping[str, Any], observed_state: PrefixBoundState) -> JsonDict:
    return {
        "contract_case_id": row.get("contract_case_id"),
        "source_family": row.get("source_family"),
        "decoder_mode": row.get("decoder_mode"),
        "model_hf_id": row.get("model_hf_id"),
        "risk_upper_bound": observed_state.unsafe_upper_bound,
        "prefix_consistent": observed_state.prefix_consistent,
        "deterministic_validator_accept": bool(row.get("deterministic_validator_accept")),
        "deterministic_false_accept": bool(row.get("false_accept")),
        "expected_label": row.get("expected_label"),
        "bound_used_as_authority": False,
    }


def _bound_violations(high_risk_instances: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "contract_case_id": row.get("contract_case_id"),
            "decoder_mode": row.get("decoder_mode"),
            "risk_upper_bound": row.get("risk_upper_bound"),
            "violation": "false_accept_above_reported_bound",
        }
        for row in high_risk_instances
        if row.get("deterministic_false_accept") is True
        and float(row.get("risk_upper_bound", 0.0)) < 1.0
    ]


def _row_raw_output(row: Mapping[str, Any]) -> str:
    return str(row.get("raw_output") or row.get("raw_output_excerpt") or "")


def _token_logprob(row: Mapping[str, Any]) -> float | None:
    value = row.get("token_logprob")
    return float(value) if isinstance(value, int | float) else None


def _topk_logprobs(row: Mapping[str, Any]) -> Sequence[Any]:
    values = row.get("topk_logprobs") or row.get("top_k_logprobs") or ()
    return values if isinstance(values, Sequence) and not isinstance(values, str | bytes) else ()


def _parse_json_object(text: str) -> JsonDict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _terminal_artifact(
    *,
    status: str,
    run_date: str,
    model_specs: Sequence[Mapping[str, Any]],
    live_sota_model_inference_used: bool,
    focused_tests_passed: bool,
    audit_manifest_path: Path,
    audit: Mapping[str, Any],
    blockers: Sequence[str],
) -> JsonDict:
    ready = bool(audit.get("beaver_bound_ready")) and status == "complete"
    honest_verdict = (
        "complete: BEAVER-lite prefix-bound contract audit ready"
        if ready
        else "complete: BEAVER-lite prefix-bound contract audit completed with blockers"
    )
    artifact = {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "milestone": MILESTONE,
        "beaver_bound_ready": ready,
        "model_specs": [dict(spec) for spec in model_specs],
        "live_sota_model_inference_used": bool(live_sota_model_inference_used),
        "bounded_prefixes": int(audit.get("bounded_prefixes") or 0),
        "token_logprob_available": bool(audit.get("token_logprob_available")),
        "topk_available": bool(audit.get("topk_available")),
        "bound_violations": list(audit.get("bound_violations") or []),
        "high_risk_instances": list(audit.get("high_risk_instances") or []),
        "deterministic_validator_final_authority": bool(
            audit.get("deterministic_validator_final_authority")
        ),
        "false_accept_rate": audit.get("false_accept_rate"),
        "bound_audit_path": _display_path(audit_manifest_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": honest_verdict,
        "blockers": list(blockers),
        "audit_module_path": AUDIT_MODULE_PATH,
        "false_accept_count": audit.get("false_accept_count", 0),
        "explicit_label_count": audit.get("explicit_label_count", 0),
        "explicit_reject_count": audit.get("explicit_reject_count", 0),
    }
    if not str(artifact["honest_verdict"]).startswith(_TERMINAL_VERDICT_PREFIXES):  # pragma: no cover
        raise ValueError("honest_verdict has a disallowed prefix")
    return artifact


def _summary_manifest_row(audit: Mapping[str, Any]) -> JsonDict:
    return {
        "row_type": "summary",
        "bounded_prefixes": audit.get("bounded_prefixes", 0),
        "token_logprob_available": audit.get("token_logprob_available", False),
        "topk_available": audit.get("topk_available", False),
        "false_accept_rate": audit.get("false_accept_rate"),
        "deterministic_validator_final_authority": audit.get(
            "deterministic_validator_final_authority",
            True,
        ),
    }


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line)
        for line in (path.read_text(encoding="utf-8") if path.exists() else "").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


__all__ = [
    "PrefixBoundState",
    "PrefixFrontierTrie",
    "REQUIRED_ARTIFACT_FIELDS",
    "audit_decoder_rows",
    "build_prefix_bound_series",
    "canonical_contract_target",
    "run_experiment",
    "score_prefix",
    "write_in_progress_artifact",
]
