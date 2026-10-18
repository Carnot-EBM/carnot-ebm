"""Safe-DSL verifier induction pack for Exp 1507.

Spec: REQ-VERIFY-1507, SCENARIO-VERIFY-1507.

AutoPyVerifier-style systems are useful because they let a model propose many
small executable verifiers and then search for a compact set.  Carnot keeps the
trust boundary tighter: model output may only describe a tiny JSON DSL.  This
module compiles that DSL into deterministic row predicates and rejects anything
that resembles Python execution, filesystem access, network access, imports,
or non-deterministic logic before a candidate can affect headline metrics.
"""

from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json"
)
DEFAULT_INDUCTION_MANIFEST_PATH = Path("results/safe_dsl_verifier_induction_1507.jsonl")
DEFAULT_CERTIFICATE_MANIFEST_PATH = Path("results/cctu_trigger_certificates_1493.jsonl")
DEFAULT_VALIDATOR_MANIFEST_PATH = Path("results/constrainprompt_validator_manifest_1494.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = cctu.MANDATED_MODEL_SPECS
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "verifier_induction_ready",
    "labeled_rows_loaded",
    "candidate_verifiers_proposed",
    "candidate_verifiers_compiled",
    "verifier_compile_rate",
    "verifier_set_size",
    "verifier_coverage_rate",
    "verifier_false_accept_rate",
    "baseline_validator_coverage_rate",
    "induction_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)

ALLOWED_RULE_PATHS: frozenset[str] = frozenset(
    {
        "row_id",
        "source",
        "family",
        "lane",
        "parser_result.parsed",
        "trigger_token_present",
        "validator_result.case_id_valid",
        "validator_result.final_answer_valid",
        "validator_result.tool_call_structure_valid",
        "validator_result.tool_result_consistent",
        "validator_result.verifier_outcome_valid",
        "verifier_result.base_valid",
        "verifier_result.false_accept",
        "validator_compiled",
        "known_good_passed",
        "known_bad_rejected",
        "compiled_validator.kind",
    }
)
ALLOWED_OPS: frozenset[str] = frozenset({"equals", "exists", "not_null", "is_true", "is_false"})
ALLOWED_TARGET_KEYS: frozenset[str] = frozenset({"source", "family", "lane"})
FORBIDDEN_TEXT_TOKENS: tuple[str, ...] = (
    "__import__",
    "import ",
    "eval(",
    "exec(",
    "open(",
    "pathlib",
    "subprocess",
    "os.",
    "socket",
    "requests.",
    "urllib",
    "http://",
    "https://",
    "random.",
    "numpy.random",
    "time.",
    "datetime.",
    "uuid.",
    "secrets.",
)

ResolverFn = Callable[[str], str | None]
CachedPairFn = Callable[..., list[JsonDict] | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectCandidateProposalsFn = Callable[[JsonDict, list["LabeledVerifierRow"]], JsonDict]


@dataclass(frozen=True)
class LabeledVerifierRow:
    """One row with a deterministic accept/reject label.

    The label is the ground truth used for false-accept accounting.  Candidate
    DSL rules are evaluated against copied manifest features, not by executing
    generated code or trusting a model's verifier self-report.
    """

    row_id: str
    source: str
    family: str
    label_accept: bool
    baseline_accept: bool
    features: JsonDict


@dataclass(frozen=True)
class LoadedRows:
    """Result of loading source manifests for the induction pack."""

    rows: list[LabeledVerifierRow]
    blockers: list[str]


@dataclass(frozen=True)
class CompiledSafeDslCandidate:
    """Compiled safe-DSL verifier candidate or a closed failure."""

    name: str
    compiled: bool
    dsl: JsonDict
    failure_reason: str | None = None
    model_hf_id: str | None = None
    model_name: str | None = None


def load_labeled_rows(
    *,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    validator_manifest_path: Path | str = DEFAULT_VALIDATOR_MANIFEST_PATH,
) -> LoadedRows:
    """Load certificate and validator manifests into deterministic labels."""

    certificate_path = Path(certificate_manifest_path)
    validator_path = Path(validator_manifest_path)
    blockers: list[str] = []
    if not certificate_path.exists():
        blockers.append(f"missing_certificate_manifest:{certificate_path}")
    if not validator_path.exists():
        blockers.append(f"missing_validator_manifest:{validator_path}")
    if blockers:
        return LoadedRows(rows=[], blockers=blockers)

    rows: list[LabeledVerifierRow] = []
    for index, row in enumerate(_read_jsonl(certificate_path)):
        row_id = f"certificate:{row.get('case_id', 'unknown')}:{row.get('lane', 'unknown')}:{index}"
        label_accept = bool(row.get("deterministic_validation_passed"))
        features = dict(row)
        features.update(
            {
                "row_id": row_id,
                "source": "certificate",
                "family": str(row.get("family") or "unknown"),
                "lane": str(row.get("lane") or "unknown"),
            }
        )
        rows.append(
            LabeledVerifierRow(
                row_id=row_id,
                source="certificate",
                family=features["family"],
                label_accept=label_accept,
                baseline_accept=label_accept,
                features=features,
            )
        )

    for index, row in enumerate(_read_jsonl(validator_path)):
        row_id = f"validator:{row.get('prompt_id', 'unknown')}:{index}"
        label_accept = bool(
            row.get("validator_compiled")
            and row.get("known_good_passed")
            and row.get("known_bad_rejected")
            and not row.get("false_accept")
            and not row.get("false_reject")
        )
        features = dict(row)
        features.update(
            {
                "row_id": row_id,
                "source": "validator",
                "family": str(row.get("family") or "unknown"),
                "lane": "validator_audit",
            }
        )
        rows.append(
            LabeledVerifierRow(
                row_id=row_id,
                source="validator",
                family=features["family"],
                label_accept=label_accept,
                baseline_accept=label_accept,
                features=features,
            )
        )
    return LoadedRows(rows=rows, blockers=[])


def baseline_validator_coverage_rate(rows: list[LabeledVerifierRow]) -> float:
    """Return the observed baseline deterministic pass rate over loaded rows."""

    if not rows:
        return 0.0
    return round(sum(row.baseline_accept for row in rows) / len(rows), 6)


def parse_candidate_proposals(output_text: str) -> list[JsonDict]:
    """Extract safe-DSL candidate objects from one model response."""

    for obj in _extract_json_objects(output_text):
        candidates = obj.get("candidates")
        if isinstance(candidates, list):
            return [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]
        if obj.get("kind") == "safe_dsl_verifier":
            return [obj]
    return []


def compile_candidate_from_model_output(
    output_text: str,
    *,
    model_hf_id: str | None = None,
    model_name: str | None = None,
) -> CompiledSafeDslCandidate:
    """Compile the first candidate in a raw model output or fail closed."""

    proposals = parse_candidate_proposals(output_text)
    if not proposals:
        return _compile_failure(
            "unparseable_model_output",
            "no_json_candidate_object",
            model_hf_id=model_hf_id,
            model_name=model_name,
        )
    return compile_candidate(
        proposals[0],
        raw_text=output_text,
        model_hf_id=model_hf_id,
        model_name=model_name,
    )


def compile_candidate(
    candidate: Mapping[str, Any],
    *,
    raw_text: str | None = None,
    model_hf_id: str | None = None,
    model_name: str | None = None,
) -> CompiledSafeDslCandidate:
    """Compile one JSON safe-DSL candidate into a deterministic predicate."""

    name = _safe_candidate_name(str(candidate.get("name") or "unnamed_candidate"))
    unsafe_reason = _unsafe_reason(raw_text) or _unsafe_reason(candidate)
    if unsafe_reason:
        return _compile_failure(
            name,
            unsafe_reason,
            model_hf_id=model_hf_id,
            model_name=model_name,
        )
    if candidate.get("kind") != "safe_dsl_verifier":
        return _compile_failure(
            name,
            "unsupported_candidate_kind",
            model_hf_id=model_hf_id,
            model_name=model_name,
        )

    target = candidate.get("target")
    if not isinstance(target, dict):
        return _compile_failure(name, "target_missing", model_hf_id=model_hf_id, model_name=model_name)
    if set(target) - set(ALLOWED_TARGET_KEYS):
        return _compile_failure(
            name,
            "target_contains_unsupported_key",
            model_hf_id=model_hf_id,
            model_name=model_name,
        )
    normalised_target = {
        key: str(value)
        for key, value in target.items()
        if isinstance(value, str) and value.strip()
    }
    if not normalised_target:
        return _compile_failure(name, "target_empty", model_hf_id=model_hf_id, model_name=model_name)

    rules = candidate.get("rules")
    if not isinstance(rules, list) or not rules or len(rules) > 12:
        return _compile_failure(
            name,
            "rules_missing_or_out_of_bounds",
            model_hf_id=model_hf_id,
            model_name=model_name,
        )

    compiled_rules: list[JsonDict] = []
    for rule in rules:
        if not isinstance(rule, dict):
            return _compile_failure(name, "rule_not_object", model_hf_id=model_hf_id, model_name=model_name)
        failure = _rule_compile_failure(rule)
        if failure:
            return _compile_failure(name, failure, model_hf_id=model_hf_id, model_name=model_name)
        compiled_rule = {"path": str(rule["path"]), "op": str(rule["op"])}
        if "value" in rule:
            compiled_rule["value"] = rule["value"]
        compiled_rules.append(compiled_rule)

    return CompiledSafeDslCandidate(
        name=name,
        compiled=True,
        dsl={
            "name": name,
            "kind": "safe_dsl_verifier",
            "target": normalised_target,
            "rules": compiled_rules,
        },
        failure_reason=None,
        model_hf_id=model_hf_id,
        model_name=model_name,
    )


def score_candidate(
    candidate: CompiledSafeDslCandidate,
    rows: list[LabeledVerifierRow],
) -> JsonDict:
    """Score one candidate against labeled rows with deterministic false-accept accounting."""

    positives = {row.row_id for row in rows if row.label_accept}
    negatives = {row.row_id for row in rows if not row.label_accept}
    if not candidate.compiled:
        return {
            "name": candidate.name,
            "compiled": False,
            "failure_reason": candidate.failure_reason,
            "coverage_rate": 0.0,
            "false_accept_rate": 0.0,
            "true_accept_count": 0,
            "false_accept_count": 0,
            "accepted_labeled_row_ids": [],
            "true_accept_row_ids": [],
            "false_accept_row_ids": [],
            "model_hf_id": candidate.model_hf_id,
            "model_name": candidate.model_name,
        }

    accepted = {row.row_id for row in rows if candidate_accepts_row(candidate, row)}
    true_accepts = sorted(accepted & positives)
    false_accepts = sorted(accepted & negatives)
    return {
        "name": candidate.name,
        "compiled": True,
        "failure_reason": None,
        "coverage_rate": round(len(true_accepts) / len(positives), 6) if positives else 0.0,
        "false_accept_rate": round(len(false_accepts) / len(negatives), 6) if negatives else 0.0,
        "true_accept_count": len(true_accepts),
        "false_accept_count": len(false_accepts),
        "accepted_labeled_row_ids": sorted(accepted),
        "true_accept_row_ids": true_accepts,
        "false_accept_row_ids": false_accepts,
        "model_hf_id": candidate.model_hf_id,
        "model_name": candidate.model_name,
    }


def candidate_accepts_row(
    candidate: CompiledSafeDslCandidate,
    row: LabeledVerifierRow,
) -> bool:
    """Return whether a compiled DSL candidate accepts one labeled row."""

    if not candidate.compiled:
        return False
    target = candidate.dsl["target"]
    for key, expected in target.items():
        if expected != "*" and str(row.features.get(key)) != expected:
            return False
    for rule in candidate.dsl["rules"]:
        value = _get_path(row.features, str(rule["path"]))
        if not _rule_accepts(value, rule):
            return False
    return True


def search_compact_verifier_set(
    scores: list[JsonDict],
    rows: list[LabeledVerifierRow],
    *,
    max_set_size: int = 5,
) -> JsonDict:
    """Find the smallest zero-false-accept set with maximum positive coverage."""

    positives = {row.row_id for row in rows if row.label_accept}
    negatives = {row.row_id for row in rows if not row.label_accept}
    usable = [
        score
        for score in scores
        if score.get("compiled")
        and int(score.get("false_accept_count") or 0) == 0
        and int(score.get("true_accept_count") or 0) > 0
    ]
    best_names: list[str] = []
    best_true: set[str] = set()
    best_false: set[str] = set()
    limit = min(max_set_size, len(usable))
    for size in range(1, limit + 1):
        for combo in itertools.combinations(usable, size):
            true_rows = set().union(*(set(score["true_accept_row_ids"]) for score in combo))
            false_rows = set().union(*(set(score["false_accept_row_ids"]) for score in combo))
            if false_rows:
                continue
            names = sorted(str(score["name"]) for score in combo)
            if len(true_rows) > len(best_true) or (
                len(true_rows) == len(best_true)
                and (not best_names or len(names) < len(best_names) or names < best_names)
            ):
                best_names = names
                best_true = true_rows
                best_false = false_rows
        if len(best_true) == len(positives) and best_names:
            break
    return {
        "row_type": "selected_set_summary",
        "candidate_names": best_names,
        "verifier_set_size": len(best_names),
        "verifier_coverage_rate": round(len(best_true) / len(positives), 6) if positives else 0.0,
        "verifier_false_accept_rate": round(len(best_false) / len(negatives), 6) if negatives else 0.0,
        "accepted_labeled_row_ids": sorted(best_true),
        "false_accept_row_ids": sorted(best_false),
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable in-progress artifact before any expensive work."""

    payload = _empty_artifact(
        status="in_progress",
        run_date=run_date,
        honest_verdict="complete: in-progress Exp 1507 bootstrap artifact",
    )
    _write_json(Path(output_path), payload)
    return payload


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    induction_manifest_path: Path | str = DEFAULT_INDUCTION_MANIFEST_PATH,
    certificate_manifest_path: Path | str = DEFAULT_CERTIFICATE_MANIFEST_PATH,
    validator_manifest_path: Path | str = DEFAULT_VALIDATOR_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] | None = None,
    collect_candidate_proposals_fn: CollectCandidateProposalsFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run bounded safe-DSL induction and write manifest plus terminal artifact."""

    output = Path(output_path)
    induction_manifest = Path(induction_manifest_path)
    write_in_progress_artifact(output, run_date=run_date)
    gpu_probe = (gpu_probe_fn or probe_gpu)()
    loaded = load_labeled_rows(
        certificate_manifest_path=certificate_manifest_path,
        validator_manifest_path=validator_manifest_path,
    )
    if loaded.blockers:
        _write_jsonl(induction_manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            rows=[],
            candidate_rows=[],
            selected_summary=search_compact_verifier_set([], []),
            manifest_path=induction_manifest,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=loaded.blockers,
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    specs = list(resolve_model_specs() if model_specs is None else model_specs)
    if not specs:
        _write_jsonl(induction_manifest, [])
        artifact = _terminal_artifact(
            run_date=run_date,
            rows=loaded.rows,
            candidate_rows=[],
            selected_summary=search_compact_verifier_set([], loaded.rows),
            manifest_path=induction_manifest,
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=["no_mandated_sota_gguf_model_available"],
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    collector = collect_candidate_proposals_fn or collect_live_candidate_proposals
    collection = collector(dict(specs[0]), loaded.rows)
    model_attempts = [dict(collection.get("summary") or {})]
    candidate_rows = _build_candidate_rows(collection, loaded.rows)
    scores = [dict(row["score"]) for row in candidate_rows]
    selected = search_compact_verifier_set(scores, loaded.rows)
    _write_jsonl(induction_manifest, [*candidate_rows, selected])

    blockers = [
        str(summary.get("blocker"))
        for summary in model_attempts
        if summary.get("model_used") is not True and summary.get("blocker")
    ]
    if not _live_sota_candidates_present(candidate_rows):
        blockers.append("live_sota_candidate_generation_unavailable")

    artifact = _terminal_artifact(
        run_date=run_date,
        rows=loaded.rows,
        candidate_rows=candidate_rows,
        selected_summary=selected,
        manifest_path=induction_manifest,
        model_attempts=model_attempts,
        gpu_probe=gpu_probe,
        blockers=list(dict.fromkeys(blockers)),
        tests_run=tests_run,
    )
    _write_json(output, artifact)
    return artifact


def resolve_model_specs(
    *,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
) -> list[JsonDict]:  # pragma: no cover - external cache discovery.
    """Resolve mandated local SOTA GGUF specs without legacy small fallbacks."""

    pair_resolver = cached_pair_fn or _cached_sota_pair
    pair = pair_resolver(gpu_indices=(0, 1))
    if pair:
        return pair
    resolver = resolver_fn or cctu._default_resolver
    specs: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        model_path = resolver(str(spec["hf_id"]))
        if model_path:
            specs.append({**spec, "model_path": model_path})
    return specs


def collect_live_candidate_proposals(
    spec: JsonDict,
    rows: list[LabeledVerifierRow],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:  # pragma: no cover - live GGUF path exercised by experiment run.
    """Ask one mandated local GGUF model for safe-DSL verifier proposals."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = spec.get("model_path") or (resolver or cctu._default_resolver)(hf_id)
    if not model_path:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "model_not_cached",
            },
            "rows": [],
        }

    env_details = (env_preparer or cctu.prepare_llama_environment)()
    ok, llama_class, import_error = (llama_importer or cctu._default_llama_importer)()
    if not ok or llama_class is None:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": import_error or "llama_cpp_import_failed",
                "env_details": env_details,
            },
            "rows": [],
        }

    load_start = time.monotonic()
    try:
        llm = llama_class(
            model_path=str(model_path),
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=8192,
            seed=1507,
            verbose=False,
        )
    except Exception as exc:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - load_start, 6),
                "env_details": env_details,
            },
            "rows": [],
        }

    started = time.monotonic()
    try:
        try:
            result = llm(
                build_induction_prompt(rows),
                max_tokens=2048,
                temperature=0.0,
                top_p=1.0,
                stop=["</s>", "<eos>"],
                echo=False,
            )
            output_text = cctu._completion_text(result)
            blocker = None if output_text.strip() else "empty_generation"
        except Exception as exc:
            output_text = ""
            blocker = f"{type(exc).__name__}: {exc}"
        proposal_rows = [
            {
                "model_hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "generation_source": "live_sota_llamacpp",
                "output_text": output_text,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "blocker": blocker,
            }
        ]
    finally:
        cctu._close_llama(llm)

    model_used = any(row.get("blocker") is None for row in proposal_rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": str(model_path),
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_candidate_generation",
            "env_details": env_details,
        },
        "rows": proposal_rows,
    }


def build_induction_prompt(rows: list[LabeledVerifierRow]) -> str:  # pragma: no cover
    """Build the bounded safe-DSL proposal prompt for the local SOTA model."""

    source_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    positive_count = 0
    for row in rows:
        source_counts[row.source] = source_counts.get(row.source, 0) + 1
        family_counts[row.family] = family_counts.get(row.family, 0) + 1
        positive_count += int(row.label_accept)
    example = {
        "candidates": [
            {
                "name": "certificate_transcript_consistency",
                "kind": "safe_dsl_verifier",
                "target": {"source": "certificate"},
                "rules": [
                    {"path": "parser_result.parsed", "op": "is_true"},
                    {"path": "validator_result.case_id_valid", "op": "is_true"},
                    {"path": "validator_result.final_answer_valid", "op": "is_true"},
                    {"path": "validator_result.tool_call_structure_valid", "op": "is_true"},
                    {"path": "validator_result.tool_result_consistent", "op": "is_true"},
                    {"path": "validator_result.verifier_outcome_valid", "op": "is_true"},
                ],
            },
            {
                "name": "compiled_validator_sanity",
                "kind": "safe_dsl_verifier",
                "target": {"source": "validator"},
                "rules": [
                    {"path": "validator_compiled", "op": "is_true"},
                    {"path": "known_good_passed", "op": "is_true"},
                    {"path": "known_bad_rejected", "op": "is_true"},
                ],
            },
        ]
    }
    return (
        "/no_think\n"
        "Return exactly one JSON object and no other text. The first character "
        "of your response must be `{`. Do not emit <think> or prose. Propose "
        "compact verifier candidates for Carnot's safe JSON DSL only. Do not "
        "write Python, imports, filesystem access, network calls, eval, exec, "
        "randomness, timestamps, or any executable code.\n"
        "Safe response shape:\n"
        f"{json.dumps(example, sort_keys=True)}\n"
        f"Loaded labeled rows: {len(rows)}; positives: {positive_count}; "
        f"source counts: {_kv_counts(source_counts)}; "
        f"family counts: {_kv_counts(family_counts)}.\n"
        f"Allowed rule paths: {json.dumps(sorted(ALLOWED_RULE_PATHS))}.\n"
        f"Allowed ops: {json.dumps(sorted(ALLOWED_OPS))}.\n"
        "Return the JSON object now."
    )


def probe_gpu() -> JsonDict:  # pragma: no cover - host hardware probe.
    """Return a JSON-safe NVIDIA GPU probe for the artifact."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"nvidia_smi_available": False, "gpu_count": 0, "gpus": [], "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {"nvidia_smi_available": False, "gpu_count": 0, "gpus": [], "error": result.stderr.strip() or "nvidia-smi failed"}
    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, _, memory = line.partition(",")
        gpus.append({"name": name.strip(), "memory_total": memory.strip()})
    return {"nvidia_smi_available": True, "gpu_count": len(gpus), "gpus": gpus}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entry point for conductor and manual runs."""

    _ = list(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment()
    print(
        "[exp1507] "
        f"ready={artifact['verifier_induction_ready']} "
        f"compiled={artifact['candidate_verifiers_compiled']} "
        f"false_accept={artifact['verifier_false_accept_rate']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


def _build_candidate_rows(collection: JsonDict, rows: list[LabeledVerifierRow]) -> list[JsonDict]:
    candidate_rows: list[JsonDict] = []
    for proposal_row in collection.get("rows") or []:
        output_text = str(proposal_row.get("output_text") or "")
        proposals = parse_candidate_proposals(output_text)
        if not proposals and output_text.strip():
            compiled = compile_candidate_from_model_output(
                output_text,
                model_hf_id=(
                    str(proposal_row["model_hf_id"]) if proposal_row.get("model_hf_id") else None
                ),
                model_name=str(proposal_row["model_name"]) if proposal_row.get("model_name") else None,
            )
            score = score_candidate(compiled, rows)
            candidate_rows.append(
                {
                    "row_type": "candidate",
                    "candidate_index": 0,
                    "candidate_name": compiled.name,
                    "model_hf_id": proposal_row.get("model_hf_id"),
                    "model_name": proposal_row.get("model_name"),
                    "generation_source": proposal_row.get("generation_source"),
                    "elapsed_seconds": proposal_row.get("elapsed_seconds"),
                    "model_blocker": proposal_row.get("blocker"),
                    "raw_candidate": {"raw_output_excerpt": output_text[:500]},
                    "compiled": False,
                    "compiled_dsl": compiled.dsl,
                    "compile_failure_reason": compiled.failure_reason,
                    "score": score,
                }
            )
            continue
        for index, proposal in enumerate(proposals):
            model_hf_id = proposal_row.get("model_hf_id")
            model_name = proposal_row.get("model_name")
            compiled = compile_candidate(
                proposal,
                raw_text=output_text,
                model_hf_id=str(model_hf_id) if model_hf_id else None,
                model_name=str(model_name) if model_name else None,
            )
            score = score_candidate(compiled, rows)
            candidate_rows.append(
                {
                    "row_type": "candidate",
                    "candidate_index": index,
                    "candidate_name": compiled.name,
                    "model_hf_id": model_hf_id,
                    "model_name": model_name,
                    "generation_source": proposal_row.get("generation_source"),
                    "elapsed_seconds": proposal_row.get("elapsed_seconds"),
                    "model_blocker": proposal_row.get("blocker"),
                    "raw_candidate": proposal,
                    "compiled": compiled.compiled,
                    "compiled_dsl": compiled.dsl,
                    "compile_failure_reason": compiled.failure_reason,
                    "score": score,
                }
            )
    return candidate_rows


def _terminal_artifact(
    *,
    run_date: str,
    rows: list[LabeledVerifierRow],
    candidate_rows: list[JsonDict],
    selected_summary: JsonDict,
    manifest_path: Path,
    model_attempts: list[JsonDict],
    gpu_probe: JsonDict,
    blockers: list[str],
    tests_run: list[str] | None,
) -> JsonDict:
    candidate_count = len(candidate_rows)
    compiled_count = sum(bool(row["compiled"]) for row in candidate_rows)
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    live_used = _live_sota_candidates_present(candidate_rows)
    false_accept_rate = selected_summary.get("verifier_false_accept_rate")
    ready = bool(live_used and compiled_count > 0 and false_accept_rate is not None and not blockers)
    artifact = _empty_artifact(
        status="complete" if ready else "blocked",
        run_date=run_date,
        honest_verdict=(
            "complete: safe-DSL verifier induction pack measured on live local SOTA GGUF proposals"
            if ready
            else "complete: blocked before safe-DSL verifier induction headline readiness"
        ),
    )
    artifact.update(
        {
            "live_sota_model_inference_used": bool(live_used),
            "verifier_induction_ready": bool(ready),
            "labeled_rows_loaded": len(rows),
            "candidate_verifiers_proposed": candidate_count,
            "candidate_verifiers_compiled": compiled_count,
            "verifier_compile_rate": round(compiled_count / candidate_count, 6) if candidate_count else 0.0,
            "verifier_set_size": int(selected_summary["verifier_set_size"]),
            "verifier_coverage_rate": float(selected_summary["verifier_coverage_rate"]),
            "verifier_false_accept_rate": float(selected_summary["verifier_false_accept_rate"]),
            "baseline_validator_coverage_rate": baseline_validator_coverage_rate(rows),
            "induction_manifest_path": _display_path(manifest_path),
            "models_used": models_used,
            "gpu_probe": gpu_probe,
            "blockers": blockers,
            "model_attempts": model_attempts,
            "manifest_rows": candidate_count + (1 if candidate_rows or rows else 0),
            "tests_run": list(tests_run or []),
        }
    )
    return artifact


def _empty_artifact(*, status: str, run_date: str, honest_verdict: str) -> JsonDict:
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "verifier_induction_ready": False,
        "labeled_rows_loaded": 0,
        "candidate_verifiers_proposed": 0,
        "candidate_verifiers_compiled": 0,
        "verifier_compile_rate": 0.0,
        "verifier_set_size": 0,
        "verifier_coverage_rate": 0.0,
        "verifier_false_accept_rate": 0.0,
        "baseline_validator_coverage_rate": 0.0,
        "induction_manifest_path": _display_path(DEFAULT_INDUCTION_MANIFEST_PATH),
        "models_used": [],
        "gpu_probe": {},
        "blockers": [],
        "honest_verdict": honest_verdict,
    }


def _compile_failure(
    name: str,
    reason: str,
    *,
    model_hf_id: str | None,
    model_name: str | None,
) -> CompiledSafeDslCandidate:
    return CompiledSafeDslCandidate(
        name=_safe_candidate_name(name),
        compiled=False,
        dsl={"kind": "compile_failure"},
        failure_reason=reason,
        model_hf_id=model_hf_id,
        model_name=model_name,
    )


def _rule_compile_failure(rule: Mapping[str, Any]) -> str | None:
    if set(rule) - {"path", "op", "value"}:
        return "rule_contains_unsupported_key"
    path = rule.get("path")
    op = rule.get("op")
    if path not in ALLOWED_RULE_PATHS:
        return "unsupported_rule_path"
    if op not in ALLOWED_OPS:
        return "unsupported_rule_op"
    if op == "equals" and "value" not in rule:
        return "equals_rule_missing_value"
    if "value" in rule and isinstance(rule["value"], (dict, list)):
        return "rule_value_must_be_scalar"
    return None


def _rule_accepts(value: Any, rule: Mapping[str, Any]) -> bool:
    op = rule["op"]
    if op == "exists":
        return value is not _MISSING
    if value is _MISSING:
        return False
    if op == "not_null":
        return value is not None
    if op == "is_true":
        return value is True
    if op == "is_false":
        return value is False
    if op == "equals":
        return _canonical(value) == _canonical(rule.get("value"))
    return False


def _unsafe_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.casefold()
        for token in FORBIDDEN_TEXT_TOKENS:
            if token in lowered:
                return f"unsafe_token:{token}"
        return None
    if isinstance(value, Mapping):
        for item in value.values():
            reason = _unsafe_reason(item)
            if reason:
                return reason
    if isinstance(value, list):
        for item in value:
            reason = _unsafe_reason(item)
            if reason:
                return reason
    return None


def _safe_candidate_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())[:96]
    return cleaned or "unnamed_candidate"


_MISSING = object()


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


def _live_sota_candidates_present(candidate_rows: list[JsonDict]) -> bool:
    mandated = {str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
    return any(
        row.get("generation_source") == "live_sota_llamacpp"
        and row.get("model_blocker") is None
        and row.get("model_hf_id") in mandated
        for row in candidate_rows
    )


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _extract_json_objects(text: str) -> list[JsonDict]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    objects: list[JsonDict] = []
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def _kv_counts(counts: Mapping[str, int]) -> str:  # pragma: no cover
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

    return cached_sota_pair(**kwargs)


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(cctu._repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_CERTIFICATE_MANIFEST_PATH",
    "DEFAULT_INDUCTION_MANIFEST_PATH",
    "DEFAULT_VALIDATOR_MANIFEST_PATH",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "CompiledSafeDslCandidate",
    "LabeledVerifierRow",
    "LoadedRows",
    "baseline_validator_coverage_rate",
    "candidate_accepts_row",
    "collect_live_candidate_proposals",
    "compile_candidate",
    "compile_candidate_from_model_output",
    "load_labeled_rows",
    "main",
    "parse_candidate_proposals",
    "probe_gpu",
    "resolve_model_specs",
    "run_experiment",
    "score_candidate",
    "search_compact_verifier_set",
    "write_in_progress_artifact",
]
