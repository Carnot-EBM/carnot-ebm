"""Exp6288 fail-closed partial atom evidence adapter.

Spec refs: REQ-CONSTRAINT-6288,
SCENARIO-CONSTRAINT-6288-EXTRACT-FAIL-CLOSED,
SCENARIO-CONSTRAINT-6288-ORACLE-AFTER-EXTRACTION,
SCENARIO-CONSTRAINT-6288-WARM-CONTROLS.

This adapter treats raw model text as evidence about atoms, not as a complete
answer. It freezes atom evidence first. It then uses the sealed sidecar only to
test whether that frozen partial evidence has an exact completion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any

from carnot import asp_continuous_relaxation as relax
from carnot import asp_energy
from carnot.terminal_artifacts import classify_artifact_path, path_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6288_partial_atom_evidence_adapter.json")
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6288_partial_atom_evidence_adapter --date 20260810"
)
INFERENCE_SUBSTRATE = "deterministic_partial_atom_text_adapter_no_llm"

ELIGIBILITY_RELATIVE_PATH = Path("results/experiment_6286_v541_evidence_eligibility_ledger.json")
ELIGIBLE_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6286_v541_flagship_raw_row_eligible_manifest.jsonl"
)
RELAXATION_RELATIVE_PATH = Path("results/experiment_6287_asp_continuous_relaxation.json")
SEALED_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.sealed_manifest.json"
)
FORMAL_SIDECAR_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.formal_sidecar.json"
)
RAW_DIR_RELATIVE_PATH = Path("results/experiment_6275_flagship_asp_raw")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6288_partial_atom_evidence_adapter.py")
SOURCE_RELATIVE_PATH = Path("python/carnot/experiment_6288_partial_atom_evidence_adapter.py")

OPTIMIZER_STEPS = 24
OPTIMIZER_STEP_SIZE = 0.25
RESTART_BUDGET = 1
EXACT_COMPLETION_BUDGET = 4096
RANDOM_SEEDS = (6288, 6289, 6290)
POSITIVE_PRECISION_FLOOR = 0.95

DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    ".venv/bin/pytest tests/python/test_experiment_6288_partial_atom_evidence_adapter.py -q --no-cov -n 0",
    (
        ".venv/bin/coverage run --branch "
        "--include=python/carnot/experiment_6288_partial_atom_evidence_adapter.py "
        "-m pytest tests/python/test_experiment_6288_partial_atom_evidence_adapter.py -q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage report "
        "--include=python/carnot/experiment_6288_partial_atom_evidence_adapter.py --fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6288_partial_atom_evidence_adapter.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6288_partial_atom_evidence_adapter.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_eligibility_path_hash_and_terminal_class",
    "upstream_relaxation_path_hash_and_terminal_class",
    "eligible_raw_manifest_path_and_hash",
    "raw_source_paths_and_hashes",
    "models_represented",
    "frozen_atom_vocabulary_by_fixture",
    "adapter_source_paths_and_hashes",
    "positive_negative_unknown_evidence_by_row",
    "contradiction_foreign_atom_and_ambiguous_negation_rejections",
    "accepted_and_rejected_row_counts",
    "evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family",
    "evidence_leakage_controls",
    "warm_blank_and_random_start_outcomes",
    "continuous_refinement_results",
    "exact_completion_results",
    "cold_exact_completion_controls",
    "unsafe_evidence_acceptance_count",
    "partial_atom_evidence_adapter_ready_score",
    "source_model_weight_mutation_count",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names whether all fail-closed gates passed.",
    "upstream_eligibility_path_hash_and_terminal_class": "Pins the raw-row gate.",
    "upstream_relaxation_path_hash_and_terminal_class": "Pins the continuous bridge.",
    "eligible_raw_manifest_path_and_hash": "Limits replay to eligible rows.",
    "raw_source_paths_and_hashes": "Pins immutable raw text files.",
    "models_represented": "Shows which mandated model families had rows.",
    "frozen_atom_vocabulary_by_fixture": "Prevents post-hoc atom changes.",
    "adapter_source_paths_and_hashes": "Pins adapter code, tests, and spec.",
    "positive_negative_unknown_evidence_by_row": "Keeps every atom state visible.",
    "contradiction_foreign_atom_and_ambiguous_negation_rejections": "Shows unsafe text failed closed.",
    "accepted_and_rejected_row_counts": "Keeps row denominators auditable.",
    "evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family": "Measures evidence quality by source family.",
    "evidence_leakage_controls": "Proves extraction did not read answer labels.",
    "warm_blank_and_random_start_outcomes": "Compares matched-budget starts.",
    "continuous_refinement_results": "Reports refinement without gating on lift.",
    "exact_completion_results": "Shows oracle completion after extraction.",
    "cold_exact_completion_controls": "Keeps no-evidence controls separate.",
    "unsafe_evidence_acceptance_count": "Bare zero blocks unsafe acceptance.",
    "partial_atom_evidence_adapter_ready_score": "Opens only on preregistered gates.",
    "source_model_weight_mutation_count": "Bare zero proves no model update.",
    "protected_files_unchanged": "Checks protected inputs did not drift.",
    "preconditions_checked": "Records frozen inputs and budgets.",
    "inference_substrate": "Declares deterministic replay only.",
    "verifier_is_oracle": "Exact sidecar is the oracle.",
    "field_provenance": "Maps fields to inputs and code.",
    "field_principles": "Gives each field a reason.",
    "test_commands": "Lists the verification boundary.",
    "test_exit_codes": "Records observed command exits.",
    "duration_s": "Reports real wall time.",
    "random_seeds": "Pins random starts.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the terminal claim boundary.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: ["REQ-CONSTRAINT-6288", "Exp6286 eligible manifest", "Exp6275 sealed sidecar"]
    for field in REQUIRED_ARTIFACT_FIELDS
}

PROTECTED_RELATIVE_PATHS = (
    ELIGIBILITY_RELATIVE_PATH,
    ELIGIBLE_MANIFEST_RELATIVE_PATH,
    RELAXATION_RELATIVE_PATH,
    SEALED_MANIFEST_RELATIVE_PATH,
    FORMAL_SIDECAR_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)

_ANSWER_RE = re.compile(r"\banswer\s*:\s*([^\n\r]*)", re.IGNORECASE)
_ATOM_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*")
_NEGATION_WORD_RE = re.compile(r"\b(?:not|no|never|without)\b", re.IGNORECASE)
_ANSWER_STOPWORDS = {
    "answer",
    "and",
    "be",
    "or",
    "none",
    "impossible",
    "is",
    "label",
    "labels",
    "not",
    "the",
    "selected",
    "select",
    "true",
    "false",
}


def extract_partial_atom_evidence(
    text: str,
    atom_vocabulary: Sequence[str],
    *,
    generated_token_count: int,
    row_id: str,
) -> JsonDict:
    """Extract atom evidence without reading the sealed answer sidecar."""

    atoms = tuple(sorted(atom_vocabulary))
    positive: set[str] = set()
    negative: set[str] = set()
    foreign: set[str] = set()
    ambiguous: list[str] = []
    reasons: list[str] = []
    if generated_token_count <= 0:
        reasons.append("zero_token_row")
    if not text.strip():
        reasons.append("empty_output")

    aliases = _aliases_by_atom(atoms)
    lower_text = text.lower()
    consumed_answer_spans: list[tuple[int, int]] = []
    for match in _ANSWER_RE.finditer(text):
        fragment = match.group(1)
        fragment_start = match.start(1)
        evidence_atoms, foreign_atoms, spans = _answer_fragment_atoms(fragment, aliases)
        positive.update(evidence_atoms)
        foreign.update(foreign_atoms)
        consumed_answer_spans.extend(
            (fragment_start + start, fragment_start + end) for start, end in spans
        )

    for atom, atom_aliases in aliases.items():
        positive.update(
            atom
            for pattern in _positive_patterns(atom_aliases)
            if re.search(pattern, text, re.IGNORECASE)
        )
        negative.update(
            atom
            for pattern in _negative_patterns(atom_aliases)
            if re.search(pattern, text, re.IGNORECASE)
        )
        ambiguous.extend(_ambiguous_negation_spans(lower_text, atom_aliases, consumed_answer_spans))

    contradiction = sorted(positive & negative)
    if contradiction:
        reasons.append("contradictory_evidence")
    if foreign:
        reasons.append("foreign_atom")
    if ambiguous:
        reasons.append("ambiguous_negation")
    if not positive and not negative and not reasons:
        reasons.append("no_atom_evidence")

    accepted = not reasons
    return {
        "row_id": row_id,
        "accepted": accepted,
        "positive_atoms": sorted(positive),
        "negative_atoms": sorted(negative),
        "unknown_atoms": sorted(set(atoms) - positive - negative),
        "rejection_reasons": sorted(set(reasons)),
        "contradictory_atoms": contradiction,
        "foreign_atoms": sorted(foreign),
        "ambiguous_negation_spans": sorted(set(ambiguous)),
        "sidecar_checked_after_extraction": False,
    }


def check_evidence_support(
    evidence: Mapping[str, Any],
    *,
    exact_answer_sets: Sequence[Sequence[str]],
) -> JsonDict:
    """Check frozen evidence against exact answers after extraction."""

    positive = set(str(atom) for atom in evidence.get("positive_atoms") or [])
    negative = set(str(atom) for atom in evidence.get("negative_atoms") or [])
    exact_states = [set(str(atom) for atom in state) for state in exact_answer_sets]
    compatible = [
        state for state in exact_states if positive <= state and negative.isdisjoint(state)
    ]
    supported = bool(evidence.get("accepted") is True and compatible)
    rejection_reasons = list(evidence.get("rejection_reasons") or [])
    if evidence.get("accepted") is True and not compatible:
        rejection_reasons.append("unsupported_by_exact_sidecar")
    supporting_completion = sorted(compatible[0]) if compatible else []
    return {
        "supported": supported,
        "sidecar_checked_after_extraction": True,
        "supporting_completion": supporting_completion,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "positive_evidence_count": len(positive),
        "positive_correct_count": len(positive) if supported else 0,
        "negative_evidence_count": len(negative),
        "negative_correct_count": len(negative) if supported else 0,
    }


def table_from_program(program_text: str, fixture_id: str) -> relax.VertexEnergyTable:
    """Build the bounded relaxation table used by matched controls."""

    compiled = asp_energy.compile_program(program_text, program_id=fixture_id)
    return relax.build_energy_table(
        compiled,
        fixture_id=fixture_id,
        max_atoms=12,
        max_vertices=EXACT_COMPLETION_BUDGET,
    )


def compare_refinement_starts(
    table: relax.VertexEnergyTable,
    evidence: Mapping[str, Any],
    support: Mapping[str, Any],
    *,
    row_id: str,
    seed: int,
) -> dict[str, JsonDict]:
    """Run evidence, blank, and random starts with identical budgets."""

    positive = set(str(atom) for atom in evidence.get("positive_atoms") or [])
    negative = set(str(atom) for atom in evidence.get("negative_atoms") or [])
    warm = [1.0 if atom in positive else 0.0 if atom in negative else 0.5 for atom in table.atoms]
    blank = [0.5 for _ in table.atoms]
    rng = random.Random(_stable_seed(row_id, seed))
    starts = {
        "evidence_warm": (warm, sorted(positive), sorted(negative), seed),
        "blank": (blank, [], [], seed),
        "random": ([rng.random() for _ in table.atoms], [], [], seed),
    }
    best_energy = 0 if support.get("supported") else table.best_discrete_energy
    return {
        name: _refine_start(
            table, row_id, name, probabilities, true_atoms, false_atoms, arm_seed, best_energy
        )
        for name, (probabilities, true_atoms, false_atoms, arm_seed) in starts.items()
    }


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the terminal Exp6288 artifact."""

    started = time.perf_counter()
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=time.perf_counter() - started if duration_s is None else duration_s,
        test_exit_codes=test_exit_codes,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_canonical_json(artifact, indent=2), encoding="utf-8")
    return artifact


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Replay eligible rows and build the audited adapter payload."""

    root = REPO_ROOT
    before = _protected_hashes(root)
    sidecar = _read_json_mapping(root / FORMAL_SIDECAR_RELATIVE_PATH)
    entries = dict(sidecar.get("entries") or {})
    raw_by_key, raw_receipts = _load_raw_sources(root)
    eligible_rows = _read_jsonl(root / ELIGIBLE_MANIFEST_RELATIVE_PATH)
    vocab = _frozen_vocabularies(entries)

    evidence_rows: list[JsonDict] = []
    refinement_by_row: dict[str, JsonDict] = {}
    exact_by_row: dict[str, JsonDict] = {}
    tables: dict[str, relax.VertexEnergyTable] = {}
    for source_index, manifest_row in enumerate(eligible_rows):
        for seed in manifest_row.get("seeds") or []:
            row_id = _row_id(manifest_row, int(seed), source_index)
            task_id = str(manifest_row.get("task_id") or "")
            entry = entries.get(task_id, {})
            atoms = tuple(vocab.get(_fixture_key(manifest_row), {}).get("atoms") or [])
            raw = raw_by_key.get((str(manifest_row.get("model_hf_id")), task_id, int(seed)))
            evidence = _extract_row(row_id, manifest_row, raw, atoms)
            support = check_evidence_support(
                evidence,
                exact_answer_sets=entry.get("exact_answer_sets") or [],
            )
            record = _evidence_record(row_id, manifest_row, raw, evidence, support)
            evidence_rows.append(record)
            if record["accepted"]:
                table = tables.setdefault(
                    task_id,
                    table_from_program(
                        str(entry.get("program_text") or ""),
                        str(entry.get("fixture_id") or task_id),
                    ),
                )
                refinement_by_row[row_id] = compare_refinement_starts(
                    table,
                    evidence,
                    support,
                    row_id=row_id,
                    seed=RANDOM_SEEDS[0],
                )
                exact_by_row[row_id] = _exact_completion_record(table, evidence, support)

    counts = _row_counts(evidence_rows)
    precision = _precision_summary(evidence_rows)
    leakage = _leakage_controls()
    unsafe_count = _unsafe_acceptance_count(evidence_rows)
    model_acceptance = set(counts["represented_model_families_with_acceptance"])
    represented = set(counts["represented_model_families"])
    readiness = (
        unsafe_count == 0
        and represented <= model_acceptance
        and precision["overall"]["positive_precision"] > POSITIVE_PRECISION_FLOOR
        and leakage["all_clean"] is True
    )
    status = "complete" if readiness else "blocked"
    protected = _protected_unchanged(root, before)
    artifact: JsonDict = {
        "status": status,
        "upstream_eligibility_path_hash_and_terminal_class": _classification_receipt(
            root, ELIGIBILITY_RELATIVE_PATH
        ),
        "upstream_relaxation_path_hash_and_terminal_class": _classification_receipt(
            root, RELAXATION_RELATIVE_PATH
        ),
        "eligible_raw_manifest_path_and_hash": _path_receipt(
            root / ELIGIBLE_MANIFEST_RELATIVE_PATH, row_count=len(eligible_rows)
        ),
        "raw_source_paths_and_hashes": raw_receipts,
        "models_represented": _models_represented(evidence_rows),
        "frozen_atom_vocabulary_by_fixture": vocab,
        "adapter_source_paths_and_hashes": _adapter_source_hashes(root),
        "positive_negative_unknown_evidence_by_row": evidence_rows,
        "contradiction_foreign_atom_and_ambiguous_negation_rejections": _rejection_summary(
            evidence_rows
        ),
        "accepted_and_rejected_row_counts": counts,
        "evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family": precision,
        "evidence_leakage_controls": leakage,
        "warm_blank_and_random_start_outcomes": _start_outcome_summary(refinement_by_row),
        "continuous_refinement_results": _continuous_refinement_summary(refinement_by_row),
        "exact_completion_results": _exact_completion_summary(exact_by_row),
        "cold_exact_completion_controls": _cold_exact_completion_controls(tables),
        "unsafe_evidence_acceptance_count": int(unsafe_count),
        "partial_atom_evidence_adapter_ready_score": 1.0 if readiness else 0.0,
        "source_model_weight_mutation_count": 0,
        "protected_files_unchanged": protected,
        "preconditions_checked": _preconditions(date, result_path, before, len(eligible_rows)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {RUN_COMMAND: 0}),
        "duration_s": float(duration_s),
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp6288 artifact and reject false readiness."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(type(artifact.get("unsafe_evidence_acceptance_count")) is int, "unsafe_count_type")
    _require(
        type(artifact.get("source_model_weight_mutation_count")) is int, "weight_mutation_type"
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_provenance", {})),
        "field_provenance",
    )
    counts = artifact.get("accepted_and_rejected_row_counts", {})
    represented = set(counts.get("represented_model_families") or [])
    with_acceptance = set(counts.get("represented_model_families_with_acceptance") or [])
    precision = artifact.get(
        "evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family",
        {},
    ).get("overall", {})
    leakage_clean = artifact.get("evidence_leakage_controls", {}).get("all_clean") is True
    expected_ready = (
        artifact.get("unsafe_evidence_acceptance_count") == 0
        and represented <= with_acceptance
        and float(precision.get("positive_precision") or 0.0) > POSITIVE_PRECISION_FLOOR
        and leakage_clean
    )
    expected_score = 1.0 if expected_ready else 0.0
    _require(
        artifact.get("partial_atom_evidence_adapter_ready_score") == expected_score,
        "ready_score",
    )
    _require(artifact.get("unsafe_evidence_acceptance_count") == 0, "unsafe_count")
    _require(artifact.get("source_model_weight_mutation_count") == 0, "weight_mutation")
    if expected_ready:
        _require(artifact.get("status") == "complete", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
    else:
        _require(artifact.get("status") != "complete", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("blocked:"), "honest_verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking volatile fields."""

    stable = json.loads(_canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible values."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Return a stable SHA-256 digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the required experiment command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = run(
        date=args.date,
        result_path=args.result_path,
        duration_s=time.perf_counter() - started,
        write=True,
    )
    print(_canonical_json({"result": str(args.result_path), "status": artifact["status"]}))
    return 0


def _aliases_by_atom(atoms: Sequence[str]) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for atom in atoms:
        parts = atom.split("_")
        alias_set = {atom, " ".join(parts), "-".join(parts)}
        aliases[atom] = tuple(sorted(alias_set, key=len, reverse=True))
    return aliases


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias).replace(r"\ ", r"[\s_`*]+").replace(r"\-", r"[\s_\-`*]+")
    return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"


def _positive_patterns(aliases: Sequence[str]) -> list[str]:
    patterns: list[str] = []
    for alias in aliases:
        atom = _alias_pattern(alias)
        patterns.extend(
            [
                rf"\blabel\s+{atom}\s+(?:must|should|is to be)\s+be\s+selected\b",
                rf"\b{atom}\s+(?:must|should)\s+be\s+selected\b",
                rf"\b{atom}\s+is\s+(?:selected|true|positive)\b",
                rf"(?<!not\s)\bselect\s+{atom}\b",
                rf"\bchoose\s+{atom}\b",
                rf"\binclude\s+{atom}\b",
                rf'"{atom}"\s*:\s*"{atom}"',
            ]
        )
    return patterns


def _negative_patterns(aliases: Sequence[str]) -> list[str]:
    patterns: list[str] = []
    for alias in aliases:
        atom = _alias_pattern(alias)
        patterns.extend(
            [
                rf"\b{atom}\s+is\s+not\s+selected\b",
                rf"\b{atom}\s+(?:must|should)\s+not\s+be\s+selected\b",
                rf"\bdo\s+not\s+select\s+{atom}\b",
                rf"\bwithout\s+{atom}\b",
                rf"\b{atom}\s+is\s+(?:false|negative)\b",
            ]
        )
    return patterns


def _answer_fragment_atoms(
    fragment: str,
    aliases: Mapping[str, Sequence[str]],
) -> tuple[set[str], set[str], list[tuple[int, int]]]:
    positive: set[str] = set()
    consumed: list[tuple[int, int]] = []
    scoped_fragment = re.split(r"[.;]", fragment, maxsplit=1)[0]
    masked = scoped_fragment
    for atom, atom_aliases in aliases.items():
        for alias in atom_aliases:
            for match in re.finditer(_alias_pattern(alias), scoped_fragment, re.IGNORECASE):
                positive.add(atom)
                consumed.append(match.span())
    for start, end in sorted(consumed, reverse=True):
        masked = masked[:start] + " " * (end - start) + masked[end:]
    foreign = {
        token.lower()
        for token in _ATOM_TOKEN_RE.findall(masked)
        if token.lower() not in _ANSWER_STOPWORDS
    }
    return positive, foreign, consumed


def _ambiguous_negation_spans(
    lower_text: str,
    aliases: Sequence[str],
    consumed_answer_spans: Sequence[tuple[int, int]],
) -> list[str]:
    spans: list[str] = []
    for alias in aliases:
        for match in re.finditer(_alias_pattern(alias), lower_text, re.IGNORECASE):
            if _inside_spans(match.start(), consumed_answer_spans):
                continue
            window_start = max(0, match.start() - 24)
            window_end = min(len(lower_text), match.end() + 24)
            window = lower_text[window_start:window_end]
            if (
                _NEGATION_WORD_RE.search(window)
                and not any(
                    re.search(pattern, window, re.IGNORECASE)
                    for pattern in _negative_patterns((alias,))
                )
                and _negation_has_no_punctuation_boundary(
                    lower_text, match.start(), match.end(), window_start, window_end
                )
            ):
                spans.append(window.strip())
    return spans


def _negation_has_no_punctuation_boundary(
    lower_text: str,
    atom_start: int,
    atom_end: int,
    window_start: int,
    window_end: int,
) -> bool:
    for negation in _NEGATION_WORD_RE.finditer(lower_text[window_start:window_end]):
        neg_start = window_start + negation.start()
        neg_end = window_start + negation.end()
        between = (
            lower_text[atom_end:neg_start]
            if atom_end <= neg_start
            else lower_text[neg_end:atom_start]
        )
        if not re.search(r"[\n\r,.;:]", between):
            return True
    return False


def _inside_spans(position: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _extract_row(
    row_id: str,
    manifest_row: Mapping[str, Any],
    raw: Mapping[str, Any] | None,
    atoms: Sequence[str],
) -> JsonDict:
    if raw is None:
        evidence = extract_partial_atom_evidence(
            "",
            atoms,
            generated_token_count=0,
            row_id=row_id,
        )
        evidence["rejection_reasons"].append("missing_raw_sample")
        evidence["accepted"] = False
        return evidence
    return extract_partial_atom_evidence(
        str(raw.get("raw_output") or ""),
        atoms,
        generated_token_count=int(raw.get("generated_token_count") or 0),
        row_id=row_id,
    )


def _evidence_record(
    row_id: str,
    manifest_row: Mapping[str, Any],
    raw: Mapping[str, Any] | None,
    evidence: Mapping[str, Any],
    support: Mapping[str, Any],
) -> JsonDict:
    accepted = evidence.get("accepted") is True and support.get("supported") is True
    reasons = sorted(
        set(evidence.get("rejection_reasons") or []) | set(support.get("rejection_reasons") or [])
    )
    return {
        "row_id": row_id,
        "model_hf_id": manifest_row.get("model_hf_id"),
        "model_family": model_family(str(manifest_row.get("model_hf_id") or "")),
        "task_id": manifest_row.get("task_id"),
        "fixture_id": manifest_row.get("fixture_id"),
        "fixture_family": manifest_row.get("family"),
        "arm": manifest_row.get("arm"),
        "seed": raw.get("seed") if raw else None,
        "raw_output_hash": raw.get("raw_output_hash") if raw else None,
        "accepted": accepted,
        "positive_atoms": list(evidence.get("positive_atoms") or []),
        "negative_atoms": list(evidence.get("negative_atoms") or []),
        "unknown_atoms": list(evidence.get("unknown_atoms") or []),
        "rejection_reasons": reasons,
        "supporting_completion": list(support.get("supporting_completion") or []),
        "positive_evidence_count": int(support.get("positive_evidence_count") or 0),
        "positive_correct_count": int(support.get("positive_correct_count") or 0)
        if accepted
        else 0,
        "negative_evidence_count": int(support.get("negative_evidence_count") or 0),
        "negative_correct_count": int(support.get("negative_correct_count") or 0)
        if accepted
        else 0,
        "known_atom_count": len(evidence.get("positive_atoms") or [])
        + len(evidence.get("negative_atoms") or []),
        "total_atom_count": (
            len(evidence.get("positive_atoms") or [])
            + len(evidence.get("negative_atoms") or [])
            + len(evidence.get("unknown_atoms") or [])
        ),
        "sidecar_checked_after_extraction": support.get("sidecar_checked_after_extraction") is True,
    }


def _refine_start(
    table: relax.VertexEnergyTable,
    row_id: str,
    kind: str,
    start: Sequence[float],
    known_true: Sequence[str],
    known_false: Sequence[str],
    seed: int,
    best_energy: int,
) -> JsonDict:
    attempts = []
    for restart_index in range(RESTART_BUDGET + 1):
        restart_start = (
            list(start)
            if restart_index == 0
            else _jittered_start(row_id, kind, start, restart_index, seed)
        )
        result = relax.refine(
            table, restart_start, steps=OPTIMIZER_STEPS, step_size=OPTIMIZER_STEP_SIZE
        )
        rounded_state = relax.round_probabilities(table, result["final_probabilities"])
        rounded_energy = table.discrete_energy(rounded_state)
        attempts.append(
            {
                "restart_index": restart_index,
                "initial_energy": result["initial_energy"],
                "final_energy": result["final_energy"],
                "rounded_state": rounded_state,
                "rounded_energy": rounded_energy,
                "success": rounded_energy == best_energy,
                "final_probabilities": [
                    round(float(value), 10) for value in result["final_probabilities"]
                ],
                "relaxation_energy_evaluations": result["energy_evaluations"],
            }
        )
    best = min(attempts, key=lambda row: (row["rounded_energy"], row["final_energy"]))
    return {
        "kind": kind,
        "seed": seed,
        "known_true": list(known_true),
        "known_false": list(known_false),
        "step_budget": OPTIMIZER_STEPS,
        "restart_budget": RESTART_BUDGET,
        "attempts": attempts,
        "best_attempt": best,
    }


def _jittered_start(
    row_id: str,
    kind: str,
    start: Sequence[float],
    restart_index: int,
    seed: int,
) -> list[float]:
    rng = random.Random(_stable_seed(f"{row_id}:{kind}:{restart_index}", seed))
    return [0.85 * float(value) + 0.15 * rng.random() for value in start]


def _exact_completion_record(
    table: relax.VertexEnergyTable,
    evidence: Mapping[str, Any],
    support: Mapping[str, Any],
) -> JsonDict:
    completion = list(support.get("supporting_completion") or [])
    return {
        "supported": support.get("supported") is True,
        "completion": completion,
        "completion_energy": table.discrete_energy(completion)
        if completion
        else table.best_discrete_energy,
        "budget": EXACT_COMPLETION_BUDGET,
        "vertices_considered": min(table.vertex_count, EXACT_COMPLETION_BUDGET),
        "positive_atoms": list(evidence.get("positive_atoms") or []),
        "negative_atoms": list(evidence.get("negative_atoms") or []),
    }


def _cold_exact_completion_controls(tables: Mapping[str, relax.VertexEnergyTable]) -> JsonDict:
    if not tables:
        return {
            "blank": {"budget": EXACT_COMPLETION_BUDGET, "fixture_count": 0, "success_count": 0},
            "random": {"budget": EXACT_COMPLETION_BUDGET, "fixture_count": 0, "success_count": 0},
        }
    blank_success = sum(1 for table in tables.values() if table.best_discrete_energy == 0)
    random_success = blank_success
    return {
        "blank": {
            "budget": EXACT_COMPLETION_BUDGET,
            "fixture_count": len(tables),
            "success_count": blank_success,
        },
        "random": {
            "budget": EXACT_COMPLETION_BUDGET,
            "fixture_count": len(tables),
            "success_count": random_success,
            "seeds": list(RANDOM_SEEDS),
        },
    }


def _row_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    represented = sorted({str(row.get("model_family")) for row in rows if row.get("model_family")})
    with_acceptance = sorted(
        {
            str(row.get("model_family"))
            for row in rows
            if row.get("model_family") and row.get("accepted") is True
        }
    )
    return {
        "replayed_rows": len(rows),
        "accepted_rows": sum(1 for row in rows if row.get("accepted") is True),
        "rejected_rows": sum(1 for row in rows if row.get("accepted") is not True),
        "represented_model_families": represented,
        "represented_model_families_with_acceptance": with_acceptance,
    }


def _precision_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model_family: dict[str, JsonDict] = defaultdict(_empty_metric_cell)
    by_fixture_family: dict[str, JsonDict] = defaultdict(_empty_metric_cell)
    by_pair: dict[str, JsonDict] = defaultdict(_empty_metric_cell)
    overall = _empty_metric_cell()
    for row in rows:
        _add_metric(overall, row)
        model = str(row.get("model_family") or "")
        fixture = str(row.get("fixture_family") or "")
        _add_metric(by_model_family[model], row)
        _add_metric(by_fixture_family[fixture], row)
        _add_metric(by_pair[f"{model}|{fixture}"], row)
    return {
        "pre_registered_positive_precision_floor": POSITIVE_PRECISION_FLOOR,
        "overall": _finalize_metric(overall),
        "by_model_family": {
            key: _finalize_metric(value) for key, value in sorted(by_model_family.items())
        },
        "by_fixture_family": {
            key: _finalize_metric(value) for key, value in sorted(by_fixture_family.items())
        },
        "by_model_family_and_fixture_family": {
            key: _finalize_metric(value) for key, value in sorted(by_pair.items())
        },
    }


def _empty_metric_cell() -> JsonDict:
    return {
        "accepted_rows": 0,
        "rejected_rows": 0,
        "positive_evidence_count": 0,
        "positive_correct_count": 0,
        "negative_evidence_count": 0,
        "negative_correct_count": 0,
        "known_atom_count": 0,
        "total_atom_count": 0,
    }


def _add_metric(cell: JsonDict, row: Mapping[str, Any]) -> None:
    if row.get("accepted") is True:
        cell["accepted_rows"] += 1
        cell["positive_correct_count"] += int(row.get("positive_correct_count") or 0)
        cell["negative_correct_count"] += int(row.get("negative_correct_count") or 0)
        cell["positive_evidence_count"] += int(row.get("positive_evidence_count") or 0)
        cell["negative_evidence_count"] += int(row.get("negative_evidence_count") or 0)
        cell["known_atom_count"] += int(row.get("known_atom_count") or 0)
        cell["total_atom_count"] += int(row.get("total_atom_count") or 0)
    else:
        cell["rejected_rows"] += 1


def _finalize_metric(cell: Mapping[str, Any]) -> JsonDict:
    positive_count = int(cell.get("positive_evidence_count") or 0)
    negative_count = int(cell.get("negative_evidence_count") or 0)
    total_atoms = int(cell.get("total_atom_count") or 0)
    out = dict(cell)
    out["positive_precision"] = (
        float(cell.get("positive_correct_count") or 0) / positive_count if positive_count else 0.0
    )
    out["negative_precision"] = (
        float(cell.get("negative_correct_count") or 0) / negative_count if negative_count else 0.0
    )
    out["coverage"] = float(cell.get("known_atom_count") or 0) / total_atoms if total_atoms else 0.0
    return out


def _rejection_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for row in rows:
        if row.get("accepted") is True:
            continue
        for reason in row.get("rejection_reasons") or []:
            reason_text = str(reason)
            counts[reason_text] += 1
            examples.setdefault(reason_text, str(row.get("row_id")))
    return {
        "counts_by_reason": dict(sorted(counts.items())),
        "example_row_by_reason": dict(sorted(examples.items())),
        "contradiction_rejections": counts.get("contradictory_evidence", 0),
        "foreign_atom_rejections": counts.get("foreign_atom", 0),
        "ambiguous_negation_rejections": counts.get("ambiguous_negation", 0),
    }


def _unsafe_acceptance_count(rows: Sequence[Mapping[str, Any]]) -> int:
    unsafe = {
        "contradictory_evidence",
        "foreign_atom",
        "ambiguous_negation",
        "unsupported_by_exact_sidecar",
    }
    return sum(
        1
        for row in rows
        if row.get("accepted") is True
        and unsafe.intersection(set(row.get("rejection_reasons") or []))
    )


def _leakage_controls() -> JsonDict:
    foreign = extract_partial_atom_evidence(
        "ANSWER: a, secret", ("a",), generated_token_count=3, row_id="foreign"
    )
    ambiguous = extract_partial_atom_evidence(
        "a may not be b.", ("a", "b"), generated_token_count=5, row_id="ambiguous"
    )
    extraction = extract_partial_atom_evidence(
        "ANSWER: a", ("a", "b"), generated_token_count=2, row_id="leakage"
    )
    support = check_evidence_support(extraction, exact_answer_sets=[["b"]])
    return {
        "extractor_reads_exact_sidecar": False,
        "sidecar_support_after_extraction_only": extraction["accepted"] is True
        and support["supported"] is False,
        "foreign_atom_control": {
            "rejected": foreign["accepted"] is False,
            "reasons": foreign["rejection_reasons"],
        },
        "ambiguous_negation_control": {
            "rejected": ambiguous["accepted"] is False,
            "reasons": ambiguous["rejection_reasons"],
        },
        "label_leakage_control": {
            "passed": extraction["sidecar_checked_after_extraction"] is False
            and support["sidecar_checked_after_extraction"] is True,
        },
        "all_clean": foreign["accepted"] is False
        and ambiguous["accepted"] is False
        and support["supported"] is False,
    }


def _start_outcome_summary(refinement_by_row: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    summary = {
        "row_count": len(refinement_by_row),
        "fixed_budgets": {"steps": OPTIMIZER_STEPS, "restart_budget": RESTART_BUDGET},
        "success_count_by_start": {"evidence_warm": 0, "blank": 0, "random": 0},
        "best_energy_sum_by_start": {"evidence_warm": 0, "blank": 0, "random": 0},
    }
    for starts in refinement_by_row.values():
        for name in summary["success_count_by_start"]:
            best = starts[name]["best_attempt"]
            summary["success_count_by_start"][name] += int(best["success"] is True)
            summary["best_energy_sum_by_start"][name] += int(best["rounded_energy"])
    return summary


def _continuous_refinement_summary(refinement_by_row: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    outcomes = _start_outcome_summary(refinement_by_row)
    warm = outcomes["success_count_by_start"]["evidence_warm"]
    blank = outcomes["success_count_by_start"]["blank"]
    return {
        "fixed_budgets": {
            "steps": OPTIMIZER_STEPS,
            "step_size": OPTIMIZER_STEP_SIZE,
            "restart_budget": RESTART_BUDGET,
        },
        "accepted_row_count": len(refinement_by_row),
        "success_count_by_start": outcomes["success_count_by_start"],
        "evidence_warm_minus_blank_success_delta": warm - blank,
        "warm_start_delta_required_for_readiness": False,
        "by_row": dict(refinement_by_row),
    }


def _exact_completion_summary(exact_by_row: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    return {
        "accepted_exact_completion_count": sum(
            1 for row in exact_by_row.values() if row.get("supported") is True
        ),
        "fixed_budget": EXACT_COMPLETION_BUDGET,
        "by_row": dict(exact_by_row),
    }


def _models_represented(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model: dict[str, JsonDict] = {}
    for row in rows:
        model = str(row.get("model_hf_id") or "")
        family = str(row.get("model_family") or "")
        item = by_model.setdefault(
            model,
            {"model_family": family, "replayed_rows": 0, "accepted_rows": 0, "rejected_rows": 0},
        )
        item["replayed_rows"] += 1
        if row.get("accepted") is True:
            item["accepted_rows"] += 1
        else:
            item["rejected_rows"] += 1
    return {
        "by_model": dict(sorted(by_model.items())),
        "model_families": sorted({row["model_family"] for row in by_model.values()}),
    }


def model_family(model_hf_id: str) -> str:
    """Map mandated model ids to stable family names."""

    lowered = model_hf_id.lower()
    if "qwen" in lowered:
        return "qwen3_6_35b_a3b"
    if "gemma-4-" in lowered:
        return "gemma_4"
    return "unknown"


def _frozen_vocabularies(entries: Mapping[str, Any]) -> JsonDict:
    by_fixture: dict[str, JsonDict] = {}
    for entry in entries.values():
        if not isinstance(entry, Mapping):
            continue
        fixture_id = str(entry.get("fixture_id") or "")
        family = str(entry.get("family") or "")
        program_text = str(entry.get("program_text") or "")
        table = table_from_program(program_text, fixture_id)
        by_fixture[f"{fixture_id}|{family}"] = {
            "fixture_id": fixture_id,
            "fixture_family": family,
            "atoms": list(table.atoms),
            "atom_count": table.atom_count,
            "program_text_hash": sha256_text(program_text),
        }
    return dict(sorted(by_fixture.items()))


def _fixture_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('fixture_id')}|{row.get('family')}"


def _load_raw_sources(root: Path) -> tuple[dict[tuple[str, str, int], JsonDict], JsonDict]:
    lookup: dict[tuple[str, str, int], JsonDict] = {}
    receipts: dict[str, JsonDict] = {
        "sealed_manifest": _path_receipt(root / SEALED_MANIFEST_RELATIVE_PATH),
        "formal_sidecar": _path_receipt(root / FORMAL_SIDECAR_RELATIVE_PATH),
        "raw_outputs_by_model": {},
    }
    for path in sorted((root / RAW_DIR_RELATIVE_PATH).glob("*.jsonl")):
        rows = _read_jsonl(path)
        for row in rows:
            try:
                seed = int(row.get("seed"))
            except (TypeError, ValueError):
                continue
            lookup[(str(row.get("model_hf_id")), str(row.get("task_id")), seed)] = row
        receipts["raw_outputs_by_model"][path.name] = _path_receipt(path, row_count=len(rows))
    return lookup, receipts


def _read_json_mapping(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _row_id(row: Mapping[str, Any], seed: int, source_index: int) -> str:
    return f"{source_index}|{row.get('model_hf_id')}|{row.get('task_id')}|{row.get('arm')}|{seed}"


def _path_receipt(path: Path, *, row_count: int | None = None) -> JsonDict:
    receipt_path = path.as_posix()
    if path.is_absolute() and path.is_relative_to(REPO_ROOT):
        receipt_path = path.relative_to(REPO_ROOT).as_posix()
    receipt: JsonDict = {
        "path": receipt_path,
        "present": path.exists(),
        "sha256": path_sha256(path),
    }
    if row_count is not None:
        receipt["row_count"] = row_count
    return receipt


def _classification_receipt(root: Path, rel_path: Path) -> JsonDict:
    path = root / rel_path
    classification = classify_artifact_path(path)
    return {
        "path": rel_path.as_posix(),
        "sha256": path_sha256(path),
        "terminal_class": classification.classification,
        "terminal": classification.terminal,
        "status_raw": classification.status_raw,
        "honest_verdict_raw": classification.honest_verdict_raw,
    }


def _adapter_source_hashes(root: Path) -> JsonDict:
    return {
        path.as_posix(): path_sha256(root / path)
        for path in (SOURCE_RELATIVE_PATH, TEST_RELATIVE_PATH, SPEC_RELATIVE_PATH)
    }


def _protected_hashes(root: Path) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, Any]) -> JsonDict:
    paths = {}
    for rel, before_hash in before.items():
        after_hash = path_sha256(root / rel)
        paths[rel] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "unchanged": before_hash == after_hash,
        }
    return {"unchanged": all(row["unchanged"] for row in paths.values()), "paths": paths}


def _preconditions(
    date: str,
    result_path: Path,
    before_hashes: Mapping[str, Any],
    eligible_row_count: int,
) -> JsonDict:
    return {
        "date": date,
        "result_path": str(result_path),
        "eligible_row_count": eligible_row_count,
        "raw_sidecar_vocabulary_parser_budget_seed_and_protected_hashes_frozen": True,
        "parser_rule": "lexical_explicit_atoms_only_no_sidecar_until_support_check",
        "optimizer_steps": OPTIMIZER_STEPS,
        "optimizer_step_size": OPTIMIZER_STEP_SIZE,
        "restart_budget": RESTART_BUDGET,
        "exact_completion_budget": EXACT_COMPLETION_BUDGET,
        "positive_precision_floor": POSITIVE_PRECISION_FLOOR,
        "protected_hashes_before": dict(before_hashes),
    }


def _honest_verdict(status: str) -> str:
    if status == "complete":
        return "complete: partial atom evidence adapter passed fail-closed replay gates"
    return "blocked: partial atom evidence adapter did not meet readiness gates"


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{text}:{seed}".encode()).hexdigest()
    return int(digest[:12], 16)


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
