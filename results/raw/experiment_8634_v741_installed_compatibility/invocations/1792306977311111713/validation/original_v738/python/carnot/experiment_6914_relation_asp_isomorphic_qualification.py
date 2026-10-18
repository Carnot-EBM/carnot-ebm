"""Replay relation semantics through two exact engines and paired changes.

Spec refs: REQ-CONSTRAINT-6914 and SCENARIO-CONSTRAINT-6914-*.

The reducer reads immutable proposal cells only after their public artifacts
pass exact hash checks. It then opens each sealed label file once. The bounded
energy compiler and clingo make separate decisions, so agreement does not hide
a shared error against the sealed expected effect.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import asp_energy


JsonDict = dict[str, Any]
ExactEngine = Callable[[str, str], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_6914_relation_asp_isomorphic_qualification.json")
PUBLIC_PATHS = {
    "exp6274": Path("results/experiment_6274_asp_energy_semantic_compiler.json"),
    "exp6886": Path("results/experiment_6886_enoki_exact_relation_fixture.json"),
    "exp6900": Path("results/experiment_6900_authentic_anchored_relation_corpus.json"),
    "exp6912": Path("results/experiment_6912_alias_safe_relation_corpus_reducer.json"),
    "compiler": Path("python/carnot/asp_energy.py"),
}
EXPECTED_PUBLIC_HASHES = {
    "exp6274": "sha256:b02c88963c4815aa0e26d451ffd60fdd9f1014d32e76f638592ac114c611e96b",
    "exp6886": "sha256:602250fbfe172f08458ea279787d992e89835f12005ba6ef59ec02f3b411d500",
    "exp6900": "sha256:beb442dfa3743bc3271150eb88d35cf0e31ac8b657e00664ed143611ed7d0c0c",
    "exp6912": "sha256:5122cd3e95c59ec116dad0d64c03484171a79fe977f9891872da03a8a7a7c2c6",
    "compiler": "sha256:0f6077bcd49aa93a6cdbde72422ecf97d905b76b31cadbc0cd401c494af015e1",
}
SIDECAR_ROOT = Path.home() / ".cache" / "carnot" / "exp6886_enoki_exact_relation_fixture"
SIDECAR_PATHS = {
    "calibration": SIDECAR_ROOT / "sealed_calibration_formal_sidecar.json",
    "held": SIDECAR_ROOT / "sealed_held_formal_sidecar.json",
}
EXPECTED_SIDECAR_HASHES = {
    "calibration": "sha256:cb8db945ac4b20e0d2e9658cd09e5263fe2240e78f999c323fd09fc3351a6cce",
    "held": "sha256:d75b4b032ed27542dd8d619922865c389f869aa11470a5cfffd90c05de6553ad",
}
SIDECAR_SCHEMA = "carnot.exp6886.sealed_formal_sidecar.v1"
SCHEMA = "carnot.exp6914.relation_asp_isomorphic_qualification.v1"
INFERENCE_SUBSTRATE = "deterministic_cpu_asp_isomorphic_qualification_no_llm"
BLOCKED_VERDICT = "complete_blocked_relation_asp_isomorphic_qualification"
READY_VERDICT = "complete_circular_positive_relation_asp_isomorphic_qualification"
DISQUALIFIED_VERDICT = "complete_disqualified_relation_asp_isomorphic_qualification"
EXPECTED_CELL_COUNT = 1_400
RANDOM_SEED = 2609036914
PERTURBATIONS = (
    "base",
    "entity_renaming",
    "relation_paraphrase",
    "relation_reversal",
    "contradiction_injection",
    "relation_omission",
    "solution_space_restructuring",
)
PARAPHRASE_BY_PREDICATE = {
    "has_color": "is colored",
    "scheduled_at": "takes place in",
    "has_condition": "suffers from",
    "has_truth_status": "is judged",
    "selects": "chooses",
}
PREFIX_BY_FAMILY = {
    "graph_coloring": "gc",
    "scheduling": "sc",
    "non_monotonic_defaults": "df",
    "contradictions": "ct",
    "cardinality_constraints": "cd",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "sealed_sidecar_hashes",
    "rows",
    "asp_compilation_rows",
    "bounded_vocabulary_rows",
    "primary_solver_rows",
    "independent_solver_rows",
    "solver_parity_rows",
    "entity_renaming_rows",
    "paraphrase_rows",
    "reversal_rows",
    "contradiction_rows",
    "omission_rows",
    "restructuring_rows",
    "arm_summary_rows",
    "model_summary_rows",
    "family_summary_rows",
    "seed_summary_rows",
    "perturbation_summary_rows",
    "proposal_coverage_by_arm",
    "exact_atom_validity_by_arm",
    "isomorphic_invariance_by_arm",
    "completeness_blind_spot_rows",
    "solver_disagreement_count",
    "held_leakage_count",
    "model_inference_call_count",
    "reported_vs_recomputed_metrics",
    "random_seed",
    "reproducibility_checksum",
    "asp_isomorphic_shard_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field explains why its evidence exists.",
    "preconditions_checked": "Public checks finish before any sealed label content opens.",
    "inference_substrate": "The fixed value records deterministic CPU work and no model calls.",
    "duration_s": "Measured wall time shows that the reducer executed.",
    "source_artifact_hashes": "Exact hashes bind every public immutable input.",
    "sealed_sidecar_hashes": "Exact hashes bind the sealed expected effects.",
    "rows": "One terminal row preserves every cell and paired change decision.",
    "asp_compilation_rows": "Compiler rows expose programs, unsupported atoms, and failures.",
    "bounded_vocabulary_rows": "Vocabulary rows name every atom allowed in a program.",
    "primary_solver_rows": "Primary rows retain bounded energy-enumeration receipts.",
    "independent_solver_rows": "Independent rows retain direct clingo receipts.",
    "solver_parity_rows": "Parity rows compare the engines without claiming correctness.",
    "entity_renaming_rows": "Renaming rows test invariance under an injective atom map.",
    "paraphrase_rows": "Paraphrase rows test only registered meaning-preserving syntax.",
    "reversal_rows": "Reversal rows test relation direction as a separate decision.",
    "contradiction_rows": "Contradiction rows require both polarities to make the program unsat.",
    "omission_rows": "Omission rows remove proposal facts and recompute the effect.",
    "restructuring_rows": "Restructuring rows separate model count from semantic outcome.",
    "arm_summary_rows": "Arm rows keep control and model denominators separate.",
    "model_summary_rows": "Model rows prevent one model from carrying another model.",
    "family_summary_rows": "Family rows prevent an easy family from hiding a weak family.",
    "seed_summary_rows": "Seed rows preserve deterministic generation identities.",
    "perturbation_summary_rows": "Perturbation rows keep each paired test visible.",
    "proposal_coverage_by_arm": "Coverage uses every immutable base cell as its denominator.",
    "exact_atom_validity_by_arm": "Atom validity is distinct from coverage and solver parity.",
    "isomorphic_invariance_by_arm": "Invariance uses only renaming and paraphrase pairs.",
    "completeness_blind_spot_rows": "Missing expected atoms remain outside verifier completeness.",
    "solver_disagreement_count": "Zero is required before the exact engines can qualify the shard.",
    "held_leakage_count": "Zero proves formal held content did not enter proposal text.",
    "model_inference_call_count": "Zero proves this experiment only reduced saved outputs.",
    "reported_vs_recomputed_metrics": "Fresh row replay detects aggregate drift.",
    "random_seed": "A fixed non-identity seed pins deterministic pairing choices.",
    "reproducibility_checksum": "A stable digest detects semantic artifact drift.",
    "asp_isomorphic_shard_ready_score": "One means the shard is complete, not universally correct.",
    "gate_check_summary": "Exact expected and observed values make every failure actionable.",
    "verifier_is_oracle": "True discloses that exact execution supplies the labels.",
    "verdict_class": "The closed class prevents an oracle result from claiming positive.",
    "honest_verdict": "A complete prefix marks a terminal result.",
    "gate_check_summary.checks": "The full list prevents later failures from being hidden.",
    "gate_check_summary.passed": "One Boolean gives the terminal gate state.",
    "gate_check_summary.failed_check": "The first failure gives an actionable check name.",
    "gate_check_summary.expected": "The failed expectation states the required value.",
    "gate_check_summary.observed": "The observation states the value that caused the failure.",
    "gate_check_summary.failed_checks": "All failures remain visible after the first one.",
    "gate_check.check": "Each check has a stable machine-readable name.",
    "gate_check.expected": "Each check records its exact required value.",
    "gate_check.observed": "Each check records its exact observed value.",
    "gate_check.passed": "Each check reports its own equality decision.",
}


class QualificationError(RuntimeError):
    """Report one fail-closed qualification condition."""


class SealedSidecarReader:
    """Open, hash, and validate one sealed sidecar at most once."""

    def __init__(self, path: Path | str, expected_sha256: str, *, expected_split: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = expected_sha256
        self.expected_split = expected_split
        self.open_count = 0
        self.observed_sha256 = "not_opened"

    def open_once(self) -> JsonDict:
        """Return one validated payload and reject a second exposure."""

        if self.open_count:
            raise QualificationError("sealed_sidecar_opened_more_than_once")
        self.open_count += 1
        raw = self.path.read_bytes()
        self.observed_sha256 = sha256_bytes(raw)
        if self.observed_sha256 != self.expected_sha256:
            raise QualificationError(
                f"sidecar_hash_drift:{self.expected_sha256}:{self.observed_sha256}"
            )
        payload = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != SIDECAR_SCHEMA
            or payload.get("split") != self.expected_split
            or not isinstance(payload.get("rows"), list)
            or not payload["rows"]
        ):
            raise QualificationError("sidecar_identity")
        return payload


def canonical_json(value: Any) -> str:
    """Serialize JSON in one stable form."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return a repository-style SHA-256 value."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    """Hash one file without interpreting its content."""

    return sha256_bytes(Path(path).read_bytes())


def _jsonable_gate_value(value: Any) -> Any:
    """Convert sets and nested containers to stable JSON gate evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_gate_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable_gate_value(item) for item in value), key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [_jsonable_gate_value(item) for item in value]
    return deepcopy(value)


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Build one exact gate row."""

    return {
        "check": check,
        "expected": _jsonable_gate_value(expected),
        "observed": _jsonable_gate_value(observed),
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every gate row while naming the first failure."""

    copied = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": copied,
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def public_precondition_summary(
    observed_hashes: Mapping[str, str],
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    expected_hashes: Mapping[str, str] = EXPECTED_PUBLIC_HASHES,
    solver_name: str | None = None,
) -> JsonDict:
    """Check all public bytes and exact-engine qualifications before sealed access."""

    checks = [
        gate_check(f"public_hash:{name}", expected, observed_hashes.get(name, "missing"))
        for name, expected in expected_hashes.items()
    ]
    exp6274 = artifacts.get("exp6274", {})
    exp6886 = artifacts.get("exp6886", {})
    exp6900 = artifacts.get("exp6900", {})
    exp6912 = artifacts.get("exp6912", {})
    checks.extend(
        [
            gate_check(
                "qualified_bounded_compiler", 1.0, exp6274.get("asp_energy_semantic_ready_score")
            ),
            gate_check("compiler_parity_failure_count", 0, exp6274.get("parity_failure_count")),
            gate_check("exact_exp6886", 1, exp6886.get("relation_fixture_ready_score")),
            gate_check("exact_exp6900", 1, exp6900.get("relation_corpus_complete_score")),
            gate_check("source_held_access_count", 0, exp6900.get("held_sidecar_access_count")),
            gate_check(
                "clean_relation_corpus_ready_score",
                1,
                exp6912.get("clean_relation_corpus_ready_score"),
            ),
            gate_check(
                "replayed_cell_count", EXPECTED_CELL_COUNT, exp6912.get("replayed_cell_count")
            ),
            gate_check(
                "independent_exact_engine",
                True,
                bool(solver_name) and not str(solver_name).endswith(":missing"),
            ),
        ]
    )
    return gate_summary(checks)


def _fixture_parts(fixture_id: str) -> tuple[str, int]:
    family, separator, ordinal = fixture_id.rpartition("_")
    if not separator or family not in PREFIX_BY_FAMILY or not ordinal.isdigit():
        raise QualificationError(f"unsupported_fixture:{fixture_id}")
    return family, int(ordinal)


def _fixture_vocabulary(fixture_id: str, vocabulary: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    family, ordinal = _fixture_parts(fixture_id)
    suffix = f"_{ordinal}"
    return [
        deepcopy(dict(row))
        for row in vocabulary
        if row.get("family") == family and str(row.get("subject_id", "")).endswith(suffix)
    ]


def map_relation_tuple(
    normalized_tuple: Sequence[Any], vocabulary: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Map one exact four-field tuple through the closed vocabulary."""

    candidate = tuple(str(value) for value in normalized_tuple)
    mapping = {
        tuple(str(value) for value in row.get("normalized_tuple", [])): str(row["asp_atom"])
        for row in vocabulary
        if len(row.get("normalized_tuple", [])) == 4 and row.get("asp_atom")
    }
    atom = mapping.get(candidate)
    if atom is None:
        return {
            "accepted": False,
            "asp_atom": None,
            "reason": "unsupported_atom",
            "normalized_tuple": list(candidate),
        }
    return {"accepted": True, "asp_atom": atom, "reason": "mapped"}


def _semantic_tuple(
    parse_row: Mapping[str, Any], fixture_vocabulary: Sequence[Mapping[str, Any]]
) -> list[str]:
    subjects = sorted({str(row["subject_id"]) for row in fixture_vocabulary})
    objects = sorted({str(row["object_id"]) for row in fixture_vocabulary})
    if len(subjects) != 1 or len(objects) != 1:
        raise QualificationError("ambiguous_fixture_vocabulary")
    return [
        subjects[0],
        str(parse_row.get("predicate", "")),
        objects[0],
        str(parse_row.get("polarity", "")),
    ]


def reverse_tuple(normalized_tuple: Sequence[Any]) -> list[str]:
    """Swap the subject and object while retaining predicate and polarity."""

    values = [str(value) for value in normalized_tuple]
    if len(values) != 4:
        raise QualificationError("invalid_tuple_arity")
    return [values[2], values[1], values[0], values[3]]


def build_paraphrase_tuple(normalized_tuple: Sequence[Any]) -> list[str]:
    """Replace one predicate with its frozen syntax-only alias."""

    values = [str(value) for value in normalized_tuple]
    if len(values) != 4 or values[1] not in PARAPHRASE_BY_PREDICATE:
        raise QualificationError("unsupported_paraphrase_source")
    values[1] = PARAPHRASE_BY_PREDICATE[values[1]]
    return values


def canonicalize_paraphrase(normalized_tuple: Sequence[Any]) -> list[str]:
    """Map a frozen alias back to its only registered predicate."""

    values = [str(value) for value in normalized_tuple]
    inverse = {alias: predicate for predicate, alias in PARAPHRASE_BY_PREDICATE.items()}
    if len(values) != 4 or values[1] not in inverse:
        raise QualificationError("semantic_changing_paraphrase")
    values[1] = inverse[values[1]]
    return values


def rename_program(program_text: str, atom_map: Mapping[str, str]) -> str:
    """Rename atoms in one pass and reject a many-to-one map."""

    if len(set(atom_map.values())) != len(atom_map):
        raise QualificationError("non_injective_renaming")
    if not atom_map:
        raise QualificationError("empty_renaming")
    pattern = re.compile(
        r"\b(?:"
        + "|".join(re.escape(atom) for atom in sorted(atom_map, key=len, reverse=True))
        + r")\b"
    )
    renamed = pattern.sub(lambda match: atom_map[match.group(0)], program_text)
    if renamed == program_text:
        raise QualificationError("entity_renaming_no_op")
    return renamed


def _rename_models(models: Sequence[Sequence[str]], atom_map: Mapping[str, str]) -> list[list[str]]:
    return sorted([sorted(atom_map.get(atom, atom) for atom in model) for model in models])


def contradiction_atoms(
    atoms: Sequence[str], vocabulary: Sequence[Mapping[str, Any]], fixture_id: str
) -> list[str]:
    """Add both frozen polarity atoms for one fixture."""

    relation_atoms = {str(row["asp_atom"]) for row in _fixture_vocabulary(fixture_id, vocabulary)}
    if len(relation_atoms) != 2:
        raise QualificationError("contradiction_vocabulary")
    return sorted(set(str(atom) for atom in atoms) | relation_atoms)


def omit_relation_atoms(_atoms: Sequence[str]) -> list[str]:
    """Return the deliberate empty proposal-fact set."""

    return []


def restructure_program(program_text: str, auxiliary_atom: str) -> str:
    """Add one independent bounded choice that changes only solution count."""

    parsed = asp_energy.parse_program(program_text, program_id="restructure_source")
    if auxiliary_atom in parsed.atoms:
        raise QualificationError("restructuring_no_op")
    return program_text.rstrip() + f"\n0 {{{auxiliary_atom}}} 1.\n"


def _canonical_models(models: Any) -> list[list[str]]:
    if not isinstance(models, list):
        raise QualificationError("engine_models_not_list")
    return sorted([sorted({str(atom) for atom in model}) for model in models])


def primary_exact_engine(program_text: str, program_id: str) -> JsonDict:
    """Enumerate zero-energy states with the qualified bounded compiler."""

    compiled = asp_energy.compile_program(program_text, program_id=program_id)
    models = compiled.zero_energy_states()
    return {
        "models": models,
        "receipt": {
            "engine": "bounded_asp_energy_enumerator",
            "program_sha256": sha256_bytes(program_text.encode("utf-8")),
            "atom_count": len(compiled.program.atoms),
            "enumerated_state_count": compiled.exact_state_count,
            "model_count": len(models),
            "status": "complete",
        },
    }


def independent_exact_engine(program_text: str, program_id: str) -> JsonDict:
    """Solve source text directly with clingo, independent of the energy parser."""

    try:
        import clingo
    except ImportError as exc:  # pragma: no cover - admission checks this environment failure.
        raise QualificationError("independent_solver_missing:clingo") from exc
    control = clingo.Control(["0", "--warn=none"])
    control.add("base", [], program_text)
    control.ground([("base", [])])
    models = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            models.append(sorted(str(symbol) for symbol in model.symbols(shown=True)))
    canonical = sorted(models)
    return {
        "models": canonical,
        "receipt": {
            "engine": "clingo_direct",
            "name_version": f"clingo {clingo.__version__}",
            "program_id": program_id,
            "program_sha256": sha256_bytes(program_text.encode("utf-8")),
            "model_count": len(canonical),
            "status": "complete",
        },
    }


def _base_program(formal: Mapping[str, Any], relation_atoms: set[str]) -> str:
    lines = []
    for line in str(formal["asp_program"]).splitlines():
        stripped = line.strip()
        if stripped.endswith(".") and stripped[:-1] in relation_atoms:
            continue
        if stripped:
            lines.append(stripped)
    return "\n".join(lines) + "\n"


def _program_with_atoms(base_program: str, atoms: Sequence[str]) -> str:
    return base_program + "".join(f"{atom}.\n" for atom in sorted(set(atoms)))


def _program_atoms(program_text: str) -> set[str]:
    return set(asp_energy.parse_program(program_text, program_id="vocabulary_check").atoms)


def _effect(models: Sequence[Sequence[str]]) -> JsonDict:
    canonical = _canonical_models(list(models))
    return {
        "satisfiable": bool(canonical),
        "model_count": len(canonical),
        "models": canonical,
        "models_sha256": sha256_json(canonical),
    }


def _not_run_effect(status: str) -> JsonDict:
    return {
        "status": status,
        "satisfiable": None,
        "model_count": None,
        "models": [],
        "models_sha256": sha256_json([]),
    }


def _evaluate_program(
    *,
    row: JsonDict,
    program_text: str,
    allowed_atoms: set[str],
    primary_engine: ExactEngine,
    independent_engine: ExactEngine,
) -> JsonDict:
    program_atoms = _program_atoms(program_text)
    escaped = sorted(program_atoms - allowed_atoms)
    if escaped:
        row.update(
            {
                "program": program_text,
                "program_sha256": sha256_bytes(program_text.encode("utf-8")),
                "compilation_status": "unsupported_atom",
                "unsupported_atoms": escaped,
                "primary_solver_receipt": {"status": "not_run_unsupported_atom"},
                "independent_solver_receipt": {"status": "not_run_unsupported_atom"},
                "primary_effect": _not_run_effect("not_run_unsupported_atom"),
                "independent_effect": _not_run_effect("not_run_unsupported_atom"),
                "observed_effect": _not_run_effect("not_run_unsupported_atom"),
                "solver_parity": None,
            }
        )
        return row
    primary = primary_engine(program_text, str(row["row_id"]))
    independent = independent_engine(program_text, str(row["row_id"]))
    primary_models = _canonical_models(primary.get("models"))
    independent_models = _canonical_models(independent.get("models"))
    primary_effect = _effect(primary_models)
    independent_effect = _effect(independent_models)
    row.update(
        {
            "program": program_text,
            "program_sha256": sha256_bytes(program_text.encode("utf-8")),
            "program_atoms": sorted(program_atoms),
            "compilation_status": "compiled",
            "unsupported_atoms": [],
            "primary_solver_receipt": deepcopy(dict(primary.get("receipt", {}))),
            "independent_solver_receipt": deepcopy(dict(independent.get("receipt", {}))),
            "primary_effect": primary_effect,
            "independent_effect": independent_effect,
            "observed_effect": primary_effect,
            "solver_parity": primary_models == independent_models,
            "satisfiable": primary_effect["satisfiable"],
            "model_count": primary_effect["model_count"],
            "projected_models": primary_models,
        }
    )
    return row


def _stub_row(
    cell: Mapping[str, Any],
    perturbation: str,
    *,
    tuples: Sequence[Sequence[str]],
    atoms: Sequence[str],
    parseable_count: int,
    nonparseable_count: int,
    exact_atom_valid: bool,
) -> JsonDict:
    arm = str(cell["arm"])
    model_id = str(cell.get("hf_id") or arm)
    seed_id: int | str = cell.get("seed") if cell.get("seed") is not None else "deterministic"
    return {
        "row_id": f"{cell['cell_identity']}::{perturbation}",
        "cell_identity": str(cell["cell_identity"]),
        "arm": arm,
        "model_id": model_id,
        "seed": seed_id,
        "fixture_id": str(cell["fixture_id"]),
        "family": str(cell["family"]),
        "split": str(cell["split"]),
        "perturbation": perturbation,
        "terminal": True,
        "parseable_tuple_count": parseable_count,
        "nonparseable_tuple_count": nonparseable_count,
        "proposal_covered": parseable_count > 0,
        "tuples": [list(values) for values in tuples],
        "atoms": sorted(set(str(atom) for atom in atoms)),
        "exact_atom_valid": exact_atom_valid,
        "program": None,
        "program_sha256": None,
        "program_atoms": [],
        "compilation_status": "not_run",
        "unsupported_atoms": [],
        "expected_effect": {},
        "observed_effect": _not_run_effect("not_run"),
        "primary_effect": _not_run_effect("not_run"),
        "independent_effect": _not_run_effect("not_run"),
        "primary_solver_receipt": {"status": "not_run"},
        "independent_solver_receipt": {"status": "not_run"},
        "solver_parity": None,
        "expected_effect_met": False,
        "pair_applicable": False,
        "pair_requirement_met": True,
        "transform_valid": True,
        "isomorphic_invariant": None,
        "shortcut_detected": False,
        "exact_outcome_decision": "not_applicable",
        "satisfiable": None,
        "model_count": None,
        "projected_models": [],
    }


def _entity_mapping(program_text: str, fixture_id: str, allowed_atoms: set[str]) -> dict[str, str]:
    family, ordinal = _fixture_parts(fixture_id)
    prefix = PREFIX_BY_FAMILY[family]
    candidates = sorted(
        {
            int(match.group(1))
            for atom in allowed_atoms
            if (match := re.fullmatch(rf"{re.escape(prefix)}_(\d+)_.+", atom))
        }
    )
    if len(candidates) < 2 or ordinal not in candidates:
        raise QualificationError("entity_renaming_target_missing")
    target = candidates[(candidates.index(ordinal) + 1) % len(candidates)]
    mapping = {}
    for atom in sorted(_program_atoms(program_text)):
        renamed = re.sub(
            rf"^{re.escape(prefix)}_{ordinal}_",
            f"{prefix}_{target}_",
            atom,
        )
        if renamed == atom or renamed not in allowed_atoms:
            raise QualificationError(f"entity_renaming_escape:{atom}:{renamed}")
        mapping[atom] = renamed
    if len(set(mapping.values())) != len(mapping):
        raise QualificationError("non_injective_renaming")
    return mapping


def _project_models(models: Sequence[Sequence[str]], removed_atom: str) -> list[list[str]]:
    return sorted(
        {tuple(sorted(atom for atom in model if atom != removed_atom)) for model in models}
    )


def validate_pair(
    perturbation: str, base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> JsonDict:
    """Check one paired semantic contract without pooling it with solver parity."""

    if perturbation == "entity_renaming":
        applicable = bool(base.get("tuples"))
        met = (not applicable) or (
            base.get("satisfiable") == candidate.get("satisfiable")
            and base.get("model_count") == candidate.get("model_count")
            and base.get("projected_models") == candidate.get("projected_models")
        )
        return {"applicable": applicable, "met": met, "shortcut_detected": False}
    if perturbation == "relation_paraphrase":
        applicable = bool(base.get("tuples"))
        met = (not applicable) or (
            base.get("tuples") == candidate.get("canonical_tuples")
            and base.get("satisfiable") == candidate.get("satisfiable")
            and base.get("projected_models") == candidate.get("projected_models")
        )
        return {"applicable": applicable, "met": met, "shortcut_detected": False}
    if perturbation == "reversal":
        applicable = bool(base.get("tuples"))
        met = (not applicable) or (
            base.get("tuples") != candidate.get("tuples")
            and candidate.get("exact_atom_valid") is False
        )
        return {"applicable": applicable, "met": met, "shortcut_detected": False}
    if perturbation == "contradiction":
        applicable = base.get("satisfiable") is True
        met = (not applicable) or (
            set(candidate.get("atoms", [])) > set(base.get("atoms", []))
            and candidate.get("satisfiable") is False
        )
        return {"applicable": applicable, "met": met, "shortcut_detected": False}
    if perturbation == "omission":
        applicable = bool(base.get("atoms"))
        met = (not applicable) or (
            not candidate.get("atoms") and candidate.get("atoms") != base.get("atoms")
        )
        return {"applicable": applicable, "met": met, "shortcut_detected": False}
    if perturbation == "solution_space_restructuring":
        applicable = base.get("satisfiable") is True
        shortcut = applicable and base.get("exact_outcome_decision") != candidate.get(
            "exact_outcome_decision"
        )
        met = (not applicable) or (
            base.get("satisfiable") == candidate.get("satisfiable")
            and base.get("model_count") != candidate.get("model_count")
            and base.get("projected_models") == candidate.get("projected_models")
            and not shortcut
        )
        return {"applicable": applicable, "met": met, "shortcut_detected": shortcut}
    raise QualificationError(f"unknown_pair:{perturbation}")


def evaluate_cell(
    cell: Mapping[str, Any],
    formal: Mapping[str, Any],
    vocabulary: Sequence[Mapping[str, Any]],
    theory_atoms: set[str],
    *,
    primary_engine: ExactEngine = primary_exact_engine,
    independent_engine: ExactEngine = independent_exact_engine,
) -> list[JsonDict]:
    """Emit one base decision and six deterministic paired decisions for a cell."""

    fixture_id = str(cell["fixture_id"])
    fixture_vocabulary = _fixture_vocabulary(fixture_id, vocabulary)
    parse_rows = [row for row in cell.get("parse_rows", []) if isinstance(row, Mapping)]
    accepted = [row for row in parse_rows if row.get("status") == "accepted"]
    nonparseable_count = len(parse_rows) - len(accepted)
    tuples = [_semantic_tuple(row, fixture_vocabulary) for row in accepted]
    mappings = [map_relation_tuple(values, fixture_vocabulary) for values in tuples]
    atoms = sorted({str(row["asp_atom"]) for row in mappings if row.get("accepted") is True})
    exact_atom_valid = bool(tuples) and all(row.get("accepted") is True for row in mappings)
    relation_atoms = {str(row["asp_atom"]) for row in fixture_vocabulary}
    base_program = _base_program(formal, relation_atoms)
    expected_models = _canonical_models(formal.get("answer_sets", []))
    allowed_atoms = set(theory_atoms)

    base = _stub_row(
        cell,
        "base",
        tuples=tuples,
        atoms=atoms,
        parseable_count=len(accepted),
        nonparseable_count=nonparseable_count,
        exact_atom_valid=exact_atom_valid,
    )
    base["expected_effect"] = {"kind": "sealed_exact_models", **_effect(expected_models)}
    if not accepted:
        base["compilation_status"] = "not_run_nonparseable"
        base["observed_effect"] = _not_run_effect("not_run_nonparseable")
        base["exact_outcome_decision"] = "not_parseable"
    elif not exact_atom_valid:
        base["compilation_status"] = "unsupported_atom"
        base["unsupported_atoms"] = [
            row.get("normalized_tuple") for row in mappings if row.get("accepted") is not True
        ]
        base["observed_effect"] = _not_run_effect("not_run_unsupported_atom")
        base["exact_outcome_decision"] = "unsupported_atom"
    else:
        _evaluate_program(
            row=base,
            program_text=_program_with_atoms(base_program, atoms),
            allowed_atoms=allowed_atoms,
            primary_engine=primary_engine,
            independent_engine=independent_engine,
        )
        base["expected_effect_met"] = base["primary_effect"]["models"] == expected_models
        base["pair_requirement_met"] = base["expected_effect_met"]
        base["exact_outcome_decision"] = (
            "qualified"
            if base["solver_parity"] is True and base["expected_effect_met"]
            else "disqualified"
        )

    rows = [base]
    if not exact_atom_valid:
        for perturbation in PERTURBATIONS[1:]:
            row = _stub_row(
                cell,
                perturbation,
                tuples=tuples,
                atoms=atoms,
                parseable_count=len(accepted),
                nonparseable_count=nonparseable_count,
                exact_atom_valid=False,
            )
            row["compilation_status"] = "not_run_base_ineligible"
            row["observed_effect"] = _not_run_effect("not_run_base_ineligible")
            rows.append(row)
        return rows

    base_models = base["primary_effect"]["models"]
    base_program_text = str(base["program"])

    rename_map = _entity_mapping(base_program_text, fixture_id, allowed_atoms)
    renamed_atoms = sorted(rename_map[atom] for atom in atoms)
    renamed = _stub_row(
        cell,
        "entity_renaming",
        tuples=tuples,
        atoms=renamed_atoms,
        parseable_count=len(accepted),
        nonparseable_count=nonparseable_count,
        exact_atom_valid=all(
            atom in {str(row["asp_atom"]) for row in vocabulary} for atom in renamed_atoms
        ),
    )
    renamed["entity_rename_map"] = rename_map
    renamed_expected = _rename_models(expected_models, rename_map)
    renamed["expected_effect"] = {"kind": "renamed_sealed_models", **_effect(renamed_expected)}
    _evaluate_program(
        row=renamed,
        program_text=rename_program(base_program_text, rename_map),
        allowed_atoms=allowed_atoms,
        primary_engine=primary_engine,
        independent_engine=independent_engine,
    )
    inverse = {value: key for key, value in rename_map.items()}
    renamed["projected_models"] = _rename_models(renamed["primary_effect"]["models"], inverse)
    renamed["expected_effect_met"] = renamed["primary_effect"]["models"] == renamed_expected
    renamed["exact_outcome_decision"] = (
        "qualified"
        if renamed["solver_parity"] is True and renamed["expected_effect_met"]
        else "disqualified"
    )
    renamed_pair = validate_pair("entity_renaming", base, renamed)
    renamed["pair_applicable"] = renamed_pair["applicable"]
    renamed["pair_requirement_met"] = renamed_pair["met"]
    renamed["isomorphic_invariant"] = renamed_pair["met"]
    rows.append(renamed)

    paraphrase_tuples = [build_paraphrase_tuple(values) for values in tuples]
    canonical_tuples = [canonicalize_paraphrase(values) for values in paraphrase_tuples]
    paraphrase = _stub_row(
        cell,
        "relation_paraphrase",
        tuples=paraphrase_tuples,
        atoms=atoms,
        parseable_count=len(accepted),
        nonparseable_count=nonparseable_count,
        exact_atom_valid=canonical_tuples == tuples,
    )
    paraphrase["canonical_tuples"] = canonical_tuples
    paraphrase["expected_effect"] = {
        "kind": "paraphrased_sealed_models",
        **_effect(expected_models),
    }
    _evaluate_program(
        row=paraphrase,
        program_text=base_program_text,
        allowed_atoms=allowed_atoms,
        primary_engine=primary_engine,
        independent_engine=independent_engine,
    )
    paraphrase["expected_effect_met"] = paraphrase["primary_effect"]["models"] == expected_models
    paraphrase["exact_outcome_decision"] = (
        "qualified"
        if paraphrase["solver_parity"] is True and paraphrase["expected_effect_met"]
        else "disqualified"
    )
    paraphrase_pair = validate_pair("relation_paraphrase", base, paraphrase)
    paraphrase["pair_applicable"] = paraphrase_pair["applicable"]
    paraphrase["pair_requirement_met"] = paraphrase_pair["met"]
    paraphrase["isomorphic_invariant"] = paraphrase_pair["met"]
    rows.append(paraphrase)

    reversed_tuples = [reverse_tuple(values) for values in tuples]
    reversed_mappings = [
        map_relation_tuple(values, fixture_vocabulary) for values in reversed_tuples
    ]
    reversal = _stub_row(
        cell,
        "relation_reversal",
        tuples=reversed_tuples,
        atoms=[],
        parseable_count=len(accepted),
        nonparseable_count=nonparseable_count,
        exact_atom_valid=all(row.get("accepted") is True for row in reversed_mappings),
    )
    reversal["compilation_status"] = "unsupported_atom"
    reversal["unsupported_atoms"] = [row["normalized_tuple"] for row in reversed_mappings]
    reversal["expected_effect"] = {"kind": "directional_vocabulary_rejection"}
    reversal["observed_effect"] = _not_run_effect("not_run_directional_rejection")
    reversal["exact_outcome_decision"] = (
        "qualified" if not reversal["exact_atom_valid"] else "disqualified"
    )
    reversal_pair = validate_pair("reversal", base, reversal)
    reversal["pair_applicable"] = reversal_pair["applicable"]
    reversal["pair_requirement_met"] = reversal_pair["met"]
    reversal["expected_effect_met"] = reversal_pair["met"]
    reversal["transform_valid"] = reversed_tuples != tuples
    rows.append(reversal)

    contradicted_atoms = contradiction_atoms(atoms, vocabulary, fixture_id)
    contradiction = _stub_row(
        cell,
        "contradiction_injection",
        tuples=tuples,
        atoms=contradicted_atoms,
        parseable_count=len(accepted),
        nonparseable_count=nonparseable_count,
        exact_atom_valid=True,
    )
    contradiction["expected_effect"] = {"kind": "contradiction", "satisfiable": False}
    _evaluate_program(
        row=contradiction,
        program_text=_program_with_atoms(base_program, contradicted_atoms),
        allowed_atoms=allowed_atoms,
        primary_engine=primary_engine,
        independent_engine=independent_engine,
    )
    contradiction["expected_effect_met"] = contradiction["satisfiable"] is False
    contradiction["exact_outcome_decision"] = (
        "qualified"
        if contradiction["solver_parity"] is True and contradiction["expected_effect_met"]
        else "disqualified"
    )
    contradiction_pair = validate_pair("contradiction", base, contradiction)
    contradiction["pair_applicable"] = contradiction_pair["applicable"]
    contradiction["pair_requirement_met"] = contradiction_pair["met"]
    contradiction["transform_valid"] = set(contradicted_atoms) >= relation_atoms
    rows.append(contradiction)

    omitted_atoms = omit_relation_atoms(atoms)
    omission = _stub_row(
        cell,
        "relation_omission",
        tuples=[],
        atoms=omitted_atoms,
        parseable_count=len(accepted),
        nonparseable_count=nonparseable_count,
        exact_atom_valid=True,
    )
    omission["expected_effect"] = {"kind": "all_proposal_relation_facts_removed"}
    _evaluate_program(
        row=omission,
        program_text=_program_with_atoms(base_program, omitted_atoms),
        allowed_atoms=allowed_atoms,
        primary_engine=primary_engine,
        independent_engine=independent_engine,
    )
    omission["expected_effect_met"] = omission["solver_parity"] is True and not omission["atoms"]
    omission["exact_outcome_decision"] = (
        "qualified" if omission["expected_effect_met"] else "disqualified"
    )
    omission_pair = validate_pair("omission", base, omission)
    omission["pair_applicable"] = omission_pair["applicable"]
    omission["pair_requirement_met"] = omission_pair["met"]
    omission["transform_valid"] = omission["atoms"] != base["atoms"]
    rows.append(omission)

    auxiliary_candidates = sorted(allowed_atoms - _program_atoms(base_program_text))
    if not auxiliary_candidates:
        raise QualificationError("restructuring_auxiliary_missing")
    auxiliary_atom = auxiliary_candidates[0]
    restructuring = _stub_row(
        cell,
        "solution_space_restructuring",
        tuples=tuples,
        atoms=atoms,
        parseable_count=len(accepted),
        nonparseable_count=nonparseable_count,
        exact_atom_valid=True,
    )
    restructuring["auxiliary_atom"] = auxiliary_atom
    restructuring["expected_effect"] = {
        "kind": "same_projected_models_and_satisfiability_different_model_count"
    }
    restructured_program = restructure_program(base_program_text, auxiliary_atom)
    _evaluate_program(
        row=restructuring,
        program_text=restructured_program,
        allowed_atoms=allowed_atoms,
        primary_engine=primary_engine,
        independent_engine=independent_engine,
    )
    restructuring["projected_models"] = _project_models(
        restructuring["primary_effect"]["models"], auxiliary_atom
    )
    restructuring["exact_outcome_decision"] = (
        "qualified" if restructuring["solver_parity"] is True else "disqualified"
    )
    restructuring_pair = validate_pair("solution_space_restructuring", base, restructuring)
    restructuring["pair_applicable"] = restructuring_pair["applicable"]
    restructuring["pair_requirement_met"] = restructuring_pair["met"]
    restructuring["expected_effect_met"] = restructuring_pair["met"]
    restructuring["shortcut_detected"] = restructuring_pair["shortcut_detected"]
    restructuring["transform_valid"] = restructured_program != base_program_text
    if not restructuring_pair["met"]:
        restructuring["exact_outcome_decision"] = "disqualified"
    rows.append(restructuring)
    return rows


def proposal_strings(value: Mapping[str, Any]) -> list[str]:
    """Decode the explicit raw proposal fields used by leakage checks."""

    if isinstance(value.get("raw_output"), str):
        return [str(value["raw_output"])]
    encoded = value.get("raw_output_b64")
    if not isinstance(encoded, str):
        return []
    try:
        return [base64.b64decode(encoded, validate=True).decode("utf-8")]
    except (ValueError, UnicodeDecodeError):
        return []


def detect_held_leakage(
    proposal_artifact: Mapping[str, Any], held_payload: Mapping[str, Any]
) -> list[JsonDict]:
    """Find direct formal strings without treating a correct relation as leakage."""

    texts: list[tuple[str, str]] = []

    def collect(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                collect(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, f"{location}[{index}]")
        elif isinstance(value, str):
            texts.append((location, value))

    collect(proposal_artifact.get("prompt_manifest", {}), "prompt_manifest")
    for cell in proposal_artifact.get("cell_manifest", []):
        if not isinstance(cell, Mapping):
            continue
        for text in proposal_strings(cell):
            texts.append((f"cell:{cell.get('cell_identity')}", text))
    rows = []
    for formal in held_payload.get("rows", []):
        if not isinstance(formal, Mapping):
            continue
        fingerprints = {
            "asp_program": str(formal.get("asp_program", "")),
            "answer_sets": canonical_json(formal.get("answer_sets", [])),
            "zero_energy_states": canonical_json(formal.get("zero_energy_states", [])),
            "solver_receipt": canonical_json(formal.get("solver_receipt", {})),
        }
        for leak_kind, fingerprint in fingerprints.items():
            if not fingerprint or fingerprint in {"[]", "{}"}:
                continue
            for location, text in texts:
                if fingerprint in text:
                    rows.append(
                        {
                            "fixture_id": formal.get("fixture_id"),
                            "leak_kind": leak_kind,
                            "location": location,
                            "fingerprint_sha256": sha256_bytes(fingerprint.encode("utf-8")),
                        }
                    )
    return rows


def _summary_rows(rows: Sequence[Mapping[str, Any]], field: str) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result = []
    for value in sorted(grouped):
        selected = grouped[value]
        base_rows = [row for row in selected if row["perturbation"] == "base"]
        compiled = [row for row in selected if row.get("compilation_status") == "compiled"]
        iso = [
            row
            for row in selected
            if row.get("perturbation") in {"entity_renaming", "relation_paraphrase"}
            and row.get("pair_applicable") is True
        ]
        directional = [
            row
            for row in selected
            if row.get("perturbation")
            in {"relation_reversal", "contradiction_injection", "relation_omission"}
            and row.get("pair_applicable") is True
        ]
        shortcuts = [
            row
            for row in selected
            if row.get("perturbation") == "solution_space_restructuring"
            and row.get("pair_applicable") is True
        ]
        all_exact = all(row.get("exact_atom_valid") is True for row in base_rows)
        all_parity = all(row.get("solver_parity") is True for row in compiled)
        all_pairs = all(
            row.get("pair_requirement_met") is True for row in iso + directional + shortcuts
        )
        result.append(
            {
                field: value,
                "cell_count": len({str(row["cell_identity"]) for row in selected}),
                "expected_terminal_row_count": len({str(row["cell_identity"]) for row in selected})
                * len({str(row["perturbation"]) for row in selected}),
                "terminal_row_count": sum(row.get("terminal") is True for row in selected),
                "proposal_covered_count": sum(
                    row.get("proposal_covered") is True for row in base_rows
                ),
                "proposal_coverage": (
                    sum(row.get("proposal_covered") is True for row in base_rows) / len(base_rows)
                    if base_rows
                    else None
                ),
                "exact_atom_valid_count": sum(
                    row.get("exact_atom_valid") is True for row in base_rows
                ),
                "exact_atom_validity": (
                    sum(row.get("exact_atom_valid") is True for row in base_rows) / len(base_rows)
                    if base_rows
                    else None
                ),
                "compiled_count": len(compiled),
                "solver_parity_count": sum(row.get("solver_parity") is True for row in compiled),
                "solver_parity_rate": (
                    sum(row.get("solver_parity") is True for row in compiled) / len(compiled)
                    if compiled
                    else None
                ),
                "isomorphic_pair_count": len(iso),
                "isomorphic_invariance_count": sum(
                    row.get("isomorphic_invariant") is True for row in iso
                ),
                "isomorphic_invariance_rate": (
                    sum(row.get("isomorphic_invariant") is True for row in iso) / len(iso)
                    if iso
                    else None
                ),
                "directional_pair_count": len(directional),
                "directional_pair_pass_count": sum(
                    row.get("pair_requirement_met") is True for row in directional
                ),
                "restructuring_shortcut_eligible_count": len(shortcuts),
                "restructuring_shortcut_count": sum(
                    row.get("shortcut_detected") is True for row in shortcuts
                ),
                "restructuring_shortcut_rate": (
                    sum(row.get("shortcut_detected") is True for row in shortcuts) / len(shortcuts)
                    if shortcuts
                    else None
                ),
                "qualification_decision": (
                    "qualified"
                    if base_rows and all_exact and all_parity and all_pairs
                    else "disqualified"
                ),
            }
        )
    return result


def _by_arm_base_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> list[JsonDict]:
    bases = [row for row in rows if row.get("perturbation") == "base"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in bases:
        grouped[str(row["arm"])].append(row)
    result = []
    for arm in sorted(grouped):
        selected = grouped[arm]
        numerator = sum(row.get(metric) is True for row in selected)
        result.append(
            {
                "arm": arm,
                "numerator": numerator,
                "denominator": len(selected),
                "rate": numerator / len(selected),
                "parseable_denominator": sum(
                    row.get("proposal_covered") is True for row in selected
                ),
            }
        )
    return result


def _isomorphic_by_arm(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("perturbation") in {"entity_renaming", "relation_paraphrase"}:
            grouped[str(row["arm"])].append(row)
    result = []
    for arm in sorted(grouped):
        eligible = [row for row in grouped[arm] if row.get("pair_applicable") is True]
        numerator = sum(row.get("isomorphic_invariant") is True for row in eligible)
        result.append(
            {
                "arm": arm,
                "numerator": numerator,
                "denominator": len(eligible),
                "rate": numerator / len(eligible) if eligible else None,
            }
        )
    return result


def _blind_spots(
    rows: Sequence[Mapping[str, Any]],
    formal_by_id: Mapping[str, Mapping[str, Any]],
    vocabulary: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    result = []
    for row in rows:
        if row.get("perturbation") != "base":
            continue
        fixture_id = str(row["fixture_id"])
        relation_atoms = {
            str(item["asp_atom"]) for item in _fixture_vocabulary(fixture_id, vocabulary)
        }
        formal_program = asp_energy.parse_program(
            str(formal_by_id[fixture_id]["asp_program"]), program_id=fixture_id
        )
        expected = {atom for _, atom, _ in formal_program.facts if atom in relation_atoms}
        for atom in sorted(expected - set(row.get("atoms", []))):
            result.append(
                {
                    "cell_identity": row["cell_identity"],
                    "arm": row["arm"],
                    "model_id": row["model_id"],
                    "seed": row["seed"],
                    "fixture_id": fixture_id,
                    "family": row["family"],
                    "missing_expected_atom": atom,
                    "reason": "proposal_omission_not_proved_complete_by_exact_solver",
                }
            )
    return result


def _row_projection(row: Mapping[str, Any], fields: Sequence[str]) -> JsonDict:
    return {field: deepcopy(row.get(field)) for field in fields}


def _derived_metrics(artifact: Mapping[str, Any]) -> JsonDict:
    rows = artifact.get("rows", [])
    return {
        "terminal_row_count": sum(row.get("terminal") is True for row in rows),
        "arm_summary_rows": _summary_rows(rows, "arm"),
        "model_summary_rows": _summary_rows(rows, "model_id"),
        "family_summary_rows": _summary_rows(rows, "family"),
        "seed_summary_rows": _summary_rows(rows, "seed"),
        "perturbation_summary_rows": _summary_rows(rows, "perturbation"),
        "proposal_coverage_by_arm": _by_arm_base_metric(rows, "proposal_covered"),
        "exact_atom_validity_by_arm": _by_arm_base_metric(rows, "exact_atom_valid"),
        "isomorphic_invariance_by_arm": _isomorphic_by_arm(rows),
        "solver_disagreement_count": sum(
            row.get("compilation_status") == "compiled" and row.get("solver_parity") is not True
            for row in rows
        ),
        "held_leakage_count": len(artifact.get("held_leakage_rows", [])),
    }


def replay_reported_metrics(artifact: Mapping[str, Any]) -> JsonDict:
    """Rebuild every reported aggregate directly from terminal rows."""

    recomputed = _derived_metrics(artifact)
    reported = {
        key: deepcopy(artifact.get(key))
        for key in (
            "arm_summary_rows",
            "model_summary_rows",
            "family_summary_rows",
            "seed_summary_rows",
            "perturbation_summary_rows",
            "proposal_coverage_by_arm",
            "exact_atom_validity_by_arm",
            "isomorphic_invariance_by_arm",
            "solver_disagreement_count",
            "held_leakage_count",
        )
    }
    comparable = {key: recomputed[key] for key in reported}
    agreement = reported == comparable
    return {
        "reported_metrics_sha256": sha256_json(reported),
        "recomputed_metrics_sha256": sha256_json(comparable),
        "agreement": agreement,
        "recomputed_ready_score": int(
            agreement
            and recomputed["solver_disagreement_count"] == 0
            and recomputed["held_leakage_count"] == 0
        ),
    }


def _attach_principles(artifact: JsonDict) -> None:
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves one required qualification receipt.")
        for key in artifact
    }
    for key in REQUIRED_ARTIFACT_FIELDS:
        artifact["field_principles"][key] = FIELD_PRINCIPLES[key]
    for key, principle in FIELD_PRINCIPLES.items():
        if key.startswith("gate_check"):
            artifact["field_principles"][key] = principle


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    included = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
    }
    return sha256_json(included)


def blocked_artifact(
    *,
    date: str,
    checks: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    sealed_sidecar_hashes: Mapping[str, Any],
    duration_s: float,
    sealed_sidecar_open_count: int = 0,
) -> JsonDict:
    """Build a complete blocked receipt without inventing semantic rows."""

    summary = gate_summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6914,
        "run_date": date,
        "status": "blocked",
        "field_principles": {},
        "preconditions_checked": {
            "public_inputs_hashed_before_sealed_open": True,
            "sealed_sidecar_open_count": sealed_sidecar_open_count,
            "gate_check_summary": summary,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "sealed_sidecar_hashes": deepcopy(dict(sealed_sidecar_hashes)),
        "rows": [],
        "asp_compilation_rows": [],
        "bounded_vocabulary_rows": [],
        "primary_solver_rows": [],
        "independent_solver_rows": [],
        "solver_parity_rows": [],
        "entity_renaming_rows": [],
        "paraphrase_rows": [],
        "reversal_rows": [],
        "contradiction_rows": [],
        "omission_rows": [],
        "restructuring_rows": [],
        "arm_summary_rows": [],
        "model_summary_rows": [],
        "family_summary_rows": [],
        "seed_summary_rows": [],
        "perturbation_summary_rows": [],
        "proposal_coverage_by_arm": [],
        "exact_atom_validity_by_arm": [],
        "isomorphic_invariance_by_arm": [],
        "completeness_blind_spot_rows": [],
        "held_leakage_rows": [],
        "sealed_access_rows": [],
        "solver_disagreement_count": 0,
        "held_leakage_count": 0,
        "model_inference_call_count": 0,
        "reported_vs_recomputed_metrics": {
            "agreement": True,
            "recomputed_ready_score": 0,
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "asp_isomorphic_shard_ready_score": 0,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    cells: Sequence[Mapping[str, Any]],
    formal_rows: Sequence[Mapping[str, Any]],
    vocabulary: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    sealed_sidecar_hashes: Mapping[str, Any],
    precondition_checks: Sequence[Mapping[str, Any]],
    sealed_access_rows: Sequence[Mapping[str, Any]],
    held_leakage_rows: Sequence[Mapping[str, Any]],
    duration_s: float,
    primary_engine: ExactEngine = primary_exact_engine,
    independent_engine: ExactEngine = independent_exact_engine,
) -> JsonDict:
    """Build all cell pairs, exact receipts, summaries, and the terminal gate."""

    formal_by_id = {str(row["fixture_id"]): row for row in formal_rows}
    theory_atoms: set[str] = set()
    for formal in formal_rows:
        theory_atoms.update(_program_atoms(str(formal["asp_program"])))
    rows = []
    for cell in cells:
        fixture_id = str(cell["fixture_id"])
        if fixture_id not in formal_by_id:
            raise QualificationError(f"missing_formal_row:{fixture_id}")
        rows.extend(
            evaluate_cell(
                cell,
                formal_by_id[fixture_id],
                vocabulary,
                theory_atoms,
                primary_engine=primary_engine,
                independent_engine=independent_engine,
            )
        )
    metrics = _derived_metrics({"rows": rows, "held_leakage_rows": list(held_leakage_rows)})
    compilation_fields = (
        "row_id",
        "cell_identity",
        "perturbation",
        "program",
        "program_sha256",
        "compilation_status",
        "program_atoms",
        "unsupported_atoms",
        "exact_atom_valid",
        "exact_outcome_decision",
    )
    solver_fields = (
        "row_id",
        "cell_identity",
        "perturbation",
        "compilation_status",
        "solver_parity",
        "expected_effect_met",
    )
    expected_rows = len(cells) * len(PERTURBATIONS)
    unique_rows = len({str(row["row_id"]) for row in rows})
    access_ok = {str(row.get("sidecar")) for row in sealed_access_rows} == set(
        SIDECAR_PATHS
    ) and all(
        row.get("open_count") == 1 and row.get("hash_match") is True for row in sealed_access_rows
    )
    transforms_ok = all(row.get("transform_valid") is True for row in rows)
    decisions_ok = bool(metrics["arm_summary_rows"] and metrics["family_summary_rows"]) and all(
        row.get("qualification_decision") in {"qualified", "disqualified"}
        for row in [*metrics["arm_summary_rows"], *metrics["family_summary_rows"]]
    )
    checks = [
        *[deepcopy(dict(row)) for row in precondition_checks],
        gate_check("expected_terminal_row_count", expected_rows, len(rows)),
        gate_check("unique_terminal_row_count", expected_rows, unique_rows),
        gate_check("all_rows_terminal", True, all(row.get("terminal") is True for row in rows)),
        gate_check("all_transforms_valid", True, transforms_ok),
        gate_check("solver_disagreement_count", 0, metrics["solver_disagreement_count"]),
        gate_check("held_leakage_count", 0, metrics["held_leakage_count"]),
        gate_check("sealed_access_protocol", True, access_ok),
        gate_check("arm_and_family_decisions", True, decisions_ok),
        gate_check("model_inference_call_count", 0, 0),
    ]
    provisional_summary = gate_summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6914,
        "run_date": date,
        "status": "complete",
        "field_principles": {},
        "preconditions_checked": {
            "public_inputs_hashed_before_sealed_open": True,
            "sealed_sidecar_open_count": sum(
                int(row.get("open_count", 0)) for row in sealed_access_rows
            ),
            "sealed_access_rows": [deepcopy(dict(row)) for row in sealed_access_rows],
            "gate_check_summary": gate_summary(precondition_checks),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "sealed_sidecar_hashes": deepcopy(dict(sealed_sidecar_hashes)),
        "rows": rows,
        "asp_compilation_rows": [_row_projection(row, compilation_fields) for row in rows],
        "bounded_vocabulary_rows": [
            {
                "asp_atom": atom,
                "relation_atom": atom in {str(row["asp_atom"]) for row in vocabulary},
                "source": "sealed_theory_or_closed_relation_vocabulary",
            }
            for atom in sorted(theory_atoms)
        ],
        "primary_solver_rows": [
            {
                **_row_projection(row, solver_fields),
                "effect": deepcopy(row["primary_effect"]),
                "receipt": deepcopy(row["primary_solver_receipt"]),
            }
            for row in rows
        ],
        "independent_solver_rows": [
            {
                **_row_projection(row, solver_fields),
                "effect": deepcopy(row["independent_effect"]),
                "receipt": deepcopy(row["independent_solver_receipt"]),
            }
            for row in rows
        ],
        "solver_parity_rows": [_row_projection(row, solver_fields) for row in rows],
        "entity_renaming_rows": [
            deepcopy(row) for row in rows if row["perturbation"] == "entity_renaming"
        ],
        "paraphrase_rows": [
            deepcopy(row) for row in rows if row["perturbation"] == "relation_paraphrase"
        ],
        "reversal_rows": [
            deepcopy(row) for row in rows if row["perturbation"] == "relation_reversal"
        ],
        "contradiction_rows": [
            deepcopy(row) for row in rows if row["perturbation"] == "contradiction_injection"
        ],
        "omission_rows": [
            deepcopy(row) for row in rows if row["perturbation"] == "relation_omission"
        ],
        "restructuring_rows": [
            deepcopy(row) for row in rows if row["perturbation"] == "solution_space_restructuring"
        ],
        "arm_summary_rows": metrics["arm_summary_rows"],
        "model_summary_rows": metrics["model_summary_rows"],
        "family_summary_rows": metrics["family_summary_rows"],
        "seed_summary_rows": metrics["seed_summary_rows"],
        "perturbation_summary_rows": metrics["perturbation_summary_rows"],
        "proposal_coverage_by_arm": metrics["proposal_coverage_by_arm"],
        "exact_atom_validity_by_arm": metrics["exact_atom_validity_by_arm"],
        "isomorphic_invariance_by_arm": metrics["isomorphic_invariance_by_arm"],
        "completeness_blind_spot_rows": _blind_spots(rows, formal_by_id, vocabulary),
        "held_leakage_rows": [deepcopy(dict(row)) for row in held_leakage_rows],
        "sealed_access_rows": [deepcopy(dict(row)) for row in sealed_access_rows],
        "solver_disagreement_count": metrics["solver_disagreement_count"],
        "held_leakage_count": metrics["held_leakage_count"],
        "model_inference_call_count": 0,
        "reported_vs_recomputed_metrics": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "asp_isomorphic_shard_ready_score": 0,
        "gate_check_summary": provisional_summary,
        "verifier_is_oracle": True,
        "verdict_class": "disqualified",
        "honest_verdict": DISQUALIFIED_VERDICT,
    }
    replay = replay_reported_metrics(artifact)
    checks.append(gate_check("reported_vs_recomputed_metrics", True, replay["agreement"]))
    final_summary = gate_summary(checks)
    ready = int(final_summary["passed"])
    artifact.update(
        {
            "reported_vs_recomputed_metrics": replay,
            "asp_isomorphic_shard_ready_score": ready,
            "gate_check_summary": final_summary,
            "verdict_class": "circular_positive" if ready else "disqualified",
            "honest_verdict": READY_VERDICT if ready else DISQUALIFIED_VERDICT,
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, row replay, verdict boundaries, and the readiness gate."""

    errors = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    principles = artifact.get("field_principles", {})
    if any(key not in principles for key in artifact) or any(
        key not in principles for key in FIELD_PRINCIPLES if key.startswith("gate_check")
    ):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in {
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("verdict_class") == "positive":
        errors.append("positive_verdict_forbidden")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("model_inference_call_count") != 0:
        errors.append("model_inference_call_count")
    if artifact.get("status") != "blocked":
        replay = replay_reported_metrics(artifact)
        if not replay["agreement"]:
            errors.append("aggregate_disagreement")
        expected_rows = len(
            {str(row.get("cell_identity")) for row in artifact.get("rows", [])}
        ) * len(PERTURBATIONS)
        if len(artifact.get("rows", [])) != expected_rows:
            errors.append("terminal_row_count")
        if artifact.get("solver_disagreement_count") != 0:
            errors.append("solver_disagreement_count")
        if artifact.get("held_leakage_count") != 0:
            errors.append("held_leakage_count")
        recomputed_ready = int(
            not errors
            and artifact.get("gate_check_summary", {}).get("passed") is True
            and replay["agreement"]
        )
        if artifact.get("asp_isomorphic_shard_ready_score") != recomputed_ready:
            errors.append("readiness_disagreement")
    return errors


def _read_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationError(f"public_artifact_not_object:{path}")
    return value


def _safe_hash(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError:
        return "missing"


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run_experiment(
    *,
    date: str,
    output_path: Path = RESULT_PATH,
    root: Path = REPO_ROOT,
    expected_public_hashes: Mapping[str, str] = EXPECTED_PUBLIC_HASHES,
) -> JsonDict:
    """Hash public inputs, open sealed labels once, and write one terminal receipt."""

    started = time.perf_counter()
    observed = {name: _safe_hash(root / path) for name, path in PUBLIC_PATHS.items()}
    source_hashes = {
        name: {
            "path": PUBLIC_PATHS[name].as_posix(),
            "expected_sha256": expected_public_hashes.get(name),
            "observed_sha256": observed[name],
        }
        for name in PUBLIC_PATHS
    }
    artifacts: dict[str, JsonDict] = {}
    public_read_errors = []
    for name in ("exp6274", "exp6886", "exp6900", "exp6912"):
        try:
            artifacts[name] = _read_object(root / PUBLIC_PATHS[name])
        except (OSError, UnicodeError, json.JSONDecodeError, QualificationError) as exc:
            artifacts[name] = {}
            public_read_errors.append(gate_check(f"public_read:{name}", None, str(exc)))
    preconditions = public_precondition_summary(
        observed,
        artifacts,
        expected_hashes=expected_public_hashes,
        solver_name=asp_energy.solver_name_version(),
    )
    public_checks = [*preconditions["checks"], *public_read_errors]
    sealed_hashes: JsonDict = {
        name: {
            "path": str(path),
            "expected_sha256": EXPECTED_SIDECAR_HASHES[name],
            "observed_sha256": "not_opened",
        }
        for name, path in SIDECAR_PATHS.items()
    }
    if not gate_summary(public_checks)["passed"]:
        artifact = blocked_artifact(
            date=date,
            checks=public_checks,
            source_artifact_hashes=source_hashes,
            sealed_sidecar_hashes=sealed_hashes,
            duration_s=time.perf_counter() - started,
        )
        _write_json(output_path, artifact)
        return artifact

    readers = {
        name: SealedSidecarReader(
            path,
            EXPECTED_SIDECAR_HASHES[name],
            expected_split=name,
        )
        for name, path in SIDECAR_PATHS.items()
    }
    payloads: dict[str, JsonDict] = {}
    try:
        for name in ("calibration", "held"):
            payloads[name] = readers[name].open_once()
            sealed_hashes[name]["observed_sha256"] = readers[name].observed_sha256
    except (OSError, UnicodeError, json.JSONDecodeError, QualificationError) as exc:
        sidecar_checks = [
            *public_checks,
            gate_check("sealed_sidecars", "valid_exact_once", str(exc)),
        ]
        artifact = blocked_artifact(
            date=date,
            checks=sidecar_checks,
            source_artifact_hashes=source_hashes,
            sealed_sidecar_hashes=sealed_hashes,
            duration_s=time.perf_counter() - started,
            sealed_sidecar_open_count=sum(reader.open_count for reader in readers.values()),
        )
        _write_json(output_path, artifact)
        return artifact

    exp6886 = artifacts["exp6886"]
    exp6900 = artifacts["exp6900"]
    exp6912 = artifacts["exp6912"]
    formal_rows = [*payloads["calibration"]["rows"], *payloads["held"]["rows"]]
    cells = [
        deepcopy(dict(row["source_cell"]))
        for row in exp6912.get("rows", [])
        if isinstance(row, Mapping)
        and row.get("row_type") == "source_cell_replay"
        and isinstance(row.get("source_cell"), Mapping)
    ]
    relevant_fixture_ids = {str(cell["fixture_id"]) for cell in cells}
    formal_by_id = {str(row["fixture_id"]): row for row in formal_rows}
    leakage = detect_held_leakage(exp6900, payloads["held"])
    sealed_access_rows = [
        {
            "sidecar": name,
            "open_count": readers[name].open_count,
            "hash_match": readers[name].observed_sha256 == EXPECTED_SIDECAR_HASHES[name],
            "observed_sha256": readers[name].observed_sha256,
        }
        for name in ("calibration", "held")
    ]
    post_open_checks = [
        *public_checks,
        gate_check(
            "sealed_calibration_hash",
            EXPECTED_SIDECAR_HASHES["calibration"],
            readers["calibration"].observed_sha256,
        ),
        gate_check(
            "sealed_held_hash", EXPECTED_SIDECAR_HASHES["held"], readers["held"].observed_sha256
        ),
        gate_check("immutable_cell_count", EXPECTED_CELL_COUNT, len(cells)),
        gate_check(
            "unique_cell_count",
            EXPECTED_CELL_COUNT,
            len({str(cell["cell_identity"]) for cell in cells}),
        ),
        gate_check(
            "formal_fixture_coverage",
            relevant_fixture_ids,
            relevant_fixture_ids & set(formal_by_id),
        ),
    ]
    artifact = build_artifact(
        date=date,
        cells=cells,
        formal_rows=formal_rows,
        vocabulary=exp6886.get("closed_vocabulary_manifest", {}).get("entries", []),
        source_artifact_hashes=source_hashes,
        sealed_sidecar_hashes=sealed_hashes,
        precondition_checks=post_open_checks,
        sealed_access_rows=sealed_access_rows,
        held_leakage_rows=leakage,
        duration_s=time.perf_counter() - started,
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise QualificationError("artifact_validation:" + ",".join(errors))
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the required date and output path, then run the reducer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    run_experiment(date=args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the required wrapper calls main.
    raise SystemExit(main())
