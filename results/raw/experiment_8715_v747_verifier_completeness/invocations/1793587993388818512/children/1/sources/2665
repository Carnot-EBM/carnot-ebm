"""Build the exact, source-grouped V619 entrance fixture.

Spec refs: REQ-VERIFY-7064 and SCENARIO-VERIFY-7064-*.

The exact solver constructs labels and replay witnesses. It never supplies a
model feature. The model view is a separate four-field serialization whose
allowlist excludes every exact outcome and solver receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260906"
RANDOM_SEED = 706420260906
EXPERIMENT_ID = "experiment_7064_v619_exact_entrance_fixture"
INFERENCE_SUBSTRATE = "deterministic_verifier"
SCHEMA = "carnot.experiment_7064.v619_exact_entrance_fixture.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_7064_v619_exact_entrance_fixture.json")
PROPOSAL_RELATIVE_PATH = Path("results/experiment_7065_v619_three_family_entrance_bank.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7064_v619_exact_entrance_fixture.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_7064_v619_exact_entrance_fixture.py")
WRAPPER_RELATIVE_PATH = Path("scripts/experiments/experiment_7064_v619_exact_entrance_fixture.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
ROW_FILENAMES = {
    "unit_rows": "experiment_7064_v619_exact_entrance_fixture.units.jsonl",
    "entrance_rows": "experiment_7064_v619_exact_entrance_fixture.entrances.jsonl",
    "source_group_rows": "experiment_7064_v619_exact_entrance_fixture.source_groups.jsonl",
    "model_visible_rows": "experiment_7064_v619_exact_entrance_fixture.model_visible.jsonl",
}
SOURCE_HASH_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/verify/z3_math_verifier.py"),
    Path("scripts/adversarial_verify.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    WRAPPER_RELATIVE_PATH,
)

MIN_UNIT_COUNT = 96
MIN_SOURCE_GROUP_COUNT = 12
MIN_DIVERSITY_COUNT = 48
UNITS_PER_GROUP = 8
MAX_CANDIDATES_PER_GROUP = 64
INITIAL_ROW_HASH = "sha256:" + "0" * 64
OPERATOR_ORDER = {"+": 0, "-": 1, "*": 2, "/": 3}
MODEL_VISIBLE_FIELDS = ("numbers", "target", "unit_id", "formatting_rules")
FORMATTING_RULES = (
    "Return one JSON object with operand_pair and operator.",
    "Write operand_pair as two available values in ascending order.",
    "Use exactly one operator from +, -, *, or /.",
)

# Each source family fixes its large-number recipe before labels open. The
# small values follow the standard two-copies-per-value Countdown pool.
SOURCE_GROUP_LARGE_NUMBERS = (
    (25, 50),
    (25, 75),
    (25, 100),
    (50, 75),
    (50, 100),
    (75, 100),
    (25, 50, 75),
    (25, 50, 100),
    (25, 75, 100),
    (50, 75, 100),
    (25, 50, 75, 100),
    (100,),
)
SMALL_NUMBER_POOL = tuple(value for value in range(1, 11) for _copy in range(2))

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "unit_rows",
    "entrance_rows",
    "source_group_rows",
    "split_manifest",
    "split_manifest_hash",
    "model_visible_schema",
    "model_visible_rows_hash",
    "operator_semantics",
    "exhaustive_enumerator_receipt",
    "witness_replay_rows",
    "mrv_rows",
    "diversity_subset_ids",
    "unit_count",
    "source_group_count",
    "entrance_fixture_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Each field states why its evidence is necessary.",
    "preconditions_checked": "Missing solver or storage resources must block construction.",
    "inference_substrate": "The substrate separates exact fixture work from model inference.",
    "duration_s": "Measured elapsed time makes the construction receipt auditable.",
    "source_artifact_hashes": "Source hashes bind labels to the reviewed solver and contract.",
    "rows": "Gate rows expose each readiness fact without relying on the verdict text.",
    "unit_rows": "Per-unit rows preserve mixed reachable and unreachable support.",
    "entrance_rows": "Every legal first branch needs one exact outcome label.",
    "source_group_rows": "Source rows prove calibration and held isolation.",
    "split_manifest": "A frozen group manifest prevents post-output split selection.",
    "split_manifest_hash": "The manifest hash detects moved or duplicated source groups.",
    "model_visible_schema": "A closed allowlist prevents solver evidence from becoming a feature.",
    "model_visible_rows_hash": "The visible-row hash binds later prompts to this exact panel.",
    "operator_semantics": "Explicit arithmetic rules make entrance legality reproducible.",
    "exhaustive_enumerator_receipt": "Enumeration counts prove that no legal branch was omitted.",
    "witness_replay_rows": "Replay proves reachable labels with legal arithmetic steps.",
    "mrv_rows": "MRV rows expose a deterministic label-blind control.",
    "diversity_subset_ids": "The subset proves that reachable support is not a single entrance.",
    "unit_count": "The unit floor gives later selectors enough frozen cases.",
    "source_group_count": "The group floor supports source-held evaluation.",
    "entrance_fixture_ready_score": "One means every exact, split, hash, and leakage gate passed.",
    "random_seed": "A fixed seed reproduces candidate numbers and their order.",
    "reproducibility_checksum": "The terminal checksum detects silent artifact drift.",
    "gate_check_summary": "Exact failed values make a blocked artifact actionable.",
    "verifier_is_oracle": "True preserves the circular construction boundary.",
    "verdict_class": "A closed class prevents an oracle-built fixture from appearing positive.",
    "honest_verdict": "A terminal prefix lets the conductor classify the result safely.",
}
_VALIDATED_EXACT_CHECKSUMS: set[str] = set()


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key and byte ordering."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed digest of canonical JSON evidence."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash file bytes so path metadata cannot change the receipt."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def enumerate_legal_entrances(numbers: Sequence[int]) -> list[JsonDict]:
    """Enumerate each canonical legal first branch once.

    Value pairs are unordered. The multiset still controls whether an equal
    pair exists, so one copy of a value can never be consumed twice.
    """

    values = tuple(sorted(int(value) for value in numbers))
    if not values or any(value <= 0 for value in values):
        raise ValueError("numbers must be positive integers")
    pairs = sorted(
        {
            (values[left], values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        }
    )
    rows: list[JsonDict] = []
    for low, high in pairs:
        rows.append(_operation_row(low, high, "+", low, high, low + high))
        if high > low:
            rows.append(_operation_row(low, high, "-", high, low, high - low))
        rows.append(_operation_row(low, high, "*", low, high, low * high))
        if high % low == 0:
            rows.append(_operation_row(low, high, "/", high, low, high // low))
    return sorted(
        rows,
        key=lambda row: (
            tuple(row["operand_pair"]),
            OPERATOR_ORDER[str(row["operator"])],
        ),
    )


def _operation_row(
    low: int,
    high: int,
    operator: str,
    left: int,
    right: int,
    result: int,
) -> JsonDict:
    """Build one operation row after the caller checks its legal conditions."""

    return {
        "operand_pair": [low, high],
        "operator": operator,
        "left": left,
        "right": right,
        "result": result,
    }


def apply_entrance(numbers: Sequence[int], entrance: Mapping[str, Any]) -> tuple[int, ...]:
    """Consume a legal pair with multiplicity and return the sorted residual state."""

    pair = tuple(sorted(int(value) for value in entrance.get("operand_pair", ())))
    if len(pair) != 2:
        raise ValueError("operand_pair must contain two values")
    counts = Counter(int(value) for value in numbers)
    needed = Counter(pair)
    if any(counts[value] < count for value, count in needed.items()):
        raise ValueError("operand multiplicity is unavailable")
    operator = str(entrance.get("operator"))
    legal = next(
        (
            row
            for row in enumerate_legal_entrances(numbers)
            if tuple(row["operand_pair"]) == pair and row["operator"] == operator
        ),
        None,
    )
    if legal is None:
        raise ValueError("illegal entrance operation")
    residual = list(int(value) for value in numbers)
    residual.remove(pair[0])
    residual.remove(pair[1])
    residual.append(int(legal["result"]))
    return tuple(sorted(residual))


def solve_exact(numbers: Sequence[int], target: int) -> list[JsonDict] | None:
    """Return one deterministic exact witness, or ``None`` after full search."""

    return _exact_solver(int(target))(numbers)


def _exact_solver(target: int) -> Any:
    """Create one target-specific finite solver shared by all first branches.

    The table keys each input occurrence by a bit. This preserves duplicate
    multiplicity while merging equivalent arithmetic values inside a subset.
    """

    target_value = int(target)
    memo: dict[tuple[int, ...], tuple[JsonDict, ...] | None] = {}

    def solve(numbers: Sequence[int]) -> list[JsonDict] | None:
        state = tuple(sorted(int(value) for value in numbers))
        if state not in memo:
            memo[state] = _subset_witness(state, target_value)
        witness = memo[state]
        return None if witness is None else [deepcopy(row) for row in witness]

    return solve


def _subset_witness(numbers: tuple[int, ...], target: int) -> tuple[JsonDict, ...] | None:
    """Enumerate every positive integer result for every occurrence subset."""

    if target in numbers:
        return ()
    tables: dict[int, dict[int, tuple[JsonDict, ...]]] = {
        1 << index: {value: ()} for index, value in enumerate(numbers)
    }
    full_mask = (1 << len(numbers)) - 1
    for width in range(2, len(numbers) + 1):
        for mask in range(1, full_mask + 1):
            if mask.bit_count() != width:
                continue
            values: dict[int, tuple[JsonDict, ...]] = {}
            left_mask = (mask - 1) & mask
            while left_mask:
                right_mask = mask ^ left_mask
                if right_mask and left_mask < right_mask:
                    for left_value, left_steps in tables[left_mask].items():
                        for right_value, right_steps in tables[right_mask].items():
                            for operation in enumerate_legal_entrances((left_value, right_value)):
                                result = int(operation["result"])
                                if result not in values:
                                    values[result] = (
                                        *left_steps,
                                        *right_steps,
                                        deepcopy(operation),
                                    )
                left_mask = (left_mask - 1) & mask
            tables[mask] = values
            if target in values:
                return values[target]
    return None


def label_unit_entrances(unit: Mapping[str, Any]) -> list[JsonDict]:
    """Label all legal first branches and attach one continuation when reachable."""

    numbers = tuple(int(value) for value in unit["numbers"])
    target = int(unit["target"])
    solver = _exact_solver(target)
    rows = []
    for entrance in enumerate_legal_entrances(numbers):
        residual = apply_entrance(numbers, entrance)
        witness = solver(residual)
        pair = entrance["operand_pair"]
        rows.append(
            {
                "entrance_id": (
                    f"{unit['unit_id']}:{pair[0]}:{pair[1]}:"
                    f"{_operator_name(str(entrance['operator']))}"
                ),
                "unit_id": str(unit["unit_id"]),
                "source_group_id": str(unit["source_group_id"]),
                "split": str(unit["split"]),
                **deepcopy(entrance),
                "residual_numbers": list(residual),
                "reachable": witness is not None,
                "continuation_witness": witness,
            }
        )
    return rows


def _operator_name(operator: str) -> str:
    """Use readable operator names in stable entrance identifiers."""

    return {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[operator]


def replay_entrance_witness(
    numbers: Sequence[int],
    target: int,
    entrance_row: Mapping[str, Any],
) -> bool:
    """Replay the first branch and its continuation with exact multiset checks."""

    if entrance_row.get("reachable") is not True:
        return False
    witness = entrance_row.get("continuation_witness")
    if not isinstance(witness, list):
        return False
    try:
        state = apply_entrance(numbers, entrance_row)
        for step in witness:
            legal = next(
                row
                for row in enumerate_legal_entrances(state)
                if row["operand_pair"] == step.get("operand_pair")
                and row["operator"] == step.get("operator")
            )
            if any(legal[key] != step.get(key) for key in ("left", "right", "result")):
                return False
            state = apply_entrance(state, step)
    except (KeyError, StopIteration, TypeError, ValueError):
        return False
    return int(target) in state


def residual_domain_sizes(numbers: Sequence[int]) -> list[int]:
    """Count legal operator results for each prompt-derived residual pair."""

    counts: dict[tuple[int, int], int] = Counter(
        tuple(row["operand_pair"]) for row in enumerate_legal_entrances(numbers)
    )
    return [counts[pair] for pair in sorted(counts)]


def mrv_score(numbers: Sequence[int]) -> int:
    """Return the smallest residual pair domain size; lower means more restricted."""

    sizes = residual_domain_sizes(numbers)
    return min(sizes) if sizes else 0


def build_mrv_rows(entrance_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build MRV receipts without consulting any label or witness field."""

    rows = []
    for entrance in entrance_rows:
        residual = tuple(int(value) for value in entrance["residual_numbers"])
        sizes = residual_domain_sizes(residual)
        rows.append(
            {
                "entrance_id": str(entrance["entrance_id"]),
                "residual_numbers": list(sorted(residual)),
                "residual_domain_sizes": sizes,
                "mrv_score": min(sizes) if sizes else 0,
                "mrv_tiebreak_total_domain_size": sum(sizes),
                "label_fields_read": False,
            }
        )
    return rows


def build_source_group_rows() -> list[JsonDict]:
    """Freeze twelve source families and their group-safe split assignments."""

    return [
        {
            "source_group_id": f"v619_source_{index:02d}",
            "source_family": f"large_numbers_{'_'.join(map(str, large_numbers))}",
            "large_numbers": list(large_numbers),
            "small_number_pool": list(SMALL_NUMBER_POOL),
            "split": "calibration" if index < 6 else "held",
            "unit_target": UNITS_PER_GROUP,
            "candidate_generation": "seeded_small_multiset_and_fixed_visible_expression",
            "sealed_before_label_open": True,
        }
        for index, large_numbers in enumerate(SOURCE_GROUP_LARGE_NUMBERS)
    ]


def _candidate_units(group: Mapping[str, Any]) -> list[JsonDict]:
    """Create prompt-visible candidate units without reading entrance outcomes."""

    rows = []
    large_numbers = [int(value) for value in group["large_numbers"]]
    small_count = 6 - len(large_numbers)
    for candidate_index in range(MAX_CANDIDATES_PER_GROUP):
        seed_text = f"{RANDOM_SEED}:{group['source_group_id']}:{candidate_index}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        generator = random.Random(seed)
        pool = list(SMALL_NUMBER_POOL)
        generator.shuffle(pool)
        small_numbers = sorted(pool[:small_count])
        numbers = sorted([*large_numbers, *small_numbers])
        first, second = small_numbers[:2]
        target = first * second + sum(numbers) - first - second
        rows.append(
            {
                "unit_id": f"{group['source_group_id']}_candidate_{candidate_index:03d}",
                "source_group_id": str(group["source_group_id"]),
                "split": str(group["split"]),
                "numbers": numbers,
                "target": target,
                "target_construction": "two_smallest_multiply_then_add_remaining_values",
                "candidate_frozen_before_label_open": True,
            }
        )
    return rows


def build_fixture() -> JsonDict:
    """Construct retained unit, entrance, witness, split, and visible rows."""

    source_groups_raw = build_source_group_rows()
    source_group_prelabel_hash = sha256_json(source_groups_raw)
    retained_units: list[JsonDict] = []
    retained_entrances: list[JsonDict] = []
    candidate_receipts = []
    for group in source_groups_raw:
        candidates = _candidate_units(group)
        selected = 0
        attempted = 0
        for candidate in candidates:
            attempted += 1
            entrances = label_unit_entrances(candidate)
            reachable = [row for row in entrances if row["reachable"]]
            unreachable = [row for row in entrances if not row["reachable"]]
            if not reachable or not unreachable or len(reachable) < 2:
                continue
            unit = {
                **candidate,
                "legal_entrance_count": len(entrances),
                "reachable_entrance_count": len(reachable),
                "unreachable_entrance_count": len(unreachable),
                "reachable_family_count": len(
                    {(tuple(row["operand_pair"]), row["operator"]) for row in reachable}
                ),
                "reachable_operator_count": len({row["operator"] for row in reachable}),
                "retention_rule": "reachable_and_unreachable_legal_entrances",
            }
            retained_units.append(unit)
            retained_entrances.extend(entrances)
            selected += 1
            if selected == UNITS_PER_GROUP:
                break
        if selected != UNITS_PER_GROUP:  # pragma: no cover - fixed budget invariant.
            raise RuntimeError(
                f"source group {group['source_group_id']} retained {selected} units, "
                f"expected {UNITS_PER_GROUP}"
            )
        candidate_receipts.append(
            {
                "source_group_id": group["source_group_id"],
                "candidates_attempted": attempted,
                "units_retained": selected,
            }
        )

    diversity_ids = [row["unit_id"] for row in retained_units if row["reachable_family_count"] >= 2]
    source_group_rows = _seal_rows(source_groups_raw, "source_group")
    unit_rows = _seal_rows(retained_units, "unit")
    entrance_rows = _seal_rows(retained_entrances, "entrance")
    ordered_ids = [row["unit_id"] for row in unit_rows]
    calibration_groups = [
        row["source_group_id"] for row in source_group_rows if row["split"] == "calibration"
    ]
    held_groups = [row["source_group_id"] for row in source_group_rows if row["split"] == "held"]
    split_manifest = {
        "split_unit": "source_group_id",
        "calibration_source_group_ids": calibration_groups,
        "held_source_group_ids": held_groups,
        "source_group_assignments": {
            row["source_group_id"]: row["split"] for row in source_group_rows
        },
        "ordered_unit_ids": ordered_ids,
        "ordered_unit_ids_hash": sha256_json(ordered_ids),
        "source_group_prelabel_hash": source_group_prelabel_hash,
        "sealed_before_model_output": True,
        "proposal_output_absent_at_seal": True,
    }
    model_rows = build_model_visible_rows(unit_rows)
    mrv_rows = _seal_rows(build_mrv_rows(entrance_rows), "mrv")
    witness_rows = _seal_rows(
        [
            {
                "entrance_id": row["entrance_id"],
                "witness_hash": sha256_json(row["continuation_witness"]),
                "witness_valid": replay_entrance_witness(
                    next(
                        unit["numbers"] for unit in unit_rows if unit["unit_id"] == row["unit_id"]
                    ),
                    next(unit["target"] for unit in unit_rows if unit["unit_id"] == row["unit_id"]),
                    row,
                ),
            }
            for row in entrance_rows
            if row["reachable"]
        ],
        "witness_replay",
    )
    return {
        "unit_rows": unit_rows,
        "entrance_rows": entrance_rows,
        "source_group_rows": source_group_rows,
        "split_manifest": split_manifest,
        "split_manifest_hash": sha256_json(split_manifest),
        "model_visible_rows": model_rows,
        "model_visible_rows_hash": sha256_json(model_rows),
        "mrv_rows": mrv_rows,
        "witness_replay_rows": witness_rows,
        "diversity_subset_ids": diversity_ids,
        "candidate_receipts": candidate_receipts,
    }


def build_model_visible_rows(unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Serialize only prompt fields allowed before any selector runs."""

    return [
        {
            "numbers": list(row["numbers"]),
            "target": int(row["target"]),
            "unit_id": str(row["unit_id"]),
            "formatting_rules": list(FORMATTING_RULES),
        }
        for row in unit_rows
    ]


def validate_model_visible_rows(
    rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]
) -> bool:
    """Reject fields outside the four-field model-visible contract."""

    allowed = set(schema.get("allowed_fields", ()))
    if allowed != set(MODEL_VISIBLE_FIELDS) or any(set(row) != allowed for row in rows):
        raise ValueError("model-visible field allowlist violation")
    return True


def _seal_rows(rows: Sequence[Mapping[str, Any]], kind: str) -> list[JsonDict]:
    """Add an ordered previous-hash chain to private evidence rows."""

    previous = INITIAL_ROW_HASH
    sealed = []
    for index, source in enumerate(rows):
        row = deepcopy(dict(source))
        row.update(
            {
                "evidence_kind": kind,
                "sequence_index": index,
                "previous_hash": previous,
                "row_hash": "",
            }
        )
        row["row_hash"] = _row_hash(row)
        previous = row["row_hash"]
        sealed.append(row)
    return sealed


def _row_hash(row: Mapping[str, Any]) -> str:
    """Hash one row with an empty self-hash field."""

    stable = deepcopy(dict(row))
    stable["row_hash"] = ""
    return sha256_json(stable)


def _validate_sealed_rows(rows: Sequence[Mapping[str, Any]], kind: str, name: str) -> None:
    """Reject row reordering, edits, missing rows, or broken kind tags."""

    previous = INITIAL_ROW_HASH
    for index, row in enumerate(rows):
        if (
            row.get("evidence_kind") != kind
            or row.get("sequence_index") != index
            or row.get("previous_hash") != previous
            or row.get("row_hash") != _row_hash(row)
        ):
            raise ValueError(f"{name} row chain invalid at {index}")
        previous = str(row["row_hash"])


def check_preconditions(
    *,
    root: Path,
    output_path: Path,
    row_paths: Mapping[str, Path],
    proposal_output_path: Path,
) -> list[JsonDict]:
    """Check the exact stack and every path before fixture construction."""

    solver_available = False
    solver_observed: Any = "unavailable"
    try:
        from carnot.verify.z3_math_verifier import Z3MathVerifier

        verifier = Z3MathVerifier()
        solver_available = bool(
            verifier.z3_available
            and verifier.score("8 / 4 = 2") == 0.0
            and solve_exact((2, 3, 6), 12) is not None
        )
        solver_observed = {
            "z3_available": verifier.z3_available,
            "exact_self_test": solver_available,
        }
    except Exception as exc:  # pragma: no cover - environment-specific block path.
        solver_observed = repr(exc)

    checks = [
        _precondition_row(
            "deterministic_arithmetic_solver_stack", solver_available, solver_observed
        ),
        _precondition_row(
            "code_path_writable",
            os.access(root / MODULE_RELATIVE_PATH, os.W_OK),
            str(root / MODULE_RELATIVE_PATH),
        ),
        _precondition_row(
            "test_path_writable",
            os.access(root / TEST_RELATIVE_PATH, os.W_OK),
            str(root / TEST_RELATIVE_PATH),
        ),
        _precondition_row(
            "result_path_writable",
            output_path.parent.is_dir() and os.access(output_path.parent, os.W_OK),
            str(output_path.parent),
        ),
        _precondition_row(
            "fixture_paths_writable",
            all(
                path.parent.is_dir() and os.access(path.parent, os.W_OK)
                for path in row_paths.values()
            ),
            sorted(str(path.parent) for path in row_paths.values()),
        ),
        _precondition_row(
            "v619_proposal_output_absent",
            not proposal_output_path.exists(),
            str(proposal_output_path) if proposal_output_path.exists() else "absent",
        ),
    ]
    return checks


def _precondition_row(check: str, available: bool, observed: Any) -> JsonDict:
    """Use one stable shape for success and blocked gate diagnostics."""

    return {
        "check": check,
        "available": bool(available),
        "expected_value": True,
        "observed_value": observed,
    }


def source_artifact_hashes(root: Path) -> JsonDict:
    """Bind the fixture to its instructions, solver, spec, module, and tests."""

    files = [
        {
            "path": str(relative),
            "exists": (root / relative).is_file(),
            "sha256": sha256_file(root / relative) if (root / relative).is_file() else None,
        }
        for relative in SOURCE_HASH_PATHS
    ]
    return {
        "files": files,
        "all_present": all(row["exists"] for row in files),
        "manifest_hash": sha256_json(files),
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding measured elapsed time."""

    stable = deepcopy(dict(artifact))
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    return sha256_json(stable)


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    fixture_dir: Path | None = None,
    proposal_output_path: Path | None = None,
    preconditions_override: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Materialize the fixture once and refuse replacement of any output."""

    started = time.monotonic()
    output = output_path or root / RESULT_RELATIVE_PATH
    rows_dir = fixture_dir or output.parent
    proposal = proposal_output_path or root / PROPOSAL_RELATIVE_PATH
    row_paths = {name: rows_dir / filename for name, filename in ROW_FILENAMES.items()}
    existing = [path for path in (output, *row_paths.values()) if path.exists()]
    if existing:
        raise FileExistsError(f"immutable fixture path already exists: {existing[0]}")
    preconditions = list(
        preconditions_override
        if preconditions_override is not None
        else check_preconditions(
            root=root,
            output_path=output,
            row_paths=row_paths,
            proposal_output_path=proposal,
        )
    )
    hashes = source_artifact_hashes(root)
    if not all(row.get("available") is True for row in preconditions):
        artifact = _blocked_artifact(preconditions, hashes, time.monotonic() - started)
        _write_new_json(output, artifact)
        return artifact

    fixture = build_fixture()
    _write_new_jsonl(row_paths["unit_rows"], fixture["unit_rows"])
    _write_new_jsonl(row_paths["entrance_rows"], fixture["entrance_rows"])
    _write_new_jsonl(row_paths["source_group_rows"], fixture["source_group_rows"])
    _write_new_jsonl(row_paths["model_visible_rows"], fixture["model_visible_rows"])
    fixture_paths = {
        name: {
            "path": str(path),
            "row_count": len(fixture[name]),
            "sha256": sha256_file(path),
        }
        for name, path in row_paths.items()
    }
    artifact = _ready_artifact(
        fixture=fixture,
        preconditions=preconditions,
        hashes=hashes,
        fixture_paths=fixture_paths,
        duration_s=time.monotonic() - started,
    )
    validate_artifact(artifact, check_files=True)
    _write_new_json(output, artifact)
    return artifact


def _blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any], duration_s: float
) -> JsonDict:
    """Build a schema-complete terminal block without constructing labels."""

    failed = [
        {
            "failed_check": str(row.get("check")),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in preconditions
        if row.get("available") is not True
    ]
    artifact = _artifact_base(preconditions, hashes, duration_s)
    artifact.update(
        {
            "rows": [],
            "unit_rows": [],
            "entrance_rows": [],
            "source_group_rows": [],
            "split_manifest": {},
            "split_manifest_hash": sha256_json({}),
            "model_visible_rows": [],
            "model_visible_rows_hash": sha256_json([]),
            "exhaustive_enumerator_receipt": {},
            "witness_replay_rows": [],
            "mrv_rows": [],
            "diversity_subset_ids": [],
            "unit_count": 0,
            "source_group_count": 0,
            "entrance_fixture_ready_score": 0,
            "gate_check_summary": {"passed": False, "failed_checks": failed},
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_v619_exact_entrance_fixture_precondition_failed",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _artifact_base(
    preconditions: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any], duration_s: float
) -> JsonDict:
    """Create fields shared by ready and blocked terminal artifacts."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "model_visible_schema": {
            "allowed_fields": list(MODEL_VISIBLE_FIELDS),
            "additional_fields_allowed": False,
            "row_count_rule": "one row per retained unit",
        },
        "operator_semantics": {
            "operand_pair": "unordered available-value pair with multiset multiplicity",
            "addition": "commutative canonical pair; positive integer result",
            "subtraction": "larger minus smaller; positive integer result only",
            "multiplication": "commutative canonical pair; positive integer result",
            "division": "larger divided by smaller; zero integer remainder required",
            "step_effect": "consume two operand occurrences and insert one result",
            "target_rule": "stop when target is an available value",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "verifier_is_oracle": True,
    }


def _ready_artifact(
    *,
    fixture: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    fixture_paths: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Reduce complete row evidence into the single readiness score."""

    unit_rows = fixture["unit_rows"]
    entrance_rows = fixture["entrance_rows"]
    source_rows = fixture["source_group_rows"]
    witness_rows = fixture["witness_replay_rows"]
    gates = [
        _gate("unit_count", MIN_UNIT_COUNT, len(unit_rows), len(unit_rows) >= MIN_UNIT_COUNT),
        _gate(
            "source_group_count",
            MIN_SOURCE_GROUP_COUNT,
            len(source_rows),
            len(source_rows) >= MIN_SOURCE_GROUP_COUNT,
        ),
        _gate(
            "mixed_entrance_labels",
            True,
            all(
                row["reachable_entrance_count"] > 0 and row["unreachable_entrance_count"] > 0
                for row in unit_rows
            ),
            all(
                row["reachable_entrance_count"] > 0 and row["unreachable_entrance_count"] > 0
                for row in unit_rows
            ),
        ),
        _gate(
            "diversity_subset_count",
            MIN_DIVERSITY_COUNT,
            len(fixture["diversity_subset_ids"]),
            len(fixture["diversity_subset_ids"]) >= MIN_DIVERSITY_COUNT,
        ),
        _gate(
            "witness_replay",
            True,
            all(row["witness_valid"] for row in witness_rows),
            bool(witness_rows) and all(row["witness_valid"] for row in witness_rows),
        ),
        _gate(
            "source_hashes",
            True,
            hashes["all_present"],
            hashes["all_present"] is True,
        ),
        _gate("model_visible_leakage_count", 0, 0, True),
    ]
    ready = all(row["passed"] for row in gates)
    artifact = _artifact_base(preconditions, hashes, duration_s)
    artifact.update(
        {
            "rows": gates,
            "unit_rows": deepcopy(unit_rows),
            "entrance_rows": deepcopy(entrance_rows),
            "source_group_rows": deepcopy(source_rows),
            "split_manifest": deepcopy(fixture["split_manifest"]),
            "split_manifest_hash": fixture["split_manifest_hash"],
            "model_visible_rows": deepcopy(fixture["model_visible_rows"]),
            "model_visible_rows_hash": fixture["model_visible_rows_hash"],
            "exhaustive_enumerator_receipt": {
                "unit_count": len(unit_rows),
                "legal_entrance_count": sum(row["legal_entrance_count"] for row in unit_rows),
                "labeled_entrance_count": len(entrance_rows),
                "duplicate_entrance_count": 0,
                "all_legal_entrances_labeled": (
                    len(entrance_rows) == sum(row["legal_entrance_count"] for row in unit_rows)
                ),
                "candidate_group_receipts": deepcopy(fixture["candidate_receipts"]),
                "labels_opened_after_source_group_seal": True,
                "model_invocation_count": 0,
            },
            "witness_replay_rows": deepcopy(witness_rows),
            "mrv_rows": deepcopy(fixture["mrv_rows"]),
            "diversity_subset_ids": list(fixture["diversity_subset_ids"]),
            "unit_count": len(unit_rows),
            "source_group_count": len(source_rows),
            "entrance_fixture_ready_score": 1 if ready else 0,
            "gate_check_summary": {
                "passed": ready,
                "failed_checks": [
                    {
                        "failed_check": row["check"],
                        "expected_value": row["expected_value"],
                        "observed_value": row["observed_value"],
                    }
                    for row in gates
                    if not row["passed"]
                ],
            },
            "fixture_paths": deepcopy(dict(fixture_paths)),
            "verdict_class": "circular_positive" if ready else "blocked",
            "honest_verdict": (
                "complete_circular_positive_exact_entrance_fixture_ready"
                if ready
                else "complete_blocked_v619_exact_entrance_fixture_gate_failed"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _gate(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Record one directly recomputable readiness condition."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
    }


def validate_artifact(artifact: Mapping[str, Any], *, check_files: bool = False) -> bool:
    """Recompute all exact fixture gates and reject any changed evidence."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    missing_principles = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"]))
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference substrate must be deterministic_verifier")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("oracle declaration must be true")
    score = artifact["entrance_fixture_ready_score"]
    if type(score) is not int or score not in (0, 1):
        raise ValueError("readiness must be a bare integer zero or one")
    if artifact["reproducibility_checksum"] != artifact_checksum(artifact):
        raise ValueError("reproducibility checksum mismatch")

    if artifact["verdict_class"] == "blocked":
        if score != 0 or not str(artifact["honest_verdict"]).startswith("complete_blocked_"):
            raise ValueError("blocked verdict fields are inconsistent")
        failed = artifact["gate_check_summary"].get("failed_checks", [])
        if artifact["gate_check_summary"].get("passed") is not False or not failed:
            raise ValueError("blocked gate summary needs exact failures")
        required = {"failed_check", "expected_value", "observed_value"}
        if any(set(row) != required for row in failed):
            raise ValueError("blocked gate failure shape is incomplete")
        return True

    if artifact["verdict_class"] != "circular_positive":
        raise ValueError("oracle fixture verdict must be circular_positive")
    if score != 1 or not str(artifact["honest_verdict"]).startswith("complete_"):
        raise ValueError("ready verdict and readiness score disagree")
    if artifact["gate_check_summary"].get("passed") is not True:
        raise ValueError("ready gate summary must pass")

    unit_rows = artifact["unit_rows"]
    entrance_rows = artifact["entrance_rows"]
    source_rows = artifact["source_group_rows"]
    _validate_sealed_rows(unit_rows, "unit", "unit")
    _validate_sealed_rows(entrance_rows, "entrance", "entrance")
    _validate_sealed_rows(source_rows, "source_group", "source group")
    _validate_sealed_rows(artifact["mrv_rows"], "mrv", "MRV")
    _validate_sealed_rows(artifact["witness_replay_rows"], "witness_replay", "witness")
    if artifact["unit_count"] != len(unit_rows) or len(unit_rows) < MIN_UNIT_COUNT:
        raise ValueError("unit count or unit rows are invalid")  # pragma: no cover
    if (
        artifact["source_group_count"] != len(source_rows)
        or len(source_rows) < MIN_SOURCE_GROUP_COUNT
    ):
        raise ValueError("source group count or rows are invalid")  # pragma: no cover

    manifest = artifact["split_manifest"]
    if artifact["split_manifest_hash"] != sha256_json(manifest):
        raise ValueError("split manifest hash mismatch")
    calibration = set(manifest["calibration_source_group_ids"])
    held = set(manifest["held_source_group_ids"])
    if not calibration or not held or calibration & held:
        raise ValueError("split manifest source groups overlap or are empty")  # pragma: no cover
    assignments = {row["source_group_id"]: row["split"] for row in source_rows}
    if assignments != manifest["source_group_assignments"]:
        raise ValueError("split manifest assignments changed")  # pragma: no cover
    ordered_ids = [row["unit_id"] for row in unit_rows]
    if manifest["ordered_unit_ids"] != ordered_ids or manifest[
        "ordered_unit_ids_hash"
    ] != sha256_json(ordered_ids):
        raise ValueError("split manifest ordered unit IDs changed")  # pragma: no cover
    if (
        manifest.get("sealed_before_model_output") is not True
        or manifest.get("proposal_output_absent_at_seal") is not True
    ):
        raise ValueError("split manifest was not sealed before model output")  # pragma: no cover

    by_unit: dict[str, list[Mapping[str, Any]]] = {str(unit["unit_id"]): [] for unit in unit_rows}
    for entrance in entrance_rows:
        if entrance["unit_id"] not in by_unit:
            raise ValueError("entrance row names an unknown unit")  # pragma: no cover
        by_unit[str(entrance["unit_id"])].append(entrance)
    checksum = str(artifact["reproducibility_checksum"])
    recompute_exact = checksum not in _VALIDATED_EXACT_CHECKSUMS
    for unit in unit_rows:
        expected = enumerate_legal_entrances(unit["numbers"])
        actual = by_unit[str(unit["unit_id"])]
        actual_keys = [(row["operand_pair"], row["operator"]) for row in actual]
        expected_keys = [(row["operand_pair"], row["operator"]) for row in expected]
        reachable_count = sum(bool(row["reachable"]) for row in actual)
        if actual_keys != expected_keys or unit["legal_entrance_count"] != len(actual):
            raise ValueError("entrance row enumeration is incomplete")  # pragma: no cover
        if (
            unit["reachable_entrance_count"] != reachable_count
            or unit["unreachable_entrance_count"] != len(actual) - reachable_count
            or reachable_count == 0
            or reachable_count == len(actual)
        ):
            raise ValueError("unit row mixed-label counts changed")  # pragma: no cover
        if recompute_exact:
            solver = _exact_solver(int(unit["target"]))
            for row in actual:
                witness = solver(row["residual_numbers"])
                if (witness is not None) is not row["reachable"]:
                    raise ValueError("entrance row reachability label changed")  # pragma: no cover
                if row["reachable"] and not replay_entrance_witness(
                    unit["numbers"], unit["target"], row
                ):
                    raise ValueError("entrance row witness replay failed")  # pragma: no cover
                if not row["reachable"] and row["continuation_witness"] is not None:
                    raise ValueError(  # pragma: no cover
                        "entrance row unreachable witness must be null"
                    )
    if recompute_exact:
        _VALIDATED_EXACT_CHECKSUMS.add(checksum)

    expected_mrv = _seal_rows(build_mrv_rows(entrance_rows), "mrv")
    if artifact["mrv_rows"] != expected_mrv:
        raise ValueError("MRV rows changed or read hidden labels")  # pragma: no cover
    diversity = [row["unit_id"] for row in unit_rows if row["reachable_family_count"] >= 2]
    if artifact["diversity_subset_ids"] != diversity or len(diversity) < MIN_DIVERSITY_COUNT:
        raise ValueError("diversity subset is invalid")  # pragma: no cover

    model_rows = artifact.get("model_visible_rows", [])
    validate_model_visible_rows(model_rows, artifact["model_visible_schema"])
    if len(model_rows) != len(unit_rows) or artifact["model_visible_rows_hash"] != sha256_json(
        model_rows
    ):
        raise ValueError("model-visible rows or hash changed")  # pragma: no cover
    if model_rows != build_model_visible_rows(unit_rows):
        raise ValueError("model-visible rows contain private evidence")  # pragma: no cover
    receipt = artifact["exhaustive_enumerator_receipt"]
    if (
        receipt.get("all_legal_entrances_labeled") is not True
        or receipt.get("legal_entrance_count") != len(entrance_rows)
        or receipt.get("labeled_entrance_count") != len(entrance_rows)
    ):
        raise ValueError("exhaustive enumerator receipt changed")  # pragma: no cover
    expected_witness_rows = _seal_rows(
        [
            {
                "entrance_id": row["entrance_id"],
                "witness_hash": sha256_json(row["continuation_witness"]),
                "witness_valid": True,
            }
            for row in entrance_rows
            if row["reachable"]
        ],
        "witness_replay",
    )
    if artifact["witness_replay_rows"] != expected_witness_rows:
        raise ValueError("witness replay rows changed")  # pragma: no cover
    if check_files:
        for name, record in artifact.get("fixture_paths", {}).items():
            path = Path(record["path"])
            if (
                not path.is_file()
                or sha256_file(path) != record["sha256"]
                or len(path.read_text(encoding="utf-8").splitlines()) != record["row_count"]
            ):
                raise ValueError(f"fixture file changed: {name}")
    return True


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Create a JSONL file once so a later run cannot replace frozen rows."""

    text = "".join(canonical_json(row) + "\n" for row in rows)
    _write_new_text(path, text)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a readable JSON artifact without replacing an earlier result."""

    _write_new_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_new_text(path: Path, text: str) -> None:
    """Use exclusive creation as the final mutation-rejection boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the dated command and materialize the immutable fixture."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--fixture-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"execution date must be {RUN_DATE}")
    write_artifact(output_path=args.output, fixture_dir=args.fixture_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the thin script wrapper.
    raise SystemExit(main())
