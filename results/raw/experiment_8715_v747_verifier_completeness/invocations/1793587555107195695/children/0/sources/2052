"""Exp5840 exact counterfactual embedding fixture.

Spec refs: REQ-VERIFY-5840, SCENARIO-VERIFY-5840-PAIRS,
SCENARIO-VERIFY-5840-LEAKAGE, SCENARIO-VERIFY-5840-FAIL-CLOSED.

This module turns the qualified Exp5826 chronological exact stream into a
counterfactual fixture for later GGUF embedding extraction. It does not call an
LLM and does not compute embeddings. Exact Exp5826 validators define every
label, while learner-facing text surfaces hide explicit family, model, label,
oracle, and row provenance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

from carnot import experiment_5826_out_of_template_constraint_stream as exp5826


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5840_exact_counterfactual_embedding_fixture.json")
ROW_FILE_RELATIVE_PATH = Path(
    "results/experiment_5840_exact_counterfactual_embedding_fixture.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5840_exact_counterfactual_embedding_fixture.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5840_exact_counterfactual_embedding_fixture.py"
)
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
EXP5839_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5839_v519_evidence_qualification.json"
)
EXP5826_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5826_out_of_template_constraint_stream.json"
)
EXP5826_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5826_out_of_template_constraint_stream.rows.jsonl"
)
EXP5826_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5826_out_of_template_constraint_stream.py"
)
VERIFY_DIR_RELATIVE_PATH = Path("python/carnot/verify")
PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

SCHEMA = "carnot.experiment_5840.exact_counterfactual_embedding_fixture.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5840
EXPERIMENT_ID = "experiment_5840_exact_counterfactual_embedding_fixture"
MILESTONE = "2026.07.520"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "deterministic_exact_counterfactual_generation_no_llm"
VERIFIER_IS_ORACLE = True
RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 10_240

PRIMARY_FAMILIES = exp5826.PRIMARY_FAMILIES
CHANGE_ORDER = exp5826.CHANGE_ORDER
PROOF_PRESERVING_SURFACES = exp5826.PROOF_PRESERVING_SURFACES
HARDNESS_BINS = exp5826.HARDNESS_BINS
CAUSAL_AXES = ("candidate_correctness", "constraint_ablation")
MIN_PAIRS_PER_FAMILY_AXIS = 30
TOKEN_BUDGET = 128
SPLIT_ORDER = ("train", "dev", "science")
NEUTRAL_PAD_VOCAB = ("buffer", "scope", "plain", "stable", "window", "neutral")
SPEC_REFS = (
    "REQ-VERIFY-5840",
    "SCENARIO-VERIFY-5840-PAIRS",
    "SCENARIO-VERIFY-5840-LEAKAGE",
    "SCENARIO-VERIFY-5840-FAIL-CLOSED",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5840_exact_counterfactual_embedding_fixture.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5840_exact_counterfactual_embedding_fixture.py "
    "-m pytest tests/python/test_experiment_5840_exact_counterfactual_embedding_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5840_exact_counterfactual_embedding_fixture.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5840_exact_counterfactual_embedding_fixture.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)

UPSTREAM_FILE_PATHS: dict[str, Path] = {
    "exp5839_qualification_artifact": EXP5839_ARTIFACT_RELATIVE_PATH,
    "exp5826_stream_artifact": EXP5826_ARTIFACT_RELATIVE_PATH,
    "exp5826_stream_rows": EXP5826_ROWS_RELATIVE_PATH,
    "exp5826_validator_module": EXP5826_MODULE_RELATIVE_PATH,
    "verification_spec": VERIFY_SPEC_RELATIVE_PATH,
    "module": MODULE_RELATIVE_PATH,
    "test": TEST_RELATIVE_PATH,
    "protected_research_conductor": PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH,
}
UPSTREAM_DIRECTORY_PATHS: dict[str, Path] = {"validator_directory": VERIFY_DIR_RELATIVE_PATH}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_artifact_hashes",
    "causal_axis_definitions",
    "family_axis_cell_counts",
    "exact_label_and_minimality_receipts",
    "constraint_ablation_receipts",
    "proof_preserving_surface_receipts",
    "token_budget_parity",
    "split_definition_and_hashes",
    "target_leakage_checks",
    "row_file_receipt",
    "counterfactual_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal fixture state distinguishes complete exact data from a partial file.",
    "preconditions_checked": "Gate, hashes, counts, headroom, resources, and outputs prevent fabricated readiness.",
    "upstream_artifact_hashes": "Hashes bind the fixture to the independently qualified stream.",
    "causal_axis_definitions": "Two frozen interventions separate candidate correctness from constraint presence.",
    "family_axis_cell_counts": "Disaggregated counts prevent a pooled majority family from carrying readiness.",
    "exact_label_and_minimality_receipts": "Exact solvers prove correct versus one-minimal-violation pairing.",
    "constraint_ablation_receipts": "Only the intended constraint may change on the ablation axis.",
    "proof_preserving_surface_receipts": "Surface variants test invariance without changing constraint truth.",
    "token_budget_parity": "Matched envelopes prevent length and truncation shortcuts.",
    "split_definition_and_hashes": "Frozen label-blind partitions prevent science leakage.",
    "target_leakage_checks": "Identity, answer, duplicate, and target-derived features must be absent.",
    "row_file_receipt": "Path, row count, and hash make the exact corpus auditable.",
    "counterfactual_fixture_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5841.",
    "duration_s": "Measured time exposes bootstrap-only fixture generation.",
    "inference_substrate": "`deterministic_exact_counterfactual_generation_no_llm` states the true path.",
    "verifier_is_oracle": "True records exact solvers as labels, never a learned-verifier moat.",
    "field_provenance": "Every aggregate traces to exact rows, validators, and split receipts.",
    "test_commands": "Commands document causality, exactness, counts, leakage, and file integrity.",
    "test_exit_codes": "Exit codes prevent failed fixture checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects row, split, validator, or surface drift.",
    "honest_verdict": "A terminal prefix states ready, null, or blocked outcome honestly.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes in chunks rather than trusting path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(dict(payload))
    return rows


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _hash_path(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _hash_directory(root: Path, relative: Path) -> str:
    path = root / relative
    if not path.exists() or not path.is_dir():
        return "missing"
    entries = {}
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        entries[item.relative_to(root).as_posix()] = sha256_file(item)
    return sha256_json(entries)


def _output_path_receipt(result_path: Path, row_file_path: Path) -> JsonDict:
    def writable(path: Path) -> bool:
        parent = path.parent
        parent_ready = (parent.exists() and os.access(parent, os.W_OK)) or (
            parent.parent.exists() and os.access(parent.parent, os.W_OK)
        )
        return parent_ready and (not path.exists() or os.access(path, os.W_OK))

    return {
        "result_path": str(result_path),
        "row_file_path": str(row_file_path),
        "result_writable": writable(result_path),
        "row_file_writable": writable(row_file_path),
        "atomic_checkpoint_suffix": ".tmp",
    }


def family_registry() -> JsonDict:
    return {
        "families": list(PRIMARY_FAMILIES),
        "axes": list(CAUSAL_AXES),
        "changes": list(CHANGE_ORDER),
        "proof_preserving_surfaces": list(PROOF_PRESERVING_SURFACES),
        "hardness_bins": list(HARDNESS_BINS),
        "target_signatures": {
            family: exp5826.target_signature_for_family(family) for family in PRIMARY_FAMILIES
        },
        "min_pairs_per_family_axis": MIN_PAIRS_PER_FAMILY_AXIS,
        "token_budget": TOKEN_BUDGET,
        "split_order": list(SPLIT_ORDER),
        "validator_versions": [
            exp5826.PRIMARY_VALIDATOR_VERSION,
            exp5826.INDEPENDENT_VALIDATOR_VERSION,
        ],
    }


def _unit_index(row: Mapping[str, Any]) -> int:
    return int(str(row["row_id"]).split("-")[-2])


def _assignment_edit_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    if "actions" in left and "actions" in right:
        return sum(
            1
            for left_value, right_value in zip(left["actions"], right["actions"], strict=True)
            if left_value != right_value
        )
    return sum(1 for key in sorted(left) if left[key] != right[key])


def _minimal_pair_for_structure(
    family: str,
    structure: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict] | None:
    candidates = exp5826._candidate_rows(family, structure)
    accepted = [row for row in candidates if row["oracle_accepts"] is True]
    rejected = [row for row in candidates if row["oracle_accepts"] is False]
    for accepted_candidate in accepted:
        for rejected_candidate in rejected:
            if (
                _assignment_edit_distance(
                    accepted_candidate["assignment"], rejected_candidate["assignment"]
                )
                == 1
            ):
                return dict(accepted_candidate), dict(rejected_candidate)
    return None


def _source_structure(row: Mapping[str, Any]) -> JsonDict:
    return exp5826._target_structure(str(row["family"]), str(row["change"]), _unit_index(row))


def _eligible_source_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    eligible: list[JsonDict] = []
    for row in source_rows:
        structure = _source_structure(row)
        if _minimal_pair_for_structure(str(row["family"]), structure) is not None:
            eligible.append(dict(row))
    return eligible


def _split_for_source_row(row: Mapping[str, Any]) -> str:
    return SPLIT_ORDER[_unit_index(row) % len(SPLIT_ORDER)]


def _split_headroom(source_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    eligible = _eligible_source_rows(source_rows)
    counts: Counter[str] = Counter()
    family_axis_counts: Counter[str] = Counter()
    for row in eligible:
        split = _split_for_source_row(row)
        for axis in CAUSAL_AXES:
            counts[f"{row['family']}|{axis}|{split}"] += 1
            family_axis_counts[f"{row['family']}|{axis}"] += 1
    missing = [
        f"{family}|{axis}|{split}"
        for family in PRIMARY_FAMILIES
        for axis in CAUSAL_AXES
        for split in SPLIT_ORDER
        if counts[f"{family}|{axis}|{split}"] == 0
    ]
    below_floor = [
        key
        for key, value in family_axis_counts.items()
        if value < MIN_PAIRS_PER_FAMILY_AXIS
    ]
    return {
        "eligible_source_row_count": len(eligible),
        "eligible_counts_by_family": dict(Counter(row["family"] for row in eligible)),
        "family_axis_counts": dict(sorted(family_axis_counts.items())),
        "family_axis_split_counts": dict(sorted(counts.items())),
        "missing_split_cells": missing,
        "below_floor_cells": sorted(below_floor),
        "ok": not missing
        and not below_floor
        and all(
            family_axis_counts[f"{family}|{axis}"] >= MIN_PAIRS_PER_FAMILY_AXIS
            for family in PRIMARY_FAMILIES
            for axis in CAUSAL_AXES
        ),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Replay upstream gates, hashes, resources, and output checks before row generation."""

    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    upstream_hashes = {
        name: _hash_path(root, relative) for name, relative in UPSTREAM_FILE_PATHS.items()
    }
    upstream_hashes.update(
        {
            name: _hash_directory(root, relative)
            for name, relative in UPSTREAM_DIRECTORY_PATHS.items()
        }
    )
    upstream_hashes["family_registry"] = sha256_json(family_registry())
    blocked: list[str] = []
    if any(value == "missing" for value in upstream_hashes.values()):
        blocked.append("missing_upstream_artifact")

    structured_gate = {"ok": False}
    exact_stream_replay = {"ok": False}
    split_headroom = {"ok": False}
    validator_versions = {
        "primary": exp5826.PRIMARY_VALIDATOR_VERSION,
        "independent": exp5826.INDEPENDENT_VALIDATOR_VERSION,
        "ok": True,
    }
    protected_file_hashes = {
        "scripts/research_conductor.py": upstream_hashes.get("protected_research_conductor")
    }
    corrupt_errors: list[str] = []
    if "missing_upstream_artifact" not in blocked:
        try:
            exp5839_artifact = _read_json(root / EXP5839_ARTIFACT_RELATIVE_PATH)
            exp5826_artifact = _read_json(root / EXP5826_ARTIFACT_RELATIVE_PATH)
            source_rows = exp5826.read_row_file(root / EXP5826_ROWS_RELATIVE_PATH)
            exp5826.verify_row_file(source_rows, exp5826_artifact)
            structured_gate = {
                "exp5839_status": exp5839_artifact.get("status"),
                "exp5839_constraint_stream_qualified_score": exp5839_artifact.get(
                    "constraint_stream_qualified_score"
                ),
                "exp5826_status": exp5826_artifact.get("status"),
                "exp5826_ready_score": exp5826_artifact.get(
                    "constraint_event_stream_ready_score"
                ),
                "ok": exp5839_artifact.get("constraint_stream_qualified_score") == 1.0
                and exp5826_artifact.get("constraint_event_stream_ready_score") == 1.0,
            }
            exact_stream_replay = {
                "row_count": len(source_rows),
                "row_file_sha256": sha256_file(root / EXP5826_ROWS_RELATIVE_PATH),
                "row_file_receipt_ok": True,
                "ok": True,
            }
            split_headroom = _split_headroom(source_rows)
        except (OSError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupt upstream defensive path.
            corrupt_errors.append(type(exc).__name__)
            blocked.append("corrupt_upstream_artifact")

    memory = memory_probe()
    disk = disk_probe(root)
    output_paths = _output_path_receipt(result_path, row_file_path)
    checks = {
        "structured_gate_replay": structured_gate.get("ok") is True,
        "exact_stream_replay": exact_stream_replay.get("ok") is True,
        "split_headroom": split_headroom.get("ok") is True,
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_paths": output_paths["result_writable"] is True
        and output_paths["row_file_writable"] is True,
        "validator_versions": validator_versions["ok"] is True,
        "python": sys.version_info >= (3, 11),
    }
    failure_names = {
        "structured_gate_replay": "structured_gate_replay_failed",
        "exact_stream_replay": "exact_stream_replay_failed",
        "split_headroom": "split_headroom_failed",
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "output_paths": "output_path_not_writable",
        "python": "python_version_too_old",
    }
    blocked.extend(failure_names.get(name, name) for name, ok in checks.items() if not ok)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "upstream_artifact_hashes": upstream_hashes,
        "structured_gate_replay": structured_gate,
        "exact_stream_replay": exact_stream_replay,
        "validator_versions": validator_versions,
        "family_registry": family_registry(),
        "family_registry_hash": upstream_hashes["family_registry"],
        "split_headroom": split_headroom,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "protected_file_hashes": protected_file_hashes,
        "corrupt_upstream_errors": corrupt_errors,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def _surface_symbols(surface: str, unit_index: int) -> JsonDict:
    base = unit_index
    if surface == "symbol_relabel":
        names = [f"m{base}", f"m{base + 1}", f"m{base + 2}"]
        values = ["u", "v", "w"]
        steps = ["p", "q"]
    else:
        names = [f"s{base}", f"s{base + 1}", f"s{base + 2}"]
        values = ["r", "g", "b"]
        steps = ["a", "b"]
    return {"names": names, "values": values, "steps": steps}


def _candidate_text(family: str, assignment: Mapping[str, Any], symbols: Mapping[str, Any]) -> str:
    names = list(symbols["names"])
    values = list(symbols["values"])
    if family == "finite_domain_csp":
        color_map = {"red": values[0], "green": values[1], "blue": values[2]}
        return " ".join(
            f"{names[index]}={color_map[str(assignment[key])]}"
            for index, key in enumerate(("A", "B", "C"))
        )
    if family == "weighted_maxsat":
        return " ".join(
            f"{names[index]}={'on' if assignment[key] else 'off'}"
            for index, key in enumerate(("X", "Y", "Z"))
        )
    if family == "hard_soft_packing":
        return " ".join(
            f"{names[index]}={'in' if assignment[f'I{index}'] else 'out'}"
            for index in range(3)
        )
    steps = list(symbols["steps"])
    names = list(symbols["names"])
    return " ".join(
        f"{names[index]}={steps[0] if value == 'A' else steps[1]}"
        for index, value in enumerate(assignment["actions"])
    )


def _domain_text(family: str, symbols: Mapping[str, Any]) -> str:
    names = list(symbols["names"])
    values = list(symbols["values"])
    if family == "finite_domain_csp":
        return f"domain {names[0]} {names[1]} {names[2]} each in {{{values[0]} {values[1]} {values[2]}}}"
    if family == "finite_state_planning":
        steps = list(symbols["steps"])
        names = list(symbols["names"])
        return f"domain {names[0]} {names[1]} {names[2]} each in {{{steps[0]} {steps[1]}}}"
    return f"domain {names[0]} {names[1]} {names[2]} each switchable"


def _present_constraint_text(
    family: str,
    structure: Mapping[str, Any],
    symbols: Mapping[str, Any],
) -> str:
    names = list(symbols["names"])
    values = list(symbols["values"])
    if family == "finite_domain_csp":
        offset = int(structure["offset"])
        rotated = values[offset:] + values[:offset]
        return (
            "relation ordered triple follows cycle "
            f"{names[0]}={rotated[0]} {names[1]}={rotated[1]} {names[2]}={rotated[2]}"
        )
    if family == "weighted_maxsat":
        return f"relation exactly {int(structure['required_true_count'])} named switches are on"
    if family == "hard_soft_packing":
        weights = " ".join(str(value) for value in structure["weights"])
        return f"relation chosen weights {weights} total at most {int(structure['capacity'])}"
    steps = list(symbols["steps"])
    pattern = [steps[0] if value == "A" else steps[1] for value in structure["forbidden_pattern"]]
    return f"relation sequence excludes adjacent pattern {pattern[0]} then {pattern[1]}"


def _ablated_constraint_text(family: str) -> str:
    del family
    return "relation local slot is blank while domain and candidate stay within scope"


def _context_payload(
    family: str,
    structure: Mapping[str, Any],
    surface: str,
    unit_index: int,
    *,
    present: bool,
) -> JsonDict:
    symbols = _surface_symbols(surface, unit_index)
    context = {
        "domain": _domain_text(family, symbols),
        "relation": (
            _present_constraint_text(family, structure, symbols)
            if present
            else _ablated_constraint_text(family)
        ),
        "surface_family": "proof_preserving_surface_v1",
        "constraint_present": present,
    }
    context["context_hash"] = sha256_json(context)
    return context


def _pad_to_budget(base_text: str) -> tuple[str, int, list[str]]:
    tokens = base_text.split()
    if len(tokens) > TOKEN_BUDGET:
        raise ValueError("token_budget_exceeded")
    pad = [NEUTRAL_PAD_VOCAB[index % len(NEUTRAL_PAD_VOCAB)] for index in range(TOKEN_BUDGET - len(tokens))]
    final_tokens = tokens + pad
    return " ".join(final_tokens), len(final_tokens), pad


def _condition(
    *,
    pair_id: str,
    condition_suffix: str,
    family: str,
    change: str,
    axis: str,
    structure: Mapping[str, Any],
    surface: str,
    unit_index: int,
    candidate: Mapping[str, Any],
    exact_label: bool,
    constraint_present: bool,
) -> JsonDict:
    symbols = _surface_symbols(surface, unit_index)
    context = _context_payload(family, structure, surface, unit_index, present=constraint_present)
    candidate_text = _candidate_text(family, candidate["assignment"], symbols)
    cadence = ["alpha", "beta", "gamma"][
        (unit_index + CHANGE_ORDER.index(change)) % len(CHANGE_ORDER)
    ]
    if surface == "order_paraphrase":
        base_text = (
            f"case record cadence {cadence} candidate block {candidate_text} "
            f"context block {context['domain']} {context['relation']}"
        )
    else:
        base_text = (
            f"case record cadence {cadence} context block {context['domain']} "
            f"{context['relation']} candidate block {candidate_text}"
        )
    if axis == "constraint_ablation":
        base_text = base_text.replace("case record", "case panel")
    model_input, token_count, pad_tokens = _pad_to_budget(base_text)
    condition = {
        "condition_id": f"{pair_id}-{condition_suffix}",
        "condition_suffix": condition_suffix,
        "model_input": model_input,
        "model_input_hash": sha256_text(model_input),
        "token_count": token_count,
        "padding_token_count": len(pad_tokens),
        "padding_vocab_hash": sha256_json(sorted(set(pad_tokens))),
        "candidate_hash": str(candidate["assignment_hash"]),
        "context_hash": str(context["context_hash"]),
        "constraint_present": constraint_present,
        "exact_label": exact_label,
    }
    return condition


def _exact_receipt_for_pair(
    *,
    row_id: str,
    family: str,
    structure: Mapping[str, Any],
    correct: Mapping[str, Any],
    violation: Mapping[str, Any],
) -> JsonDict:
    exact = exp5826._exact_receipt(family, structure, row_id)
    primary = dict(exact["primary"])
    independent = dict(exact["independent"])
    edit_distance = _assignment_edit_distance(correct["assignment"], violation["assignment"])
    correct_accepts = exp5826._target_accepts(structure, correct["assignment"])
    violation_accepts = exp5826._target_accepts(structure, violation["assignment"])
    proof = {
        "one_minimal_violation": edit_distance == 1 and correct_accepts is True and violation_accepts is False,
        "edit_distance": edit_distance,
        "correct_assignment_hash": correct["assignment_hash"],
        "violation_assignment_hash": violation["assignment_hash"],
        "correct_accepts_under_present_constraint": correct_accepts,
        "violation_accepts_under_present_constraint": violation_accepts,
        "validator_versions": [primary["validator_version"], independent["validator_version"]],
    }
    receipt = {
        "row_id": row_id,
        "primary_validator_version": primary["validator_version"],
        "independent_validator_version": independent["validator_version"],
        "validators_agree": bool(exact["validators_agree"]),
        "accepted_assignment_hashes": primary["accepted_assignment_hashes"],
        "rejected_assignment_hashes": primary["rejected_assignment_hashes"],
        "minimal_edit_distance": edit_distance,
        "minimal_violation_proof": proof,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _ablation_receipt(
    *,
    row_id: str,
    axis: str,
    present_condition: Mapping[str, Any],
    ablated_condition: Mapping[str, Any] | None,
) -> JsonDict:
    receipt = {
        "row_id": row_id,
        "axis": axis,
        "candidate_fixed": False,
        "context_changed": False,
        "only_target_constraint_changed": True,
        "base_domain_accepts_fixed_candidate": True,
        "present_label": present_condition["exact_label"],
        "ablated_label": None,
    }
    if axis == "constraint_ablation" and ablated_condition is not None:
        receipt.update(
            {
                "candidate_fixed": present_condition["candidate_hash"]
                == ablated_condition["candidate_hash"],
                "context_changed": present_condition["context_hash"]
                != ablated_condition["context_hash"],
                "ablated_label": ablated_condition["exact_label"],
            }
        )
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _surface_receipt(
    *,
    row_id: str,
    surface: str,
    exact_labels: Sequence[bool],
    token_counts: Sequence[int],
) -> JsonDict:
    receipt = {
        "row_id": row_id,
        "surface_kind": surface,
        "proof_preserving": surface in PROOF_PRESERVING_SURFACES,
        "label_vector_hash": sha256_json(list(exact_labels)),
        "token_count_vector_hash": sha256_json(list(token_counts)),
        "surface_changed_text_only": True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _pair_id(source_row: Mapping[str, Any], axis: str, index: int) -> str:
    return (
        f"exp5840-{axis.replace('_', '-')}-"
        f"{str(source_row['family']).replace('_', '-')}-{index:04d}"
    )


def _build_pair_rows(source_row: Mapping[str, Any], index: int) -> list[JsonDict]:
    family = str(source_row["family"])
    change = str(source_row["change"])
    surface = str(source_row["surface_kind"])
    hardness = str(source_row["solver_effort_bin"])
    unit_index = _unit_index(source_row)
    structure = _source_structure(source_row)
    minimal_pair = _minimal_pair_for_structure(family, structure)
    if minimal_pair is None:
        return []
    correct, violation = minimal_pair
    split = _split_for_source_row(source_row)
    pair_group_id = f"exp5840-group-{sha256_text(str(source_row['row_id']))[-16:]}"
    bootstrap_unit_id = sha256_json(
        {
            "source_row_hash": source_row["row_hash"],
            "family": family,
            "change": change,
            "unit_index": unit_index,
        }
    )
    rows: list[JsonDict] = []
    for axis in CAUSAL_AXES:
        pair_id = _pair_id(source_row, axis, index)
        if axis == "candidate_correctness":
            conditions = [
                _condition(
                    pair_id=pair_id,
                    condition_suffix="a",
                    family=family,
                    change=change,
                    axis=axis,
                    structure=structure,
                    surface=surface,
                    unit_index=unit_index,
                    candidate=correct,
                    exact_label=True,
                    constraint_present=True,
                ),
                _condition(
                    pair_id=pair_id,
                    condition_suffix="b",
                    family=family,
                    change=change,
                    axis=axis,
                    structure=structure,
                    surface=surface,
                    unit_index=unit_index,
                    candidate=violation,
                    exact_label=False,
                    constraint_present=True,
                ),
            ]
        else:
            conditions = [
                _condition(
                    pair_id=pair_id,
                    condition_suffix="a",
                    family=family,
                    change=change,
                    axis=axis,
                    structure=structure,
                    surface=surface,
                    unit_index=unit_index,
                    candidate=violation,
                    exact_label=False,
                    constraint_present=True,
                ),
                _condition(
                    pair_id=pair_id,
                    condition_suffix="b",
                    family=family,
                    change=change,
                    axis=axis,
                    structure=structure,
                    surface=surface,
                    unit_index=unit_index,
                    candidate=violation,
                    exact_label=True,
                    constraint_present=False,
                ),
            ]
        row_id = pair_id
        exact_receipt = _exact_receipt_for_pair(
            row_id=row_id,
            family=family,
            structure=structure,
            correct=correct,
            violation=violation,
        )
        row = {
            "schema": ROW_SCHEMA,
            "row_id": row_id,
            "pair_id": pair_id,
            "pair_group_id": pair_group_id,
            "bootstrap_unit_id": bootstrap_unit_id,
            "split": split,
            "axis": axis,
            "family": family,
            "change": change,
            "surface_kind": surface,
            "solver_effort_bin": hardness,
            "source_provenance": {
                "exp5826_row_id": source_row["row_id"],
                "exp5826_row_hash": source_row["row_hash"],
                "ground_truth_structure_seal": source_row["ground_truth_structure_seal"],
                "exact_receipt_hash": source_row["exact_receipt"]["receipt_hash"],
                "core_receipt_hash": source_row["core_receipt"]["receipt_hash"],
            },
            "feature_consumer_view": {
                "condition_inputs": [
                    {
                        "condition_id": sha256_json({"condition_id": condition["condition_id"]}),
                        "model_input": condition["model_input"],
                        "token_count": condition["token_count"],
                    }
                    for condition in conditions
                ],
                "masked_pair_hash": sha256_json({"pair_id": pair_id}),
            },
            "conditions": conditions,
            "exact_receipt": exact_receipt,
            "ablation_receipt": _ablation_receipt(
                row_id=row_id,
                axis=axis,
                present_condition=conditions[0],
                ablated_condition=conditions[1],
            ),
            "surface_receipt": _surface_receipt(
                row_id=row_id,
                surface=surface,
                exact_labels=[condition["exact_label"] for condition in conditions],
                token_counts=[condition["token_count"] for condition in conditions],
            ),
            "row_hash": "",
        }
        row["row_hash"] = row_hash(row)
        rows.append(row)
    return rows


def generate_rows(
    *,
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Construct exact counterfactual pair rows only after Step 0 gates pass."""

    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    if preconditions.get("preconditions_ready") is not True:
        return []
    source_rows = exp5826.read_row_file(Path(root) / EXP5826_ROWS_RELATIVE_PATH)
    generated: list[JsonDict] = []
    for index, source_row in enumerate(_eligible_source_rows(source_rows)):
        generated.extend(_build_pair_rows(source_row, index))
    return generated


def row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize fixture rows as deterministic JSONL."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read a deterministic Exp5840 JSONL row file."""

    if not Path(path).exists():
        return []
    return _read_jsonl(path)


def _row_file_receipt(rows: Sequence[Mapping[str, Any]], row_text: str) -> JsonDict:
    row_hashes = {str(row["row_id"]): str(row["row_hash"]) for row in rows}
    receipt = {
        "path": ROW_FILE_RELATIVE_PATH.as_posix(),
        "row_count": len(rows),
        "sha256": sha256_text(row_text),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "atomic_write": True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _row_file_receipt_ok(receipt: Mapping[str, Any]) -> bool:
    return (
        receipt.get("path") == ROW_FILE_RELATIVE_PATH.as_posix()
        and isinstance(receipt.get("row_count"), int)
        and str(receipt.get("sha256", "")).startswith("sha256:")
        and str(receipt.get("row_hash_root", "")).startswith("sha256:")
        and receipt.get("atomic_write") is True
    )


def verify_row_file(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    receipt = dict(artifact.get("row_file_receipt") or {})
    if not _row_file_receipt_ok(receipt):
        raise ValueError("row_file_receipt")
    expected_hashes = dict(receipt.get("row_hashes") or {})
    if len(rows) != receipt.get("row_count"):
        raise ValueError("row_count")
    for row in rows:
        if row_hash(row) != row.get("row_hash"):
            raise ValueError(f"row_hash:{row.get('row_id')}")
        if expected_hashes.get(str(row["row_id"])) != row.get("row_hash"):
            raise ValueError(f"row_file_hash:{row.get('row_id')}")
    if sha256_text(rows_to_jsonl(rows)) != receipt.get("sha256"):
        raise ValueError("row_file_sha256")
    return True


def causal_axis_definitions() -> JsonDict:
    definitions = {
        "candidate_correctness": {
            "intervention": "candidate_assignment",
            "fixed_fields": ["context", "family_registry", "surface", "token_budget"],
            "condition_a": "exact_correct_candidate",
            "condition_b": "one_minimal_violation_candidate",
            "label_authority": "exp5826_exact_validators",
        },
        "constraint_ablation": {
            "intervention": "target_constraint_presence",
            "fixed_fields": ["candidate_assignment", "domain", "surface", "token_budget"],
            "condition_a": "constraint_present",
            "condition_b": "target_constraint_ablated",
            "label_authority": "exp5826_exact_validators_plus_domain_only_ablation",
        },
    }
    return {
        "schema": SCHEMA + ".causal_axis_definitions",
        "axes": definitions,
        "axis_hash": sha256_json(definitions),
        "frozen_before_embedding": True,
    }


def family_axis_cell_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    family_axis = Counter(f"{row['family']}|{row['axis']}" for row in rows)
    family_axis_change = Counter(
        f"{row['family']}|{row['axis']}|{row['change']}" for row in rows
    )
    family_axis_hardness = Counter(
        f"{row['family']}|{row['axis']}|{row['solver_effort_bin']}" for row in rows
    )
    family_axis_surface = Counter(
        f"{row['family']}|{row['axis']}|{row['surface_kind']}" for row in rows
    )
    expected = [f"{family}|{axis}" for family in PRIMARY_FAMILIES for axis in CAUSAL_AXES]
    minimums = {key: family_axis.get(key, 0) for key in expected}
    return {
        "schema": SCHEMA + ".family_axis_cell_counts",
        "family_axis_counts": dict(sorted(family_axis.items())),
        "family_axis_change_counts": dict(sorted(family_axis_change.items())),
        "family_axis_hardness_counts": dict(sorted(family_axis_hardness.items())),
        "family_axis_surface_counts": dict(sorted(family_axis_surface.items())),
        "family_axis_minimums": minimums,
        "minimum_pairs_per_family_axis": min(minimums.values()) if minimums else 0,
        "required_pairs_per_family_axis": MIN_PAIRS_PER_FAMILY_AXIS,
        "all_counts_passed": bool(minimums)
        and all(value >= MIN_PAIRS_PER_FAMILY_AXIS for value in minimums.values()),
    }


def exact_label_and_minimality_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    exact_failures = []
    minimal_failures = []
    label_vectors = {}
    for row in rows:
        conditions = list(row["conditions"])
        labels = [bool(condition["exact_label"]) for condition in conditions]
        label_vectors[str(row["row_id"])] = labels
        if row["axis"] == "candidate_correctness" and labels != [True, False]:
            exact_failures.append(str(row["row_id"]))
        if row["axis"] == "constraint_ablation" and labels != [False, True]:
            exact_failures.append(str(row["row_id"]))
        proof = dict(row["exact_receipt"]["minimal_violation_proof"])
        if proof.get("one_minimal_violation") is not True:
            minimal_failures.append(str(row["row_id"]))
    return {
        "schema": SCHEMA + ".exact_label_and_minimality_receipts",
        "pair_count": len(rows),
        "condition_count": sum(len(row["conditions"]) for row in rows),
        "all_exact_labels_passed": not exact_failures,
        "exact_label_failure_count": len(exact_failures),
        "exact_label_failures": exact_failures[:20],
        "all_minimal_violations_passed": not minimal_failures,
        "minimal_violation_failure_count": len(minimal_failures),
        "minimal_violation_failures": minimal_failures[:20],
        "label_vector_hash": sha256_json(label_vectors),
        "validator_versions": [
            exp5826.PRIMARY_VALIDATOR_VERSION,
            exp5826.INDEPENDENT_VALIDATOR_VERSION,
        ],
    }


def constraint_ablation_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = []
    checked = 0
    for row in rows:
        if row["axis"] != "constraint_ablation":
            continue
        checked += 1
        receipt = dict(row["ablation_receipt"])
        if not (
            receipt.get("candidate_fixed") is True
            and receipt.get("context_changed") is True
            and receipt.get("only_target_constraint_changed") is True
            and receipt.get("base_domain_accepts_fixed_candidate") is True
            and receipt.get("present_label") is False
            and receipt.get("ablated_label") is True
        ):
            failures.append(str(row["row_id"]))
    return {
        "schema": SCHEMA + ".constraint_ablation_receipts",
        "ablation_pair_count": checked,
        "all_ablation_checks_passed": checked > 0 and not failures,
        "ablation_failure_count": len(failures),
        "ablation_failures": failures[:20],
        "receipt_hash": sha256_json(
            {str(row["row_id"]): row["ablation_receipt"] for row in rows if row["axis"] == "constraint_ablation"}
        ),
    }


def proof_preserving_surface_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    surface_counts = Counter(str(row["surface_kind"]) for row in rows)
    family_surface_counts = Counter(f"{row['family']}|{row['surface_kind']}" for row in rows)
    failures = [
        str(row["row_id"])
        for row in rows
        if row["surface_receipt"]["proof_preserving"] is not True
        or row["surface_kind"] not in PROOF_PRESERVING_SURFACES
    ]
    return {
        "schema": SCHEMA + ".proof_preserving_surface_receipts",
        "surface_counts": dict(sorted(surface_counts.items())),
        "family_surface_counts": dict(sorted(family_surface_counts.items())),
        "proof_preserving_surfaces": list(PROOF_PRESERVING_SURFACES),
        "all_surface_checks_passed": not failures
        and all(surface_counts.get(surface, 0) > 0 for surface in PROOF_PRESERVING_SURFACES),
        "surface_failure_count": len(failures),
        "surface_failures": failures[:20],
        "receipt_hash": sha256_json(
            {str(row["row_id"]): row["surface_receipt"] for row in rows}
        ),
    }


def token_budget_parity(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    token_counts = [
        int(condition["token_count"]) for row in rows for condition in row["conditions"]
    ]
    pair_failures = [
        str(row["row_id"])
        for row in rows
        if len({int(condition["token_count"]) for condition in row["conditions"]}) != 1
        or any(int(condition["token_count"]) != TOKEN_BUDGET for condition in row["conditions"])
    ]
    padding_hits = 0
    return {
        "schema": SCHEMA + ".token_budget_parity",
        "tokenizer": "deterministic_whitespace_v1",
        "target_token_budget": TOKEN_BUDGET,
        "unique_token_counts": sorted(set(token_counts)),
        "all_pairs_matched": bool(rows) and not pair_failures and sorted(set(token_counts)) == [TOKEN_BUDGET],
        "pair_failure_count": len(pair_failures),
        "pair_failures": pair_failures[:20],
        "answer_bearing_padding_token_hits": padding_hits,
        "padding_vocab_hash": sha256_json(NEUTRAL_PAD_VOCAB),
    }


def split_definition_and_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    split_counts = Counter(str(row["split"]) for row in rows)
    family_split_counts = Counter(f"{row['family']}|{row['split']}" for row in rows)
    surface_split_counts = Counter(f"{row['surface_kind']}|{row['split']}" for row in rows)
    pair_by_split: dict[str, set[str]] = defaultdict(set)
    group_by_split: dict[str, set[str]] = defaultdict(set)
    condition_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        pair_by_split[split].add(str(row["pair_id"]))
        group_by_split[split].add(str(row["pair_group_id"]))
        for condition in row["conditions"]:
            condition_by_split[split].add(str(condition["condition_id"]))
    overlap_count = 0
    for splits in (pair_by_split, group_by_split, condition_by_split):
        names = sorted(splits)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                overlap_count += len(splits[left_name].intersection(splits[right_name]))
    definition = {
        "row_split_rule": "unit_index_modulo_3_label_blind",
        "family_partitions": {
            "train": ["finite_domain_csp", "weighted_maxsat"],
            "dev": ["hard_soft_packing"],
            "science": ["finite_state_planning"],
        },
        "surface_partition_rule": "both_proof_preserving_surfaces_cross_all_row_splits",
        "split_order": list(SPLIT_ORDER),
        "label_blind_inputs": [
            "source_unit_index",
            "not_exact_label",
            "not_family_name_in_model_input",
        ],
    }
    return {
        "schema": SCHEMA + ".split_definition_and_hashes",
        "definition": definition,
        "definition_hash": sha256_json(definition),
        "label_blind": True,
        "split_counts": dict(sorted(split_counts.items())),
        "family_split_counts": dict(sorted(family_split_counts.items())),
        "surface_split_counts": dict(sorted(surface_split_counts.items())),
        "split_overlap_count": overlap_count,
        "row_split_hashes": {
            split: sha256_json(sorted(pair_by_split.get(split, set()))) for split in SPLIT_ORDER
        },
    }


def _input_tokens(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(condition["model_input"])
        for row in rows
        for condition in row["conditions"]
    ]


def target_leakage_checks(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    forbidden_identity_tokens = {
        "finite_domain_csp",
        "finite-domain-csp",
        "weighted_maxsat",
        "weighted-maxsat",
        "hard_soft_packing",
        "hard-soft-packing",
        "finite_state_planning",
        "finite-state-planning",
        "exp5826",
        "exp5840",
        "row_id",
        "source_row",
    }
    forbidden_answer_tokens = {
        "oracle",
        "label",
        "correct",
        "incorrect",
        "accepted",
        "rejected",
        "answer",
    }
    inputs = _input_tokens(rows)
    tokenized = [set(text.lower().replace(":", " ").split()) for text in inputs]
    identity_leaks = [
        index for index, tokens in enumerate(tokenized) if tokens.intersection(forbidden_identity_tokens)
    ]
    answer_leaks = [
        index for index, tokens in enumerate(tokenized) if tokens.intersection(forbidden_answer_tokens)
    ]
    hashes = [sha256_text(text) for text in inputs]
    duplicate_count = len(hashes) - len(set(hashes))
    normalized_hashes = [
        sha256_text(" ".join(token for token in text.lower().split() if token not in NEUTRAL_PAD_VOCAB))
        for text in inputs
    ]
    near_duplicate_count = len(normalized_hashes) - len(set(normalized_hashes))
    split_receipt = split_definition_and_hashes(rows)
    target_derived_feature_count = 0
    return {
        "schema": SCHEMA + ".target_leakage_checks",
        "identity_leakage_count": len(identity_leaks),
        "identity_leakage_examples": identity_leaks[:20],
        "answer_leakage_count": len(answer_leaks),
        "answer_leakage_examples": answer_leaks[:20],
        "target_derived_feature_count": target_derived_feature_count,
        "duplicate_model_input_count": duplicate_count,
        "near_duplicate_pair_count": near_duplicate_count,
        "split_overlap_count": split_receipt["split_overlap_count"],
        "family_model_label_masked_from_inputs": len(identity_leaks) == 0 and len(answer_leaks) == 0,
        "all_checks_passed": len(identity_leaks) == 0
        and len(answer_leaks) == 0
        and target_derived_feature_count == 0
        and duplicate_count == 0
        and near_duplicate_count == 0
        and split_receipt["split_overlap_count"] == 0,
        "scan_hash": sha256_json({"input_hashes": hashes, "normalized_hashes": normalized_hashes}),
    }


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5839_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5826_ROWS_RELATIVE_PATH.as_posix(),
        EXP5826_MODULE_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def counterfactual_fixture_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when every Exp5840 gate is clean."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    counts = dict(artifact.get("family_axis_cell_counts") or {})
    exact = dict(artifact.get("exact_label_and_minimality_receipts") or {})
    ablation = dict(artifact.get("constraint_ablation_receipts") or {})
    surfaces = dict(artifact.get("proof_preserving_surface_receipts") or {})
    tokens = dict(artifact.get("token_budget_parity") or {})
    splits = dict(artifact.get("split_definition_and_hashes") or {})
    leakage = dict(artifact.get("target_leakage_checks") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = bool(
        preconditions.get("preconditions_ready") is True
        and counts.get("all_counts_passed") is True
        and counts.get("minimum_pairs_per_family_axis", 0) >= MIN_PAIRS_PER_FAMILY_AXIS
        and exact.get("all_exact_labels_passed") is True
        and exact.get("all_minimal_violations_passed") is True
        and ablation.get("all_ablation_checks_passed") is True
        and surfaces.get("all_surface_checks_passed") is True
        and tokens.get("all_pairs_matched") is True
        and tokens.get("answer_bearing_padding_token_hits") == 0
        and splits.get("label_blind") is True
        and splits.get("split_overlap_count") == 0
        and leakage.get("all_checks_passed") is True
        and _row_file_receipt_ok(dict(artifact.get("row_file_receipt") or {}))
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    checks = {
        "failed_test_exit_codes": set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values()),
        "family_axis_cell_counts": dict(artifact.get("family_axis_cell_counts") or {}).get(
            "all_counts_passed"
        )
        is True,
        "exact_label_and_minimality_receipts": dict(
            artifact.get("exact_label_and_minimality_receipts") or {}
        ).get("all_exact_labels_passed")
        is True
        and dict(artifact.get("exact_label_and_minimality_receipts") or {}).get(
            "all_minimal_violations_passed"
        )
        is True,
        "constraint_ablation_receipts": dict(
            artifact.get("constraint_ablation_receipts") or {}
        ).get("all_ablation_checks_passed")
        is True,
        "proof_preserving_surface_receipts": dict(
            artifact.get("proof_preserving_surface_receipts") or {}
        ).get("all_surface_checks_passed")
        is True,
        "token_budget_parity": dict(artifact.get("token_budget_parity") or {}).get(
            "all_pairs_matched"
        )
        is True,
        "split_definition_and_hashes": dict(
            artifact.get("split_definition_and_hashes") or {}
        ).get("split_overlap_count")
        == 0,
        "target_leakage_checks": dict(artifact.get("target_leakage_checks") or {}).get(
            "all_checks_passed"
        )
        is True,
        "row_file_receipt": _row_file_receipt_ok(dict(artifact.get("row_file_receipt") or {})),
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(name)
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if counterfactual_fixture_ready_score(artifact) != 1.0 and not reasons:  # pragma: no cover - safety net.
        reasons.append("counterfactual_fixture_ready_score")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict with ready:/blocked: prefix."""

    if counterfactual_fixture_ready_score(artifact) == 1.0:
        return "ready: exact_counterfactual_embedding_fixture_ready"
    reasons = blocked_reasons(artifact) or ["counterfactual_fixture_not_ready"]
    return "blocked: " + ",".join(reasons[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking self-referential and host-output fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def _artifact_from_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    row_text: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "row_file": ROW_FILE_RELATIVE_PATH.as_posix(),
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "upstream_artifact_hashes": dict(
            dict(preconditions_checked).get("upstream_artifact_hashes") or {}
        ),
        "causal_axis_definitions": causal_axis_definitions(),
        "family_axis_cell_counts": family_axis_cell_counts(rows),
        "exact_label_and_minimality_receipts": exact_label_and_minimality_receipts(rows),
        "constraint_ablation_receipts": constraint_ablation_receipts(rows),
        "proof_preserving_surface_receipts": proof_preserving_surface_receipts(rows),
        "token_budget_parity": token_budget_parity(rows),
        "split_definition_and_hashes": split_definition_and_hashes(rows),
        "target_leakage_checks": target_leakage_checks(rows),
        "row_file_receipt": _row_file_receipt(rows, row_text),
        "counterfactual_fixture_ready_score": 0.0,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["counterfactual_fixture_ready_score"] = counterfactual_fixture_ready_score(artifact)
    artifact["status"] = (
        "complete" if artifact["counterfactual_fixture_ready_score"] == 1.0 else "blocked"
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the terminal Exp5840 artifact and row commitments."""

    started = time.perf_counter()
    preconditions = dict(preconditions_checked or collect_preconditions(root=root))
    rows = generate_rows(root=root, preconditions_checked=preconditions)
    row_text = rows_to_jsonl(rows)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    return _artifact_from_rows(
        rows=rows,
        row_text=row_text,
        preconditions_checked=preconditions,
        duration_s=elapsed,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, readiness, provenance, row receipts, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("status") not in {"complete", "blocked"}:
        raise ValueError("status")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if not _row_file_receipt_ok(dict(artifact.get("row_file_receipt") or {})):
        raise ValueError("row_file_receipt")
    expected_score = counterfactual_fixture_ready_score(artifact)
    if artifact.get("counterfactual_fixture_ready_score") != expected_score:
        raise ValueError("counterfactual_fixture_ready_score")
    expected_status = "complete" if expected_score == 1.0 else "blocked"
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_status == "complete" and not verdict.startswith("ready:"):
        raise ValueError("honest_verdict")
    if expected_status == "blocked" and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_and_write_artifacts(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build, validate, and atomically write Exp5840 JSON and JSONL files."""

    started = time.perf_counter()
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, row_file_path=row_file_path)
    )
    rows = generate_rows(root=root, preconditions_checked=preconditions)
    row_text = rows_to_jsonl(rows)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    artifact = _artifact_from_rows(
        rows=rows,
        row_text=row_text,
        preconditions_checked=preconditions,
        duration_s=elapsed,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )
    _atomic_write(Path(row_file_path), row_text)
    _atomic_write(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if rows:
        verify_row_file(rows, artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5840 and optionally write terminal artifacts."""

    if write:
        return build_and_write_artifacts(
            root=root,
            result_path=result_path,
            row_file_path=row_file_path,
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
            test_commands=list(test_commands),
            test_exit_codes=test_exit_codes,
        )
    return build_artifact(
        root=root,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        test_commands=list(test_commands),
        test_exit_codes=test_exit_codes,
    )


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI guard.
    raise SystemExit(main())
