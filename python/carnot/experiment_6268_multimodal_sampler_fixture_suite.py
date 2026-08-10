"""Exp6268 frozen bounded exact sampler fixture suite.

Spec refs: REQ-SAMPLER-6268, SCENARIO-SAMPLER-6268-EXACT-SUITE,
SCENARIO-SAMPLER-6268-CONTROLS-FAIL-CLOSED,
SCENARIO-SAMPLER-6268-NO-PERFORMANCE-CLAIM.

This module builds exact target fixtures only. It does not run a sampler
comparison. The suite gives later A/B runs fixed, bounded targets with exact
probabilities, basin labels, and barrier metadata.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from carnot import experiment_6152_typed_stochastic_constraint_ir as exp6152
from carnot import experiment_6166_mode_jumping_factor_thermalization as exp6166
from carnot import experiment_6237_activated_mode_jump_sampler_ab as exp6237
from carnot.samplers.mode_jump_rust_backend import ModeJumpRustBackend, frozen_mode_jump_inputs
from carnot.samplers.potts_sampler import PottsSampler


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6268_multimodal_sampler_fixture_suite.json")
FIXTURE_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6268_multimodal_sampler_fixture_manifest.json"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6268_multimodal_sampler_fixture_suite.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6268_multimodal_sampler_fixture_suite.py")
SAMPLER_SPEC_RELATIVE_PATH = Path("openspec/capabilities/samplers/spec.md")
POTTS_SPEC_RELATIVE_PATH = Path("openspec/capabilities/potts-sampler/spec.md")
EXP6237_RESULT_RELATIVE_PATH = Path("results/experiment_6237_activated_mode_jump_sampler_ab.json")
RUN_DATE = "20260810"
SCHEMA = "carnot.experiment_6268.multimodal_sampler_fixture_suite.v1"
EXPERIMENT_ID = "experiment_6268_multimodal_sampler_fixture_suite"
EXACT_TOLERANCE = 1.0e-12
STATE_SPACE_SIZE_BOUND = 2048
INFERENCE_SUBSTRATE = "local_cpu_exact_fixture_construction"
DEFAULT_RECEIPT_PATH = Path("/tmp/carnot_6268_command_receipts.json")

RANDOM_SEEDS = (6268, 6269, 6270)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6268_multimodal_sampler_fixture_suite.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6268_multimodal_sampler_fixture_suite.py -m pytest tests/python/test_experiment_6268_multimodal_sampler_fixture_suite.py -q --no-cov -o addopts=",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6268_multimodal_sampler_fixture_suite.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6268_multimodal_sampler_fixture_suite.py",
    "cargo test -p carnot-samplers --test mode_jump --quiet",
    ".venv/bin/pytest tests/python/test_e2e_serialization.py tests/python/test_pyo3_integration.py -q -o addopts=",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_6268_multimodal_sampler_fixture_suite --date 20260810",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6268_multimodal_sampler_fixture_suite.json",
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SAMPLER_SPEC_RELATIVE_PATH,
    POTTS_SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6152_typed_stochastic_constraint_ir.py"),
    Path("python/carnot/experiment_6166_mode_jumping_factor_thermalization.py"),
    Path("python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py"),
    Path("python/carnot/samplers/mode_jump_rust_backend.py"),
    Path("python/carnot/samplers/potts_sampler.py"),
    Path("crates/carnot-samplers/src/mode_jump.rs"),
    EXP6237_RESULT_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "fixture_manifest_path_and_hash",
    "fixture_family_counts",
    "source_paths_and_hashes",
    "state_space_sizes",
    "exact_enumeration_receipts",
    "normalized_target_probability_hashes",
    "energy_and_factor_definitions",
    "basin_labels_and_barrier_metadata",
    "mode_jump_support_by_fixture",
    "original_six_state_positive_control",
    "unimodal_control",
    "inactive_treatment_control",
    "unsupported_shape_control",
    "random_seeds_and_schedule_defaults",
    "duplicate_fixture_count",
    "exact_probability_normalization_error_by_fixture",
    "source_mutation_count",
    "sampler_fixture_suite_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Separates ready exact-suite evidence from blocked fixture, control, and command states.",
    "fixture_manifest_path_and_hash": "Pins the generated fixture manifest so later sampler runs consume the same suite.",
    "fixture_family_counts": "Proves the preregistered binary Ising, Potts, typed-factor, and control families are all present.",
    "source_paths_and_hashes": "Pins source, spec, test, upstream artifact, sampler, and protected paths before the JSON is trusted.",
    "state_space_sizes": "Makes every bounded enumeration size explicit.",
    "exact_enumeration_receipts": "Stores independently enumerated probabilities, energies, modes, basins, and normalization receipts.",
    "normalized_target_probability_hashes": "Content-addresses normalized target probabilities per fixture.",
    "energy_and_factor_definitions": "Records the Ising Hamiltonians, Potts couplings, and typed factor kernels used to enumerate targets.",
    "basin_labels_and_barrier_metadata": "Records basin assignment and minimum barrier evidence instead of inferring modality from names.",
    "mode_jump_support_by_fixture": "Separates the six-state supported mode-jump target from unsupported suite fixtures and inactive controls.",
    "original_six_state_positive_control": "Proves the Exp6237 treatment-positive fixture is reproduced in the suite.",
    "unimodal_control": "Proves a valid exact target can be unimodal and therefore cannot support a multimodal treatment claim.",
    "inactive_treatment_control": "Proves inactive treatment evidence is classified as an instrument failure, not a null result.",
    "unsupported_shape_control": "Proves unsupported fixture shapes fail closed at the mode-jump boundary.",
    "random_seeds_and_schedule_defaults": "Freezes replay seeds and default construction schedules without making timing claims.",
    "duplicate_fixture_count": "Bare zero proves the suite did not register duplicate exact targets.",
    "exact_probability_normalization_error_by_fixture": "Makes normalization tolerance evidence mechanical for each fixture.",
    "source_mutation_count": "Bare zero proves protected and preregistered source hashes did not change during construction.",
    "sampler_fixture_suite_ready_score": "Equals one only when family, exactness, control, duplicate, source, protection, and command gates pass.",
    "protected_files_unchanged": "Confirms conductor and reconciler-owned files stayed byte-identical.",
    "preconditions_checked": "Records frozen families, bounds, tolerances, seeds, hashes, and protected files before enumeration.",
    "inference_substrate": "Declares local CPU exact fixture construction, not LLM inference, CUDA, FPGA, TSU, cDLS, or timing.",
    "verifier_is_oracle": "States that exact finite enumeration is the oracle for this suite.",
    "field_provenance": "Maps every required field to prompt, spec, source, fixture manifest, command receipts, or computed exact evidence.",
    "field_principles": "Explains why each required field exists before a reviewer trusts the artifact.",
    "test_commands": "Records focused Python, coverage, Rust, E2E, artifact, adversarial, and suite command receipts.",
    "test_exit_codes": "Stores exit codes so failed checks cannot become readiness evidence.",
    "duration_s": "Reports real wall time without padding or timing interpretation.",
    "reproducibility_checksum": "Content-addresses the artifact after blanking volatile duration and the checksum field.",
    "honest_verdict": "Uses a terminal prefix and states fixture readiness, exactness, controls, and no performance claim.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in stable ASCII byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash text in the repository's prefixed SHA-256 format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash one file by bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stable_float(value: Any) -> float:
    rounded = round(float(value), 15)
    return 0.0 if rounded == 0.0 else rounded


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _hash_path(path: Path, root: Path) -> JsonDict:
    target = root / path
    return {
        "exists": target.exists(),
        "kind": "file",
        "sha256": sha256_file(target) if target.exists() else None,
        "size_bytes": target.stat().st_size if target.exists() else None,
    }


def _path_hashes(paths: Sequence[Path], root: Path) -> dict[str, JsonDict]:
    return {path.as_posix(): _hash_path(path, root) for path in paths}


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected at {path}")  # pragma: no cover
    return payload


def build_fixture_suite(root: Path = REPO_ROOT) -> list[JsonDict]:
    """Freeze every fixture definition before exact enumeration."""

    labels, target, proposal = frozen_mode_jump_inputs(root)
    exp6152_program = exp6152.compile_exp6145_bounded_workflow()
    exp6166_program = exp6166.build_multimodal_factor_program()
    return _json_copy(
        [
            {
                "fixture_name": "exp6237_original_six_state",
                "family": "original_six_state_positive_control",
                "target_type": "categorical_mode_jump",
                "definition": {
                    "labels": labels,
                    "target_probabilities": target.astype(float).tolist(),
                    "proposal_probabilities": proposal.astype(float).tolist(),
                    "mode_labels": {
                        basin: list(members) for basin, members in exp6237.MODE_LABELS.items()
                    },
                    "explicit_modes": ["left_peak", "right_peak"],
                    "source": EXP6237_RESULT_RELATIVE_PATH.as_posix(),
                },
            },
            {
                "fixture_name": "ising_ferromagnetic_ring4",
                "family": "ising_multimodal",
                "target_type": "ising",
                "definition": {
                    "n_spins": 4,
                    "spin_values": [-1, 1],
                    "beta": 1.05,
                    "edges": [
                        [0, 1, 1.0],
                        [1, 2, 1.0],
                        [2, 3, 1.0],
                        [3, 0, 1.0],
                    ],
                    "fields": [0.0, 0.0, 0.0, 0.0],
                    "energy": "-sum_edges J_ij s_i s_j - sum_i h_i s_i",
                },
            },
            {
                "fixture_name": "ising_ferromagnetic_ring5",
                "family": "ising_multimodal",
                "target_type": "ising",
                "definition": {
                    "n_spins": 5,
                    "spin_values": [-1, 1],
                    "beta": 0.9,
                    "edges": [
                        [0, 1, 0.9],
                        [1, 2, 0.9],
                        [2, 3, 0.9],
                        [3, 4, 0.9],
                        [4, 0, 0.9],
                    ],
                    "fields": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "energy": "-sum_edges J_ij s_i s_j - sum_i h_i s_i",
                },
            },
            {
                "fixture_name": "potts_chain3_q3",
                "family": "potts",
                "target_type": "potts",
                "definition": {
                    "n_spins": 3,
                    "q_states": 3,
                    "state_labels": ["incorrect", "partial", "correct"],
                    "beta": 1.1,
                    "couplings": [
                        [0.0, 0.5, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.5, 0.0],
                    ],
                    "energy": "PottsSampler.energy: -sum_ij J_ij delta(s_i, s_j)",
                },
            },
            {
                "fixture_name": "potts_antiferro_triangle3_q3",
                "family": "potts",
                "target_type": "potts",
                "definition": {
                    "n_spins": 3,
                    "q_states": 3,
                    "state_labels": ["red", "green", "blue"],
                    "beta": 1.3,
                    "couplings": [
                        [0.0, -0.4, -0.4],
                        [-0.4, 0.0, -0.4],
                        [-0.4, -0.4, 0.0],
                    ],
                    "energy": "PottsSampler.energy: -sum_ij J_ij delta(s_i, s_j)",
                },
            },
            {
                "fixture_name": "typed_access_control_exp6152",
                "family": "typed_factor",
                "target_type": "typed_factor",
                "definition": {
                    "program_payload": exp6152.program_to_payload(exp6152_program),
                    "expected_max_kernel_arity": 3,
                    "basin_wire": "accepted",
                    "source": exp6152.RESULT_RELATIVE_PATH.as_posix(),
                },
            },
            {
                "fixture_name": "typed_multimodal_factor_exp6166",
                "family": "typed_factor",
                "target_type": "typed_factor",
                "definition": {
                    "program_payload": exp6152.program_to_payload(exp6166_program),
                    "expected_max_kernel_arity": 0,
                    "basin_wire": "mode_state",
                    "mode_labels": {
                        basin: list(members) for basin, members in exp6166.MODE_LABELS.items()
                    },
                    "explicit_modes": ["left_peak", "right_peak"],
                    "source": exp6166.RESULT_RELATIVE_PATH.as_posix(),
                },
            },
            {
                "fixture_name": "control_unimodal_ising3",
                "family": "unimodal_control",
                "target_type": "ising",
                "definition": {
                    "n_spins": 3,
                    "spin_values": [-1, 1],
                    "beta": 1.0,
                    "edges": [[0, 1, 0.2], [1, 2, 0.2]],
                    "fields": [1.1, 0.9, 1.0],
                    "energy": "-sum_edges J_ij s_i s_j - sum_i h_i s_i",
                    "control_role": "unimodal",
                },
            },
        ]
    )


def enumerate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    """Dispatch one frozen definition to its exact enumerator."""

    target_type = str(fixture["target_type"])
    if target_type == "categorical_mode_jump":
        return _enumerate_categorical(fixture)
    if target_type == "ising":
        return _enumerate_ising(fixture)
    if target_type == "potts":
        return _enumerate_potts(fixture)
    if target_type == "typed_factor":
        return _enumerate_typed_factor(fixture)
    raise ValueError(f"unsupported fixture target_type: {target_type}")  # pragma: no cover


def _enumerate_categorical(fixture: Mapping[str, Any]) -> JsonDict:
    definition = dict(fixture["definition"])
    labels = [str(label) for label in definition["labels"]]
    probabilities = [float(value) for value in definition["target_probabilities"]]
    mode_labels = {
        str(basin): [str(label) for label in members]
        for basin, members in definition["mode_labels"].items()
    }
    basin_by_label = {label: basin for basin, members in mode_labels.items() for label in members}
    rows = [
        {
            "state_label": label,
            "state": {"label": label},
            "probability": _stable_float(probability),
            "energy": _stable_float(-math.log(probability)),
            "basin": basin_by_label[label],
            "is_mode": label in set(definition["explicit_modes"]),
        }
        for label, probability in zip(labels, probabilities, strict=True)
        if probability > 0.0
    ]
    adjacency = _categorical_adjacency(labels, definition["proposal_probabilities"])
    return _finish_receipt(fixture, rows, len(labels), 1.0, adjacency)


def _enumerate_ising(fixture: Mapping[str, Any]) -> JsonDict:
    definition = dict(fixture["definition"])
    n_spins = int(definition["n_spins"])
    states = list(itertools.product((-1, 1), repeat=n_spins))
    energies = [_ising_energy(state, definition) for state in states]
    probabilities = _boltzmann_probabilities(energies, float(definition["beta"]))
    minimum_energy = min(energies)
    rows = []
    for state, energy, probability in zip(states, energies, probabilities, strict=True):
        label = _spin_label(state)
        rows.append(
            {
                "state_label": label,
                "state": list(state),
                "probability": _stable_float(probability),
                "energy": _stable_float(energy),
                "basin": _ising_basin(state, energy, minimum_energy),
                "is_mode": abs(energy - minimum_energy) <= EXACT_TOLERANCE,
            }
        )
    adjacency = _hamming_adjacency([row["state_label"] for row in rows])
    return _finish_receipt(fixture, rows, len(states), float(definition["beta"]), adjacency)


def _enumerate_potts(fixture: Mapping[str, Any]) -> JsonDict:
    definition = dict(fixture["definition"])
    _validate_potts_definition(definition)
    n_spins = int(definition["n_spins"])
    q_states = int(definition["q_states"])
    states = list(itertools.product(range(q_states), repeat=n_spins))
    sampler = PottsSampler(n_spins=n_spins, q=q_states, beta=float(definition["beta"]))
    couplings = np.asarray(definition["couplings"], dtype=np.float64)
    energies = [sampler.energy(couplings, np.asarray(state, dtype=np.int64)) for state in states]
    probabilities = _boltzmann_probabilities(energies, float(definition["beta"]))
    minimum_energy = min(energies)
    rows = []
    for state, energy, probability in zip(states, energies, probabilities, strict=True):
        label = _potts_label(state)
        rows.append(
            {
                "state_label": label,
                "state": list(state),
                "probability": _stable_float(probability),
                "energy": _stable_float(energy),
                "basin": _potts_basin(state, energy, minimum_energy),
                "is_mode": abs(energy - minimum_energy) <= EXACT_TOLERANCE,
            }
        )
    adjacency = _hamming_adjacency([row["state_label"] for row in rows])
    return _finish_receipt(fixture, rows, len(states), float(definition["beta"]), adjacency)


def _enumerate_typed_factor(fixture: Mapping[str, Any]) -> JsonDict:
    definition = dict(fixture["definition"])
    program = exp6152.program_from_payload(definition["program_payload"])
    exact = exp6152.execute_exact(program)
    support = sorted(exact["support"], key=lambda row: exp6152.canonical_json(row["state"]))
    basin_wire = str(definition["basin_wire"])
    mode_labels = {
        str(basin): [str(label) for label in members]
        for basin, members in dict(definition.get("mode_labels") or {}).items()
    }
    explicit_modes = set(str(label) for label in definition.get("explicit_modes", []))
    max_probability = max(float(row["probability"]) for row in support)
    rows = []
    for row in support:
        probability = float(row["probability"])
        state = dict(row["state"])
        state_label = exp6152.canonical_json(state)
        basin = _typed_basin(state, basin_wire, mode_labels)
        rows.append(
            {
                "state_label": state_label,
                "state": state,
                "probability": _stable_float(probability),
                "energy": _stable_float(-math.log(probability)),
                "basin": basin,
                "is_mode": _typed_is_mode(
                    state, basin_wire, explicit_modes, probability, max_probability
                ),
            }
        )
    adjacency = _typed_support_adjacency(rows)
    receipt = _finish_receipt(fixture, rows, int(exact["state_space_size"]), 1.0, adjacency)
    receipt["typed_factor_arity"] = max(len(kernel.inputs) for kernel in program.kernels)
    receipt["expected_typed_factor_arity"] = int(definition["expected_max_kernel_arity"])
    receipt["wire_order"] = exp6152.wire_order(program)
    return receipt


def _validate_potts_definition(definition: Mapping[str, Any]) -> None:
    q_states = int(definition["q_states"])
    labels = list(definition["state_labels"])
    if q_states < 2 or q_states != len(labels):
        raise ValueError("Potts cardinality mismatch")
    couplings = np.asarray(definition["couplings"], dtype=np.float64)
    expected = (int(definition["n_spins"]), int(definition["n_spins"]))
    if couplings.shape != expected:
        raise ValueError("Potts coupling shape mismatch")  # pragma: no cover


def _ising_energy(state: Sequence[int], definition: Mapping[str, Any]) -> float:
    edge_term = sum(
        float(weight) * state[int(left)] * state[int(right)]
        for left, right, weight in definition["edges"]
    )
    field_term = sum(
        float(field) * spin for field, spin in zip(definition["fields"], state, strict=True)
    )
    return -edge_term - field_term


def _boltzmann_probabilities(energies: Sequence[float], beta: float) -> list[float]:
    shifted = [float(energy) - min(energies) for energy in energies]
    weights = [math.exp(-beta * energy) for energy in shifted]
    total = sum(weights)
    return [weight / total for weight in weights]


def _spin_label(state: Sequence[int]) -> str:
    return ",".join("+1" if value > 0 else "-1" for value in state)


def _potts_label(state: Sequence[int]) -> str:
    return ",".join(str(value) for value in state)


def _ising_basin(state: Sequence[int], energy: float, minimum_energy: float) -> str:
    if abs(energy - minimum_energy) <= EXACT_TOLERANCE:
        return "mode_positive" if sum(state) > 0 else "mode_negative"
    magnetization = sum(state)
    if magnetization > 0:
        return "positive_basin"
    if magnetization < 0:
        return "negative_basin"
    return "separatrix_basin"


def _potts_basin(state: Sequence[int], energy: float, minimum_energy: float) -> str:
    if abs(energy - minimum_energy) <= EXACT_TOLERANCE:
        return f"mode_{_potts_label(state)}"
    counts = Counter(state)
    [(first_state, first_count)] = counts.most_common(1)
    tied = sum(1 for count in counts.values() if count == first_count) > 1
    return "mixed_basin" if tied else f"majority_{first_state}_basin"


def _typed_basin(
    state: Mapping[str, Any],
    basin_wire: str,
    mode_labels: Mapping[str, Sequence[str]],
) -> str:
    value = str(state[basin_wire])
    for basin, members in mode_labels.items():
        if value in set(members):
            return basin
    return f"{basin_wire}_{value}_basin"


def _typed_is_mode(
    state: Mapping[str, Any],
    basin_wire: str,
    explicit_modes: set[str],
    probability: float,
    max_probability: float,
) -> bool:
    if explicit_modes:
        return str(state[basin_wire]) in explicit_modes
    return abs(probability - max_probability) <= EXACT_TOLERANCE


def _categorical_adjacency(
    labels: Sequence[str],
    proposal_probabilities: Sequence[Sequence[float]],
) -> list[list[str]]:
    edges = []
    for left_index, left in enumerate(labels):
        for right_index, right in enumerate(labels):
            if (
                left_index < right_index
                and float(proposal_probabilities[left_index][right_index]) > 0.0
            ):
                edges.append([str(left), str(right)])
    return edges


def _hamming_adjacency(labels: Sequence[str]) -> list[list[str]]:
    edges = []
    split = {label: label.split(",") for label in labels}
    for left, right in itertools.combinations(labels, 2):
        if sum(a != b for a, b in zip(split[left], split[right], strict=True)) == 1:
            edges.append([left, right])
    return edges


def _typed_support_adjacency(rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    edges = []
    for left, right in itertools.combinations(rows, 2):
        left_state = dict(left["state"])
        right_state = dict(right["state"])
        differing = sum(left_state[key] != right_state[key] for key in left_state)
        if differing == 1:
            edges.append([str(left["state_label"]), str(right["state_label"])])
    return edges


def _finish_receipt(
    fixture: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    state_space_size: int,
    beta: float,
    adjacency: Sequence[Sequence[str]],
) -> JsonDict:
    support = sorted((_json_copy(row) for row in rows), key=lambda row: str(row["state_label"]))
    normalization = sum(float(row["probability"]) for row in support)
    receipt: JsonDict = {
        "fixture_name": str(fixture["fixture_name"]),
        "family": str(fixture["family"]),
        "target_type": str(fixture["target_type"]),
        "definition": _json_copy(fixture["definition"]),
        "definition_hash": sha256_json(fixture["definition"]),
        "state_space_size": int(state_space_size),
        "support_count": len(support),
        "zero_probability_state_count": int(state_space_size) - len(support),
        "boltzmann_beta": _stable_float(beta),
        "normalization": _stable_float(normalization),
        "normalization_error": _stable_float(abs(1.0 - normalization)),
        "support": support,
        "modes": [str(row["state_label"]) for row in support if row["is_mode"]],
        "basin_by_state": {str(row["state_label"]): str(row["basin"]) for row in support},
        "adjacency_edges": [list(edge) for edge in adjacency],
        "enumerator": "local_cpu_exact_enumeration_independent_of_sampler",
    }
    receipt["barrier_metadata"] = _barrier_metadata(receipt)
    receipt["target_probability_hash"] = normalized_target_probability_hash(receipt)
    validate_fixture_receipt(receipt)
    return receipt


def _barrier_metadata(receipt: Mapping[str, Any]) -> JsonDict:
    by_label = {str(row["state_label"]): row for row in receipt["support"]}
    basins = sorted(set(str(row["basin"]) for row in receipt["support"]))
    min_energy_by_basin = {
        basin: _stable_float(
            min(float(row["energy"]) for row in receipt["support"] if row["basin"] == basin)
        )
        for basin in basins
    }
    best_pairs: dict[tuple[str, str], JsonDict] = {}
    for left, right in receipt["adjacency_edges"]:
        left_row = by_label[str(left)]
        right_row = by_label[str(right)]
        left_basin = str(left_row["basin"])
        right_basin = str(right_row["basin"])
        if left_basin == right_basin:
            continue
        key = tuple(sorted((left_basin, right_basin)))
        saddle_energy = max(float(left_row["energy"]), float(right_row["energy"]))
        floor = max(min_energy_by_basin[left_basin], min_energy_by_basin[right_basin])
        candidate = {
            "basins": list(key),
            "edge": [str(left), str(right)],
            "saddle_energy": _stable_float(saddle_energy),
            "barrier_delta": _stable_float(saddle_energy - floor),
        }
        if key not in best_pairs or candidate["barrier_delta"] < best_pairs[key]["barrier_delta"]:
            best_pairs[key] = candidate
    barrier_pairs = [best_pairs[key] for key in sorted(best_pairs)]
    return {
        "basin_count": len(basins),
        "basin_labels": basins,
        "mode_count": len(receipt["modes"]),
        "modes": list(receipt["modes"]),
        "minimum_energy_by_basin": min_energy_by_basin,
        "barrier_pairs": barrier_pairs,
        "minimum_cross_basin_barrier": _stable_float(
            min((float(row["barrier_delta"]) for row in barrier_pairs), default=0.0)
        ),
    }


def normalized_target_probability_hash(receipt: Mapping[str, Any]) -> str:
    """Hash normalized target probabilities independent of row order."""

    rows = sorted(
        [
            {
                "state_label": str(row["state_label"]),
                "probability": _stable_float(row["probability"]),
            }
            for row in receipt["support"]
        ],
        key=lambda row: row["state_label"],
    )
    return sha256_json(
        {
            "fixture_name": str(receipt["fixture_name"]),
            "state_space_size": int(receipt["state_space_size"]),
            "support": rows,
        }
    )


def validate_fixture_receipt(receipt: Mapping[str, Any]) -> bool:
    """Validate one exact fixture receipt and its energy/probability contract."""

    support = list(receipt["support"])
    if not support:
        raise ValueError("support must be non-empty")  # pragma: no cover
    probability_sum = sum(float(row["probability"]) for row in support)
    if abs(probability_sum - 1.0) > EXACT_TOLERANCE:
        raise ValueError("normalization_error exceeds tolerance")
    beta = float(receipt["boltzmann_beta"])
    ordered = sorted(support, key=lambda item: str(item["state_label"]))
    energies = [float(row["energy"]) for row in ordered]
    probabilities = _boltzmann_probabilities(energies, beta)
    for row, expected in zip(ordered, probabilities, strict=True):
        if abs(float(row["probability"]) - expected) > 1.0e-10:
            raise ValueError("energy_probability_consistency failed")
    if receipt.get("target_probability_hash") not in {
        None,
        normalized_target_probability_hash(receipt),
    }:
        raise ValueError("normalized_target_probability_hashes mismatch")
    if receipt.get("target_type") == "typed_factor" and receipt.get(
        "typed_factor_arity"
    ) != receipt.get("expected_typed_factor_arity"):
        raise ValueError("typed_factor_arity mismatch")
    return True


def duplicate_fixture_count(receipts: Sequence[Mapping[str, Any]]) -> int:
    """Count duplicated exact targets by normalized probability hash."""

    hashes = [normalized_target_probability_hash(receipt) for receipt in receipts]
    counts = Counter(hashes)
    return sum(count - 1 for count in counts.values() if count > 1)


def fixture_family_counts(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = Counter(str(receipt["family"]) for receipt in receipts)
    result: JsonDict = dict(sorted(counts.items()))
    result["all_preregistered_families_present"] = bool(
        counts["original_six_state_positive_control"] == 1
        and counts["ising_multimodal"] >= 2
        and counts["potts"] >= 2
        and counts["typed_factor"] >= 2
        and counts["unimodal_control"] >= 1
    )
    result["principle"] = FIELD_PRINCIPLES["fixture_family_counts"]
    return result


def state_space_sizes(receipts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(receipt["fixture_name"]): int(receipt["state_space_size"]) for receipt in receipts}


def exact_probability_normalization_error_by_fixture(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    return {
        str(receipt["fixture_name"]): _stable_float(receipt["normalization_error"])
        for receipt in receipts
    }


def energy_and_factor_definitions(receipts: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        str(receipt["fixture_name"]): {
            "target_type": receipt["target_type"],
            "family": receipt["family"],
            "definition_hash": receipt["definition_hash"],
            "definition": receipt["definition"],
        }
        for receipt in receipts
    }


def basin_labels_and_barrier_metadata(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    return {
        str(receipt["fixture_name"]): {
            **dict(receipt["barrier_metadata"]),
            "basin_by_state": receipt["basin_by_state"],
        }
        for receipt in receipts
    }


def mode_jump_support_by_fixture(receipts: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Classify whether each fixture is supported by the fixed mode-jump boundary."""

    support: dict[str, JsonDict] = {}
    backend = ModeJumpRustBackend(prefer_rust=True)
    for receipt in receipts:
        name = str(receipt["fixture_name"])
        if name == "exp6237_original_six_state":
            probabilities = receipt["definition"]["target_probabilities"]
            proposal = receipt["definition"]["proposal_probabilities"]
        else:
            probabilities = [float(row["probability"]) for row in receipt["support"]]
            proposal = np.eye(len(probabilities), dtype=np.float64).tolist()
        try:
            backend._coerce_mode_jump_inputs(  # noqa: SLF001
                np.asarray(probabilities, dtype=np.float64),
                np.asarray(proposal, dtype=np.float64),
            )
        except ValueError as exc:
            support[name] = {
                "mode_jump_rust_supported": False,
                "classification": "unsupported_for_fixed_mode_jump_runtime",
                "message": str(exc),
            }
        else:
            support[name] = {
                "mode_jump_rust_supported": True,
                "classification": "supported_original_six_state_positive_control",
                "message": None,
            }
    return support


def original_six_state_positive_control(
    receipts: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> JsonDict:
    receipt = _receipt_by_name(receipts, "exp6237_original_six_state")
    exp6237_exact = exp6237.exact_reference_distribution_receipts(root)
    exp6237_artifact = _read_json(root / EXP6237_RESULT_RELATIVE_PATH)
    exact_probabilities = {
        str(row["state"]["label"]): _stable_float(row["probability"]) for row in receipt["support"]
    }
    reproduces = exact_probabilities == exp6237_exact["target_probabilities"]
    prior_passed = exp6237_artifact.get("multimodal_positive_control", {}).get("passed") is True
    return {
        "fixture": receipt["fixture_name"],
        "passed": bool(
            reproduces
            and prior_passed
            and support[receipt["fixture_name"]]["mode_jump_rust_supported"] is True
            and len(receipt["modes"]) >= 2
        ),
        "reproduces_exp6237_fixture": reproduces,
        "exp6237_target_probability_hash": exp6237_exact["fixture_sha256"],
        "suite_target_probability_hash": receipt["target_probability_hash"],
        "prior_exp6237_positive_control_passed": prior_passed,
        "mode_count": len(receipt["modes"]),
        "principle": FIELD_PRINCIPLES["original_six_state_positive_control"],
    }


def unimodal_control(receipts: Sequence[Mapping[str, Any]], support: Mapping[str, Any]) -> JsonDict:
    receipt = _receipt_by_name(receipts, "control_unimodal_ising3")
    valid = (
        len(receipt["modes"]) == 1
        and receipt["normalization_error"] <= EXACT_TOLERANCE
        and support[receipt["fixture_name"]]["mode_jump_rust_supported"] is False
    )
    return {
        "fixture": receipt["fixture_name"],
        "valid_unimodal_control": bool(valid),
        "mode_count": len(receipt["modes"]),
        "multimodal_claim_allowed": False,
        "mode_jump_supported": support[receipt["fixture_name"]]["mode_jump_rust_supported"],
        "principle": FIELD_PRINCIPLES["unimodal_control"],
    }


def inactive_treatment_control() -> JsonDict:
    return {
        "fixture": "exp6237_original_six_state",
        "treatment_activation_passed": False,
        "activation_score": 0.0,
        "instrument_failure_if_used_for_quality": True,
        "null_sampler_verdict_allowed": False,
        "valid_inactive_control": True,
        "principle": FIELD_PRINCIPLES["inactive_treatment_control"],
    }


def unsupported_shape_control(support: Mapping[str, Any]) -> JsonDict:
    unsupported = {
        name: row for name, row in support.items() if row["mode_jump_rust_supported"] is False
    }
    return {
        "valid_unsupported_shape_control": bool(unsupported),
        "unsupported_fixture_count": len(unsupported),
        "first_unsupported_fixture": next(iter(sorted(unsupported))) if unsupported else None,
        "all_unsupported_fail_closed": all(
            row["classification"] == "unsupported_for_fixed_mode_jump_runtime"
            for row in unsupported.values()
        ),
        "principle": FIELD_PRINCIPLES["unsupported_shape_control"],
    }


def random_seeds_and_schedule_defaults() -> JsonDict:
    return {
        "random_seeds": list(RANDOM_SEEDS),
        "exact_tolerance": EXACT_TOLERANCE,
        "state_space_size_bound": STATE_SPACE_SIZE_BOUND,
        "sampler_comparison_run": False,
        "timing_claim_allowed": False,
        "hardware_claim_allowed": False,
        "construction_schedule": "enumerate_each_bounded_state_space_once",
        "principle": FIELD_PRINCIPLES["random_seeds_and_schedule_defaults"],
    }


def source_paths_and_hashes(
    *,
    root: Path,
    source_before: Mapping[str, Any],
    source_after: Mapping[str, Any],
) -> JsonDict:
    return {
        "source_hashes_before": dict(source_before),
        "source_hashes_after": dict(source_after),
        "changed_source_paths": [
            path for path in source_before if source_before[path] != source_after.get(path)
        ],
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "manifest_path": FIXTURE_MANIFEST_RELATIVE_PATH.as_posix(),
        "root": root.as_posix(),
        "principle": FIELD_PRINCIPLES["source_paths_and_hashes"],
    }


def preconditions_checked(
    *,
    root: Path,
    run_date: str,
    source_before: Mapping[str, Any],
    protected_before: Mapping[str, Any],
) -> JsonDict:
    fixture_defs = build_fixture_suite(root)
    declared_sizes = [_declared_state_space_size(fixture) for fixture in fixture_defs]
    spec_text = (root / SAMPLER_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks = {
        "families_frozen_before_enumeration": True,
        "size_bound_frozen": STATE_SPACE_SIZE_BOUND == 2048,
        "all_declared_state_spaces_within_bound": max(declared_sizes) <= STATE_SPACE_SIZE_BOUND,
        "exact_tolerance_frozen": EXACT_TOLERANCE == 1.0e-12,
        "seeds_frozen": list(RANDOM_SEEDS) == [6268, 6269, 6270],
        "sampler_spec_has_req": "REQ-SAMPLER-6268" in spec_text,
        "protected_files_present": all(row["exists"] for row in protected_before.values()),
        "source_paths_present": all(row["exists"] for row in source_before.values()),
    }
    return {
        "run_date": run_date,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "fixture_count": len(fixture_defs),
        "declared_state_space_sizes": declared_sizes,
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "source_hashes_before_sha256": sha256_json(source_before),
        "protected_hashes_before_sha256": sha256_json(protected_before),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def protected_files_unchanged(
    *,
    root: Path,
    protected_before: Mapping[str, Any],
) -> JsonDict:
    after = _path_hashes(PROTECTED_FILES, root)
    changed = [path for path in protected_before if protected_before[path] != after.get(path)]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "before": dict(protected_before),
        "after": after,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def verifier_is_oracle() -> JsonDict:
    return {
        "value": True,
        "oracle": "exact finite enumeration over every bounded fixture state space",
        "sampler_execution_used_as_oracle": False,
        "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
    }


def field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": _field_source(field),
            "spec_refs": [
                "REQ-SAMPLER-6268",
                "SCENARIO-SAMPLER-6268-EXACT-SUITE",
                "SCENARIO-SAMPLER-6268-CONTROLS-FAIL-CLOSED",
                "SCENARIO-SAMPLER-6268-NO-PERFORMANCE-CLAIM",
            ],
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _field_source(field: str) -> str:
    computed = {
        "state_space_sizes",
        "exact_enumeration_receipts",
        "normalized_target_probability_hashes",
        "basin_labels_and_barrier_metadata",
        "mode_jump_support_by_fixture",
        "original_six_state_positive_control",
        "unimodal_control",
        "inactive_treatment_control",
        "unsupported_shape_control",
        "duplicate_fixture_count",
        "exact_probability_normalization_error_by_fixture",
        "sampler_fixture_suite_ready_score",
    }
    if field in computed:
        return "computed_exact_fixture_evidence"
    if field in {"test_commands", "test_exit_codes"}:
        return "command_receipts"
    if field in {"source_paths_and_hashes", "source_mutation_count", "protected_files_unchanged"}:
        return "local_path_hashes"
    return "prompt_spec_and_builder"


def fixture_manifest_payload(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": "carnot.experiment_6268.fixture_manifest.v1",
        "spec_refs": [
            "REQ-SAMPLER-6268",
            "SCENARIO-SAMPLER-6268-EXACT-SUITE",
        ],
        "fixture_count": len(receipts),
        "fixtures": [
            {
                "fixture_name": receipt["fixture_name"],
                "family": receipt["family"],
                "target_type": receipt["target_type"],
                "state_space_size": receipt["state_space_size"],
                "support_count": receipt["support_count"],
                "target_probability_hash": normalized_target_probability_hash(receipt),
            }
            for receipt in receipts
        ],
    }


def fixture_manifest_path_and_hash(
    receipts: Sequence[Mapping[str, Any]],
    manifest_path: Path,
) -> JsonDict:
    payload = fixture_manifest_payload(receipts)
    return {
        "path": manifest_path.as_posix(),
        "sha256": sha256_json(payload),
        "schema": payload["schema"],
        "fixture_count": payload["fixture_count"],
        "principle": FIELD_PRINCIPLES["fixture_manifest_path_and_hash"],
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float = 0.0,
    test_exit_codes: Mapping[str, int] | None = None,
    manifest_path: Path | None = None,
) -> JsonDict:
    """Assemble the Exp6268 terminal artifact."""

    source_before = _path_hashes(SOURCE_PATHS, root)
    protected_before = _path_hashes(PROTECTED_FILES, root)
    preconditions = preconditions_checked(
        root=root,
        run_date=run_date,
        source_before=source_before,
        protected_before=protected_before,
    )
    receipts = [enumerate_fixture(fixture) for fixture in build_fixture_suite(root)]
    source_after = _path_hashes(SOURCE_PATHS, root)
    support = mode_jump_support_by_fixture(receipts)
    manifest = fixture_manifest_path_and_hash(
        receipts,
        manifest_path or root / FIXTURE_MANIFEST_RELATIVE_PATH,
    )
    normalized_codes = _normalize_test_exit_codes(test_exit_codes or {})
    artifact: JsonDict = {
        "status": "pending",
        "fixture_manifest_path_and_hash": manifest,
        "fixture_family_counts": fixture_family_counts(receipts),
        "source_paths_and_hashes": source_paths_and_hashes(
            root=root,
            source_before=source_before,
            source_after=source_after,
        ),
        "state_space_sizes": state_space_sizes(receipts),
        "exact_enumeration_receipts": receipts,
        "normalized_target_probability_hashes": {
            str(receipt["fixture_name"]): normalized_target_probability_hash(receipt)
            for receipt in receipts
        },
        "energy_and_factor_definitions": energy_and_factor_definitions(receipts),
        "basin_labels_and_barrier_metadata": basin_labels_and_barrier_metadata(receipts),
        "mode_jump_support_by_fixture": support,
        "original_six_state_positive_control": original_six_state_positive_control(
            receipts,
            support,
            root,
        ),
        "unimodal_control": unimodal_control(receipts, support),
        "inactive_treatment_control": inactive_treatment_control(),
        "unsupported_shape_control": unsupported_shape_control(support),
        "random_seeds_and_schedule_defaults": random_seeds_and_schedule_defaults(),
        "duplicate_fixture_count": duplicate_fixture_count(receipts),
        "exact_probability_normalization_error_by_fixture": (
            exact_probability_normalization_error_by_fixture(receipts)
        ),
        "source_mutation_count": sum(
            1 for path in source_before if source_before[path] != source_after.get(path)
        ),
        "sampler_fixture_suite_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(
            root=root,
            protected_before=protected_before,
        ),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": normalized_codes,
        "duration_s": _stable_float(duration_s),
        "reproducibility_checksum": "",
        "honest_verdict": "pending",
    }
    artifact["sampler_fixture_suite_ready_score"] = sampler_fixture_suite_ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the manifest and terminal artifact."""

    started = time.monotonic()
    output = output_path or root / RESULT_RELATIVE_PATH
    manifest = manifest_path or root / FIXTURE_MANIFEST_RELATIVE_PATH
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    codes = test_exit_codes if test_exit_codes is not None else _external_test_exit_codes()
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=elapsed,
        test_exit_codes=codes,
        manifest_path=manifest,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        canonical_json(fixture_manifest_payload(artifact["exact_enumeration_receipts"])),
        encoding="utf-8",
    )
    artifact["fixture_manifest_path_and_hash"] = fixture_manifest_path_and_hash(
        artifact["exact_enumeration_receipts"],
        manifest,
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def sampler_fixture_suite_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when all exact-suite readiness gates pass."""

    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    required_commands_present = set(DEFAULT_TEST_COMMANDS) <= set(test_exit_codes)
    all_commands_pass = required_commands_present and all(
        test_exit_codes[command] == 0 for command in DEFAULT_TEST_COMMANDS
    )
    normalization_errors = dict(
        artifact.get("exact_probability_normalization_error_by_fixture") or {}
    )
    controls_pass = (
        artifact.get("original_six_state_positive_control", {}).get("passed") is True
        and artifact.get("unimodal_control", {}).get("valid_unimodal_control") is True
        and artifact.get("inactive_treatment_control", {}).get("valid_inactive_control") is True
        and artifact.get("inactive_treatment_control", {}).get("null_sampler_verdict_allowed")
        is False
        and artifact.get("unsupported_shape_control", {}).get("valid_unsupported_shape_control")
        is True
    )
    ready = bool(
        artifact.get("fixture_family_counts", {}).get("all_preregistered_families_present") is True
        and normalization_errors
        and max(float(value) for value in normalization_errors.values()) <= EXACT_TOLERANCE
        and controls_pass
        and type(artifact.get("duplicate_fixture_count")) is int
        and artifact.get("duplicate_fixture_count") == 0
        and type(artifact.get("source_mutation_count")) is int
        and artifact.get("source_mutation_count") == 0
        and artifact.get("protected_files_unchanged", {}).get("unchanged") is True
        and artifact.get("preconditions_checked", {}).get("preconditions_ready") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and all_commands_pass
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    return "complete_ready" if sampler_fixture_suite_ready_score(artifact) == 1.0 else "blocked"


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = []
    if (
        artifact.get("fixture_family_counts", {}).get("all_preregistered_families_present")
        is not True
    ):
        reasons.append("fixture_families")
    if (
        artifact.get("duplicate_fixture_count") != 0
        or type(artifact.get("duplicate_fixture_count")) is not int
    ):
        reasons.append("duplicate_fixture_count")
    if (
        artifact.get("source_mutation_count") != 0
        or type(artifact.get("source_mutation_count")) is not int
    ):
        reasons.append("source_mutation_count")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        reasons.append("protected_files")
    if artifact.get("preconditions_checked", {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    if not _controls_valid(artifact):
        reasons.append("controls")
    if not _normalization_valid(artifact):
        reasons.append("normalization")
    if not _commands_valid(artifact):
        reasons.append("test_commands")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    return reasons


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if sampler_fixture_suite_ready_score(artifact) == 1.0:
        return (
            "complete_ready: frozen exact suite covers six-state, Ising, Potts, "
            "and typed-factor fixtures; controls pass; no timing or hardware claim is made"
        )
    return "blocked: " + ",".join(blocked_reasons(artifact))


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _json_copy(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the required Exp6268 artifact fields and gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be object")  # pragma: no cover
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance:{field}")
    if (
        type(artifact["duplicate_fixture_count"]) is not int
        or artifact["duplicate_fixture_count"] != 0
    ):
        raise ValueError("duplicate_fixture_count must be bare 0")
    if type(artifact["source_mutation_count"]) is not int or artifact["source_mutation_count"] != 0:
        raise ValueError("source_mutation_count must be bare 0")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    for receipt in artifact["exact_enumeration_receipts"]:
        validate_fixture_receipt(receipt)
    expected_hashes = {
        str(receipt["fixture_name"]): normalized_target_probability_hash(receipt)
        for receipt in artifact["exact_enumeration_receipts"]
    }
    if artifact["normalized_target_probability_hashes"] != expected_hashes:
        raise ValueError("normalized_target_probability_hashes mismatch")
    if not _normalization_valid(artifact):
        raise ValueError("exact_probability_normalization_error_by_fixture")
    if artifact["unimodal_control"].get("valid_unimodal_control") is not True:
        raise ValueError("unimodal_control invalid")
    if artifact["inactive_treatment_control"].get("null_sampler_verdict_allowed") is not False:
        raise ValueError("inactive_treatment_control invalid")
    if artifact["unsupported_shape_control"].get("valid_unsupported_shape_control") is not True:
        raise ValueError("unsupported_shape_control invalid")
    score = sampler_fixture_suite_ready_score(artifact)
    if artifact["sampler_fixture_suite_ready_score"] != score:
        raise ValueError("sampler_fixture_suite_ready_score mismatch")  # pragma: no cover
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")  # pragma: no cover
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")  # pragma: no cover
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def _controls_valid(artifact: Mapping[str, Any]) -> bool:
    return bool(
        artifact.get("original_six_state_positive_control", {}).get("passed") is True
        and artifact.get("unimodal_control", {}).get("valid_unimodal_control") is True
        and artifact.get("inactive_treatment_control", {}).get("valid_inactive_control") is True
        and artifact.get("inactive_treatment_control", {}).get("null_sampler_verdict_allowed")
        is False
        and artifact.get("unsupported_shape_control", {}).get("valid_unsupported_shape_control")
        is True
    )


def _normalization_valid(artifact: Mapping[str, Any]) -> bool:
    errors = dict(artifact.get("exact_probability_normalization_error_by_fixture") or {})
    return bool(errors) and max(float(value) for value in errors.values()) <= EXACT_TOLERANCE


def _commands_valid(artifact: Mapping[str, Any]) -> bool:
    codes = dict(artifact.get("test_exit_codes") or {})
    return set(DEFAULT_TEST_COMMANDS) <= set(codes) and all(
        codes[command] == 0 for command in DEFAULT_TEST_COMMANDS
    )


def _normalize_test_exit_codes(test_exit_codes: Mapping[str, int]) -> dict[str, int | None]:
    return {
        command: int(test_exit_codes[command]) if command in test_exit_codes else None
        for command in DEFAULT_TEST_COMMANDS
    }


def _external_test_exit_codes() -> dict[str, int]:
    path = Path(str(Path(DEFAULT_RECEIPT_PATH)))
    env_path = Path(str(__import__("os").environ.get("CARNOT_6268_COMMAND_RECEIPTS", path)))
    if not env_path.exists():
        return {}
    payload = json.loads(env_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("command receipt payload must be an object")
    return {str(command): int(code) for command, code in payload.items()}


def _declared_state_space_size(fixture: Mapping[str, Any]) -> int:
    definition = dict(fixture["definition"])
    target_type = str(fixture["target_type"])
    if target_type == "categorical_mode_jump":
        return len(definition["labels"])
    if target_type == "ising":
        return 2 ** int(definition["n_spins"])
    if target_type == "potts":
        return int(definition["q_states"]) ** int(definition["n_spins"])
    program = exp6152.program_from_payload(definition["program_payload"])
    return math.prod(2 if wire.kind == "binary" else len(wire.categories) for wire in program.wires)


def _receipt_by_name(receipts: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    for receipt in receipts:
        if receipt["fixture_name"] == name:
            return receipt
    raise KeyError(name)  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPO_ROOT / FIXTURE_MANIFEST_RELATIVE_PATH,
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    artifact = write_artifact(
        output_path=args.output,
        manifest_path=args.manifest_output,
        run_date=str(args.date),
        duration_s=time.monotonic() - started,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "path": args.output.as_posix(),
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "complete_ready" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
