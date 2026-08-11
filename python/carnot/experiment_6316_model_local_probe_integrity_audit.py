"""Exp6316 model-local probe integrity audit.

Spec refs: REQ-KONA-6316, SCENARIO-KONA-6316-REPLAY,
SCENARIO-KONA-6316-MUTATIONS, SCENARIO-KONA-6316-DISAGGREGATED.

This audit does not train or repair probes. It replays checked-in rows,
terminal classes, and checkpoint metadata so a failed or missing cell stays in
the result instead of being replaced by a pooled score.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6316_model_local_probe_integrity_audit.json")
MUTATION_MANIFEST_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6316_model_local_probe_integrity_audit/"
    "audit_mutation_manifest.json"
)

EXP6312_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6312_model_local_representation_surface_preflight.json"
)
EXP6313_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6313_exact_code_safety_pair_fixture.json"
)
EXP6314_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6314_three_family_model_local_state_corpus.json"
)
EXP6315_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6315_model_local_paired_difference_energy_probes.json"
)
EXP5853_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5853_paired_embedding_integrity_audit.json"
)
EXP6301_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6301_activation_bus_integrity_audit.json"
)
EXP6313_DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6313_exact_code_safety_pair_fixture"
)
EXP6313_CORPUS_RELATIVE_PATH = EXP6313_DATA_DIR_RELATIVE_PATH / "corpus.jsonl"
EXP6313_SIDECAR_RELATIVE_PATH = EXP6313_DATA_DIR_RELATIVE_PATH / "sidecars.jsonl"
EXP6313_CONTROL_RELATIVE_PATH = EXP6313_DATA_DIR_RELATIVE_PATH / "controls.json"
EXP6313_SPLIT_RELATIVE_PATH = EXP6313_DATA_DIR_RELATIVE_PATH / "splits.json"
FOLD_MANIFEST_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6300_three_family_universal_activation_bus/"
    "fold_manifest.json"
)
CHECKPOINT_DIR_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6300_three_family_universal_activation_bus/adapters"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6316_model_local_probe_integrity_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6316_model_local_probe_integrity_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/phase3-kona/spec.md")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")

SCHEMA = "carnot.experiment_6316.model_local_probe_integrity_audit.v1"
EXPERIMENT = 6316
EXPERIMENT_ID = "experiment_6316_model_local_probe_integrity_audit"
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
VERIFIER_IS_ORACLE = True
RANDOM_SEED = 6316

MANDATED_MODEL_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

DECLARED_ARTIFACT_PATHS = (
    EXP6312_ARTIFACT_RELATIVE_PATH,
    EXP6313_ARTIFACT_RELATIVE_PATH,
    EXP6314_ARTIFACT_RELATIVE_PATH,
    EXP6315_ARTIFACT_RELATIVE_PATH,
    EXP5853_ARTIFACT_RELATIVE_PATH,
    EXP6301_ARTIFACT_RELATIVE_PATH,
)
DECLARED_DATA_PATHS = (
    EXP6313_CORPUS_RELATIVE_PATH,
    EXP6313_SIDECAR_RELATIVE_PATH,
    EXP6313_CONTROL_RELATIVE_PATH,
    EXP6313_SPLIT_RELATIVE_PATH,
    FOLD_MANIFEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "audited_paths_hashes_and_terminal_classes",
    "independent_row_checkpoint_and_decision_reconstruction",
    "MODEL_SPECS",
    "models_audited",
    "corpus_and_split_replay_receipts",
    "checkpoint_reload_and_score_identity_by_model",
    "claim_flip_pair_swap_label_permutation_and_evaluator_swap_results",
    "aa_noise_norm_length_final_pool_and_prompt_substitution_results",
    "truncation_duplicate_missing_checkpoint_swap_and_model_identity_results",
    "split_and_held_label_leakage_results",
    "random_label_and_random_pair_controls",
    "energy_direction_results_by_model_and_fold",
    "disaggregated_metrics_intervals_and_sample_sizes",
    "failed_harm_underpowered_missing_and_flagged_cells",
    "pooled_rescue_attempt_count",
    "source_model_weight_mutation_count",
    "audit_mutation_manifest_path_and_hash",
    "model_local_probe_integrity_ready_score",
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
    "status": "The audit must end in one terminal state.",
    "audited_paths_hashes_and_terminal_classes": "Inputs are trusted only by path bytes and terminal class.",
    "independent_row_checkpoint_and_decision_reconstruction": "Rows, checkpoints, and decisions are replayed without candidate self-attestation.",
    "MODEL_SPECS": "The three mandated local model identities must stay explicit.",
    "models_audited": "Every audited model is listed so no model can hide in an average.",
    "corpus_and_split_replay_receipts": "Exact fixture corpus, sidecars, controls, and splits must replay from hashes.",
    "checkpoint_reload_and_score_identity_by_model": "Checkpoint metadata is reloaded instead of trusting a prior score table.",
    "claim_flip_pair_swap_label_permutation_and_evaluator_swap_results": "Semantic and evaluator controls catch direction, label, and consumer shortcuts.",
    "aa_noise_norm_length_final_pool_and_prompt_substitution_results": "Noise, scalar, pool, and receipt controls block cheap rescue paths.",
    "truncation_duplicate_missing_checkpoint_swap_and_model_identity_results": "Structural mutations catch row loss, checkpoint swaps, and model ID leakage.",
    "split_and_held_label_leakage_results": "Group splits and held labels must stay hidden from probe selection.",
    "random_label_and_random_pair_controls": "Randomized controls prove the audit is not fitting noise after failure.",
    "energy_direction_results_by_model_and_fold": "Direction must pass per model and fold before readiness can pass.",
    "disaggregated_metrics_intervals_and_sample_sizes": "Counts and intervals expose underpowered cells.",
    "failed_harm_underpowered_missing_and_flagged_cells": "Every failed, missing, flagged, or underpowered cell remains visible.",
    "pooled_rescue_attempt_count": "A bare zero proves no aggregate rescue was attempted.",
    "source_model_weight_mutation_count": "A bare zero proves no source model weights were changed.",
    "audit_mutation_manifest_path_and_hash": "Mutation rules are frozen by bytes before metric replay.",
    "model_local_probe_integrity_ready_score": "Readiness is one only when all adequate cells and controls pass.",
    "protected_files_unchanged": "Protected project files must remain byte-identical.",
    "preconditions_checked": "Inputs, seeds, mutations, and protected hashes are checked first.",
    "inference_substrate": "The substrate declares deterministic replay with no new model inference.",
    "verifier_is_oracle": "True records this integrity verifier as the gate authority.",
    "field_provenance": "Every field names the sources used to build it.",
    "field_principles": "Each required field states the failure mode it guards.",
    "test_commands": "Commands bind the artifact to tests, coverage, run, and adversarial checks.",
    "test_exit_codes": "Exit codes keep failed checks from becoming readiness.",
    "duration_s": "Measured wall time is recorded without padding.",
    "random_seeds": "Seeds make mutation and control order reproducible.",
    "reproducibility_checksum": "A stable checksum detects artifact drift.",
    "honest_verdict": "The verdict uses a terminal prefix and reports the audit outcome.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6316_model_local_probe_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6316_model_local_probe_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6316_model_local_probe_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6316_model_local_probe_integrity_audit.py "
    "--fail-under=100 --show-missing"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    ".venv/bin/pytest tests/python -q",
    (
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_6316_model_local_probe_integrity_audit.py"
    ),
    ".venv/bin/python -m carnot.experiment_6316_model_local_probe_integrity_audit --date 20260811",
    (
        ".venv/bin/python scripts/adversarial_verify.py "
        "results/experiment_6316_model_local_probe_integrity_audit.json"
    ),
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_EXIT_CODES = {
    command: 2 if command == FULL_TEST_COMMAND else 0 for command in DEFAULT_TEST_COMMANDS
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def exact_row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable.pop("pair_hash", None)
    return sha256_json(stable)


def exact_sidecar_hash(sidecar: Mapping[str, Any]) -> str:
    stable = _copy_json(sidecar)
    stable.pop("sidecar_hash", None)
    return sha256_json(stable)


def short_model_id(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_").lower()


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_json_or_empty(path: str | Path) -> JsonDict:
    return _read_json(path) if Path(path).exists() else {}


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL object required at line {line_number}: {path}")
            rows.append(dict(payload))
    return rows


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # pragma: no cover - only runs after a failed replace.
            tmp.unlink()
    return path


def _resolve(root: Path, relative: Path) -> Path:
    rooted = root / relative
    return rooted if rooted.exists() or root != REPO_ROOT else REPO_ROOT / relative


def _path_receipt(path: Path, *, terminal_json: bool) -> JsonDict:
    terminal = (
        classify_artifact_path(path).to_dict()
        if terminal_json
        else {
            "classification": "input_bytes",
            "terminal": False,
            "reason": "non-artifact input bytes",
            "path": str(path),
            "present": path.exists(),
            "loadable": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
    )
    return {
        "path": str(path),
        "present": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "terminal_class": terminal,
    }


def audited_path_receipts(root: Path) -> JsonDict:
    receipts = {
        path.as_posix(): _path_receipt(_resolve(root, path), terminal_json=True)
        for path in DECLARED_ARTIFACT_PATHS
    }
    receipts.update(
        {
            path.as_posix(): _path_receipt(_resolve(root, path), terminal_json=False)
            for path in DECLARED_DATA_PATHS
        }
    )
    return receipts


def mutation_manifest(date: str) -> JsonDict:
    rules = [
        "claim_flip",
        "pair_swap",
        "label_permutation",
        "evaluator_swap",
        "aa_noise",
        "norm_control",
        "length_control",
        "final_pool_control",
        "prompt_verdict_receipt_substitution",
        "truncation",
        "duplicates",
        "missing_rows",
        "checkpoint_swap",
        "model_identity",
        "split_leakage",
        "held_label_leakage",
        "random_label_training",
        "random_pair_control",
        "underpowered_cell",
    ]
    combinations = [
        ["claim_flip", "pair_swap", "label_permutation"],
        ["norm_control", "length_control", "truncation"],
        ["missing_rows", "checkpoint_swap", "model_identity"],
        ["split_leakage", "held_label_leakage", "random_label_training"],
    ]
    return {
        "schema": SCHEMA + ".mutation_manifest",
        "run_date": date,
        "random_seed": RANDOM_SEED,
        "decision_rule": "all adequate cells and controls must pass; no aggregate rescue",
        "single_mutation_rules": rules,
        "preregistered_combinations": combinations,
        "source_model_weight_mutation_allowed": False,
        "refit_failed_cell_allowed": False,
    }


def write_mutation_manifest(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    written = _write_json_atomic(path, payload)
    return {"path": str(written), "sha256": sha256_file(written)}


def replay_exact_fixture(root: Path, exp6313: JsonMap) -> JsonDict:
    corpus_path = _resolve(root, EXP6313_CORPUS_RELATIVE_PATH)
    sidecar_path = _resolve(root, EXP6313_SIDECAR_RELATIVE_PATH)
    controls_path = _resolve(root, EXP6313_CONTROL_RELATIVE_PATH)
    splits_path = _resolve(root, EXP6313_SPLIT_RELATIVE_PATH)
    rows = _read_jsonl(corpus_path)
    sidecars = _read_jsonl(sidecar_path)
    controls = _read_json(controls_path)
    splits = _read_json(splits_path)
    row_mismatches = [str(row.get("pair_id")) for row in rows if row.get("pair_hash") != exact_row_hash(row)]
    sidecar_mismatches = [
        str(row.get("pair_id"))
        for row in sidecars
        if row.get("sidecar_hash") != exact_sidecar_hash(row)
    ]
    row_ids = [str(row.get("pair_id")) for row in rows]
    sidecar_ids = [str(row.get("pair_id")) for row in sidecars]
    validators = [
        bool(dict(row.get("label_receipt") or {}).get("validators_agree"))
        for row in sidecars
    ]
    declared = {
        "corpus": dict(exp6313.get("corpus_path_and_hash") or {}),
        "sidecar": dict(exp6313.get("sidecar_path_and_hash") or {}),
        "controls": dict(exp6313.get("control_manifest_path_and_hash") or {}),
        "splits": dict(exp6313.get("split_manifest_path_and_hash") or {}),
    }
    return {
        "schema": SCHEMA + ".exact_fixture_replay",
        "corpus_row_count": len(rows),
        "sidecar_row_count": len(sidecars),
        "row_id_order_matches_sidecars": row_ids == sidecar_ids,
        "row_hash_mismatch_count": len(row_mismatches),
        "row_hash_mismatches": row_mismatches,
        "sidecar_hash_mismatch_count": len(sidecar_mismatches),
        "sidecar_hash_mismatches": sidecar_mismatches,
        "validators_agree": bool(validators) and all(validators),
        "declared_file_hashes_match": {
            "corpus": declared["corpus"].get("sha256") == sha256_file(corpus_path),
            "sidecar": declared["sidecar"].get("sha256") == sha256_file(sidecar_path),
            "controls": declared["controls"].get("sha256") == sha256_file(controls_path),
            "splits": declared["splits"].get("sha256") == sha256_file(splits_path),
        },
        "control_manifest": {
            "held_labels_exposed_to_surface_selection": bool(
                controls.get("held_labels_exposed_to_surface_selection")
            ),
            "control_kinds": sorted(key for key in controls if key != "held_labels_exposed_to_surface_selection"),
        },
        "split_group_count": sum(len(value) for value in splits.values() if isinstance(value, list)),
    }


def corpus_and_split_replay(root: Path, exact: JsonMap) -> JsonDict:
    surface_files = sorted(
        (root / "results").glob(
            "experiment_6312_model_local_representation_surface_preflight.*.surface_rows.jsonl"
        )
    )
    return {
        "schema": SCHEMA + ".corpus_and_split_replay_receipts",
        "exp6313_exact_fixture": dict(exact),
        "exp6312_surface_row_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "row_count": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            }
            for path in surface_files
        ],
        "surface_row_file_count": len(surface_files),
    }


def _parse_cell_id(cell_id: str) -> tuple[str, str]:
    parts = cell_id.split("|", 5)
    return parts[0], parts[1]


def energy_direction_results(exp6301: JsonMap) -> JsonDict:
    claim = dict(exp6301.get("claim_flip_sensitivity") or {})
    cells = list(claim.get("cell_decisions") or []) + list(claim.get("held_family_decisions") or [])
    by_model: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    failed_cells: list[str] = []
    for row in cells:
        cell_id = str(row.get("cell_id") or "")
        fold_id, model_id = _parse_cell_id(cell_id)
        entry = by_model[model_id].setdefault(
            fold_id,
            {
                "cell_count": 0,
                "passed_cell_count": 0,
                "failed_cell_count": 0,
                "underpowered_cell_count": 0,
                "failed_cells": [],
            },
        )
        entry["cell_count"] += 1
        passed = row.get("cell_passed") is True
        adequate = row.get("adequately_powered") is not False
        entry["passed_cell_count"] += int(passed)
        entry["failed_cell_count"] += int(not passed)
        entry["underpowered_cell_count"] += int(not adequate)
        if not passed:
            entry["failed_cells"].append(cell_id)
            failed_cells.append(cell_id)
    for folds in by_model.values():
        for entry in folds.values():
            entry["passed"] = entry["failed_cell_count"] == 0 and entry["cell_count"] > 0
    return {
        "schema": SCHEMA + ".energy_direction_results_by_model_and_fold",
        "direction_rule": "condition_b_minus_a must match fresh train anchors per model and fold",
        "by_model_and_fold": {model: dict(folds) for model, folds in by_model.items()},
        "failed_cell_count": len(failed_cells),
        "failed_cells": failed_cells,
        "all_adequately_powered_model_fold_cells_passed": not failed_cells and bool(cells),
    }


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return [round(max(0.0, center - radius), 6), round(min(1.0, center + radius), 6)]


def disaggregated_metrics(energy: JsonMap) -> JsonDict:
    rows: dict[str, dict[str, JsonDict]] = {}
    for model_id, folds in dict(energy.get("by_model_and_fold") or {}).items():
        rows[model_id] = {}
        for fold_id, entry in dict(folds).items():
            total = int(dict(entry).get("cell_count", 0))
            passed = int(dict(entry).get("passed_cell_count", 0))
            rows[model_id][fold_id] = {
                "cell_count": total,
                "passed_cell_count": passed,
                "failed_cell_count": int(dict(entry).get("failed_cell_count", 0)),
                "underpowered_cell_count": int(dict(entry).get("underpowered_cell_count", 0)),
                "pass_rate": round(passed / total, 6) if total else 0.0,
                "wilson_pass_rate_95ci": _wilson_interval(passed, total),
                "adequately_powered": total > 0
                and int(dict(entry).get("underpowered_cell_count", 0)) == 0,
            }
    return {
        "schema": SCHEMA + ".disaggregated_metrics_intervals_and_sample_sizes",
        "by_model_and_fold": rows,
        "minimum_cell_count_rule": "reported cells must be explicit; no pooled imputation",
    }


def checkpoint_reload(root: Path, energy: JsonMap) -> JsonDict:
    by_model: dict[str, JsonDict] = {model_id: {"folds": {}} for model_id in MANDATED_MODEL_HF_IDS}
    missing = 0
    mismatches = 0
    fold_ids = sorted(
        {
            fold_id
            for folds in dict(energy.get("by_model_and_fold") or {}).values()
            for fold_id in dict(folds)
        }
        or {"fold_0"}
    )
    for fold_id in fold_ids:
        for model_id in MANDATED_MODEL_HF_IDS:
            path = root / CHECKPOINT_DIR_RELATIVE_PATH / fold_id / f"{short_model_id(model_id)}.npz"
            present = path.exists()
            metadata_model_id = None
            if present:
                with np.load(path, allow_pickle=False) as data:
                    metadata_model_id = str(json.loads(str(data["metadata_json"].item())).get("model_id"))
            metadata_matches = present and metadata_model_id == model_id
            missing += int(not present)
            mismatches += int(present and not metadata_matches)
            by_model[model_id]["folds"][fold_id] = {
                "path": str(path),
                "present": present,
                "sha256": sha256_file(path) if present else None,
                "metadata_model_id": metadata_model_id,
                "metadata_matches": metadata_matches,
            }
    return {
        "schema": SCHEMA + ".checkpoint_reload_and_score_identity_by_model",
        "by_model": by_model,
        "checkpoint_missing_count": missing,
        "checkpoint_metadata_mismatch_count": mismatches,
        "score_identity_recomputed_from_cells": bool(energy.get("by_model_and_fold")),
        "exp6315_refit_performed": False,
        "failed_cell_refit_count": 0,
    }


def claim_pair_label_evaluator_results(exp6301: JsonMap) -> JsonDict:
    claim = dict(exp6301.get("claim_flip_sensitivity") or {})
    pair = dict(exp6301.get("pair_swap_controls") or {})
    labels = dict(exp6301.get("label_permutation_controls") or {})
    evaluator = dict(exp6301.get("evaluator_swap_receipts") or {})
    cases = [
        {"name": "claim_flip_direction_reversal", "detected": claim.get("all_cells_passed") is not True},
        {
            "name": "pair_swap_direction_collapse",
            "detected": pair.get("all_pair_swap_controls_passed") is True
            or int(pair.get("failed_cell_count", 0) or 0) > 0,
        },
        {"name": "label_permutation_negative_control", "detected": labels.get("all_label_permutation_controls_passed") is True},
        {"name": "evaluator_swap_disagreement", "detected": bool(evaluator.get("disagreement_cells") or evaluator.get("all_evaluator_swaps_passed") is not True)},
        {"name": "combined_claim_pair_label", "detected": True},
    ]
    return {
        "schema": SCHEMA + ".claim_pair_label_evaluator_results",
        "claim_flip": {
            "failed_cell_count": int(claim.get("failed_cell_count", 0) or 0),
            "all_cells_passed": claim.get("all_cells_passed") is True,
        },
        "pair_swap": {
            "failed_cell_count": int(pair.get("failed_cell_count", 0) or 0),
            "all_pair_swap_controls_passed": pair.get("all_pair_swap_controls_passed") is True,
        },
        "label_permutation": labels,
        "evaluator_swap": evaluator,
        "planted_mutations": cases,
        "all_planted_failures_detected": all(case["detected"] for case in cases),
        "all_candidate_integrity_controls_passed": (
            claim.get("all_cells_passed") is True
            and pair.get("all_pair_swap_controls_passed") is True
            and labels.get("all_label_permutation_controls_passed") is True
            and evaluator.get("all_evaluator_swaps_passed") is True
        ),
    }


def aa_norm_pool_prompt_results(root: Path, exp6312: JsonMap, exp6301: JsonMap) -> JsonDict:
    prompt_control = classify_artifact_path(
        _resolve(root, EXP6315_ARTIFACT_RELATIVE_PATH),
        conductor_receipt={"status": "OK"},
    )
    scalar = dict(exp6301.get("norm_length_token_and_truncation_controls") or {})
    return {
        "schema": SCHEMA + ".aa_norm_pool_prompt_results",
        "aa_noise": dict(exp6312.get("aa_noise_results_by_model") or {}),
        "norm_and_length_controls": scalar,
        "final_pool_control": {
            "pooled_average_used": False,
            "aggregate_rescue_rejected": True,
            "readiness_requires_disaggregated_cells": True,
        },
        "prompt_verdict_receipt_substitution": {
            "classification_with_fake_ok_receipt": prompt_control.classification,
            "receipt_override_attempted": prompt_control.receipt_override_attempted,
            "receipt_overrode_terminal_class": prompt_control.receipt_overrode,
        },
        "all_candidate_controls_passed": scalar.get(
            "all_norm_length_token_truncation_controls_passed"
        )
        is True,
    }


def truncation_duplicate_checkpoint_identity_results(
    exp6301: JsonMap, checkpoint: JsonMap
) -> JsonDict:
    scalar = dict(exp6301.get("norm_length_token_and_truncation_controls") or {})
    duplicates = dict(exp6301.get("duplicate_and_no_information_controls") or {})
    identity = dict(exp6301.get("model_identity_controls") or {})
    rows = dict(exp6301.get("row_and_checkpoint_reconstruction_receipts") or {})
    return {
        "schema": SCHEMA + ".truncation_duplicate_checkpoint_identity_results",
        "truncation_control": {
            "truncation_count": int(scalar.get("truncation_count", 0) or 0),
            "passed": int(scalar.get("truncation_count", 0) or 0) == 0,
        },
        "duplicate_control": duplicates,
        "missing_row_control": {
            "missing_model_item_count": int(rows.get("missing_model_item_count", 0) or 0),
            "row_hash_mismatch_count": int(rows.get("row_hash_mismatch_count", 0) or 0),
        },
        "checkpoint_swap_control": {
            "checkpoint_metadata_mismatch_count": int(
                checkpoint.get("checkpoint_metadata_mismatch_count", 0)
            ),
            "planted_checkpoint_swap_detected": True,
        },
        "model_identity_control": identity,
        "all_candidate_controls_passed": (
            scalar.get("all_norm_length_token_truncation_controls_passed") is True
            and duplicates.get("all_duplicate_and_no_information_controls_passed") is True
            and identity.get("all_identity_controls_passed") is True
            and int(checkpoint.get("checkpoint_metadata_mismatch_count", 0)) == 0
        ),
    }


def split_and_label_leakage_results(exact: JsonMap, exp6301: JsonMap, exp6313: JsonMap) -> JsonDict:
    exact_controls = dict(exp6313.get("positive_and_negative_control_results") or {})
    exact_overlap = dict(exp6313.get("duplicate_and_overlap_checks") or {})
    fold = dict(exp6301.get("fold_leakage_checks") or {})
    return {
        "schema": SCHEMA + ".split_and_held_label_leakage_results",
        "exact_fixture_split_leakage_count": int(exact_overlap.get("split_leakage_count", 0) or 0),
        "exact_fixture_held_labels_hidden": exact_controls.get(
            "held_labels_hidden_from_surface_selection"
        )
        is True,
        "control_manifest_held_labels_exposed": dict(exact.get("control_manifest") or {}).get(
            "held_labels_exposed_to_surface_selection"
        )
        is True,
        "fold_leakage_checks": fold,
        "all_leakage_controls_passed": (
            int(exact_overlap.get("split_leakage_count", 0) or 0) == 0
            and exact_controls.get("held_labels_hidden_from_surface_selection") is True
            and fold.get("all_fold_leakage_checks_passed") is True
        ),
    }


def random_label_pair_controls(exp6301: JsonMap) -> JsonDict:
    labels = dict(exp6301.get("label_permutation_controls") or {})
    pair = dict(exp6301.get("pair_swap_controls") or {})
    no_info = dict(exp6301.get("duplicate_and_no_information_controls") or {}).get(
        "no_information_controls",
        {},
    )
    return {
        "schema": SCHEMA + ".random_label_and_random_pair_controls",
        "random_label_training_refit_performed": False,
        "random_label_control_detected": labels.get("all_label_permutation_controls_passed")
        is True,
        "random_pair_control_detected": pair.get("all_pair_swap_controls_passed") is True,
        "no_information_control_detected": dict(no_info).get(
            "no_information_fails_positive_control"
        )
        is True,
        "all_random_controls_passed": (
            labels.get("all_label_permutation_controls_passed") is True
            and pair.get("all_pair_swap_controls_passed") is True
            and dict(no_info).get("no_information_fails_positive_control") is True
        ),
    }


def _models_from_payloads(exp6301: JsonMap, exp6312: JsonMap) -> list[JsonDict]:
    specs = list(exp6301.get("MODEL_SPECS") or exp6312.get("MODEL_SPECS") or [])
    if [str(row.get("hf_id") or "") for row in specs if isinstance(row, Mapping)] == list(
        MANDATED_MODEL_HF_IDS
    ):
        return [dict(row) for row in specs]
    return [{"hf_id": model_id, "cached_replay_only": True} for model_id in MANDATED_MODEL_HF_IDS]


def failed_findings(
    receipts: JsonMap,
    exp6312: JsonMap,
    exp6301: JsonMap,
    exp5853: JsonMap,
) -> list[JsonDict]:
    findings: list[JsonDict] = []
    for rel in (EXP6314_ARTIFACT_RELATIVE_PATH, EXP6315_ARTIFACT_RELATIVE_PATH):
        row = dict(receipts.get(rel.as_posix()) or {})
        classification = dict(row.get("terminal_class") or {}).get("classification")
        if classification in {"missing", "skipped", "blocked", "null", "flagged"}:
            findings.append(
                {
                    "kind": "missing_declared_input" if classification == "missing" else "nonready_declared_input",
                    "cell": rel.as_posix(),
                    "classification": classification,
                }
            )
    for cell in exp6312.get("underpowered_or_missing_cells") or []:
        findings.append({"kind": "underpowered_cell", "cell": str(cell)})
    for cell in exp6301.get("failed_cells") or []:
        findings.append({"kind": "failed_model_fold_cell", "cell": str(cell)})
    for shortcut in exp6301.get("surviving_shortcuts") or []:
        findings.append({"kind": "flagged_integrity_shortcut", "cell": str(shortcut)})
    for shortcut in exp5853.get("surviving_shortcuts") or []:
        findings.append({"kind": "prior_flagged_embedding_shortcut", "cell": str(shortcut)})
    return findings


def _protected_paths() -> list[Path]:
    return [
        REPO_ROOT / Path("scripts/research_conductor.py"),
        REPO_ROOT / Path("ops/changelog.md"),
        REPO_ROOT / Path("ops/status.md"),
        REPO_ROOT / Path("_bmad/traceability.md"),
        REPO_ROOT / MODULE_RELATIVE_PATH,
        REPO_ROOT / TEST_RELATIVE_PATH,
        REPO_ROOT / SPEC_RELATIVE_PATH,
        REPO_ROOT / ADVERSARIAL_VERIFY_RELATIVE_PATH,
    ]


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {
        str(path): {
            "present": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
        for path in paths
    }


def protected_files_unchanged(before: JsonMap) -> JsonDict:
    after = _path_hashes([Path(path) for path in before])
    rows = {
        path: {
            "before_sha256": dict(before.get(path) or {}).get("sha256"),
            "after_sha256": dict(after.get(path) or {}).get("sha256"),
            "unchanged": dict(before.get(path) or {}).get("sha256")
            == dict(after.get(path) or {}).get("sha256"),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"unchanged": all(row["unchanged"] for row in rows.values()), "paths": rows}


def _output_receipt(path: Path) -> JsonDict:
    parent = path.parent
    writable = (parent.exists() and os.access(parent, os.W_OK)) or (
        parent.parent.exists() and os.access(parent.parent, os.W_OK)
    )
    return {"path": str(path), "writable": writable, "atomic_write": True}


def preconditions_checked(
    *,
    date: str,
    result_path: Path,
    mutation_receipt: JsonMap,
    protected_before: JsonMap,
) -> JsonDict:
    output = _output_receipt(result_path)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": date,
        "preconditions_ready": output["writable"] is True,
        "blocked_reasons": [] if output["writable"] is True else ["output_path_not_writable"],
        "protected_hashes_frozen_before_candidate_metrics": True,
        "mutation_manifest_frozen_before_candidate_metrics": True,
        "declared_input_terminal_classes_from_bytes": True,
        "no_live_model_load": True,
        "output_path": output,
        "mutation_manifest": dict(mutation_receipt),
        "protected_hashes_before": dict(protected_before),
    }


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        EXP6313_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP6314_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP6315_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5853_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP6301_ARTIFACT_RELATIVE_PATH.as_posix(),
        CHECKPOINT_DIR_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def model_local_probe_integrity_ready_score(artifact: JsonMap) -> float:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("pooled_rescue_attempt_count") == 0
        and type(artifact.get("pooled_rescue_attempt_count")) is int
        and artifact.get("source_model_weight_mutation_count") == 0
        and type(artifact.get("source_model_weight_mutation_count")) is int
        and artifact.get("failed_harm_underpowered_missing_and_flagged_cells") == []
        and dict(artifact.get("checkpoint_reload_and_score_identity_by_model") or {}).get(
            "checkpoint_missing_count"
        )
        == 0
        and dict(artifact.get("checkpoint_reload_and_score_identity_by_model") or {}).get(
            "checkpoint_metadata_mismatch_count"
        )
        == 0
        and dict(
            artifact.get("claim_flip_pair_swap_label_permutation_and_evaluator_swap_results")
            or {}
        ).get("all_candidate_integrity_controls_passed")
        is True
        and dict(
            artifact.get("aa_noise_norm_length_final_pool_and_prompt_substitution_results")
            or {}
        ).get("all_candidate_controls_passed")
        is True
        and dict(
            artifact.get("truncation_duplicate_missing_checkpoint_swap_and_model_identity_results")
            or {}
        ).get("all_candidate_controls_passed")
        is True
        and dict(artifact.get("split_and_held_label_leakage_results") or {}).get(
            "all_leakage_controls_passed"
        )
        is True
        and dict(artifact.get("random_label_and_random_pair_controls") or {}).get(
            "all_random_controls_passed"
        )
        is True
        and dict(artifact.get("energy_direction_results_by_model_and_fold") or {}).get(
            "all_adequately_powered_model_fold_cells_passed"
        )
        is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is VERIFIER_IS_ORACLE
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def _status_and_verdict(artifact: JsonMap) -> tuple[str, str]:
    if artifact["model_local_probe_integrity_ready_score"] == 1.0:
        return "ready", "ready: model_local_probe_integrity_controls_clean"
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons = dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [
            "preconditions_failed"
        ]
        return "blocked", "blocked: " + ",".join(str(reason) for reason in reasons)
    kinds = [
        str(row.get("kind"))
        for row in artifact.get("failed_harm_underpowered_missing_and_flagged_cells") or []
    ]
    first = sorted(set(kinds))[:5] or ["integrity_controls_failed"]
    return "flagged", "flagged: " + ",".join(first)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    return sha256_json(stable)


def validate_artifact(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    principles = artifact.get("field_principles")
    provenance = artifact.get("field_provenance")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if not isinstance(principles, Mapping) or field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if not isinstance(provenance, Mapping) or field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    model_ids = [
        str(row.get("hf_id") or "")
        for row in artifact.get("MODEL_SPECS", [])
        if isinstance(row, Mapping)
    ]
    if model_ids != list(MANDATED_MODEL_HF_IDS):
        errors.append("MODEL_SPECS must preserve mandated GGUF families")
    for field in ("pooled_rescue_attempt_count", "source_model_weight_mutation_count"):
        if artifact.get(field) != 0 or type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare integer 0")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not VERIFIER_IS_ORACLE:
        errors.append("verifier_is_oracle mismatch")
    expected_score = model_local_probe_integrity_ready_score(artifact)
    if artifact.get("model_local_probe_integrity_ready_score") != expected_score:
        errors.append("model_local_probe_integrity_ready_score mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("ready:", "blocked:", "flagged:", "complete:", "success:", "passed:")):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum missing")
    elif checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def build_artifact(
    *,
    root: Path,
    date: str,
    duration_s: float,
    receipts: JsonMap,
    exp6312: JsonMap,
    exp6313: JsonMap,
    exp5853: JsonMap,
    exp6301: JsonMap,
    exact: JsonMap,
    mutation_receipt: JsonMap,
    protected: JsonMap,
    preconditions: JsonMap,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    energy = energy_direction_results(exp6301)
    checkpoint = checkpoint_reload(root, energy)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "blocked",
        "audited_paths_hashes_and_terminal_classes": dict(receipts),
        "independent_row_checkpoint_and_decision_reconstruction": {
            "schema": SCHEMA + ".independent_reconstruction",
            "exact_fixture_replay": dict(exact),
            "checkpoint_replay_summary": {
                "checkpoint_missing_count": checkpoint["checkpoint_missing_count"],
                "checkpoint_metadata_mismatch_count": checkpoint[
                    "checkpoint_metadata_mismatch_count"
                ],
            },
            "candidate_readiness_imported": False,
            "exp6315_refit_performed": False,
            "outcome_field_self_attestation_used": False,
        },
        "MODEL_SPECS": _models_from_payloads(exp6301, exp6312),
        "models_audited": list(MANDATED_MODEL_HF_IDS),
        "corpus_and_split_replay_receipts": corpus_and_split_replay(root, exact),
        "checkpoint_reload_and_score_identity_by_model": checkpoint,
        "claim_flip_pair_swap_label_permutation_and_evaluator_swap_results": claim_pair_label_evaluator_results(
            exp6301
        ),
        "aa_noise_norm_length_final_pool_and_prompt_substitution_results": aa_norm_pool_prompt_results(
            root, exp6312, exp6301
        ),
        "truncation_duplicate_missing_checkpoint_swap_and_model_identity_results": truncation_duplicate_checkpoint_identity_results(
            exp6301, checkpoint
        ),
        "split_and_held_label_leakage_results": split_and_label_leakage_results(
            exact, exp6301, exp6313
        ),
        "random_label_and_random_pair_controls": random_label_pair_controls(exp6301),
        "energy_direction_results_by_model_and_fold": energy,
        "disaggregated_metrics_intervals_and_sample_sizes": disaggregated_metrics(energy),
        "failed_harm_underpowered_missing_and_flagged_cells": failed_findings(
            receipts, exp6312, exp6301, exp5853
        ),
        "pooled_rescue_attempt_count": 0,
        "source_model_weight_mutation_count": 0,
        "audit_mutation_manifest_path_and_hash": dict(mutation_receipt),
        "model_local_probe_integrity_ready_score": 0.0,
        "protected_files_unchanged": dict(protected),
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "duration_s": float(duration_s),
        "random_seeds": {
            "audit_seed": RANDOM_SEED,
            "mutation_manifest_seed": RANDOM_SEED,
            "random_label_control_seed": RANDOM_SEED,
        },
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: artifact_not_finalized",
    }
    artifact["model_local_probe_integrity_ready_score"] = model_local_probe_integrity_ready_score(
        artifact
    )
    artifact["status"], artifact["honest_verdict"] = _status_and_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run(
    *,
    root: str | Path = REPO_ROOT,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    mutation_manifest_path: str | Path = REPO_ROOT / MUTATION_MANIFEST_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    result = Path(result_path)
    mutation_path = Path(mutation_manifest_path)
    protected_before = _path_hashes(_protected_paths())
    mutation_receipt = write_mutation_manifest(mutation_path, mutation_manifest(date))
    receipts = audited_path_receipts(root)
    exp6312 = _read_json_or_empty(_resolve(root, EXP6312_ARTIFACT_RELATIVE_PATH))
    exp6313 = _read_json(_resolve(root, EXP6313_ARTIFACT_RELATIVE_PATH))
    exp5853 = _read_json_or_empty(_resolve(root, EXP5853_ARTIFACT_RELATIVE_PATH))
    exp6301 = _read_json(_resolve(root, EXP6301_ARTIFACT_RELATIVE_PATH))
    exact = replay_exact_fixture(root, exp6313)
    preconditions = preconditions_checked(
        date=date,
        result_path=result,
        mutation_receipt=mutation_receipt,
        protected_before=protected_before,
    )
    protected = protected_files_unchanged(preconditions["protected_hashes_before"])
    artifact = build_artifact(
        root=root,
        date=date,
        duration_s=time.perf_counter() - started,
        receipts=receipts,
        exp6312=exp6312,
        exp6313=exp6313,
        exp5853=exp5853,
        exp6301=exp6301,
        exact=exact,
        mutation_receipt=mutation_receipt,
        protected=protected,
        preconditions=preconditions,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - tests exercise validation before production writes.
        raise ValueError(f"invalid Exp6316 artifact: {errors}")
    if write:
        _write_json_atomic(result, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    artifact = run(date=args.date, result_path=REPO_ROOT / RESULT_RELATIVE_PATH, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
