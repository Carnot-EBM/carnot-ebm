"""Exp5745 lossless scalar gate corrigendum for Exp5740.

This module repairs a schema mismatch, not a scientific result. It verifies the
checked-in Exp5740 bytes and trace receipts, copies retained primitive evidence
by content hash, separates rejected leak canaries from admitted leakage, and
emits the scalar fields expected by the downstream conductor gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5745_arc_causal_gate_schema_corrigendum.json")
SOURCE_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5740_arc_game_blind_primitive_causal_audit.json"
)
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

SOURCE_SCHEMA_VERSION = "carnot.experiment_5740.game_blind_primitive_causal_audit.legacy_object_gate.v1"
NORMALIZED_SCHEMA_VERSION = "carnot.experiment_5745.arc_causal_gate_schema_corrigendum.v1"
EXPECTED_SOURCE_ARTIFACT_HASH = (
    "sha256:c38b1f700a8c253c87ef69a90571836a94383f84c53fb7708b89a795650400e4"
)
EXPECTED_FROZEN_EFFECT_HASH = (
    "sha256:669d865183509414339a7406c16dbf7c332c78e35906045244078d7a48f6949b"
)
EXPECTED_ORIGINAL_COVERAGE_HASH = (
    "sha256:015e6989641f63ffff220554052c8cb8c3f7ce0e8718f97aa5028db979a1d5a7"
)
EXPECTED_RETAINED_CANDIDATES_HASH = (
    "sha256:8c541284bca553ad35bef66d01a56adc7de951cf62aca1dfc5a61ed71de57221"
)
EXPECTED_REGISTRY_HASH = (
    "sha256:990b559eee6d2d05723c25bf66d8d4d977d9e6865111bd3ce03618103b5ea36e"
)
EXPECTED_TRACE_MANIFEST_HASHES = {
    "results/arc_live_oracle_gap.json": (
        "sha256:4f3f36b4cce99c4c3ec1c0930b90020cf96547a3498de93e649cc28c072999ab"
    ),
    "results/experiment_5727_arc_generalization_live_oracle_gap_v511.json": (
        "sha256:46347c420f76cba180e8a5f8b07bf7db7a2d9f2861f8e44e82130127e7b1b63b"
    ),
    "results/experiment_5727_perception_action_effect_adequacy.json": (
        "sha256:57a08a92bf5c2b0665d7f4ef0025a33fb5d90f5fe0f5f24a9a461d6727c0110b"
    ),
}

FROZEN_PRIMITIVE_IDS = (
    "object_displacement",
    "reversible_or_noop_action",
    "boundary_or_collision",
    "inventory_or_state_toggle",
    "agent_relative_motion",
    "repeated_action_loop",
    "delayed_effect",
)
MIN_PAIRED_REPLAYS = 30
EXPECTED_POSITIVE_COUNT = 7
EXPECTED_PAIRED_REPLAY_COUNT = 20759
EXPECTED_TRACE_STEP_COUNT = 9975

SOURCE_LEAK_KEYS = frozenset(
    {
        "source_file",
        "source_rule",
        "game_source",
        "solution_code",
        "hidden_state",
        "per_game_adapter",
        "adapter_label",
        "outer_loop_bfs",
        "hand_authored_model",
    }
)
GAME_IDENTITY_KEYS = frozenset(
    {
        "game",
        "game_id",
        "game_name",
        "source_game",
        "registry_game",
        "registry_provenance",
    }
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "every normalized field carries its own audit rationale so the corrigendum is schema-stable and principle-grounded.",
    "preconditions_checked": "fail-closed checks prove the source artifact, trace receipts, effects, registry state, and code immutability before normalization.",
    "source_artifact_path": "the normalized gate is explicitly tied to the Exp5740 source JSON rather than an inferred experiment number.",
    "source_artifact_hash": "the corrigendum is valid only for the exact checked-in Exp5740 bytes.",
    "source_schema_version": "records the legacy object-valued gate shape being normalized.",
    "normalized_schema_version": "names the scalar gate contract consumed by downstream conductor gates.",
    "positive_causal_primitive_count": "copies the Exp5740 retained primitive count without re-mining or effect edits.",
    "frozen_primitive_ids": "freezes the credited primitive identities in deterministic order for downstream live hardening.",
    "frozen_effect_hash": "content-addresses the retained deletion effects so no effect size can be silently changed.",
    "original_counterfactual_receipt_coverage": "preserves the complete Exp5740 coverage object losslessly while adding a scalar gate field.",
    "counterfactual_receipt_coverage_score": "equals 1.0 only when all credited primitives meet paired-replay and trace-receipt existence gates.",
    "detected_source_leak_canary_count": "counts source leak canaries that the negative-control harness detected and rejected.",
    "detected_game_identity_leak_canary_count": "counts game-identity canaries that the negative-control harness detected and rejected.",
    "admitted_source_leak_count": "counts only source leakage that entered credited primitives or live state, never rejected canaries.",
    "admitted_game_identity_leak_count": "counts only game-identity leakage that entered credited primitives or live state, never rejected canaries.",
    "normalization_rules": "documents the deterministic object-to-scalar and detected-vs-admitted leak transformations.",
    "registry_precheck": "records that all 25 public games and all 183 registry levels were already complete before the corrigendum.",
    "solve_provenance": "development_proxy marks this as schema repair evidence, not live hidden-game self-discovery.",
    "arc_registry_delta": "zero prevents schema repair from inflating the public solve registry.",
    "arc_solve_credited": "false prevents a gate corrigendum from claiming a solve.",
    "science_rerun": "false proves the artifact did not rerun mining, replay science, or live games.",
    "live_policy_modified": "false keeps schema normalization out of the submitted policy path.",
    "test_commands": "records the verification commands used to validate the corrigendum.",
    "test_exit_codes": "records the exit code of every verification command instead of relying on prose.",
    "reproducibility_checksum": "content-addresses the normalized artifact while excluding its self-checksum.",
    "honest_verdict": "terminal complete: or blocked: verdict states whether the lossless scalar schema repair is usable.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def stable_json(value: Any) -> str:
    """Serialize JSON deterministically for byte-stable hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    """Return Carnot's prefixed SHA-256 digest for raw bytes."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    """Return Carnot's prefixed SHA-256 digest for JSON-compatible content."""

    return sha256_bytes(stable_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    """Hash a file byte-for-byte."""

    return sha256_bytes(path.read_bytes())


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def read_json(path: Path) -> JsonDict:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> JsonDict:
    """Read a YAML object, returning an empty mapping for empty files."""

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _source_payload_checksum(source: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in source.items() if key != "reproducibility_checksum"}
    return sha256_json(payload)


def _retained_candidates(source: Mapping[str, Any]) -> list[JsonDict]:
    rows = source.get("primitive_candidates")
    if not isinstance(rows, list):
        raise ValueError("primitive_candidates must be a list")
    retained = [dict(row) for row in rows if isinstance(row, Mapping) and row.get("causal_retained")]
    primitive_ids = [str(row.get("primitive")) for row in retained]
    if primitive_ids != list(FROZEN_PRIMITIVE_IDS):
        raise ValueError("frozen_primitive_ids mismatch")
    return retained


def _retained_effects(source: Mapping[str, Any]) -> JsonDict:
    utilities = source.get("counterfactual_trajectory_utility")
    if not isinstance(utilities, Mapping):
        raise ValueError("counterfactual_trajectory_utility must be a mapping")
    retained_effects: JsonDict = {}
    for primitive in FROZEN_PRIMITIVE_IDS:
        effect = utilities.get(primitive)
        if not isinstance(effect, Mapping):
            raise ValueError(f"counterfactual_trajectory_utility missing {primitive}")
        retained_effects[primitive] = dict(effect)
    effect_hash = sha256_json(retained_effects)
    if effect_hash != EXPECTED_FROZEN_EFFECT_HASH:
        raise ValueError("frozen_effect_hash mismatch")
    return retained_effects


def _exact_replay_receipts(retained_effects: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    receipts: JsonDict = {}
    for primitive, effect in retained_effects.items():
        baseline = str(effect.get("baseline_decision_hash", ""))
        deletion = str(effect.get("deletion_decision_hash", ""))
        if not baseline.startswith("sha256:") or not deletion.startswith("sha256:"):
            raise ValueError("exact_replay_receipts missing decision hashes")
        if baseline == deletion:
            raise ValueError("exact_replay_receipts require changed deletion hash")
        if int(effect.get("downstream_decision_hash_changed_count", 0) or 0) <= 0:
            raise ValueError("exact_replay_receipts missing downstream changes")
        receipts[primitive] = {
            "baseline_decision_hash": baseline,
            "deletion_decision_hash": deletion,
            "paired_replay_count": int(effect.get("paired_replay_count", 0) or 0),
            "downstream_decision_hash_changed_count": int(
                effect.get("downstream_decision_hash_changed_count", 0) or 0
            ),
        }
    return {
        "verified": True,
        "receipt_count": len(receipts),
        "receipt_hash": sha256_json(receipts),
        "receipts": receipts,
    }


def derive_leak_counts(source: Mapping[str, Any]) -> JsonDict:
    """Separate rejected canary detections from admitted leakage."""

    controls = source.get("negative_controls")
    if not isinstance(controls, list):
        raise ValueError("negative_controls must be a list")
    detected_source = 0
    detected_identity = 0
    for row in controls:
        if not isinstance(row, Mapping):
            raise ValueError("negative_controls rows must be mappings")
        classes = {str(item) for item in row.get("leak_classes", [])}
        if classes and not (row.get("detected") is True and row.get("rejected") is True):
            raise ValueError("negative_controls leak canary was not detected and rejected")
        if "source" in classes:
            detected_source += 1
        if "game_identity" in classes:
            detected_identity += 1

    admitted_source = 0
    admitted_identity = 0
    for row in _retained_candidates(source):
        classes = set()
        for key in ("admitted_leak_classes", "leak_classes", "live_state_leak_classes"):
            value = row.get(key, [])
            if isinstance(value, list):
                classes.update(str(item) for item in value)
        if set(row) & SOURCE_LEAK_KEYS:
            classes.add("source")
        if set(row) & GAME_IDENTITY_KEYS:
            classes.add("game_identity")
        learner_visible = row.get("learner_visible")
        if isinstance(learner_visible, Mapping):
            if set(learner_visible) & SOURCE_LEAK_KEYS:
                classes.add("source")
            if set(learner_visible) & GAME_IDENTITY_KEYS:
                classes.add("game_identity")
        if "source" in classes:
            admitted_source += 1
        if "game_identity" in classes:
            admitted_identity += 1

    return {
        "detected_source_leak_canary_count": detected_source,
        "detected_game_identity_leak_canary_count": detected_identity,
        "admitted_source_leak_count": admitted_source,
        "admitted_game_identity_leak_count": admitted_identity,
    }


def verify_trace_manifest_hashes(source: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Verify every trace receipt path and hash recorded by Exp5740."""

    manifest = source.get("trace_manifest")
    if not isinstance(manifest, list):
        raise ValueError("trace_manifest must be a list")
    verified_hashes: JsonDict = {}
    used_for_mining: list[str] = []
    for row in manifest:
        if not isinstance(row, Mapping):
            raise ValueError("trace_manifest rows must be mappings")
        rel = str(row.get("path", ""))
        expected_hash = EXPECTED_TRACE_MANIFEST_HASHES.get(rel)
        recorded_hash = str(row.get("sha256", ""))
        full_path = root / rel
        if expected_hash is None or recorded_hash != expected_hash:
            raise ValueError("trace_manifest recorded hash mismatch")
        if not full_path.exists():
            raise ValueError("trace_manifest referenced receipt is missing")
        actual_hash = file_sha256(full_path)
        if actual_hash != recorded_hash:
            raise ValueError("trace_manifest file hash mismatch")
        verified_hashes[rel] = actual_hash
        if row.get("used_for_mining") is True:
            used_for_mining.append(rel)
    if verified_hashes != EXPECTED_TRACE_MANIFEST_HASHES:
        raise ValueError("trace_manifest did not verify the full expected receipt set")
    return {
        "verified": True,
        "receipt_count": len(verified_hashes),
        "trace_hashes": verified_hashes,
        "used_for_mining_paths": used_for_mining,
    }


def normalize_counterfactual_coverage(
    source: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> float:
    """Normalize Exp5740's coverage object into the scalar gate value."""

    coverage = source.get("counterfactual_receipt_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("counterfactual_receipt_coverage must be a mapping")
    retained = _retained_candidates(source)
    paired_counts = [int(row.get("paired_replay_count", 0) or 0) for row in retained]
    if any(count < MIN_PAIRED_REPLAYS for count in paired_counts):
        raise ValueError("paired_replay threshold failed for a retained primitive")
    if min(paired_counts) != int(coverage.get("minimum_positive_candidate_paired_replay_count", -1)):
        raise ValueError("paired_replay coverage minimum mismatch")
    if sum(paired_counts) != int(coverage.get("paired_replay_count", -1)):
        raise ValueError("paired_replay coverage total mismatch")
    if int(coverage.get("trace_step_count", -1)) != EXPECTED_TRACE_STEP_COUNT:
        raise ValueError("counterfactual_receipt_coverage trace_step_count mismatch")
    if coverage.get("meets_minimum_n") is not True:
        raise ValueError("counterfactual_receipt_coverage minimum flag mismatch")
    verify_trace_manifest_hashes(source, root=root)
    return 1.0


def registry_precheck(*, root: Path = REPO_ROOT) -> JsonDict:
    """Read the ARC registry and prove there is no solve-credit headroom."""

    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_hash = file_sha256(registry_path)
    registry = read_yaml(registry_path)
    games = registry.get("games")
    if not isinstance(games, list):
        raise ValueError("registry_precheck games must be a list")
    public_game_count = len(games)
    completed_levels = sum(int(row.get("levels_reproduced", 0) or 0) for row in games)
    full_clear_count = sum(1 for row in games if isinstance(row, Mapping) and row.get("full_game_clear") is True)
    total_levels = int(registry.get("reproducible_total_levels", completed_levels) or 0)
    total_games = int(registry.get("reproducible_total_games", public_game_count) or 0)
    precheck = {
        "registry_path": str(REGISTRY_RELATIVE_PATH),
        "registry_hash": registry_hash,
        "public_game_count": public_game_count,
        "reproducible_total_games": total_games,
        "reproducible_total_levels": total_levels,
        "completed_levels": completed_levels,
        "full_game_clear_count": full_clear_count,
        "all_public_games_complete": (
            registry_hash == EXPECTED_REGISTRY_HASH
            and public_game_count == 25
            and total_games == 25
            and total_levels == 183
            and completed_levels == 183
            and full_clear_count == 25
        ),
        "registry_delta_allowed": 0,
        "arc_solve_credit_allowed": False,
    }
    if precheck["all_public_games_complete"] is not True:
        raise ValueError("registry_precheck expected 25 games and 183/183 complete levels")
    return precheck


def _research_conductor_modified(*, root: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", str(RESEARCH_CONDUCTOR_RELATIVE_PATH)],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode != 0


def _normalization_rules() -> list[JsonDict]:
    return [
        {
            "rule": "source_hash_gate",
            "input": str(SOURCE_ARTIFACT_RELATIVE_PATH),
            "output": "source_artifact_hash",
            "description": "Normalize only the exact checked-in Exp5740 bytes.",
        },
        {
            "rule": "coverage_object_to_scalar",
            "input": "counterfactual_receipt_coverage",
            "output": "counterfactual_receipt_coverage_score",
            "description": "Emit 1.0 only when every retained primitive and trace receipt passes.",
        },
        {
            "rule": "detected_vs_admitted_leaks",
            "input": "negative_controls and retained primitive rows",
            "output": "detected_*_canary_count and admitted_*_leak_count",
            "description": "Rejected canaries stay visible but do not count as admitted leakage.",
        },
        {
            "rule": "effect_freeze",
            "input": "counterfactual_trajectory_utility for retained primitive IDs",
            "output": "frozen_effect_hash",
            "description": "Hash retained effect payloads without changing any effect size.",
        },
        {
            "rule": "no_credit",
            "input": "ops/arc_solve_registry.yaml",
            "output": "development_proxy, arc_registry_delta=0, arc_solve_credited=false",
            "description": "Schema repair cannot bank a public-game or hidden-game solve.",
        },
    ]


def _preconditions(
    *,
    source_hash: str,
    source: Mapping[str, Any],
    retained: Sequence[Mapping[str, Any]],
    retained_effects: Mapping[str, Mapping[str, Any]],
    trace_receipt: Mapping[str, Any],
    replay_receipt: Mapping[str, Any],
    registry_receipt: Mapping[str, Any],
    root: Path,
) -> JsonDict:
    retained_hash = sha256_json(list(retained))
    paired_replay_count = sum(int(row.get("paired_replay_count", 0) or 0) for row in retained)
    checksum_verified = source.get("reproducibility_checksum") == _source_payload_checksum(source)
    return {
        "source_artifact_hash_verified": source_hash == EXPECTED_SOURCE_ARTIFACT_HASH,
        "source_artifact_reproducibility_checksum_verified": checksum_verified,
        "trace_manifest_hashes_verified": trace_receipt.get("verified") is True,
        "trace_manifest_receipt_count": trace_receipt.get("receipt_count"),
        "primitive_count_verified": len(retained) == EXPECTED_POSITIVE_COUNT,
        "frozen_primitive_ids_verified": [row.get("primitive") for row in retained]
        == list(FROZEN_PRIMITIVE_IDS),
        "retained_candidates_hash": retained_hash,
        "retained_candidates_hash_verified": retained_hash == EXPECTED_RETAINED_CANDIDATES_HASH,
        "deletion_effect_hash_verified": sha256_json(retained_effects)
        == EXPECTED_FROZEN_EFFECT_HASH,
        "paired_replay_count": paired_replay_count,
        "paired_replay_count_verified": paired_replay_count == EXPECTED_PAIRED_REPLAY_COUNT,
        "exact_replay_receipts_verified": replay_receipt.get("verified") is True,
        "exact_replay_receipt_hash": replay_receipt.get("receipt_hash"),
        "negative_controls_verified": derive_leak_counts(source)["admitted_source_leak_count"] == 0
        and derive_leak_counts(source)["admitted_game_identity_leak_count"] == 0,
        "registry_precheck_passed": registry_receipt.get("all_public_games_complete") is True,
        "science_rerun": False,
        "live_policy_modified": False,
        "scripts_research_conductor_modified": _research_conductor_modified(root=root),
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the normalized Exp5745 artifact from checked-in receipts only."""

    source_path = root / SOURCE_ARTIFACT_RELATIVE_PATH
    source_hash = file_sha256(source_path)
    if source_hash != EXPECTED_SOURCE_ARTIFACT_HASH:
        raise ValueError("source_artifact_hash mismatch")
    source = read_json(source_path)
    retained = _retained_candidates(source)
    retained_effects = _retained_effects(source)
    replay_receipt = _exact_replay_receipts(retained_effects)
    trace_receipt = verify_trace_manifest_hashes(source, root=root)
    coverage_score = normalize_counterfactual_coverage(source, root=root)
    leak_counts = derive_leak_counts(source)
    registry_receipt = registry_precheck(root=root)
    preconditions = _preconditions(
        source_hash=source_hash,
        source=source,
        retained=retained,
        retained_effects=retained_effects,
        trace_receipt=trace_receipt,
        replay_receipt=replay_receipt,
        registry_receipt=registry_receipt,
        root=root,
    )
    artifact: JsonDict = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "source_artifact_path": str(SOURCE_ARTIFACT_RELATIVE_PATH),
        "source_artifact_hash": source_hash,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "normalized_schema_version": NORMALIZED_SCHEMA_VERSION,
        "positive_causal_primitive_count": len(retained),
        "frozen_primitive_ids": [row["primitive"] for row in retained],
        "frozen_effect_hash": sha256_json(retained_effects),
        "original_counterfactual_receipt_coverage": dict(
            source["counterfactual_receipt_coverage"]
        ),
        "counterfactual_receipt_coverage_score": coverage_score,
        **leak_counts,
        "normalization_rules": _normalization_rules(),
        "registry_precheck": registry_receipt,
        "solve_provenance": "development_proxy",
        "arc_registry_delta": 0,
        "arc_solve_credited": False,
        "science_rerun": False,
        "live_policy_modified": False,
        "test_commands": list(test_commands or []),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: exp5740_lossless_scalar_gate_corrigendum_positive_count_7_"
            "admitted_leaks_0_registry_delta_0"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed on malformed normalized gate artifacts."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    extra = sorted(set(artifact) - set(FIELD_PRINCIPLES))
    if extra:
        raise ValueError(f"field_principles missing top-level fields: {extra}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact.get("source_artifact_hash") != EXPECTED_SOURCE_ARTIFACT_HASH:
        raise ValueError("source_artifact_hash mismatch")
    if artifact.get("source_schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("source_schema_version mismatch")
    if artifact.get("normalized_schema_version") != NORMALIZED_SCHEMA_VERSION:
        raise ValueError("normalized_schema_version mismatch")
    if artifact.get("positive_causal_primitive_count") != EXPECTED_POSITIVE_COUNT:
        raise ValueError("positive_causal_primitive_count mismatch")
    if artifact.get("frozen_primitive_ids") != list(FROZEN_PRIMITIVE_IDS):
        raise ValueError("frozen_primitive_ids mismatch")
    if artifact.get("frozen_effect_hash") != EXPECTED_FROZEN_EFFECT_HASH:
        raise ValueError("frozen_effect_hash mismatch")
    if (
        sha256_json(artifact.get("original_counterfactual_receipt_coverage"))
        != EXPECTED_ORIGINAL_COVERAGE_HASH
    ):
        raise ValueError("original_counterfactual_receipt_coverage mismatch")
    if artifact.get("counterfactual_receipt_coverage_score") != 1.0:
        raise ValueError("counterfactual_receipt_coverage_score mismatch")
    if artifact.get("detected_source_leak_canary_count") != 1:
        raise ValueError("detected_source_leak_canary_count mismatch")
    if artifact.get("detected_game_identity_leak_canary_count") != 2:
        raise ValueError("detected_game_identity_leak_canary_count mismatch")
    if artifact.get("admitted_source_leak_count") != 0:
        raise ValueError("admitted_source_leak_count mismatch")
    if artifact.get("admitted_game_identity_leak_count") != 0:
        raise ValueError("admitted_game_identity_leak_count mismatch")
    registry = artifact.get("registry_precheck")
    if not isinstance(registry, Mapping) or registry.get("all_public_games_complete") is not True:
        raise ValueError("registry_precheck mismatch")
    if registry.get("public_game_count") != 25 or registry.get("completed_levels") != 183:
        raise ValueError("registry_precheck count mismatch")
    if artifact.get("solve_provenance") != "development_proxy":
        raise ValueError("solve_provenance mismatch")
    if artifact.get("arc_registry_delta") != 0:
        raise ValueError("arc_registry_delta mismatch")
    if artifact.get("arc_solve_credited") is not False:
        raise ValueError("arc_solve_credited mismatch")
    if artifact.get("science_rerun") is not False:
        raise ValueError("science_rerun mismatch")
    if artifact.get("live_policy_modified") is not False:
        raise ValueError("live_policy_modified mismatch")
    preconditions = artifact.get("preconditions_checked")
    required_preconditions = (
        "source_artifact_hash_verified",
        "source_artifact_reproducibility_checksum_verified",
        "trace_manifest_hashes_verified",
        "primitive_count_verified",
        "frozen_primitive_ids_verified",
        "retained_candidates_hash_verified",
        "deletion_effect_hash_verified",
        "paired_replay_count_verified",
        "exact_replay_receipts_verified",
        "negative_controls_verified",
        "registry_precheck_passed",
    )
    if not isinstance(preconditions, Mapping):
        raise ValueError("preconditions_checked mismatch")
    for key in required_preconditions:
        if preconditions.get(key) is not True:
            raise ValueError(f"preconditions_checked {key} mismatch")
    if preconditions.get("scripts_research_conductor_modified") is not False:
        raise ValueError("preconditions_checked scripts_research_conductor_modified mismatch")
    commands = artifact.get("test_commands")
    exit_codes = artifact.get("test_exit_codes")
    if not isinstance(commands, list) or not isinstance(exit_codes, Mapping):
        raise ValueError("test_commands/test_exit_codes mismatch")
    for command in commands:
        if command not in exit_codes or int(exit_codes[command]) != 0:
            raise ValueError("test_exit_codes mismatch")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked:")):
        raise ValueError("honest_verdict lacks terminal prefix")


def write_output(results_dir: Path, artifact: Mapping[str, Any]) -> Path:
    """Write the normalized artifact into a results directory."""

    validate_artifact(artifact)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / RESULT_RELATIVE_PATH.name
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = list(argv if argv is not None else sys.argv[1:])
    results_dir = REPO_ROOT / "results"
    if "--results-dir" in args:
        results_dir = Path(args[args.index("--results-dir") + 1])
    artifact = build_artifact(root=REPO_ROOT)
    target = write_output(results_dir, artifact)
    print(f"wrote {target} -- honest_verdict={artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
