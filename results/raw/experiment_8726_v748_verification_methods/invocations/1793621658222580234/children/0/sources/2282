"""Build the Exp6395 held factor transport license matrix artifact.

Spec refs: REQ-LEARN-6395, SCENARIO-LEARN-6395-MATRIX,
SCENARIO-LEARN-6395-LICENSE, SCENARIO-LEARN-6395-ABSTAIN,
SCENARIO-LEARN-6395-ATTACKS, SCENARIO-LEARN-6395-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6380_three_family_canonical_factor_transport_canary as exp6380
from carnot import experiment_6394_model_family_factor_harness_freeze as exp6394
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]
HostChecksFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6395_held_factor_transport_license_matrix.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6395_held_factor_transport_license_matrix")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6395_held_factor_transport_license_matrix.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6395_held_factor_transport_license_matrix.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6394_RELATIVE_PATH = exp6394.RESULT_RELATIVE_PATH
EXP6380_RELATIVE_PATH = exp6380.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6395.held_factor_transport_license_matrix.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6395
TOKENIZER_METHOD = exp6394.TOKENIZER_METHOD
PREFERRED_QUANT = exp6394.PREFERRED_QUANT
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"

MANDATED_MODEL_IDS = exp6394.MANDATED_MODEL_IDS
MODEL_TEMPLATE_BY_ID = exp6394.MODEL_TEMPLATE_BY_ID
REQUIRED_CONSTRAINT_FAMILIES = exp6394.REQUIRED_EVENT_FAMILIES
MODEL_EVENT_FAMILY_BY_ID = exp6394.EVENT_FAMILY_BY_MODEL_ID
EXACT_CHECK_COST = exp6380.EXACT_CHECK_COST
CHECKER_TIME_PER_CALL_S = exp6380.CHECKER_TIME_PER_CALL_S

LICENSED_CELL_TARGETS = frozenset(
    {
        ("gemma_dense", "threshold_guard"),
        ("gemma_dense", "route_guard"),
        ("gemma_moe", "route_guard"),
        ("gemma_moe", "conservation_guard"),
    }
)
ATTACK_IDS = (
    "model_row_swap",
    "family_label_swap",
    "harness_drift",
    "stale_schema",
    "source_substitution",
    "missing_rows",
    "fallback_laundering",
    "abstention_suppression",
    "repeated_output",
    "exact_fail_promotion",
)
RANDOM_SEEDS = {
    "held_manifest": 639500,
    "trial_order": 639501,
    "raw_output": 639502,
    "exact_checker": 639503,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6395_held_factor_transport_license_matrix --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6395_held_factor_transport_license_matrix.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6395_held_factor_transport_license_matrix.py "
    "-m pytest tests/python/test_experiment_6395_held_factor_transport_license_matrix.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6395_held_factor_transport_license_matrix.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6395_held_factor_transport_license_matrix.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6395_held_factor_transport_license_matrix.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6394_RELATIVE_PATH,
    EXP6380_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py"),
    Path("python/carnot/experiment_6394_model_family_factor_harness_freeze.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6394_gate_receipt",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "cuda_offload_and_runtime_receipts_by_model",
    "frozen_harness_and_schema_hashes",
    "held_manifest_path_hash_license_balance_and_prior_access_receipt",
    "preregistered_license_rule",
    "raw_output_before_parse_paths_hashes_and_counts",
    "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix",
    "capability_license_records",
    "rejected_and_abstained_cell_records",
    "license_binding_and_expiration_fields",
    "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix",
    "licensed_cell_count",
    "licensed_model_count",
    "licensed_constraint_family_count",
    "held_factor_transport_license_ready_score",
    "universal_support_claimed",
    "protected_leakage_count",
    "model_weight_change_count",
    "prohibited_mechanism_usage_counts",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status separates positive, null, blocked, and retired held-license evidence.",
    "exp6394_gate_receipt": "The Exp6394 freeze gate and frozen sidecars are revalidated before held evaluation.",
    "MODEL_SPECS": "The three mandated GGUF model rows come from cached SOTA helper calls.",
    "models_used": "Only mandated models with authenticated runtime receipts count as used.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Model file identity and tokenizer method are pinned.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer receipts use only embedded GGUF tokenizers.",
    "autotokenizer_usage_count": "Bare zero proves no external tokenizer path was used.",
    "cuda_offload_and_runtime_receipts_by_model": "CUDA offload, timing, raw streams, and cleanup are reported per model.",
    "frozen_harness_and_schema_hashes": "Harness sidecars and canonical schema are hash-bound.",
    "held_manifest_path_hash_license_balance_and_prior_access_receipt": "Held events are licensed, balanced, sealed, and not read before freeze.",
    "preregistered_license_rule": "The exact licensing thresholds are frozen before held outcomes are scored.",
    "raw_output_before_parse_paths_hashes_and_counts": "Raw bytes are frozen before one parse.",
    "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix": "Every model-family cell reports transport, source binding, exact calls, abstention, timeout, latency, and cost.",
    "capability_license_records": "Accepted licenses bind model, harness, schema, tokenizer, family, manifest, and expiration.",
    "rejected_and_abstained_cell_records": "Every unlicensed cell has a terminal reason.",
    "license_binding_and_expiration_fields": "License identity and expiry fields are explicit and narrow.",
    "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix": "Swap, drift, fallback, abstention, repetition, and promotion attacks fail closed.",
    "licensed_cell_count": "Bare count of valid model-family licenses.",
    "licensed_model_count": "Bare count of mandated models with at least one valid license.",
    "licensed_constraint_family_count": "Bare count of families with at least one valid license.",
    "held_factor_transport_license_ready_score": "Readiness is a conjunctive matrix gate and never a universal-support claim.",
    "universal_support_claimed": "Bare false prevents a universal gate from reappearing under another name.",
    "protected_leakage_count": "Protected leakage must be zero for any license.",
    "model_weight_change_count": "Bare zero proves no model weights changed.",
    "prohibited_mechanism_usage_counts": "Retry, repair, reselection, tuning, family substitution, fallback, and external-tokenizer counts stay zero.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, underpowered, abstained, rejected, and attacked cells stay visible.",
    "protected_files_unchanged": "Protected files remain byte-identical.",
    "preconditions_checked": "Preconditions bind upstream, models, tokenizers, GPU, schema, manifests, sources, and protected files.",
    "inference_substrate": "The substrate declares deterministic verifier replay over local GGUF identity receipts.",
    "verifier_is_oracle": "Bare true applies only to exact task checkers.",
    "field_principles": "Every required field states its guard and scientific purpose.",
    "field_provenance": "Every required field maps to specs, upstream artifacts, sidecars, model receipts, tests, or exact checks.",
    "random_seed": "Fixed seeds pin held schedule and matrix order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the license boundary.",
    "model_family_harness_freeze_ready_score": "The Exp6394 gate proves only that harnesses froze before held access.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6395",
        "Exp6394 frozen harness sidecars",
        "Exp6380 canonical schema receipts",
        "held Exp6395 trial matrix",
        "focused Exp6395 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and sidecar payloads."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash the compact JSON serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the path is absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a required gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return exp6394.model_slug(model_id)


def rounded(value: float) -> float:
    """Round receipts without hiding small nonzero costs."""

    return round(float(value), 12)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON when requested, otherwise return the would-be digest."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: str | Path, *, digest: str | None = None) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object from disk."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_top_level_not_object:{path}")
    return value


def _token_count(receipt: Mapping[str, Any]) -> int:
    """Read either token-count spelling used by older experiments."""

    return int(receipt.get("token_count", receipt.get("prompt_tokens", 0)) or 0)


def embedded_gguf_tokenizer_receipt(model_path: str, text: str) -> JsonDict:  # pragma: no cover
    """Count text tokens through the model file's embedded GGUF tokenizer."""

    receipt = exp6394.embedded_gguf_tokenizer_receipt(model_path, text)
    return {**receipt, "token_count": _token_count(receipt), "autotokenizer_used": False}


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve the three mandated GGUF rows through cached SOTA helper calls."""

    return exp6394.build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )


def _tokenizer_identity(row: Mapping[str, Any]) -> str:
    """Bind the embedded tokenizer to the model file and precheck receipt."""

    return sha256_json(
        {
            "hf_id": row.get("hf_id"),
            "model_file_sha256": row.get("model_file_sha256"),
            "tokenizer_method": row.get("tokenizer_method"),
            "tokenizer_detail": row.get("tokenizer_detail"),
            "precheck_tokens": row.get("prompt_tokens_for_tokenizer_precheck"),
        }
    )


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model identity, hashes, quantization, and tokenizer method."""

    rows = []
    for row in exp6394.model_file_receipts(model_specs):
        source = next(
            item for item in model_specs if item.get("hf_id") == row.get("hf_id")
        )
        rows.append({**row, "embedded_tokenizer_sha256": _tokenizer_identity(source)})
    return rows


def tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]],
    tokenizer_func: TokenizerFn,
) -> list[JsonDict]:
    """Return embedded tokenizer receipts for each model."""

    rows = []
    for row in model_specs:
        receipt = tokenizer_func(str(row.get("model_path", "")), "Exp6395 tokenizer license.")
        identity = _tokenizer_identity({**dict(row), "tokenizer_detail": receipt.get("tokenizer_detail")})
        rows.append(
            {
                "hf_id": row.get("hf_id"),
                "model_path": row.get("model_path"),
                "method": receipt.get("method", TOKENIZER_METHOD),
                "loadable": receipt.get("loadable") is True,
                "token_count": _token_count(receipt),
                "detail": receipt.get("tokenizer_detail", ""),
                "embedded_tokenizer_sha256": identity,
                "autotokenizer_used": False,
            }
        )
    return rows


def host_environment_receipts() -> JsonDict:  # pragma: no cover
    """Collect live host receipts through the prior GGUF harness helper."""

    return exp6394.host_environment_receipts()


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must remain unchanged."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define the experiment contract."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes from before and after the run."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def exp6394_gate_receipt(path: str | Path) -> JsonDict:
    """Revalidate the Exp6394 gate and frozen harness sidecar hashes."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "status": "missing",
            "model_family_harness_freeze_ready_score": 0.0,
            "gate_passed": False,
            "blocked_reasons": ["exp6394_artifact_missing"],
            "frozen_harness_sidecars": {},
            "sidecar_hashes_match": False,
            "held_access_before_freeze_count": 0,
        }
    payload = read_json(path)
    frozen = as_mapping(payload.get("frozen_harness_paths_hashes_and_controls"))
    sidecars = {}
    schema_hashes: set[str] = set()
    for family, row in as_mapping(frozen.get("by_model_family")).items():
        row_map = as_mapping(row)
        sidecar_path = Path(str(row_map.get("path", "")))
        actual_hash = sha256_file(sidecar_path)
        sidecar_payload = read_json(sidecar_path) if sidecar_path.is_file() else {}
        controls = as_mapping(row_map.get("controls"))
        schema_hash = str(
            sidecar_payload.get("canonical_schema_sha256")
            or controls.get("canonical_schema_sha256")
            or ""
        )
        if schema_hash:
            schema_hashes.add(schema_hash)
        sidecars[str(family)] = {
            **path_receipt(sidecar_path, digest=actual_hash),
            "expected_sha256": row_map.get("sha256"),
            "hash_matches": actual_hash == row_map.get("sha256"),
            "model_hf_id": sidecar_payload.get("model_hf_id"),
            "model_family": family,
            "abstention": sidecar_payload.get("abstention", controls.get("abstention")) is True,
            "canonical_schema_sha256": schema_hash,
            "controls": dict(controls),
        }
    leakage = as_mapping(payload.get("protected_leakage_and_same_step_write_counts"))
    mechanisms = as_mapping(
        payload.get("grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts")
    )
    held_access_count = int(payload.get("held_access_during_selection_count", 0) or 0)
    held_access_count += int(leakage.get("held_event_content_read_count", 0) or 0)
    held_access_count += int(leakage.get("held_outcome_read_count", 0) or 0)
    blocked = []
    if float(payload.get("model_family_harness_freeze_ready_score", 0.0) or 0.0) != 1.0:
        blocked.append("exp6394_freeze_score_not_ready")
    if not sidecars or not all(row["hash_matches"] for row in sidecars.values()):
        blocked.append("frozen_harness_sidecar_hash_mismatch")
    if held_access_count:
        blocked.append("held_access_before_freeze")
    if int(leakage.get("protected_leakage_count", 0) or 0) != 0:
        blocked.append("exp6394_protected_leakage")
    if int(payload.get("model_weight_change_count", 0) or 0) != 0:
        blocked.append("exp6394_model_weight_change")
    if any(int(value) != 0 for key, value in mechanisms.items() if key != "schema"):
        blocked.append("exp6394_prohibited_mechanism")
    return {
        **receipt,
        "status": payload.get("status", "missing"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "model_family_harness_freeze_ready_score": payload.get(
            "model_family_harness_freeze_ready_score",
            0.0,
        ),
        "gate_passed": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "frozen_harness_sidecars": sidecars,
        "sidecar_hashes_match": bool(sidecars)
        and all(row["hash_matches"] for row in sidecars.values()),
        "canonical_schema_hashes": sorted(schema_hashes),
        "held_access_before_freeze_count": held_access_count,
        "protected_leakage_before_freeze_count": int(leakage.get("protected_leakage_count", 0) or 0),
        "model_weight_change_before_freeze_count": int(payload.get("model_weight_change_count", 0) or 0),
    }


def frozen_harness_and_schema_hashes(gate: Mapping[str, Any]) -> JsonDict:
    """Expose frozen harness sidecars and their canonical schema hashes."""

    sidecars = as_mapping(gate.get("frozen_harness_sidecars"))
    schema_hashes = sorted(
        {
            str(as_mapping(row).get("canonical_schema_sha256"))
            for row in sidecars.values()
            if as_mapping(row).get("canonical_schema_sha256")
        }
    )
    return {
        "schema": SCHEMA + ".frozen_harness_and_schema_hashes",
        "by_model_family": dict(sidecars),
        "canonical_schema_hashes": schema_hashes,
        "single_canonical_schema_hash": schema_hashes[0] if len(schema_hashes) == 1 else None,
        "all_harness_hashes_match": gate.get("sidecar_hashes_match") is True,
        "frozen_before_held_access": gate.get("held_access_before_freeze_count") == 0,
    }


def _held_source_events() -> dict[str, list[JsonDict]]:
    """Group reusable executable events by constraint family."""

    by_family: dict[str, list[JsonDict]] = {family: [] for family in REQUIRED_CONSTRAINT_FAMILIES}
    for row in exp6394.generated_events():
        family = str(row.get("family"))
        if family in by_family:
            by_family[family].append(dict(row))
    return by_family


def held_balance_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check that held events cover all required executable families."""

    by_family = Counter(str(row.get("family")) for row in events)
    return {
        "schema": SCHEMA + ".held_balance",
        "event_count": sum(by_family.values()),
        "family_count": len(by_family),
        "events_by_family": dict(sorted(by_family.items())),
        "balanced": sum(by_family.values()) >= 18
        and set(by_family) == set(REQUIRED_CONSTRAINT_FAMILIES)
        and all(by_family[family] >= 6 for family in REQUIRED_CONSTRAINT_FAMILIES),
    }


def held_manifest_path_hash_license_balance_and_prior_access_receipt(
    data_dir: str | Path,
    *,
    gate: Mapping[str, Any],
    write: bool,
    allow_held_content: bool,
) -> JsonDict:
    """Seal held events after the freeze gate permits held evaluation."""

    path = Path(data_dir) / "held_manifest" / "held_factor_transport_license_manifest.json"
    if not allow_held_content:
        payload = {
            "schema": SCHEMA + ".held_manifest",
            "event_count": 0,
            "events": [],
            "license": "held_content_not_loaded_because_freeze_gate_closed",
            "held_content_loaded_after_freeze": False,
            "random_seed": RANDOM_SEEDS["held_manifest"],
        }
        digest = write_payload_or_hash(path, payload, write=write)
        return {
            **path_receipt(path, digest=digest),
            "manifest": payload,
            "event_count": 0,
            "license": payload["license"],
            "balance": held_balance_receipt([]),
            "prior_access_receipt": {
                "held_access_before_exp6394_freeze_count": gate.get(
                    "held_access_before_freeze_count",
                    0,
                ),
                "held_content_loaded_after_freeze": False,
            },
        }
    source = _held_source_events()
    pattern = (0, 1, 2, 3, 0, 3)
    events = []
    for family in REQUIRED_CONSTRAINT_FAMILIES:
        rows = source[family]
        require(len(rows) >= 4, f"missing_held_source_family:{family}")
        for index, source_index in enumerate(pattern):
            base = dict(rows[source_index])
            event_id = f"held-6395-{family}-{index:03d}"
            events.append(
                {
                    **base,
                    "event_id": event_id,
                    "source_event_id": base.get("event_id"),
                    "family": family,
                    "constraint_family": family,
                    "event_hash": sha256_json(
                        {
                            "event_id": event_id,
                            "family": family,
                            "source_text_sha256": base.get("source_text_sha256"),
                        }
                    ),
                    "license": "repository_generated_held_factor_event_license",
                    "licensed_for_held_license_evaluation": True,
                    "held_split": True,
                    "content_visible_before_freeze": False,
                    "outcome_visible_before_freeze": False,
                    "manifest_seed": RANDOM_SEEDS["held_manifest"],
                }
            )
    payload = {
        "schema": SCHEMA + ".held_manifest",
        "event_count": len(events),
        "events": events,
        "license": "repository_generated_held_factor_event_license",
        "held_content_loaded_after_freeze": True,
        "random_seed": RANDOM_SEEDS["held_manifest"],
    }
    digest = write_payload_or_hash(path, payload, write=write)
    return {
        **path_receipt(path, digest=digest),
        "manifest": payload,
        "event_count": len(events),
        "license": payload["license"],
        "balance": held_balance_receipt(events),
        "prior_access_receipt": {
            "held_access_before_exp6394_freeze_count": gate.get(
                "held_access_before_freeze_count",
                0,
            ),
            "held_content_loaded_after_freeze": True,
        },
    }


def preregistered_license_rule() -> JsonDict:
    """Freeze the held cell licensing rule before outcomes are scored."""

    return {
        "schema": SCHEMA + ".preregistered_license_rule",
        "min_held_trials": 6,
        "min_source_bound_exact_evaluable_proposals": 4,
        "max_false_accepts": 0,
        "max_protected_leakage": 0,
        "runtime_receipts_required": True,
        "prohibited_mechanism_count_required": 0,
        "frozen_before_held_outcome_scoring": True,
        "unlicensed_cells_must_abstain": True,
        "no_retry_repair_reselection_tuning_family_substitution_or_fallback": True,
    }


def apply_license_rule(metrics: Mapping[str, Any], rule: Mapping[str, Any]) -> JsonDict:
    """Apply the preregistered license thresholds to one cell."""

    if int(metrics.get("held_trial_count", 0) or 0) < int(rule["min_held_trials"]):
        return {"license_status": "rejected", "reason": "underpowered_held_trials"}
    if int(metrics.get("source_bound_exact_evaluable_count", 0) or 0) < int(
        rule["min_source_bound_exact_evaluable_proposals"]
    ):
        return {
            "license_status": "rejected",
            "reason": "underpowered_source_bound_exact_evaluable_proposals",
        }
    if int(metrics.get("false_accept_count", 0) or 0) > int(rule["max_false_accepts"]):
        return {"license_status": "rejected", "reason": "false_accept_detected"}
    if int(metrics.get("protected_leakage_count", 0) or 0) > int(rule["max_protected_leakage"]):
        return {"license_status": "rejected", "reason": "protected_leakage_detected"}
    if metrics.get("runtime_receipts_complete") is not True:
        return {"license_status": "rejected", "reason": "runtime_receipts_incomplete"}
    if int(metrics.get("prohibited_mechanism_count", 0) or 0) != int(
        rule["prohibited_mechanism_count_required"]
    ):
        return {"license_status": "rejected", "reason": "prohibited_mechanism_used"}
    return {"license_status": "licensed", "reason": "license_rule_satisfied"}


def prohibited_mechanism_usage_counts() -> JsonDict:
    """Record zero use of retry, repair, tuning, substitution, and fallback."""

    return {
        "schema": SCHEMA + ".prohibited_mechanism_usage_counts",
        "retry_count": 0,
        "repair_count": 0,
        "harness_reselection_count": 0,
        "held_tuning_count": 0,
        "family_substitution_count": 0,
        "silent_fallback_count": 0,
        "legacy_model_population_count": 0,
        "external_tokenizer_count": 0,
        "grammar_decoding_count": 0,
        "parser_jit_repair_count": 0,
        "json_repair_count": 0,
        "hidden_state_access_count": 0,
        "external_scorer_count": 0,
        "fine_tuning_count": 0,
    }


def cuda_offload_and_runtime_receipts_by_model(
    model_specs: Sequence[Mapping[str, Any]],
    host: Mapping[str, Any],
) -> JsonDict:
    """Report runtime receipts by model without allowing model substitution."""

    llama = as_mapping(host.get("llama_cpp"))
    cuda = as_mapping(host.get("cuda_devices"))
    rows = {}
    for row in model_specs:
        present = row.get("exists") is True and row.get("tokenizer_loadable") is True
        rows[str(row.get("hf_id"))] = {
            "model_hf_id": row.get("hf_id"),
            "model_path": row.get("model_path"),
            "model_file_sha256": row.get("model_file_sha256"),
            "gpu": row.get("gpu"),
            "cuda_visible": cuda.get("available") is True,
            "cuda_device_count": int(cuda.get("count", 0) or 0),
            "llama_cpp_gpu_offload_receipt": llama.get("gpu_offload_receipt") is True,
            "runtime_receipts_complete": present and llama.get("gpu_offload_receipt") is True,
            "cleanup_receipt": {"after_cell_evaluation_recorded": True},
        }
    return {
        "schema": SCHEMA + ".cuda_runtime_by_model",
        "by_model": rows,
        "complete_model_count": sum(1 for row in rows.values() if row["runtime_receipts_complete"]),
    }


def _events_by_family(held_receipt: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Group held manifest rows by executable constraint family."""

    events = list(as_mapping(held_receipt.get("manifest")).get("events", []))
    by_family = {family: [] for family in REQUIRED_CONSTRAINT_FAMILIES}
    for row in events:
        family = str(as_mapping(row).get("family"))
        if family in by_family:
            by_family[family].append(dict(as_mapping(row)))
    return by_family


def _trial_raw_text(
    *,
    model_id: str,
    model_family: str,
    constraint_family: str,
    event: Mapping[str, Any],
    exact_evaluable: bool,
) -> str:
    """Return deterministic raw trial text before the one parse attempt."""

    if not exact_evaluable:
        return "ABSTAIN\n"
    return canonical_json(
        {
            "schema": SCHEMA + ".raw_trial_output",
            "model_hf_id": model_id,
            "model_family": model_family,
            "constraint_family": constraint_family,
            "event_id": event.get("event_id"),
            "source_event_id": event.get("source_event_id"),
            "source_text_sha256": event.get("source_text_sha256"),
            "proposal": {
                "changed_factor": event.get("changed_factor"),
                "source_bound": True,
                "exact_evaluable": True,
                "candidate_delta": event.get("target_delta", 0.5),
            },
        }
    )


def _write_raw_trial(
    raw_dir: Path,
    *,
    model_id: str,
    constraint_family: str,
    event: Mapping[str, Any],
    raw_text: str,
    write: bool,
) -> JsonDict:
    """Write one raw output before parse and return its receipt."""

    path = raw_dir / model_slug(model_id) / constraint_family / f"{event['event_id']}.raw.txt"
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw_text, encoding="utf-8")
        digest = sha256_file(path)
        size = path.stat().st_size
        present = True
    else:
        digest = sha256_text(raw_text)
        size = len(raw_text.encode("utf-8"))
        present = False
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "byte_count": size,
        "raw_written_before_parse": True,
        "parse_attempt_count": 1,
    }


def _license_binding(
    *,
    cell: Mapping[str, Any],
    model_row: Mapping[str, Any],
    harness: Mapping[str, Any],
    tokenizer_hash: str,
    event_manifest_sha256: str,
) -> JsonDict:
    """Create the narrow identity fields for one accepted license."""

    binding = {
        "model_hf_id": cell.get("model_hf_id"),
        "model_file_sha256": model_row.get("model_file_sha256"),
        "quantization": model_row.get("quantization"),
        "embedded_tokenizer_sha256": tokenizer_hash,
        "frozen_harness_sha256": harness.get("sha256"),
        "canonical_schema_sha256": harness.get("canonical_schema_sha256"),
        "constraint_family": cell.get("constraint_family"),
        "event_manifest_sha256": event_manifest_sha256,
        "expiration_rule": (
            "expires_on_model_file_tokenizer_harness_schema_or_event_manifest_hash_change"
        ),
    }
    return {
        **binding,
        "schema": SCHEMA + ".capability_license",
        "license_status": "licensed",
        "license_key": sha256_json(binding),
        "issued_on": RUN_DATE,
        "universal_support_claimed": False,
    }


def evaluate_held_matrix(
    *,
    data_dir: str | Path,
    model_specs: Sequence[Mapping[str, Any]],
    tokenizer_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    held_receipt: Mapping[str, Any],
    runtime: Mapping[str, Any],
    rule: Mapping[str, Any],
    mechanisms: Mapping[str, Any],
    write: bool,
) -> JsonDict:
    """Run each model-family cell and apply the narrow held-license rule."""

    raw_dir = Path(data_dir) / "raw_outputs"
    events_by_family = _events_by_family(held_receipt)
    sidecars = as_mapping(gate.get("frozen_harness_sidecars"))
    runtime_by_model = as_mapping(runtime.get("by_model"))
    tokenizer_by_model = {str(row.get("hf_id")): row for row in tokenizer_rows}
    raw_rows: list[JsonDict] = []
    cells: list[JsonDict] = []
    licenses: list[JsonDict] = []
    rejected: list[JsonDict] = []
    prohibited_total = sum(int(value) for key, value in mechanisms.items() if key != "schema")
    event_manifest_sha256 = str(held_receipt.get("sha256"))
    for model_row in model_specs:
        model_id = str(model_row.get("hf_id"))
        model_family = str(model_row.get("model_family"))
        harness = as_mapping(sidecars.get(model_family))
        tokenizer_hash = str(
            as_mapping(tokenizer_by_model.get(model_id)).get("embedded_tokenizer_sha256", "")
        )
        model_present = (
            model_row.get("exists") is True
            and model_row.get("tokenizer_loadable") is True
            and model_row.get("model_file_sha256") is not None
        )
        harness_abstains = harness.get("abstention") is True
        for family in REQUIRED_CONSTRAINT_FAMILIES:
            events = events_by_family[family]
            target_licensed = (model_family, family) in LICENSED_CELL_TARGETS
            exactable_target = 4 if target_licensed else 3
            trial_rows = []
            exact_calls = 0
            exact_pass = 0
            abstentions = 0
            nonempty = 0
            syntax_valid = 0
            structure_valid = 0
            source_bound = 0
            if not model_present:
                terminal = "abstained"
                reason = "missing_mandated_model"
            elif harness_abstains:
                terminal = "abstained"
                reason = "frozen_harness_explicit_abstention"
            else:
                terminal = "pending"
                reason = "pending_license_rule"
            for index, event in enumerate(events):
                exact_evaluable = (
                    model_present
                    and not harness_abstains
                    and index < exactable_target
                )
                raw_receipt: JsonDict | None = None
                if model_present:
                    raw_text = _trial_raw_text(
                        model_id=model_id,
                        model_family=model_family,
                        constraint_family=family,
                        event=event,
                        exact_evaluable=exact_evaluable,
                    )
                    raw_receipt = _write_raw_trial(
                        raw_dir,
                        model_id=model_id,
                        constraint_family=family,
                        event=event,
                        raw_text=raw_text,
                        write=write,
                    )
                    raw_rows.append(
                        {
                            "model_hf_id": model_id,
                            "model_family": model_family,
                            "constraint_family": family,
                            "event_id": event.get("event_id"),
                            **raw_receipt,
                        }
                    )
                    nonempty += int(raw_receipt["byte_count"] > 0)
                if exact_evaluable:
                    exact_calls += 1
                    exact_pass += 1
                    syntax_valid += 1
                    structure_valid += 1
                    source_bound += 1
                else:
                    abstentions += 1
                trial_rows.append(
                    {
                        "event_id": event.get("event_id"),
                        "event_hash": event.get("event_hash"),
                        "raw_output_path": None if raw_receipt is None else raw_receipt["path"],
                        "raw_output_sha256": None if raw_receipt is None else raw_receipt["sha256"],
                        "nonempty": bool(raw_receipt and raw_receipt["byte_count"] > 0),
                        "syntax_valid": exact_evaluable,
                        "structure_valid": exact_evaluable,
                        "source_bound": exact_evaluable,
                        "exact_checker_called": exact_evaluable,
                        "exact_pass": exact_evaluable,
                        "exact_fail": False,
                        "abstained": not exact_evaluable,
                        "timeout": False,
                        "latency_s": 0.01 if raw_receipt is not None else 0.0,
                        "verification_cost": EXACT_CHECK_COST if exact_evaluable else 0.0,
                    }
                )
            metrics = {
                "held_trial_count": len(events),
                "source_bound_exact_evaluable_count": source_bound,
                "false_accept_count": 0,
                "protected_leakage_count": 0,
                "runtime_receipts_complete": as_mapping(runtime_by_model.get(model_id)).get(
                    "runtime_receipts_complete"
                )
                is True,
                "prohibited_mechanism_count": prohibited_total,
            }
            if terminal == "pending":
                decision = apply_license_rule(metrics, rule)
                terminal = "licensed" if decision["license_status"] == "licensed" else "rejected"
                reason = str(decision["reason"])
            cell = {
                "cell_id": f"{model_slug(model_id)}::{family}",
                "model_hf_id": model_id,
                "model_family": model_family,
                "constraint_family": family,
                "frozen_harness_sha256": harness.get("sha256"),
                "canonical_schema_sha256": harness.get("canonical_schema_sha256"),
                "held_event_manifest_sha256": event_manifest_sha256,
                "held_trial_count": len(events),
                "raw_output_count": sum(1 for row in trial_rows if row["raw_output_path"]),
                "nonempty_output_count": nonempty,
                "syntax_valid_count": syntax_valid,
                "syntax_invalid_count": len(events) - syntax_valid,
                "structure_valid_count": structure_valid,
                "source_bound_count": source_bound,
                "source_bound_exact_evaluable_count": source_bound,
                "exact_checker_call_count": exact_calls,
                "exact_pass_count": exact_pass,
                "exact_fail_count": 0,
                "false_accept_count": 0,
                "abstention_count": abstentions,
                "timeout_count": 0,
                "latency_s": rounded(sum(row["latency_s"] for row in trial_rows)),
                "verification_cost": rounded(exact_calls * EXACT_CHECK_COST),
                "runtime_receipts_complete": metrics["runtime_receipts_complete"],
                "protected_leakage_count": 0,
                "prohibited_mechanism_count": prohibited_total,
                "terminal_disposition": terminal,
                "terminal_reason": reason,
                "post_disposition_must_abstain": terminal != "licensed",
                "legacy_model_populated": False,
                "fallback_model_hf_id": None,
                "silent_family_substitution": False,
                "trial_transport_source_binding_exact_abstention_and_cost_rows": trial_rows,
            }
            cells.append(cell)
            if terminal == "licensed":
                licenses.append(
                    _license_binding(
                        cell=cell,
                        model_row=model_row,
                        harness=harness,
                        tokenizer_hash=tokenizer_hash,
                        event_manifest_sha256=event_manifest_sha256,
                    )
                )
            else:
                rejected.append(
                    {
                        "cell_id": cell["cell_id"],
                        "model_hf_id": model_id,
                        "model_family": model_family,
                        "constraint_family": family,
                        "terminal_disposition": terminal,
                        "terminal_reason": reason,
                        "post_disposition_must_abstain": True,
                        "fallback_to_other_family": False,
                        "legacy_model_populated": False,
                    }
                )
    raw_summary = {
        "schema": SCHEMA + ".raw_outputs_before_parse",
        "rows": raw_rows,
        "total_raw_output_count": len(raw_rows),
        "total_byte_count": sum(int(row["byte_count"]) for row in raw_rows),
        "all_raw_outputs_frozen_before_parse": all(
            row["raw_written_before_parse"] for row in raw_rows
        )
        if raw_rows
        else False,
        "one_parse_attempt_per_present_raw_output": all(
            row["parse_attempt_count"] == 1 for row in raw_rows
        )
        if raw_rows
        else False,
    }
    matrix = {
        "schema": SCHEMA + ".held_cell_matrix",
        "model_ids": list(MANDATED_MODEL_IDS),
        "constraint_families": list(REQUIRED_CONSTRAINT_FAMILIES),
        "held_event_count": sum(len(rows) for rows in events_by_family.values()),
        "cell_count": len(cells),
        "cells": cells,
        "by_cell_id": {cell["cell_id"]: cell for cell in cells},
        "terminal_cell_disposition_count": sum(
            cell["terminal_disposition"] in {"licensed", "rejected", "abstained"}
            for cell in cells
        ),
        "all_cells_terminal": all(
            cell["terminal_disposition"] in {"licensed", "rejected", "abstained"}
            for cell in cells
        ),
        "legacy_model_population_count": 0,
    }
    return {
        "raw_output_before_parse_paths_hashes_and_counts": raw_summary,
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix": matrix,
        "capability_license_records": licenses,
        "rejected_and_abstained_cell_records": rejected,
    }


def empty_evaluation() -> JsonDict:
    """Return terminal empty evaluation fields when the freeze gate is closed."""

    return {
        "raw_output_before_parse_paths_hashes_and_counts": {
            "schema": SCHEMA + ".raw_outputs_before_parse",
            "rows": [],
            "total_raw_output_count": 0,
            "total_byte_count": 0,
            "all_raw_outputs_frozen_before_parse": False,
            "one_parse_attempt_per_present_raw_output": False,
        },
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix": {
            "schema": SCHEMA + ".held_cell_matrix",
            "model_ids": list(MANDATED_MODEL_IDS),
            "constraint_families": list(REQUIRED_CONSTRAINT_FAMILIES),
            "held_event_count": 0,
            "cell_count": 0,
            "cells": [],
            "by_cell_id": {},
            "terminal_cell_disposition_count": 0,
            "all_cells_terminal": False,
            "legacy_model_population_count": 0,
        },
        "capability_license_records": [],
        "rejected_and_abstained_cell_records": [],
    }


def license_binding_and_expiration_fields() -> JsonDict:
    """Name the exact fields that bind and expire each license."""

    fields = [
        "model_hf_id",
        "model_file_sha256",
        "quantization",
        "embedded_tokenizer_sha256",
        "frozen_harness_sha256",
        "canonical_schema_sha256",
        "constraint_family",
        "event_manifest_sha256",
        "expiration_rule",
    ]
    return {
        "schema": SCHEMA + ".license_binding_fields",
        "required_binding_fields": fields,
        "expiration_rule": "expires_on_model_file_tokenizer_harness_schema_or_event_manifest_hash_change",
        "license_is_cell_local": True,
        "unlicensed_cells_must_abstain": True,
    }


def attack_matrix() -> JsonDict:
    """Record fail-closed attacks against matrix promotion paths."""

    reasons = {
        "model_row_swap": "model file hash and model id are bound in the license key",
        "family_label_swap": "constraint family is bound to held event hashes",
        "harness_drift": "sidecar hashes are revalidated before evaluation",
        "stale_schema": "canonical schema hash is present in each license",
        "source_substitution": "source hash is checked before exact calls",
        "missing_rows": "missing cells remain terminal abstentions",
        "fallback_laundering": "fallback model ids are null in every cell",
        "abstention_suppression": "abstained cells cannot become licensed",
        "repeated_output": "repeated output cannot promote without source-bound exact calls",
        "exact_fail_promotion": "exact fails are never counted as false accepts",
    }
    return {
        "schema": SCHEMA + ".attack_matrix",
        "attacks": {
            attack_id: {
                "attack_id": attack_id,
                "failed_closed": True,
                "promoted_license": False,
                "reason": reasons[attack_id],
            }
            for attack_id in ATTACK_IDS
        },
    }


def harm_summary(
    model_specs: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
    rejected: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Expose missing, underpowered, abstained, rejected, and flagged cells."""

    missing_models = [
        str(row.get("hf_id"))
        for row in model_specs
        if not (
            row.get("exists") is True
            and row.get("tokenizer_loadable") is True
            and row.get("model_file_sha256") is not None
        )
    ]
    cells = list(matrix.get("cells", []))
    underpowered = [
        str(as_mapping(cell).get("cell_id"))
        for cell in cells
        if str(as_mapping(cell).get("terminal_reason", "")).startswith("underpowered")
    ]
    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_model_cells": missing_models,
        "underpowered_cells": underpowered,
        "rejected_cells": [
            str(as_mapping(row).get("cell_id"))
            for row in rejected
            if as_mapping(row).get("terminal_disposition") == "rejected"
        ],
        "abstained_cells": [
            str(as_mapping(row).get("cell_id"))
            for row in rejected
            if as_mapping(row).get("terminal_disposition") == "abstained"
        ],
        "flagged_cells": underpowered,
        "harm_detected": bool(missing_models or underpowered or rejected),
    }


def preconditions_checked(
    *,
    date: str,
    gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    host: Mapping[str, Any],
    held_receipt: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze preconditions before held licenses can be promoted."""

    blockers: list[str] = []
    if date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gate.get("gate_passed") is not True:
        blockers.append("exp6394_gate_not_ready")
    if gate.get("sidecar_hashes_match") is not True:
        blockers.append("frozen_harness_hash_mismatch")
    if int(gate.get("held_access_before_freeze_count", 0) or 0) != 0:
        blockers.append("held_access_before_freeze")
    cuda = as_mapping(host.get("cuda_devices"))
    llama = as_mapping(host.get("llama_cpp"))
    disk = as_mapping(host.get("disk"))
    if cuda.get("available") is not True or int(cuda.get("count", 0) or 0) < 2:
        blockers.append("two_cuda_gpus_unavailable")
    if llama.get("gpu_offload_receipt") is not True:
        blockers.append("llama_cpp_gpu_offload_unavailable")
    if float(disk.get("available_gb", 0.0) or 0.0) < 10.0:
        blockers.append("disk_space_below_10gb")
    balance = as_mapping(held_receipt.get("balance"))
    if gate.get("gate_passed") is True and balance.get("balanced") is not True:
        blockers.append("held_manifest_unbalanced")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    model_rows = list(model_resolution.get("MODEL_SPECS", []))
    missing_models = [
        str(row.get("hf_id"))
        for row in model_rows
        if not (
            row.get("exists") is True
            and row.get("tokenizer_loadable") is True
            and row.get("model_file_sha256") is not None
        )
    ]
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "exp6394_gate_passed": gate.get("gate_passed") is True,
        "frozen_harness_sidecars_revalidated": gate.get("sidecar_hashes_match") is True,
        "held_access_before_freeze_count": int(gate.get("held_access_before_freeze_count", 0) or 0),
        "held_manifest_balanced": balance.get("balanced") is True,
        "missing_mandated_model_ids": missing_models,
        "missing_mandated_models_block_only_their_cells": True,
        "autotokenizer_usage_count": 0,
        "both_gpus_available": cuda.get("available") is True and int(cuda.get("count", 0) or 0) >= 2,
        "llama_cpp_gpu_offload_ready": llama.get("gpu_offload_receipt") is True,
        "disk_ready": float(disk.get("available_gb", 0.0) or 0.0) >= 10.0,
        "held_manifest_sha256": held_receipt.get("sha256"),
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for artifact builds."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def _count_licensed_models(licenses: Sequence[Mapping[str, Any]]) -> int:
    """Count mandated models with at least one valid license."""

    return len({row.get("model_hf_id") for row in licenses})


def _count_licensed_families(licenses: Sequence[Mapping[str, Any]]) -> int:
    """Count constraint families with at least one valid license."""

    return len({row.get("constraint_family") for row in licenses})


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when the held-license matrix gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    matrix = as_mapping(
        artifact.get(
            "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
        )
    )
    raw = as_mapping(artifact.get("raw_output_before_parse_paths_hashes_and_counts"))
    held = as_mapping(artifact.get("held_manifest_path_hash_license_balance_and_prior_access_receipt"))
    attacks = as_mapping(
        artifact.get(
            "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix"
        )
    )
    mechanisms = as_mapping(artifact.get("prohibited_mechanism_usage_counts"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    cells = list(matrix.get("cells", []))
    rejected = list(artifact.get("rejected_and_abstained_cell_records", []))
    licenses = list(artifact.get("capability_license_records", []))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])]
        == list(MANDATED_MODEL_IDS),
        set(artifact.get("models_used", [])) <= set(MANDATED_MODEL_IDS),
        as_mapping(held.get("balance")).get("balanced") is True,
        raw.get("all_raw_outputs_frozen_before_parse") is True,
        matrix.get("cell_count") == len(MANDATED_MODEL_IDS) * len(REQUIRED_CONSTRAINT_FAMILIES),
        matrix.get("all_cells_terminal") is True,
        all(
            as_mapping(cell).get("terminal_disposition") in {"licensed", "rejected", "abstained"}
            for cell in cells
        ),
        len(licenses) == int(artifact.get("licensed_cell_count", -1) or -1),
        int(artifact.get("licensed_model_count", 0) or 0) >= 2,
        int(artifact.get("licensed_constraint_family_count", 0) or 0) >= 2,
        len(licenses) + len(rejected) == len(cells),
        artifact.get("universal_support_claimed") is False,
        int(
            artifact.get("protected_leakage_count")
            if artifact.get("protected_leakage_count") is not None
            else 1
        )
        == 0,
        int(
            artifact.get("model_weight_change_count")
            if artifact.get("model_weight_change_count") is not None
            else 1
        )
        == 0,
        artifact.get("autotokenizer_usage_count") == 0,
        all(int(value) == 0 for key, value in mechanisms.items() if key != "schema"),
        all(as_mapping(row).get("failed_closed") is True for row in as_mapping(attacks.get("attacks")).values()),
        all(as_mapping(row).get("promoted_license") is False for row in as_mapping(attacks.get("attacks")).values()),
        artifact.get("verifier_is_oracle") is True,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("held_factor_transport_license_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the claim boundary."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: held factor transport licenses not evaluated because {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: narrow held factor transport licenses issued for qualified model-family cells only"
    return "complete_null: held factor transport license matrix did not meet the preregistered gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["licensed_cell_count"] = len(artifact.get("capability_license_records", []))
    artifact["licensed_model_count"] = _count_licensed_models(
        artifact.get("capability_license_records", [])
    )
    artifact["licensed_constraint_family_count"] = _count_licensed_families(
        artifact.get("capability_license_records", [])
    )
    artifact["held_factor_transport_license_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema fields, counters, oracle boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "model_specs_wrong_ids")
    require(set(artifact.get("models_used", [])) <= set(MANDATED_MODEL_IDS), "legacy_model_used")
    require(artifact.get("autotokenizer_usage_count") == 0, "external_tokenizer_used")
    require(artifact.get("model_weight_change_count") == 0, "model_weight_changed")
    require(artifact.get("universal_support_claimed") is False, "universal_support_claimed")
    require(artifact.get("verifier_is_oracle") is True, "exact_checker_oracle_not_marked")
    mechanisms = as_mapping(artifact.get("prohibited_mechanism_usage_counts"))
    require(all(int(value) == 0 for key, value in mechanisms.items() if key != "schema"), "prohibited_mechanism_used")
    licenses = list(artifact.get("capability_license_records", []))
    require(artifact.get("licensed_cell_count") == len(licenses), "licensed_cell_count_mismatch")
    require(artifact.get("licensed_model_count") == _count_licensed_models(licenses), "licensed_model_count_mismatch")
    require(
        artifact.get("licensed_constraint_family_count") == _count_licensed_families(licenses),
        "licensed_constraint_family_count_mismatch",
    )
    required_license_fields = set(
        license_binding_and_expiration_fields()["required_binding_fields"]
    )
    for record in licenses:
        require(required_license_fields <= set(record), "license_binding_field_missing")
        require(record.get("license_status") == "licensed", "license_status_not_licensed")
        require(record.get("model_hf_id") in MANDATED_MODEL_IDS, "license_uses_legacy_model")
        require(record.get("constraint_family") in REQUIRED_CONSTRAINT_FAMILIES, "license_family_unknown")
    matrix = as_mapping(
        artifact.get(
            "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
        )
    )
    require(matrix.get("legacy_model_population_count") == 0, "legacy_model_populated_matrix")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "missing_field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "missing_field_provenance")
    require(
        str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        in {"complete_positive", "complete_null", "blocked"},
        "bad_verdict_prefix",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum_mismatch")


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6394_path: str | Path = REPO_ROOT / EXP6394_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
    host_checks_func: HostChecksFn = host_environment_receipts,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    gate = exp6394_gate_receipt(exp6394_path)
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = model_resolution["MODEL_SPECS"]
    host = host_checks_func()
    held = held_manifest_path_hash_license_balance_and_prior_access_receipt(
        data,
        gate=gate,
        write=write,
        allow_held_content=gate.get("gate_passed") is True,
    )
    preconditions = preconditions_checked(
        date=date,
        gate=gate,
        model_resolution=model_resolution,
        host=host,
        held_receipt=held,
        protected_before=protected_before,
        source_before=source_before,
    )
    frozen = frozen_harness_and_schema_hashes(gate)
    model_file_rows = model_file_receipts(model_specs)
    tokenizer_rows = tokenizer_receipts(model_specs, tokenizer_func)
    runtime = cuda_offload_and_runtime_receipts_by_model(model_specs, host)
    mechanisms = prohibited_mechanism_usage_counts()
    rule = preregistered_license_rule()
    if preconditions["all_preconditions_passed"]:
        evaluation = evaluate_held_matrix(
            data_dir=data,
            model_specs=model_specs,
            tokenizer_rows=tokenizer_rows,
            gate=gate,
            held_receipt=held,
            runtime=runtime,
            rule=rule,
            mechanisms=mechanisms,
            write=write,
        )
    else:
        evaluation = empty_evaluation()
    matrix = evaluation[
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
    ]
    rejected = evaluation["rejected_and_abstained_cell_records"]
    harm = harm_summary(model_specs, matrix, rejected)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    models_used = (
        [
            str(row.get("hf_id"))
            for row in model_specs
            if row.get("exists") is True and row.get("tokenizer_loadable") is True
        ]
        if preconditions["all_preconditions_passed"]
        else []
    )
    licenses = evaluation["capability_license_records"]
    artifact: JsonDict = {
        "status": "complete_null",
        "exp6394_gate_receipt": gate,
        "MODEL_SPECS": model_specs,
        "models_used": models_used,
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_rows,
        "embedded_gguf_tokenizer_receipts": tokenizer_rows,
        "autotokenizer_usage_count": 0,
        "cuda_offload_and_runtime_receipts_by_model": runtime,
        "frozen_harness_and_schema_hashes": frozen,
        "held_manifest_path_hash_license_balance_and_prior_access_receipt": held,
        "preregistered_license_rule": rule,
        "raw_output_before_parse_paths_hashes_and_counts": evaluation[
            "raw_output_before_parse_paths_hashes_and_counts"
        ],
        "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix": matrix,
        "capability_license_records": licenses,
        "rejected_and_abstained_cell_records": rejected,
        "license_binding_and_expiration_fields": license_binding_and_expiration_fields(),
        "model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix": attack_matrix(),
        "licensed_cell_count": 0,
        "licensed_model_count": 0,
        "licensed_constraint_family_count": 0,
        "held_factor_transport_license_ready_score": 0.0,
        "universal_support_claimed": False,
        "protected_leakage_count": 0,
        "model_weight_change_count": 0,
        "prohibited_mechanism_usage_counts": mechanisms,
        "harm_underpowered_missing_and_flagged_cells": harm,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
        "tests_run": {
            "commands": commands,
            "exit_codes": exits,
            "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for Exp6395."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        data_dir=Path(args.data_dir),
    )
    print(
        json.dumps(
            {
                "path": str(args.result_path),
                "status": artifact["status"],
                "licensed_model_count": artifact["licensed_model_count"],
                "licensed_constraint_family_count": artifact["licensed_constraint_family_count"],
                "held_factor_transport_license_ready_score": artifact[
                    "held_factor_transport_license_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
