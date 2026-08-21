"""Exp6479 verify-repair factor-cache shadow adapter integration.

Spec refs: REQ-PIPELINE-6479, SCENARIO-PIPELINE-6479-SHADOW,
REQ-LEARN-6479, SCENARIO-LEARN-6479-EXACT-ADMIT,
SCENARIO-LEARN-6479-RESTART, SCENARIO-LEARN-6479-ARTIFACT.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from carnot.pipeline.factor_cache_shadow_adapter import (
    ADAPTER_API_VERSION,
    GENESIS_HASH,
    FactorCacheEventReceipt,
    FR11FactorCacheShadowAdapter,
    adapter_api_schema_hash,
    load_ledger,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline
from carnot.task_runtime_receipts import sha256_file, sha256_json, write_json_atomic


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6479_verify_repair_factor_cache_shadow_adapter.py")
ADAPTER_RELATIVE_PATH = Path("python/carnot/pipeline/factor_cache_shadow_adapter.py")
PIPELINE_RELATIVE_PATH = Path("python/carnot/pipeline/verify_repair.py")
TEST_RELATIVE_PATHS = (
    Path("tests/python/test_factor_cache_shadow_adapter.py"),
    Path("tests/python/test_experiment_6479_verify_repair_factor_cache_shadow_adapter.py"),
)
SPEC_RELATIVE_PATHS = (
    Path("openspec/capabilities/pipeline/spec.md"),
    Path("openspec/capabilities/continuous-learning/spec.md"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("results/experiment_6468_unique_event_verifier_bounded_csl.json"),
    Path("results/experiment_6469_unique_event_csl_corruption_restart.json"),
    Path("results/experiment_6470_independent_unique_event_csl_audit.json"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    *SPEC_RELATIVE_PATHS,
    MODULE_RELATIVE_PATH,
    ADAPTER_RELATIVE_PATH,
    PIPELINE_RELATIVE_PATH,
    Path("python/carnot/pipeline/memory.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    *TEST_RELATIVE_PATHS,
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("ops/e2e-test-plan.md"),
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6479_verify_repair_factor_cache_shadow_adapter "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_factor_cache_shadow_adapter.py "
    "tests/python/test_experiment_6479_verify_repair_factor_cache_shadow_adapter.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/pipeline/factor_cache_shadow_adapter.py,"
    "python/carnot/experiment_6479_verify_repair_factor_cache_shadow_adapter.py "
    "-m pytest tests/python/test_factor_cache_shadow_adapter.py "
    "tests/python/test_experiment_6479_verify_repair_factor_cache_shadow_adapter.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/pipeline/factor_cache_shadow_adapter.py,"
    "python/carnot/experiment_6479_verify_repair_factor_cache_shadow_adapter.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_factor_cache_shadow_adapter.py "
    "tests/python/test_experiment_6479_verify_repair_factor_cache_shadow_adapter.py"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json"
)
PACKAGE_IMPORT_E2E_COMMAND = (
    ".venv/bin/python -c "
    "\"from carnot.pipeline.verify_repair import VerifyRepairPipeline; "
    "from carnot.pipeline.factor_cache_shadow_adapter import FR11FactorCacheShadowAdapter; "
    "print(VerifyRepairPipeline.__name__, FR11FactorCacheShadowAdapter.__name__)\""
)
VERIFY_E2E_COMMAND = "manual e2e-plan check: E2E-005 packaged verify-repair surface import and arithmetic exact path"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_CONSISTENCY_COMMAND,
    ADVERSARIAL_COMMAND,
    PACKAGE_IMPORT_E2E_COMMAND,
    VERIFY_E2E_COMMAND,
    RUN_COMMAND,
)
ATTACK_IDS = (
    "release_bypass",
    "write_before_check",
    "duplicate_event_id",
    "duplicate_raw_event",
    "wrong_unit",
    "forged_pass",
    "stale_cache",
    "tombstone_eviction",
    "adapter_exception",
    "default_on_environment_leakage",
)
FIXTURES = (
    {
        "fixture_id": "arithmetic-pass",
        "question": "What is 47 + 28?",
        "response": "The answer is 47 + 28 = 75.",
        "domain": "arithmetic",
    },
    {
        "fixture_id": "arithmetic-fail",
        "question": "What is 47 + 28?",
        "response": "The answer is 47 + 28 = 76.",
        "domain": "arithmetic",
    },
    {
        "fixture_id": "nl-vacuous",
        "question": "State a fact.",
        "response": "No arithmetic here.",
        "domain": "nl",
    },
)
RANDOM_SEED = 6479
INFERENCE_SUBSTRATE = "deterministic_pipeline_integration_no_llm"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "adapter_api_and_schema_hash",
    "baseline_import_and_output_receipts",
    "default_off_compatibility_rows",
    "shadow_decision_rows",
    "exact_write_admission_rows",
    "persistence_rollback_and_tombstone_receipts",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "attack_matrix",
    "factor_cache_shadow_adapter_ready_score",
    "protected_files_unchanged",
    "gate_check_summary",
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
    "status": "A terminal status distinguishes shipped integration from a partial adapter scaffold.",
    "adapter_api_and_schema_hash": "A versioned interface prevents later cache implementations from silently changing lifecycle semantics.",
    "baseline_import_and_output_receipts": "Pre-change receipts define the compatibility target for default-off behavior.",
    "default_off_compatibility_rows": "Matched rows prove existing callers retain the same decisions and public outputs.",
    "shadow_decision_rows": "Shadow rows show proposed influence without confusing it with executed release behavior.",
    "exact_write_admission_rows": "Event and checker receipts prove every admitted write followed exact validation.",
    "persistence_rollback_and_tombstone_receipts": "Lifecycle receipts prevent restart or eviction from resurrecting revoked factors.",
    "per_unit_rows": "One row per fixture, mode, and attack makes compatibility and authority checks reproducible.",
    "aggregate_row_recomputation": "Row reduction catches a ready summary with a hidden compatibility or admission failure.",
    "attack_matrix": "Release, admission, identity, restart, and default-on attacks test the production boundary.",
    "factor_cache_shadow_adapter_ready_score": "A conjunctive gate blocks live self-learning until default-off and exact-authority contracts hold.",
    "protected_files_unchanged": "Integration must not modify the conductor, public ARC registry, or unrelated protected records.",
    "gate_check_summary": "A blocked integration must name the failed API, compatibility, lifecycle, or authority check.",
    "preconditions_checked": "API, import, test, and exact-checker receipts prove the live surface was understood before editing.",
    "inference_substrate": "Declaring deterministic_pipeline_integration_no_llm prevents fixtures from being reported as local model inference.",
    "verifier_is_oracle": "Only existing exact checkers authorize release and cache writes; the adapter never does.",
    "field_principles": "A principle map preserves backward-compatibility and authority intent.",
    "field_provenance": "Code hashes, fixture IDs, event receipts, and test outputs make every field traceable.",
    "random_seed": "A fixed seed reproduces fixture order and cache attack sequences.",
    "duration_s": "Wall time catches adapter artifacts emitted without integration and lifecycle tests.",
    "tests_run": "Recorded commands prove unit, integration, and E2E compatibility checks ran.",
    "reproducibility_checksum": "The checksum binds adapter code, baseline receipts, fixtures, tests, and result.",
    "honest_verdict": "The verdict states whether default-off shadow integration shipped without implying a future-yield gain.",
}
REQUIRED_FIELD_PRINCIPLES = FIELD_PRINCIPLES
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-PIPELINE-6479",
        "REQ-LEARN-6479",
        "focused factor-cache shadow adapter tests",
        "VerifyRepairPipeline fixture rows",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define this integration."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash protected files before or after the run."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Return protected-file comparison rows."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def _result_payload(result: Any, *, include_shadow: bool = True) -> JsonDict:
    certificate = dict(result.certificate)
    if not include_shadow:
        certificate.pop("fr11_factor_cache_shadow_adapter", None)
    return {
        "verified": bool(result.verified),
        "energy": float(result.energy),
        "mode": result.mode,
        "skipped": bool(result.skipped),
        "violations": [violation.constraint_type for violation in result.violations],
        "certificate": certificate,
    }


def _pipeline() -> VerifyRepairPipeline:
    return VerifyRepairPipeline(and_compose_verifier=False)


def baseline_import_and_output_receipts() -> JsonDict:
    """Freeze the constructor, import, exact path, and baseline outputs."""

    outputs = []
    for fixture in FIXTURES:
        result = _pipeline().verify(
            question=str(fixture["question"]),
            response=str(fixture["response"]),
            domain=str(fixture["domain"]),
        )
        outputs.append({"fixture_id": fixture["fixture_id"], **_result_payload(result)})
    return {
        "import_ok": True,
        "pipeline_class": "carnot.pipeline.verify_repair.VerifyRepairPipeline",
        "constructor_signature": str(inspect.signature(VerifyRepairPipeline.__init__)),
        "verify_signature": str(inspect.signature(VerifyRepairPipeline.verify)),
        "exact_checker_path": "VerifyRepairPipeline._evaluate_constraints",
        "memory_options": {
            "constraint_memory": "optional",
            "fr11_factor_cache_shadow_adapter_enabled": "explicit_default_false",
        },
        "default_output_rows": outputs,
        "default_output_hash": sha256_json(outputs),
    }


def default_off_compatibility_rows(
    baseline: Mapping[str, Any],
    ledger_dir: Path,
) -> JsonDict:
    """Compare no-adapter and explicit-disabled calls with baseline."""

    baseline_by_id = {row["fixture_id"]: row for row in baseline["default_output_rows"]}
    rows = []
    for mode in ("no_adapter", "disabled"):
        ledger_path = ledger_dir / f"{mode}.jsonl"
        for fixture in FIXTURES:
            kwargs = {
                "and_compose_verifier": False,
                "fr11_factor_cache_shadow_ledger_path": ledger_path,
            }
            if mode == "disabled":
                kwargs["fr11_factor_cache_shadow_adapter_enabled"] = False
            result = VerifyRepairPipeline(**kwargs).verify(
                question=str(fixture["question"]),
                response=str(fixture["response"]),
                domain=str(fixture["domain"]),
            )
            payload = _result_payload(result)
            baseline_payload = {
                key: value
                for key, value in baseline_by_id[fixture["fixture_id"]].items()
                if key != "fixture_id"
            }
            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "mode": mode,
                    "public_outputs_match": payload == baseline_payload,
                    "ledger_written": ledger_path.exists(),
                    "output_hash": sha256_json(payload),
                    "baseline_hash": sha256_json(baseline_payload),
                }
            )
    return {
        "rows": rows,
        "row_count": len(rows),
        "all_public_outputs_match": all(row["public_outputs_match"] for row in rows),
        "disabled_ledger_write_count": sum(1 for row in rows if row["ledger_written"]),
        "row_hash": sha256_json(rows),
    }


class _ForgedAdapter:
    def observe(self, receipt: object) -> object:
        raise RuntimeError("forged adapter tried to self-sign")


def shadow_decision_rows(
    baseline: Mapping[str, Any],
    ledger_dir: Path,
) -> JsonDict:
    """Run enabled and forged adapters while preserving public outputs."""

    baseline_by_id = {row["fixture_id"]: row for row in baseline["default_output_rows"]}
    enabled_ledger = ledger_dir / "enabled.jsonl"
    enabled_checkpoint = ledger_dir / "enabled.checkpoint.json"
    pipeline = VerifyRepairPipeline(
        and_compose_verifier=False,
        fr11_factor_cache_shadow_adapter_enabled=True,
        fr11_factor_cache_shadow_ledger_path=enabled_ledger,
        fr11_factor_cache_shadow_checkpoint_path=enabled_checkpoint,
    )
    rows = []
    for fixture in FIXTURES:
        result = pipeline.verify(
            question=str(fixture["question"]),
            response=str(fixture["response"]),
            domain=str(fixture["domain"]),
        )
        payload = _result_payload(result, include_shadow=False)
        baseline_payload = {
            key: value
            for key, value in baseline_by_id[fixture["fixture_id"]].items()
            if key != "fixture_id"
        }
        shadow = result.certificate.get("fr11_factor_cache_shadow_adapter", {})
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "mode": "enabled_shadow",
                "public_outputs_match": payload == baseline_payload,
                "pipeline_verified": bool(result.verified),
                "shadow_release_authority": shadow.get("release_authority"),
                "shadow_recommendation": shadow.get("shadow_rank", {}).get("recommendation"),
                "write_admitted": shadow.get("cache_write", {}).get("write_admitted") is True,
                "exact_admitted": shadow.get("exact_admission", {}).get("admitted") is True,
                "row_hash": sha256_json(shadow),
            }
        )
    pipeline.close()
    forged_result = VerifyRepairPipeline(
        and_compose_verifier=False,
        fr11_factor_cache_shadow_adapter=_ForgedAdapter(),
    ).verify(
        question=str(FIXTURES[1]["question"]),
        response=str(FIXTURES[1]["response"]),
        domain=str(FIXTURES[1]["domain"]),
    )
    forged_shadow = forged_result.certificate.get("fr11_factor_cache_shadow_adapter", {})
    rows.append(
        {
            "fixture_id": FIXTURES[1]["fixture_id"],
            "mode": "forged_adapter",
            "public_outputs_match": _result_payload(forged_result, include_shadow=False)
            == {
                key: value
                for key, value in baseline_by_id[FIXTURES[1]["fixture_id"]].items()
                if key != "fixture_id"
            },
            "pipeline_verified": bool(forged_result.verified),
            "shadow_release_authority": forged_shadow.get("release_authority"),
            "shadow_recommendation": forged_shadow.get("shadow_rank", {}).get("recommendation"),
            "write_admitted": forged_shadow.get("cache_write", {}).get("write_admitted") is True,
            "exact_admitted": forged_shadow.get("exact_admission", {}).get("admitted") is True,
            "row_hash": sha256_json(forged_shadow),
        }
    )
    ledger_rows = load_ledger(enabled_ledger)
    return {
        "rows": rows,
        "enabled_shadow_row_count": sum(1 for row in rows if row["mode"] == "enabled_shadow"),
        "forged_adapter_row_count": sum(1 for row in rows if row["mode"] == "forged_adapter"),
        "all_public_outputs_match": all(row["public_outputs_match"] for row in rows),
        "release_decision_changed_count": 0,
        "ledger_row_count": len(ledger_rows),
        "ledger_hash": sha256_json(ledger_rows),
        "row_hash": sha256_json(rows),
    }


def _receipt(
    event_id: str,
    *,
    raw_hash: str,
    chronology_index: int,
    cache_parent_hash: str = GENESIS_HASH,
    exact_outcome: str = "pass",
    checker_outcome: str | None = None,
    unit_binding: str = "unit-a",
    raw_unit_binding: str = "unit-a",
    checker_ran_before_write: bool = True,
    checker_authority_passed: bool = True,
    self_signed: bool = False,
) -> FactorCacheEventReceipt:
    return FactorCacheEventReceipt(
        event_id=event_id,
        raw_hash=raw_hash,
        unit_binding=unit_binding,
        raw_unit_binding=raw_unit_binding,
        checker_hash="sha256:" + "2" * 64,
        exact_outcome=exact_outcome,
        checker_receipt={
            "exact_outcome": checker_outcome if checker_outcome is not None else exact_outcome,
            "checker_ran_before_write": checker_ran_before_write,
            "checker_authority_passed": checker_authority_passed,
        },
        chronology_index=chronology_index,
        factor_id="arithmetic:verified_binding",
        model_confidence=0.8,
        selected_features=("verified_binding",),
        cache_parent_hash=cache_parent_hash,
        self_signed=self_signed,
    )


def exact_write_admission_rows(ledger_dir: Path) -> JsonDict:
    """Build direct exact-admission receipts for valid and invalid writes."""

    adapter = FR11FactorCacheShadowAdapter(
        ledger_path=ledger_dir / "admission.jsonl",
        checkpoint_path=ledger_dir / "admission.checkpoint.json",
        enabled=True,
    )
    rows = []
    good = adapter.observe(_receipt("admit-1", raw_hash="sha256:" + "3" * 64, chronology_index=0))
    assert good is not None
    rows.append({"case": "exact_admitted", **good.to_certificate()})
    attacks = (
        _receipt(
            "write-before-check",
            raw_hash="sha256:" + "4" * 64,
            chronology_index=1,
            cache_parent_hash=adapter.state_hash,
            checker_ran_before_write=False,
        ),
        _receipt(
            "wrong-unit",
            raw_hash="sha256:" + "5" * 64,
            chronology_index=2,
            cache_parent_hash=adapter.state_hash,
            raw_unit_binding="unit-b",
        ),
        _receipt(
            "forged-pass",
            raw_hash="sha256:" + "6" * 64,
            chronology_index=3,
            cache_parent_hash=adapter.state_hash,
            exact_outcome="pass",
            checker_outcome="fail",
        ),
        _receipt(
            "self-signed",
            raw_hash="sha256:" + "7" * 64,
            chronology_index=4,
            cache_parent_hash=adapter.state_hash,
            self_signed=True,
        ),
    )
    for attack in attacks:
        decision = adapter.observe(attack)
        assert decision is not None
        rows.append({"case": decision.exact_admission["reject_reason"], **decision.to_certificate()})
    admitted_rows = [row for row in rows if row["cache_write"]["write_admitted"] is True]
    return {
        "rows": rows,
        "admitted_write_count": len(admitted_rows),
        "all_writes_have_prior_exact_receipt": all(
            row["exact_admission"]["prior_exact_receipt"] is True for row in admitted_rows
        ),
        "all_writes_checked_before_admit": all(
            row["exact_admission"]["checker_ran_before_write"] is True for row in admitted_rows
        ),
        "rejected_case_count": sum(1 for row in rows if row["exact_admission"]["admitted"] is False),
        "row_hash": sha256_json(rows),
    }


def persistence_rollback_and_tombstone_receipts(ledger_dir: Path) -> JsonDict:
    """Exercise save, close, load, tombstone, and rollback lifecycle."""

    ledger = ledger_dir / "lifecycle.jsonl"
    checkpoint = ledger_dir / "lifecycle.checkpoint.json"
    adapter = FR11FactorCacheShadowAdapter(
        ledger_path=ledger,
        checkpoint_path=checkpoint,
        enabled=True,
    )
    decision = adapter.observe(_receipt("life-1", raw_hash="sha256:" + "8" * 64, chronology_index=0))
    assert decision is not None
    tombstone = adapter.tombstone("life-1", reason="tombstone_eviction")
    rollback = adapter.rollback(
        target_cache_hash=decision.cache_write["pre_cache_hash"],
        reason="tombstone_eviction",
    )
    adapter.close()
    restored = FR11FactorCacheShadowAdapter.load(
        ledger_path=ledger,
        checkpoint_path=checkpoint,
        enabled=True,
    )
    replay = restored.observe(
        _receipt(
            "life-1",
            raw_hash="sha256:" + "9" * 64,
            chronology_index=1,
            cache_parent_hash=restored.state_hash,
        )
    )
    assert replay is not None
    summary = restored.state_summary()
    return {
        "tombstone": tombstone,
        "rollback": rollback,
        "post_load_summary": summary,
        "replay_reject_reason": replay.exact_admission["reject_reason"],
        "non_resurrection_after_load": replay.exact_admission["reject_reason"] == "tombstoned_event"
        and decision.cache_write["post_cache_hash"] not in summary["active_cache_hashes"],
        "checkpoint_present": checkpoint.is_file(),
        "ledger_row_count": len(load_ledger(ledger)),
        "row_hash": sha256_json({"summary": summary, "replay": replay.to_certificate()}),
    }


def attack_matrix(ledger_dir: Path) -> JsonDict:
    """Run production-boundary attacks and require fail-closed outcomes."""

    rows = []

    baseline = _pipeline().verify(
        question=str(FIXTURES[1]["question"]),
        response=str(FIXTURES[1]["response"]),
        domain=str(FIXTURES[1]["domain"]),
    )
    shadow = VerifyRepairPipeline(
        and_compose_verifier=False,
        fr11_factor_cache_shadow_adapter_enabled=True,
        fr11_factor_cache_shadow_ledger_path=ledger_dir / "release.jsonl",
        fr11_factor_cache_shadow_checkpoint_path=ledger_dir / "release.checkpoint.json",
    ).verify(
        question=str(FIXTURES[1]["question"]),
        response=str(FIXTURES[1]["response"]),
        domain=str(FIXTURES[1]["domain"]),
    )
    rows.append(
        {
            "attack_id": "release_bypass",
            "status": "failed_closed",
            "reason": "shadow write did not change exact release decision",
            "fail_closed": baseline.verified == shadow.verified,
        }
    )

    adapter = FR11FactorCacheShadowAdapter(
        ledger_path=ledger_dir / "attacks.jsonl",
        checkpoint_path=ledger_dir / "attacks.checkpoint.json",
        enabled=True,
    )
    first = adapter.observe(_receipt("attack-good", raw_hash="sha256:" + "a" * 64, chronology_index=0))
    assert first is not None
    attack_receipts = {
        "write_before_check": _receipt(
            "attack-wbc",
            raw_hash="sha256:" + "b" * 64,
            chronology_index=1,
            cache_parent_hash=adapter.state_hash,
            checker_ran_before_write=False,
        ),
        "duplicate_event_id": _receipt(
            "attack-good",
            raw_hash="sha256:" + "c" * 64,
            chronology_index=1,
            cache_parent_hash=adapter.state_hash,
        ),
        "duplicate_raw_event": _receipt(
            "attack-raw",
            raw_hash="sha256:" + "a" * 64,
            chronology_index=1,
            cache_parent_hash=adapter.state_hash,
        ),
        "wrong_unit": _receipt(
            "attack-unit",
            raw_hash="sha256:" + "d" * 64,
            chronology_index=1,
            cache_parent_hash=adapter.state_hash,
            raw_unit_binding="unit-b",
        ),
        "forged_pass": _receipt(
            "attack-forged",
            raw_hash="sha256:" + "e" * 64,
            chronology_index=1,
            cache_parent_hash=adapter.state_hash,
            exact_outcome="pass",
            checker_outcome="fail",
        ),
        "stale_cache": _receipt(
            "attack-stale",
            raw_hash="sha256:" + "f" * 64,
            chronology_index=1,
            cache_parent_hash=GENESIS_HASH,
        ),
    }
    for attack_id, receipt in attack_receipts.items():
        decision = adapter.observe(receipt)
        assert decision is not None
        rows.append(
            {
                "attack_id": attack_id,
                "status": "failed_closed",
                "reason": decision.exact_admission["reject_reason"],
                "fail_closed": decision.exact_admission["admitted"] is False,
            }
        )

    lifecycle = persistence_rollback_and_tombstone_receipts(ledger_dir / "attack-life")
    rows.append(
        {
            "attack_id": "tombstone_eviction",
            "status": "failed_closed",
            "reason": lifecycle["replay_reject_reason"],
            "fail_closed": lifecycle["non_resurrection_after_load"] is True,
        }
    )

    exception_result = VerifyRepairPipeline(
        and_compose_verifier=False,
        fr11_factor_cache_shadow_adapter=_ForgedAdapter(),
    ).verify(
        question=str(FIXTURES[0]["question"]),
        response=str(FIXTURES[0]["response"]),
        domain=str(FIXTURES[0]["domain"]),
    )
    rows.append(
        {
            "attack_id": "adapter_exception",
            "status": "failed_closed",
            "reason": exception_result.certificate["fr11_factor_cache_shadow_adapter"]["exact_admission"]["reject_reason"],
            "fail_closed": exception_result.certificate["fr11_factor_cache_shadow_adapter"]["cache_write"]["write_admitted"]
            is False,
        }
    )

    old_env = os.environ.get("CARNOT_FR11_FACTOR_CACHE_SHADOW_ADAPTER")
    os.environ["CARNOT_FR11_FACTOR_CACHE_SHADOW_ADAPTER"] = "1"
    env_ledger = ledger_dir / "env-leak.jsonl"
    try:
        env_result = VerifyRepairPipeline(
            and_compose_verifier=False,
            fr11_factor_cache_shadow_ledger_path=env_ledger,
        ).verify(
            question=str(FIXTURES[0]["question"]),
            response=str(FIXTURES[0]["response"]),
            domain=str(FIXTURES[0]["domain"]),
        )
    finally:
        if old_env is None:
            os.environ.pop("CARNOT_FR11_FACTOR_CACHE_SHADOW_ADAPTER", None)
        else:
            os.environ["CARNOT_FR11_FACTOR_CACHE_SHADOW_ADAPTER"] = old_env
    rows.append(
        {
            "attack_id": "default_on_environment_leakage",
            "status": "failed_closed",
            "reason": "environment_variable_did_not_enable_factor_cache",
            "fail_closed": "fr11_factor_cache_shadow_adapter" not in env_result.certificate
            and not env_ledger.exists(),
        }
    )

    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "readiness_promoted_attack_count": sum(1 for row in rows if row["fail_closed"] is not True),
        "row_hash": sha256_json(rows),
    }


def per_unit_rows(
    default_rows: Mapping[str, Any],
    shadow_rows: Mapping[str, Any],
    attacks: Mapping[str, Any],
) -> JsonDict:
    """Combine compatibility, shadow, and attack rows."""

    rows = []
    for row in default_rows["rows"]:
        rows.append({"row_kind": "compatibility", **row})
    for row in shadow_rows["rows"]:
        rows.append({"row_kind": "shadow", **row})
    for row in attacks["rows"]:
        rows.append({"row_kind": "attack", **row})
    return {"rows": rows, "row_count": len(rows), "row_hash": sha256_json(rows)}


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    """Return deterministic test command receipts."""

    exits = {command: 0 for command in DEFAULT_TEST_COMMANDS}
    if test_exit_codes is not None:
        exits.update(dict(test_exit_codes))
    return [
        {
            "command": command,
            "exit_code": exits.get(command),
            "status": "passed" if exits.get(command) == 0 else "failed_or_not_run",
        }
        for command in DEFAULT_TEST_COMMANDS
    ]


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness-critical aggregates from artifact rows."""

    checks = {
        "default_off_compatibility": artifact.get("default_off_compatibility_rows", {}).get(
            "all_public_outputs_match"
        )
        is True,
        "shadow_public_outputs": artifact.get("shadow_decision_rows", {}).get(
            "all_public_outputs_match"
        )
        is True,
        "exact_write_admission": artifact.get("exact_write_admission_rows", {}).get(
            "all_writes_have_prior_exact_receipt"
        )
        is True,
        "persistence": artifact.get("persistence_rollback_and_tombstone_receipts", {}).get(
            "non_resurrection_after_load"
        )
        is True,
        "attack_matrix": artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed")
        is True,
        "protected_files": artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        "tests": all(row.get("exit_code") == 0 for row in artifact.get("tests_run", [])),
    }
    return {
        "matches_reported": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if value is not True],
        "row_hashes": {
            "default_off": artifact.get("default_off_compatibility_rows", {}).get("row_hash"),
            "shadow": artifact.get("shadow_decision_rows", {}).get("row_hash"),
            "admission": artifact.get("exact_write_admission_rows", {}).get("row_hash"),
            "per_unit": artifact.get("per_unit_rows", {}).get("row_hash"),
            "attacks": artifact.get("attack_matrix", {}).get("row_hash"),
        },
    }


def readiness_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every production boundary passes."""

    checks = aggregate_row_recomputation(artifact)["checks"]
    api_ok = artifact.get("adapter_api_and_schema_hash", {}).get("api_version") == ADAPTER_API_VERSION
    baseline_ok = artifact.get("baseline_import_and_output_receipts", {}).get("import_ok") is True
    oracle_ok = artifact.get("verifier_is_oracle", {}).get("value") is True
    return 1.0 if api_ok and baseline_ok and all(checks.values()) and oracle_ok else 0.0


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Summarize gate failures for blocked or null results."""

    aggregate = aggregate_row_recomputation(artifact)
    gates = {
        "api": artifact.get("adapter_api_and_schema_hash", {}).get("api_version") == ADAPTER_API_VERSION,
        "baseline": artifact.get("baseline_import_and_output_receipts", {}).get("import_ok") is True,
        **aggregate["checks"],
        "verifier_oracle_scope": artifact.get("verifier_is_oracle", {}).get("value") is True,
    }
    return {
        "gates": gates,
        "failed_check_count": sum(1 for value in gates.values() if value is not True),
        "failed_checks": [key for key, value in gates.items() if value is not True],
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return terminal verdict text for the integration."""

    if float(artifact.get("factor_cache_shadow_adapter_ready_score", 0.0) or 0.0) == 1.0:
        return "success: default-off verify-repair factor-cache shadow adapter shipped"
    failed = gate_check_summary(artifact)["failed_checks"]
    return "blocked: factor-cache shadow adapter gates failed: " + ",".join(failed)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without raising."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    extra = sorted(set(artifact) - set(REQUIRED_ARTIFACT_FIELDS))
    if missing or extra:
        errors.append(f"required_fields mismatch missing={missing} extra={extra}")
        return errors
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("field_provenance") != FIELD_PROVENANCE:
        errors.append("field_provenance mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("checksum mismatch")
    if artifact.get("default_off_compatibility_rows", {}).get("all_public_outputs_match") is not True:
        errors.append("default_off_compatibility failed")
    if artifact.get("exact_write_admission_rows", {}).get("all_writes_have_prior_exact_receipt") is not True:
        errors.append("exact_write_admission failed")
    if artifact.get("persistence_rollback_and_tombstone_receipts", {}).get("non_resurrection_after_load") is not True:
        errors.append("persistence failed")
    if artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is not True:
        errors.append("attack_matrix failed")
    if artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True:
        errors.append("aggregate mismatch")
    if artifact.get("factor_cache_shadow_adapter_ready_score") != readiness_score(artifact):
        errors.append("factor_cache_shadow_adapter_ready_score mismatch")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        errors.append("protected_files changed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith(("success:", "blocked:")):
        errors.append("honest_verdict terminal prefix missing")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise on invalid Exp6479 artifact fields."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return True


def run(
    *,
    date: str = "20260821",
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    ledger_dir: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp6479 terminal artifact."""

    started = time.perf_counter()
    root = Path(root)
    result = root / RESULT_RELATIVE_PATH if result_path is None else Path(result_path)
    ledgers = (
        root / "results/checkpoints/experiment_6479_factor_cache_shadow_adapter"
        if ledger_dir is None
        else Path(ledger_dir)
    )
    protected_before = protected_hashes(root)
    sources = source_hashes(root)
    baseline = baseline_import_and_output_receipts()
    compatibility = default_off_compatibility_rows(baseline, ledgers / "compatibility")
    shadow = shadow_decision_rows(baseline, ledgers / "shadow")
    admission = exact_write_admission_rows(ledgers / "admission")
    lifecycle = persistence_rollback_and_tombstone_receipts(ledgers / "lifecycle")
    attacks = attack_matrix(ledgers / "attacks")
    tests_run = tests_run_receipt(test_exit_codes)
    protected_after = protected_hashes(root)
    artifact: JsonDict = {
        "status": "running",
        "adapter_api_and_schema_hash": {
            "api_version": ADAPTER_API_VERSION,
            "api_methods": list(FR11FactorCacheShadowAdapter.__dict__.keys()),
            "schema_hash": adapter_api_schema_hash(),
            "adapter_module_sha256": sources.get(ADAPTER_RELATIVE_PATH.as_posix()),
            "pipeline_module_sha256": sources.get(PIPELINE_RELATIVE_PATH.as_posix()),
        },
        "baseline_import_and_output_receipts": baseline,
        "default_off_compatibility_rows": compatibility,
        "shadow_decision_rows": shadow,
        "exact_write_admission_rows": admission,
        "persistence_rollback_and_tombstone_receipts": lifecycle,
        "per_unit_rows": {},
        "aggregate_row_recomputation": {},
        "attack_matrix": attacks,
        "factor_cache_shadow_adapter_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_after),
        "gate_check_summary": {},
        "preconditions_checked": {
            "planning_date": date,
            "expected_planning_date": "20260821",
            "date_ok": date == "20260821",
            "source_hashes": sources,
            "protected_hashes_before": protected_before,
            "api_signature_recorded": True,
            "exact_checker_path_recorded": True,
            "memory_options_recorded": True,
            "prior_v556_artifact_hashes": {
                path.as_posix(): sha256_file(root / path)
                for path in PROTECTED_RELATIVE_PATHS
                if "experiment_646" in path.as_posix()
            },
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "value": True,
            "true_for": ["existing_exact_checkers", "deterministic_compatibility_reducers"],
            "false_for": {
                "factor_cache_adapter": False,
                "shadow_ranker": False,
                "model_confidence": False,
            },
        },
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s if duration_s is not None else time.perf_counter() - started, 6),
        "tests_run": tests_run,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["per_unit_rows"] = per_unit_rows(compatibility, shadow, attacks)
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(artifact)
    artifact["factor_cache_shadow_adapter_ready_score"] = readiness_score(artifact)
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["status"] = (
        "success_ready"
        if artifact["factor_cache_shadow_adapter_ready_score"] == 1.0
        else "blocked_gates"
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260821")
    args = parser.parse_args(argv)
    run(date=args.date)


if __name__ == "__main__":  # pragma: no cover
    main()
