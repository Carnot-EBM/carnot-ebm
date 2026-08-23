"""Tests for Exp6541 V566 direct-source contract.

Spec refs: REQ-REPORT-6541, SCENARIO-REPORT-6541-DIRECT,
SCENARIO-REPORT-6541-ADVISORY, SCENARIO-REPORT-6541-CACHE,
SCENARIO-REPORT-6541-FIELDS, SCENARIO-REPORT-6541-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6541_v566_direct_source_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6541_v566_direct_source_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6541_v566_direct_source_contract.py "
    "-m pytest tests/python/test_experiment_6541_v566_direct_source_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6541_v566_direct_source_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6541_v566_direct_source_contract.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6541_v566_direct_source_contract "
    "--date 20260823"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6541_v566_direct_source_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6541_v566_direct_source_contract --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _problem(domain: str, split: str, index: int) -> dict[str, Any]:
    entity = f"{domain}_entity_{index}"
    return {
        "problem_id": f"{domain}_{index:03d}",
        "domain": domain,
        "split": split,
        "num_entities": 1,
        "entities": [entity],
        "turns": [
            {
                "turn_number": 1,
                "user_message": f"{entity} has a fixed value.",
                "new_constraints": [{"type": "assign", "args": [entity, "color", "red"]}],
                "cumulative_constraints": [{"type": "assign", "args": [entity, "color", "red"]}],
                "gold_solution": {entity: {"color": "red"}},
                "is_satisfiable": True,
            }
        ],
    }


def _write_drift_checkout(tmp_path: Path, *, problem_count: int = 1020) -> Path:
    root = tmp_path / "drift-bench"
    (root / "data" / "problems" / "dev").mkdir(parents=True)
    (root / "data" / "problems" / "test").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "README.md").write_text(
        "DRIFT-Bench with 1,020 problems. The original run's SQLite databases suffered "
        "filesystem corruption and are not redistributed.\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("MIT License\nCopyright (c) 2026 Kaons\n", encoding="utf-8")
    (root / "data" / "problems" / "README.md").write_text(
        "Every problem has problem_id, domain, split, entities, turns, "
        "new_constraints, cumulative_constraints, gold_solution, and is_satisfiable.\n",
        encoding="utf-8",
    )
    (root / "src" / "z3_checker.py").write_text(
        "from z3 import Solver, sat\n\ndef check_problem(problem):\n    return sat\n",
        encoding="utf-8",
    )
    domains = ["logic_grid", "scheduling", "seating"]
    written = 0
    for split, per_domain in (("dev", 68), ("test", 272)):
        for domain in domains:
            for idx in range(per_domain):
                if written >= problem_count:
                    return root
                _write_json(
                    root / "data" / "problems" / split / f"{domain}_{idx:03d}.json",
                    _problem(domain, split, idx),
                )
                written += 1
    return root


def _receipt(url: str, source_id: str, *, ok: bool = False) -> mod.JsonDict:
    return {
        "ok": ok,
        "status_code": 503 if not ok else 200,
        "url": url,
        "headers": {"content-type": "text/plain"},
        "body": "" if not ok else f"fixture content for {source_id}",
        "error": "fixture unavailable" if not ok else None,
    }


def _advisory_fetcher(url: str, source_id: str) -> mod.JsonDict:
    return _receipt(url, source_id, ok=False)


def _source_metadata(root: Path) -> dict[str, Any]:
    return {
        "repo_url": mod.DRIFT_REPO_URL,
        "commit": mod.DRIFT_EXPECTED_COMMIT,
        "commit_date": "2026-04-25T13:18:49-07:00",
        "commit_subject": "public release",
        "root_tree_git_sha": "35055da27d798c526c53af63e039bd9f095511af",
        "problems_tree_git_sha": "9f83505881082b4aefd3e2b8aff563ac18cac606",
        "checkout_path": str(root),
        "ls_remote_head": mod.DRIFT_EXPECTED_COMMIT,
    }


def _model_resolvers(
    tmp_path: Path,
) -> tuple[mod.ModelPairResolver, mod.GgufResolver, dict[str, int]]:
    calls = {"pair": 0, "gguf": 0}
    paths: dict[str, Path] = {}
    for model in mod.SOTA_GGUF_MODELS:
        path = tmp_path / f"{model['name']}-{model['quantization']}.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{model['hf_id']}".encode())
        paths[model["hf_id"]] = path

    def pair_resolver(
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        del preferred_quant, model_indices
        calls["pair"] += 1
        return [
            {
                "name": mod.SOTA_GGUF_MODELS[0]["name"],
                "hf_id": mod.SOTA_GGUF_MODELS[0]["hf_id"],
                "gpu": gpu_indices[0],
                "model_path": str(paths[mod.SOTA_GGUF_MODELS[0]["hf_id"]]),
            },
            {
                "name": mod.SOTA_GGUF_MODELS[1]["name"],
                "hf_id": mod.SOTA_GGUF_MODELS[1]["hf_id"],
                "gpu": gpu_indices[1],
                "model_path": str(paths[mod.SOTA_GGUF_MODELS[1]["hf_id"]]),
            },
        ]

    def gguf_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        del preferred_quant
        calls["gguf"] += 1
        return str(paths[hf_id])

    return pair_resolver, gguf_resolver, calls


def _missing_pair_resolver(
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
    model_indices: tuple[int, int] | None = None,
) -> None:
    del gpu_indices, preferred_quant, model_indices
    return None


def _missing_gguf_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> None:
    del hf_id, preferred_quant
    return None


def _artifact(
    tmp_path: Path, *, problem_count: int = 1020, missing_models: bool = False
) -> dict[str, Any]:
    source_root = _write_drift_checkout(tmp_path / "source", problem_count=problem_count)
    pair_resolver, gguf_resolver, _calls = _model_resolvers(tmp_path / "models")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        run_date="20260823",
        drift_source_root=source_root,
        drift_git_metadata=_source_metadata(source_root),
        advisory_fetcher=_advisory_fetcher,
        cached_pair_resolver=_missing_pair_resolver if missing_models else pair_resolver,
        gguf_resolver=_missing_gguf_resolver if missing_models else gguf_resolver,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_report_6541_spec_declares_contract() -> None:
    """REQ-REPORT-6541: OpenSpec owns the Exp6541 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6541") :]

    for token in (
        "SCENARIO-REPORT-6541-DIRECT",
        "SCENARIO-REPORT-6541-ADVISORY",
        "SCENARIO-REPORT-6541-CACHE",
        "SCENARIO-REPORT-6541-FIELDS",
        "SCENARIO-REPORT-6541-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "Advisory-channel unavailability SHALL NOT set that score to zero.",
        "No `requires` or `gated_on` edge SHALL name Exp6528, Exp6529, or another retired task.",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6541_direct_source_and_advisory_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6541-DIRECT/ADVISORY: direct DRIFT source gates readiness."""

    artifact = _artifact(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_v566_direct_source_contract_ready"
    assert artifact["honest_verdict"].startswith("complete_v566_direct_source_contract_ready")
    assert artifact["verdict_class"] is None
    assert artifact["v566_direct_source_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    contract = artifact["drift_revision_license_schema_contract"]
    assert contract["repo_url"] == mod.DRIFT_REPO_URL
    assert contract["immutable_revision"] == mod.DRIFT_EXPECTED_COMMIT
    assert contract["commit_date"] == "2026-04-25T13:18:49-07:00"
    assert contract["license"] == "MIT"
    assert contract["problem_file_count"] == 1020
    assert contract["problem_file_census_matches_expected"] is True
    assert contract["schema_verified"] is True
    assert contract["z3_replay_code_present"] is True
    assert contract["contract_ready"] is True

    assert artifact["source_tree_hashes"]["problem_file_count"] == 1020
    assert artifact["source_tree_hashes"]["problems_manifest_sha256"].startswith("sha256:")
    assert artifact["source_tree_hashes"]["required_file_sha256"]["src/z3_checker.py"].startswith(
        "sha256:"
    )
    assert artifact["upstream_corruption_boundary"]["sqlite_corruption_warning_present"] is True
    assert artifact["upstream_corruption_boundary"]["upstream_sqlite_results_inherited"] is False

    advisory = artifact["advisory_discovery_rows"]
    assert {row["channel"] for row in advisory} == set(mod.ADVISORY_CHANNELS)
    assert all(row["retrieval_state"] == "blocked" for row in advisory)
    assert all(row["mandatory_for_exp6541_ready"] is False for row in advisory)
    assert all(row["failure_can_zero_direct_source_ready"] is False for row in advisory)
    assert (
        artifact["aggregate_row_recomputation"]["advisory_failures_ignored_for_direct_ready"]
        is True
    )


def test_scenario_report_6541_cache_frozen_contracts_and_dependencies(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6541-CACHE/FIELDS: cache, split, and dependency fields are frozen."""

    source_root = _write_drift_checkout(tmp_path / "source")
    pair_resolver, gguf_resolver, calls = _model_resolvers(tmp_path / "models")
    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        run_date="20260823",
        drift_source_root=source_root,
        drift_git_metadata=_source_metadata(source_root),
        advisory_fetcher=_advisory_fetcher,
        cached_pair_resolver=pair_resolver,
        gguf_resolver=gguf_resolver,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )

    assert calls["pair"] == 1
    assert calls["gguf"] == 1
    rows = artifact["model_cache_resolution_rows"]
    assert len(rows) == len(mod.SOTA_GGUF_MODELS)
    assert {row["hf_id"] for row in rows} == {model["hf_id"] for model in mod.SOTA_GGUF_MODELS}
    assert all(row["cache_hit"] is True for row in rows)
    assert all(row["model_loaded_or_run"] is False for row in rows)
    assert all(row["model_file_sha256"].startswith("sha256:") for row in rows)
    assert rows[0]["selected_by_cached_sota_pair"] is True
    assert rows[2]["selected_by_cached_sota_pair"] is False

    load_contract = artifact["gguf_load_contract"]
    assert load_contract["all_required_hub_ids_present"] is True
    assert load_contract["all_load_plans_use_model_path"] is True
    assert load_contract["weights_loaded"] is False
    assert load_contract["transformers_tokenizer_on_gguf_repo_id_allowed"] is False
    assert (
        load_contract["embedded_tokenizer_preflight_helper"]
        == "gguf_tokenizer_loadable(model_path)"
    )

    assert artifact["frozen_external_split_contract"]["downstream_field_spelling"] == [
        "split_name",
        "base_problem_id",
        "domain",
        "turn_index",
        "source_row_hash",
        "chronology_index",
    ]
    assert artifact["frozen_structural_contract"]["candidate_set_preserved"] is True
    assert artifact["frozen_cost_guard_contract"]["solver_conflict_is_correctness_label"] is False
    assert artifact["frozen_router_contract"]["learned_advice_may_prune"] is False
    assert artifact["frozen_reversible_memory_contract"]["commit_after_exact_validation"] is True
    assert artifact["frozen_arc_contract"]["arc_solver_firing_allowed"] is False
    assert (
        artifact["hardware_stop_contract"]["gatemate_command_allowed_without_new_receipt"] is False
    )

    dependency_rows = [
        row for row in artifact["dependency_and_gate_rows"] if row["row_type"] == "dependency_gate"
    ]
    assert dependency_rows
    assert all(row["upstream_task_exists"] is True for row in dependency_rows)
    assert all(row["field_declared_verbatim"] is True for row in dependency_rows)
    assert all(row["retired_id_dependency"] is False for row in dependency_rows)
    assert not any("exp6528" in row["upstream_task_id"] for row in dependency_rows)
    assert not any("exp6529" in row["upstream_task_id"] for row in dependency_rows)

    attacks = {
        row["attack_id"]: row
        for row in artifact["dependency_and_gate_rows"]
        if row["row_type"] == "attack"
    }
    assert set(attacks) >= {
        "unavailable_advisory_channel",
        "moving_branch",
        "license_ambiguity",
        "missing_source_files",
        "gguf_repo_id_tokenizer_misuse",
        "renamed_readiness_field",
        "retired_id_dependency",
        "status_only_success",
    }
    assert all(row["attack_passed"] is True for row in attacks.values())

    assert artifact["v565_boundary_receipts"]["exp6528"]["honest_verdict"].startswith("blocked_")
    assert artifact["immutable_evidence_receipts"][0]["path"].endswith(
        "experiment_6527_v565_evidence_eligibility_corrigendum.json"
    )
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert {row["row_type"] for row in artifact["per_unit_rows"]} >= {
        "direct_source",
        "advisory_discovery",
        "model_cache",
        "dependency_gate",
        "attack",
        "gate",
    }


def test_scenario_report_6541_validation_and_blocked_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-6541-SCHEMA: malformed contracts fail closed."""

    artifact = _artifact(tmp_path)

    assert mod.tests_run_receipts(TESTS_RUN) == TESTS_RUN
    assert all(row["exit_code"] == 0 for row in mod.tests_run_receipts(None))
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._load_json(tmp_path / "missing.json") == {}
    assert mod._retrieval_state(200, "ok") == "available"
    assert mod._retrieval_state(429, "rate limit") == "rate_limited"
    assert mod._retrieval_state(404, "not found") == "not_found"
    assert mod._retrieval_state(503, "blocked") == "blocked"
    assert mod._safe_cache_root({"HF_HOME": "/tmp/hf", "TOKEN": "secret"})["HF_HOME"] == "/tmp/hf"
    assert "TOKEN" not in mod._safe_cache_root({"HF_HOME": "/tmp/hf", "TOKEN": "secret"})
    assert mod._extract_required_fields("no field marker") == set()

    blocked = _artifact(tmp_path / "blocked", problem_count=1019)
    assert blocked["status"] == "blocked_v566_direct_source_contract"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["v566_direct_source_ready_score"] == 0.0
    assert (
        blocked["gate_check_summary"]["failed_checks"][0]["check"] == "direct_source_contract_ready"
    )
    assert mod.validate_artifact(blocked) == []

    bad = deepcopy(artifact)
    del bad["status"]
    _with_checksum(bad)
    assert any("required field set mismatch" in error for error in mod.validate_artifact(bad))

    bad = deepcopy(artifact)
    bad["unexpected"] = True
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "positive"
    bad["inference_substrate"] = "live_llm"
    bad["verifier_is_oracle"] = False
    bad["honest_verdict"] = "ok"
    _with_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert any("required field set mismatch" in error for error in errors)
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6541 enum" in errors
    assert "honest_verdict terminal prefix mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be true" in errors

    mutations = [
        (
            "ready score mismatch",
            lambda item: item.__setitem__("v566_direct_source_ready_score", 0.0),
        ),
        (
            "direct source contract must be ready",
            lambda item: item["drift_revision_license_schema_contract"].__setitem__(
                "contract_ready", False
            ),
        ),
        (
            "advisory rows must not gate direct readiness",
            lambda item: item["advisory_discovery_rows"][0].__setitem__(
                "failure_can_zero_direct_source_ready", True
            ),
        ),
        (
            "model cache contract must cover all mandated models",
            lambda item: item.__setitem__("model_cache_resolution_rows", []),
        ),
        (
            "GGUF load contract must forbid repo-id tokenizer misuse",
            lambda item: item["gguf_load_contract"].__setitem__(
                "transformers_tokenizer_on_gguf_repo_id_allowed", True
            ),
        ),
        (
            "frozen contracts must expose exact downstream field spelling",
            lambda item: item["frozen_router_contract"].__setitem__(
                "downstream_field_spelling", []
            ),
        ),
        (
            "dependency gates must avoid retired IDs",
            lambda item: item["dependency_and_gate_rows"].append(
                {
                    "row_type": "dependency_gate",
                    "upstream_task_id": "exp6528-v565-source-model-method-contract",
                    "artifact_field": "v565_method_contract_ready_score",
                    "upstream_task_exists": False,
                    "field_declared_verbatim": False,
                    "retired_id_dependency": True,
                }
            ),
        ),
        (
            "protected files changed",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
        (
            "status-only success attack must be detected",
            lambda item: [
                row.__setitem__("attack_passed", False)
                for row in item["dependency_and_gate_rows"]
                if row.get("attack_id") == "status_only_success"
            ],
        ),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        _with_checksum(broken)
        assert expected in mod.validate_artifact(broken)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(blocked)
    bad["gate_check_summary"]["failed_checks"] = [{"check": "direct_source_contract_ready"}]
    _with_checksum(bad)
    assert "blocked verdict must name failed check and observed value" in mod.validate_artifact(bad)

    assert (
        mod.main(["--validate", "--result-path", str(tmp_path / mod.RESULT_RELATIVE_PATH.name)])
        == 0
    )
    assert capsys.readouterr().out.strip() == "OK"

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps({"status": "bad"}), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(invalid_path)]) == 1
    assert "required field set mismatch" in capsys.readouterr().out
