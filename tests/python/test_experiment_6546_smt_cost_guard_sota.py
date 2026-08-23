"""Tests for Exp6546 SMT cost guard on mandated SOTA GGUFs.

Spec refs: REQ-BENCH-6546, SCENARIO-BENCH-6546-GATE,
SCENARIO-BENCH-6546-CHALLENGE, SCENARIO-BENCH-6546-DISPATCH,
SCENARIO-BENCH-6546-RUNTIME, SCENARIO-BENCH-6546-EFFECTS,
SCENARIO-BENCH-6546-ATTACKS, SCENARIO-BENCH-6546-CHECKPOINT,
SCENARIO-BENCH-6546-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6546_smt_cost_guard_sota as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6546_smt_cost_guard_sota.py -q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6546_smt_cost_guard_sota.py "
    "-m pytest tests/python/test_experiment_6546_smt_cost_guard_sota.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6546_smt_cost_guard_sota.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6546_smt_cost_guard_sota.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6546_smt_cost_guard_sota.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6546_smt_cost_guard_sota.json"
)
LOCAL_E2E_COMMAND = ".venv/bin/python -m carnot.experiment_6546_smt_cost_guard_sota --validate"
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6546_smt_cost_guard_sota --date 20260823"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": LOCAL_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
]


class FakeBackend:
    """Small llama.cpp-shaped backend for REQ-BENCH-6546 unit coverage."""

    def __init__(self, *, timeout_first: bool = False) -> None:
        self.timeout_first = timeout_first
        self.calls = 0

    def load_model(self, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "loader": "llama_cpp.Llama",
            "load_ok": True,
            "load_s": 0.1,
            "smoke_ok": True,
            "smoke_s": 0.01,
            "embedded_tokenizer_ok": True,
            "error": "",
        }

    def close(self) -> None:
        self.closed = True

    def tokenize(self, _spec: dict[str, Any], text: str) -> int:
        return max(1, len(text.split()))

    def infer(
        self,
        *,
        spec: dict[str, Any],
        prompt: str,
        max_tokens: int,
        timeout_s: float,
        unit_key: str,
    ) -> dict[str, Any]:
        self.calls += 1
        if self.timeout_first and self.calls == 1:
            return {
                "terminal_status": "timeout",
                "timeout": True,
                "parse_failure": True,
                "output_text": "",
                "output_tokens": 0,
                "wall_time_s": timeout_s,
                "first_token_time_s": None,
                "error": f"timeout:{unit_key}",
            }
        prompt_tokens = self.tokenize(spec, prompt)
        conflict = 3 if "CONFLICT_STRATUM: high" in prompt else 1
        surface_penalty = 2 if "SURFACE: relabeled" in prompt else 0
        output_tokens = min(max_tokens, 16 + conflict + surface_penalty)
        return {
            "terminal_status": "terminal",
            "timeout": False,
            "parse_failure": False,
            "output_text": "Reasoning omitted. FINAL: SATISFIABLE",
            "output_tokens": output_tokens,
            "wall_time_s": round(0.2 + 0.02 * prompt_tokens + 0.05 * conflict, 6),
            "first_token_time_s": 0.03,
            "error": "",
        }


@pytest.fixture()
def fake_model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    """REQ-BENCH-6546: local GGUF paths are supplied without external downloads."""

    specs: list[dict[str, Any]] = []
    for index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"fake gguf {hf_id}".encode())
        specs.append(
            {
                "name": mod.MODEL_NAMES_BY_HF_ID[hf_id],
                "hf_id": hf_id,
                "role": "dense" if "31B" in hf_id else "moe",
                "gpu": index % 2,
                "quantization": "Q4_K_M",
                "model_path": str(path),
            }
        )
    return specs


@pytest.fixture()
def artifact(
    tmp_path: Path,
    fake_model_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """REQ-BENCH-6546: build a positive artifact with an injected local backend."""

    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_path=tmp_path / "checkpoint.json",
        write=True,
        duration_s=120.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        inference_backend=FakeBackend(),
    )


def test_req_bench_6546_spec_declares_smt_cost_guard_contract() -> None:
    """REQ-BENCH-6546: OpenSpec owns the SOTA cost-guard contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6546") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6546-GATE",
        "SCENARIO-BENCH-6546-CHALLENGE",
        "SCENARIO-BENCH-6546-DISPATCH",
        "SCENARIO-BENCH-6546-RUNTIME",
        "SCENARIO-BENCH-6546-EFFECTS",
        "SCENARIO-BENCH-6546-ATTACKS",
        "SCENARIO-BENCH-6546-CHECKPOINT",
        "SCENARIO-BENCH-6546-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "smt_cost_guard_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6546_gate_model_specs_and_dispatch(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6546-GATE/DISPATCH: identity and guard are frozen first."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_smt_cost_guard_positive"
    assert artifact["honest_verdict"].startswith("complete_smt_cost_guard_positive")
    assert artifact["verdict_class"] == "positive"
    assert artifact["smt_cost_guard_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"] == mod.EXP6543_RELATIVE_PATH.as_posix()
    assert gate["field"] == "external_constraint_corpus_audited_ready_score"
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["input_hashes"]["fixture"] == mod.sha256_file(REPO / mod.FIXTURE_RELATIVE_PATH)
    assert gate["cached_sota_pair_gpu_0_1"]
    assert gate["budgets"]["max_new_tokens"] == mod.MAX_NEW_TOKENS
    assert "scripts/research_conductor.py" in gate["protected_file_hashes_before"]

    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_HF_IDS)
    assert artifact["models_used"] == list(mod.MANDATED_HF_IDS)
    assert all(row["model_path"].endswith(".gguf") for row in artifact["MODEL_SPECS"])
    assert all(row["gguf_sha256"].startswith("sha256:") for row in artifact["MODEL_SPECS"])
    assert all(row["loader"] == "llama_cpp.Llama" for row in artifact["MODEL_SPECS"])
    assert all(row["load_ok"] for row in artifact["model_cache_and_load_receipts"]["rows"])
    assert artifact["model_cache_and_load_receipts"]["all_mandated_models_loaded"] is True

    dispatch = artifact["frozen_dispatch_contract"]
    assert dispatch["training_splits_used"] == ["development", "train"]
    assert dispatch["held_rows_used_for_threshold"] is False
    assert dispatch["target_answers_used_for_threshold"] is False
    assert dispatch["model_cost_used_for_threshold"] is False
    assert dispatch["route_rule"] == "z3_direct_when_conflict_count_ge_threshold"
    assert dispatch["conflict_threshold"] >= 0


def test_scenario_bench_6546_challenge_surfaces_and_runtime_rows(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6546-CHALLENGE/RUNTIME: rows are matched and terminal."""

    challenge = artifact["frozen_challenge_contract"]
    assert challenge["surface_ids"] == list(mod.SURFACE_IDS)
    assert challenge["logical_instance_count"] >= 3
    assert set(challenge["domain_counts"]) >= {"logic_grid", "scheduling", "seating"}
    assert challenge["proof_preserving_surface_count"] == len(mod.SURFACE_IDS)

    logical = artifact["logical_instance_rows"]
    assert {row["split_name"] for row in logical} == {"held"}
    assert {row["domain"] for row in logical} >= {"logic_grid", "scheduling", "seating"}
    assert all(row["constraints_sha256"].startswith("sha256:") for row in logical)
    assert all(row["exact_label"] == "satisfiable" for row in logical)
    assert all(row["conflict_stratum"] in {"low", "medium", "high"} for row in logical)

    surfaces = artifact["proof_preserving_surface_receipts"]
    assert surfaces["all_surfaces_equivalent"] is True
    assert {row["surface_id"] for row in surfaces["rows"]} == set(mod.SURFACE_IDS)
    assert all(row["constraints_hash_unchanged"] is True for row in surfaces["rows"])
    assert all(row["exact_label_unchanged"] is True for row in surfaces["rows"])

    conflicts = artifact["solver_conflict_rows"]
    assert len(conflicts) == len(logical)
    assert all("conflict_count" in row for row in conflicts)
    assert all(row["z3_replay_status"] in {"sat", "unsat", "unknown"} for row in conflicts)

    rows = artifact["per_unit_rows"]
    expected = len(logical) * len(mod.SURFACE_IDS) * len(mod.MANDATED_HF_IDS) * len(mod.ARM_IDS)
    assert len(rows) == expected
    assert {row["model_hf_id"] for row in rows} == set(mod.MANDATED_HF_IDS)
    assert {row["surface_id"] for row in rows} == set(mod.SURFACE_IDS)
    assert {row["arm_id"] for row in rows} == set(mod.ARM_IDS)
    assert all(row["terminal_status"] == "terminal" for row in rows)
    assert all(row["row_hash"].startswith("sha256:") for row in rows)

    direct_rows = [row for row in rows if row["dispatch"] == "z3_direct"]
    model_rows = [row for row in rows if row["dispatch"] == "llama_cpp"]
    assert direct_rows
    assert model_rows
    assert all(row["arm_id"] == "guarded" for row in direct_rows)
    assert all(row["tool_time_s"] > 0.0 for row in direct_rows)
    assert all(row["model_wall_time_s"] == 0.0 for row in direct_rows)
    assert all(row["prompt_tokens"] == 0 for row in direct_rows)
    assert all(row["exact_valid"] is True for row in rows)
    assert all(row["timeout"] is False for row in rows)
    assert all(
        row["charged_time_s"] == row["model_wall_time_s"] + row["tool_time_s"] for row in rows
    )


def test_scenario_bench_6546_effects_attacks_and_checksum(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6546-EFFECTS/ATTACKS/TERMINAL: claims are row-derived."""

    aggregate = artifact["aggregate_row_recomputation"]
    recompute = artifact["token_and_time_recomputation"]
    completion = artifact["exact_completion_receipt"]
    censoring = artifact["censoring_and_timeout_receipts"]
    attacks = artifact["confound_attack_matrix"]

    assert artifact["model_and_surface_effect_rows"]
    assert artifact["conflict_cost_association_rows"]
    assert artifact["guarded_versus_unguarded_rows"]
    assert aggregate["ready_score_from_rows"] == 1.0
    assert aggregate["verdict_class_from_rows"] == "positive"
    assert aggregate["supporting_model_family_count"] >= 2
    assert aggregate["surface_controlled_audit_passed"] is True

    assert recompute["all_token_and_time_totals_match_rows"] is True
    assert recompute["guarded_total_charged_tokens"] < recompute["unguarded_total_charged_tokens"]
    assert completion["guarded_noninferior_exact_completion"] is True
    assert completion["guarded_exact_valid_count"] >= completion["unguarded_exact_valid_count"]
    assert censoring["all_units_terminal"] is True
    assert censoring["checkpoint_receipt"]["checkpointing_enabled"] is True

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.CONFOUND_ATTACK_IDS)
    assert attacks["all_confounds_fail_closed"] is True
    assert all(row["fail_closed"] is True for row in attacks["rows"])
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    source = (REPO / mod.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "AutoTokenizer" not in source
    assert ".from_pretrained(" not in source


def test_scenario_bench_6546_blocked_gate_and_missing_models(
    tmp_path: Path,
    fake_model_specs: list[dict[str, Any]],
) -> None:
    """SCENARIO-BENCH-6546-GATE: gate/cache failures produce blocked artifacts."""

    blocked_gate = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked-gate.json",
        audit_path=tmp_path / "missing-audit.json",
        checkpoint_path=tmp_path / "blocked-gate.ckpt.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        inference_backend=FakeBackend(),
    )
    assert blocked_gate["status"] == "blocked_smt_cost_guard"
    assert blocked_gate["verdict_class"] == "blocked"
    assert blocked_gate["smt_cost_guard_ready_score"] == 0.0
    assert "upstream_gate_passed" in blocked_gate["gate_check_summary"]["failed_checks"]
    assert blocked_gate["per_unit_rows"] == []
    assert json.loads((tmp_path / "blocked-gate.json").read_text(encoding="utf-8")) == blocked_gate
    assert mod.validate_artifact(blocked_gate) == []

    blocked_models = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked-models.json",
        checkpoint_path=tmp_path / "blocked-models.ckpt.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs[:2],
        inference_backend=FakeBackend(),
    )
    assert blocked_models["status"] == "blocked_smt_cost_guard"
    assert blocked_models["verdict_class"] == "blocked"
    assert (
        "all_mandated_model_paths_resolved" in blocked_models["gate_check_summary"]["failed_checks"]
    )
    assert blocked_models["model_cache_and_load_receipts"]["all_mandated_models_loaded"] is False
    assert mod.validate_artifact(blocked_models) == []


def test_scenario_bench_6546_checkpoint_timeout_validation_and_cli(
    tmp_path: Path,
    fake_model_specs: list[dict[str, Any]],
    artifact: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6546-CHECKPOINT/TERMINAL: resume and validation fail closed."""

    checkpoint = tmp_path / "manual-checkpoint.json"
    payload = {
        "schema": mod.CHECKPOINT_SCHEMA,
        "challenge_hash": "sha256:manual",
        "rows_by_key": {"k": {"row_hash": "sha256:r", "terminal_status": "terminal"}},
    }
    mod.save_checkpoint(checkpoint, payload)
    assert mod.load_checkpoint(checkpoint, "sha256:manual") == payload
    assert mod.load_checkpoint(checkpoint, "sha256:other") == {
        "schema": mod.CHECKPOINT_SCHEMA,
        "challenge_hash": "sha256:other",
        "rows_by_key": {},
    }
    bad_checkpoint = tmp_path / "bad-checkpoint.json"
    bad_checkpoint.write_text("{bad", encoding="utf-8")
    assert mod.load_checkpoint(bad_checkpoint, "sha256:x")["rows_by_key"] == {}

    timeout_artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "timeout.json",
        checkpoint_path=tmp_path / "timeout.ckpt.json",
        write=False,
        duration_s=120.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        inference_backend=FakeBackend(timeout_first=True),
    )
    timeout_rows = [row for row in timeout_artifact["per_unit_rows"] if row["timeout"]]
    assert timeout_rows
    assert timeout_rows[0]["terminal_status"] == "timeout"
    assert timeout_artifact["censoring_and_timeout_receipts"]["timeout_count"] == 1
    assert mod.validate_artifact(timeout_artifact) == []

    malformed = deepcopy(artifact)
    malformed.pop("status")
    malformed["field_principles"] = {}
    malformed["field_provenance"] = {}
    malformed["verdict_class"] = "wrong"
    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "wrong"
    malformed["verifier_is_oracle"] = True
    malformed["smt_cost_guard_ready_score"] = 0.5
    malformed["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    malformed["gate_check_summary"]["all_gates_passed"] = False
    malformed["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(malformed)
    assert "required field set mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6546 enum" in errors
    assert "honest_verdict terminal prefix mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "ready score mismatch" in errors
    assert "positive score requires all gates passed" in errors
    assert "reproducibility_checksum mismatch" in errors

    result_path = tmp_path / "cli-artifact.json"
    monkeypatch.setattr(mod, "resolve_mandated_model_specs", lambda: fake_model_specs)
    monkeypatch.setattr(mod, "LlamaCppBackend", lambda: FakeBackend())
    assert (
        mod.main(
            [
                "--result-path",
                str(result_path),
                "--checkpoint-path",
                str(tmp_path / "cli.ckpt"),
                "--date",
                "20260823",
            ]
        )
        == 0
    )
    assert result_path.is_file()
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    assert "validated" in capsys.readouterr().out

    bad_path = tmp_path / "bad.json"
    bad_payload = deepcopy(artifact)
    bad_payload["reproducibility_checksum"] = "sha256:bad"
    bad_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1
    assert "reproducibility_checksum mismatch" in capsys.readouterr().out

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced validation error"])
    forced_path = tmp_path / "forced-main-failure.json"
    assert (
        mod.main(
            [
                "--result-path",
                str(forced_path),
                "--checkpoint-path",
                str(tmp_path / "forced.ckpt"),
                "--date",
                "20260823",
            ]
        )
        == 1
    )
    assert "forced validation error" in capsys.readouterr().out


def test_scenario_bench_6546_defensive_helper_edges(
    tmp_path: Path,
    fake_model_specs: list[dict[str, Any]],
) -> None:
    """SCENARIO-BENCH-6546-GATE/CHECKPOINT/TERMINAL: helper edges are explicit."""

    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text('\n[]\n{"a": 1}\n', encoding="utf-8")
    assert mod._load_jsonl(mixed) == [{"value": []}, {"a": 1}]

    corrupt_audit = tmp_path / "corrupt-audit.json"
    corrupt_audit.write_text("{bad", encoding="utf-8")
    receipt = mod.upstream_gate_receipt(
        repo_root=REPO,
        audit_path=corrupt_audit,
        fixture_path=REPO / mod.FIXTURE_RELATIVE_PATH,
        protected_before={},
        runtime_state={},
    )
    assert receipt["parse_status"] == "corrupt_json"
    assert receipt["gate_passed"] is False

    class KeyStats:
        def __len__(self) -> int:
            return 1

        def key(self, _index: int) -> str:
            return "conflicts"

        def get_key_value(self, _key: str) -> int:
            return 7

    class BadStats:
        def __len__(self) -> int:
            raise RuntimeError("bad stats")

    assert mod._stat_value(KeyStats(), ("conflicts",)) == 7
    assert mod._stat_value(BadStats(), ("conflicts",)) == 0

    bad_conflicts = mod.solver_conflict_rows(
        fixture_rows=[
            {
                "local_unit_id": "u",
                "split_name": "held",
                "domain": "logic_grid",
                "source_turn_id": "p:turn:1",
                "solver_effort": {"solver_assertion_count": 9},
            }
        ],
        source_root=tmp_path / "missing-source",
    )
    assert bad_conflicts[0]["error"].startswith("RuntimeError")
    assert bad_conflicts[0]["solver_assertion_count"] == 9

    tiny_fixture = [
        {
            "local_unit_id": f"u{i}",
            "split_name": "held",
            "domain": "logic_grid",
            "source_row_hash": f"h{i}",
        }
        for i in range(4)
    ]
    tiny_conflicts = [
        {
            "local_unit_id": f"u{i}",
            "conflict_count": i,
            "conflict_quantile": i / 4,
            "conflict_stratum": "low",
        }
        for i in range(4)
    ]
    assert (
        len(mod.logical_instance_rows(fixture_rows=tiny_fixture, conflict_rows=tiny_conflicts))
        == mod.MAX_LOGICAL_INSTANCES
    )
    assert mod._parse_final_label("FINAL: CONTRADICTION") == "contradiction"

    invalid_rows_checkpoint = tmp_path / "invalid-rows.ckpt"
    invalid_rows_checkpoint.write_text(
        json.dumps({"schema": mod.CHECKPOINT_SCHEMA, "challenge_hash": "h", "rows_by_key": []}),
        encoding="utf-8",
    )
    assert mod.load_checkpoint(invalid_rows_checkpoint, "h")["rows_by_key"] == {}

    no_label_backend = FakeBackend()

    def no_label_infer(**_kwargs: Any) -> dict[str, Any]:
        return {
            "terminal_status": "terminal",
            "timeout": False,
            "parse_failure": False,
            "output_text": "no final label",
            "output_tokens": 3,
            "wall_time_s": 0.1,
            "first_token_time_s": None,
            "error": "",
        }

    no_label_backend.infer = no_label_infer  # type: ignore[method-assign]
    surface = {"prompt": "Solve", "prompt_sha256": "sha256:p"}
    parse_row = mod._model_unit_row(
        backend=no_label_backend,
        spec=fake_model_specs[0],
        logical={"exact_label": "satisfiable"},
        surface=surface,
        arm_id="unguarded",
        unit_key="u",
    )
    assert parse_row["terminal_status"] == "parse_failure"

    checkpoint_path = tmp_path / "reuse.ckpt"
    first = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "reuse-1.json",
        checkpoint_path=checkpoint_path,
        write=False,
        duration_s=120.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        inference_backend=FakeBackend(),
    )
    second = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "reuse-2.json",
        checkpoint_path=checkpoint_path,
        write=False,
        duration_s=120.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
        model_specs_override=fake_model_specs,
        inference_backend=FakeBackend(),
    )
    assert first["censoring_and_timeout_receipts"]["checkpoint_receipt"]["saved_row_count"] > 0
    assert second["censoring_and_timeout_receipts"]["checkpoint_receipt"]["reused_row_count"] > 0

    assert mod._slope([1.0], [2.0]) == 0.0
    assert mod._slope([1.0, 1.0], [2.0, 3.0]) == 0.0

    true = {"gate_passed": True}
    protected = {"all_protected_files_unchanged": True}
    aggregate_model_block = mod.aggregate_row_recomputation(
        gate=true,
        preconditions={"failed_preconditions": []},
        model_receipts={"all_mandated_models_loaded": False},
        rows=[{}],
        surface_receipts={"all_surfaces_equivalent": True},
        effects=[],
        guarded_rows=[],
        recomputation={},
        exact_completion={},
        censoring={},
        attacks={"all_confounds_fail_closed": True},
        protected=protected,
    )
    assert aggregate_model_block["verdict_class_from_rows"] == "blocked"

    partial = mod.aggregate_row_recomputation(
        gate=true,
        preconditions={"failed_preconditions": []},
        model_receipts={"all_mandated_models_loaded": True},
        rows=[{}],
        surface_receipts={"all_surfaces_equivalent": True},
        effects=[],
        guarded_rows=[{"supports_benefit": True}],
        recomputation={"guarded_total_charged_tokens": 1, "unguarded_total_charged_tokens": 2},
        exact_completion={"guarded_noninferior_exact_completion": True},
        censoring={},
        attacks={"all_confounds_fail_closed": True},
        protected=protected,
    )
    assert partial["verdict_class_from_rows"] == "partial"
    assert mod._status_and_honest_verdict(partial)[2] == "partial"

    null = mod.aggregate_row_recomputation(
        gate=true,
        preconditions={"failed_preconditions": []},
        model_receipts={"all_mandated_models_loaded": True},
        rows=[{}],
        surface_receipts={"all_surfaces_equivalent": True},
        effects=[],
        guarded_rows=[],
        recomputation={"guarded_total_charged_tokens": 2, "unguarded_total_charged_tokens": 2},
        exact_completion={"guarded_noninferior_exact_completion": True},
        censoring={},
        attacks={"all_confounds_fail_closed": True},
        protected=protected,
    )
    assert null["verdict_class_from_rows"] is None
    assert mod._status_and_honest_verdict(null)[2] is None

    live_preconditions = mod.preconditions_checked(
        repo_root=REPO,
        result_path=tmp_path / "r.json",
        checkpoint_path=tmp_path / "c.json",
        model_specs=mod.normalize_model_specs(fake_model_specs),
        runtime_state={
            "gpu": {"available": False},
            "llama_cpp": {
                "available": False,
                "cuda_backend_available": False,
                "gpu_offload_supported": False,
            },
        },
        live_runtime_required=True,
        source_root=tmp_path,
        run_date="20260823",
    )
    assert live_preconditions["failed_live_runtime_preconditions"] == [
        "gpu_unavailable",
        "llama_cpp_unavailable",
        "llama_cpp_cuda_backend_unavailable",
    ]
