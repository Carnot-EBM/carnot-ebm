"""Tests for Exp6505 local SOTA formal challenge mutations.

Spec refs: REQ-BENCH-6505, SCENARIO-BENCH-6505-PROVENANCE,
SCENARIO-BENCH-6505-ONE-SHOT, SCENARIO-BENCH-6505-NO-ANSWER,
SCENARIO-BENCH-6505-ADMISSION, SCENARIO-BENCH-6505-SCORES,
SCENARIO-BENCH-6505-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6505_sota_formal_challenge_mutations as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6505_sota_formal_challenge_mutations.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6505_sota_formal_challenge_mutations.py "
    "-m pytest tests/python/test_experiment_6505_sota_formal_challenge_mutations.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6505_sota_formal_challenge_mutations.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6505_sota_formal_challenge_mutations.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6505_sota_formal_challenge_mutations --date 20260822"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6505_sota_formal_challenge_mutations.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6505_sota_formal_challenge_mutations.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6505_sota_formal_challenge_mutations --validate"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6505_sota_formal_challenge_mutations.py "
    "tests/python/test_experiment_6505_sota_formal_challenge_mutations.py "
    "scripts/adversarial_verify.py"
)
GIT_STATUS_COMMAND = "git status --short"
TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUFF_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


class FakeMutationRunner:
    """Deterministic stand-in for llama_cpp generation in unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        model_spec: dict[str, Any],
        requests: list[dict[str, Any]],
        decode_config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model_spec["hf_id"],
                "request_count": len(requests),
                "decode_config": dict(decode_config),
            }
        )
        rows = []
        for request in requests:
            request_id = str(request["request_id"])
            if model_spec["hf_id"].endswith(
                ("Qwen3.6-35B-A3B-GGUF", "gemma-4-31B-it-GGUF")
            ):
                text = "BEGIN_MUTATION\nADD_CLAUSE -1\nSHIFT_COEFFICIENT 1 2\nEND_MUTATION\n"
            else:
                text = "ANSWER sat\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n"
            rows.append(
                {
                    "request_id": request_id,
                    "response_text": text,
                    "response_bytes": text.encode("utf-8"),
                    "terminal_disposition": "generated",
                    "finish_reason": "stop",
                    "truncated": False,
                    "generated_token_count": 12,
                    "decode_time_s": 0.01,
                    "error": "",
                }
            )
        return {
            "model_runtime_receipt": {
                "model_id": model_spec["hf_id"],
                "runtime_backend": "fake_llama_cpp",
                "llama_cpp_import_ok": True,
                "llama_cpp_supports_gpu_offload": True,
                "embedded_tokenizer": {
                    "source": "embedded_gguf_tokenizer",
                    "loadable": True,
                    "probe_token_count": 4,
                },
                "offload": {"n_gpu_layers": -1, "main_gpu": 0},
                "vram": {"before_free_mib": 24000, "after_free_mib": 23900},
                "timing": {"load_time_s": 0.01, "total_time_s": 0.02},
                "terminal_disposition": "complete",
                "request_count": len(requests),
            },
            "rows": rows,
        }


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    result_path = tmp_path_factory.mktemp("exp6505") / mod.RESULT_RELATIVE_PATH.name
    return mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
        generation_runner=FakeMutationRunner(),
    )


def test_req_bench_6505_spec_model_specs_and_required_fields() -> None:
    """REQ-BENCH-6505: OpenSpec owns the one-shot mutation contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6505") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6505-PROVENANCE",
        "SCENARIO-BENCH-6505-ONE-SHOT",
        "SCENARIO-BENCH-6505-NO-ANSWER",
        "SCENARIO-BENCH-6505-ADMISSION",
        "SCENARIO-BENCH-6505-SCORES",
        "SCENARIO-BENCH-6505-ARTIFACT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`AutoTokenizer`",
    ):
        assert marker in section
    assert [row["hf_id"] for row in mod.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert "AutoTokenizer" not in mod.MODULE_RELATIVE_PATH.read_text(encoding="utf-8")
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6505_no_answer_boundary_rejects_prohibited_text() -> None:
    """SCENARIO-BENCH-6505-NO-ANSWER: prohibited content fails closed."""

    source = mod.select_development_sources(REPO, count=1)[0]
    clean = "BEGIN_MUTATION\nADD_CLAUSE -1\nRELBL 1 2\nEND_MUTATION\n"
    parsed = mod.parse_edit_script(clean.encode("utf-8"), source)
    assert parsed["parse_ok"] is True
    assert {op["op"] for op in parsed["operations"]} == {"ADD_CLAUSE", "RELBL"}

    attacks = {
        "ANSWER 42\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n": "answer",
        "LABEL unsat\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n": "label",
        "Use unit propagation first.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n": "solver_advice",
        "Translate the story into ConstraintIR.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n": "semantic_translation",
        "Release this to held.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n": "release_decision",
        "BEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\nBEGIN_MUTATION\nADD_CLAUSE 2\nEND_MUTATION\n": "retry",
        "Use held split row 001.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n": "held_access",
        "Qwen should choose this.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n": "model_identity",
    }
    for text, expected in attacks.items():
        parsed = mod.parse_edit_script(text.encode("utf-8"), source)
        assert parsed["parse_ok"] is False
        assert expected in parsed["prohibited_reasons"]

    malformed = [
        b"ADD_CLAUSE 1\n",
        b"BEGIN_MUTATION\nBOGUS 1\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nADD_CLAUSE\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nADD_CLAUSE nope\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nADD_CLAUSE 999\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nADD_JOB 1\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nDROP_CLAUSE\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nADD_EDGE 1\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nADD_EDGE nope 1\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nADD_EDGE 0 1\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nSHIFT_COEFFICIENT 1\nEND_MUTATION\n",
        b"BEGIN_MUTATION\nSHIFT_COEFFICIENT 1 nope\nEND_MUTATION\n",
        (
            b"BEGIN_MUTATION\n"
            b"ADD_CLAUSE 1\nADD_CLAUSE 2\nADD_CLAUSE 3\nADD_CLAUSE 4\n"
            b"ADD_CLAUSE 5\nADD_CLAUSE 6\nADD_CLAUSE -1\nADD_CLAUSE -2\n"
            b"ADD_CLAUSE -3\nEND_MUTATION\n"
        ),
    ]
    for payload in malformed:
        parsed = mod.parse_edit_script(payload, source)
        assert parsed["parse_ok"] is False
        assert parsed["parse_errors"]


def test_scenario_bench_6505_exact_admission_accepts_and_quarantines() -> None:
    """SCENARIO-BENCH-6505-ADMISSION: exact tools decide admission."""

    source = mod.select_development_sources(REPO, count=1)[0]
    model = mod.MODEL_SPECS[0]
    response = b"BEGIN_MUTATION\nADD_CLAUSE -1\nSHIFT_COEFFICIENT 1 2\nEND_MUTATION\n"
    first = mod.admit_response(
        request_row={
            "request_id": "req-1",
            "source_instance_id": source["instance_id"],
            "source_raw_instance_hash": source["raw_instance_hash"],
            "model_id": model["hf_id"],
            "model_family": model["family"],
            "seed": 6505001,
            "truncated": False,
        },
        source=source,
        response_bytes=response,
        seen_mutation_hashes=set(),
    )
    assert first["accepted"] is True
    assert first["parse_ok"] is True
    assert first["changed"] is True
    assert first["novel"] is True
    assert first["exact_label"] in {"sat", "unsat"}
    assert first["model_or_proof_valid"] is True
    assert first["no_prohibited_output"] is True
    assert first["duplicate"] is False
    assert first["mutation_hash"].startswith("sha256:")

    duplicate_seen = {first["mutation_hash"]}
    duplicate = mod.admit_response(
        request_row={
            "request_id": "req-2",
            "source_instance_id": source["instance_id"],
            "source_raw_instance_hash": source["raw_instance_hash"],
            "model_id": model["hf_id"],
            "model_family": model["family"],
            "seed": 6505002,
            "truncated": False,
        },
        source=source,
        response_bytes=response,
        seen_mutation_hashes=duplicate_seen,
    )
    assert duplicate["accepted"] is False
    assert duplicate["duplicate"] is True
    assert duplicate["quarantine_reason"] == "duplicate_mutation"

    unchanged = mod.admit_response(
        request_row={
            "request_id": "req-3",
            "source_instance_id": source["instance_id"],
            "source_raw_instance_hash": source["raw_instance_hash"],
            "model_id": model["hf_id"],
            "model_family": model["family"],
            "seed": 6505003,
            "truncated": False,
        },
        source=source,
        response_bytes=b"BEGIN_MUTATION\nSHIFT_COEFFICIENT 1 0\nEND_MUTATION\n",
        seen_mutation_hashes=set(),
    )
    assert unchanged["accepted"] is False
    assert unchanged["changed"] is False
    assert unchanged["quarantine_reason"] == "unchanged_mutation"

    rich = mod.admit_response(
        request_row={
            "request_id": "req-4",
            "source_instance_id": source["instance_id"],
            "source_raw_instance_hash": source["raw_instance_hash"],
            "model_id": model["hf_id"],
            "model_family": model["family"],
            "seed": 6505004,
            "truncated": False,
        },
        source=source,
        response_bytes=(
            b"BEGIN_MUTATION\n"
            b"ADD_JOB\n"
            b"ADD_EDGE 1 2\n"
            b"DROP_EDGE 1 2\n"
            b"SWAP_COLOR 1 2\n"
            b"ADD_PRECEDENCE 1 2\n"
            b"DROP_CLAUSE 0\n"
            b"END_MUTATION\n"
        ),
        seen_mutation_hashes=set(),
    )
    assert rich["parse_ok"] is True
    assert rich["accepted"] is True
    assert {"ADD_JOB", "ADD_EDGE", "DROP_EDGE", "SWAP_COLOR", "ADD_PRECEDENCE", "DROP_CLAUSE"} <= set(
        rich["edit_types"]
    )

    invalid_ops = [
        [{"op": "ADD_CLAUSE", "literals": [999]}],
        [{"op": "DROP_CLAUSE", "index": 999}],
        [{"op": "ADD_EDGE", "left": 1, "right": 1}],
        [{"op": "DROP_EDGE", "left": 1, "right": 2}],
        [{"op": "RELBL", "left": 1, "right": 1}],
        [{"op": "ADD_PRECEDENCE", "left": 1, "right": 1}],
        [{"op": "SHIFT_COEFFICIENT", "var": 999, "delta": 1}],
        [{"op": "SHIFT_COEFFICIENT", "var": 1, "delta": 10}],
    ]
    for ops in invalid_ops:
        replayed = mod.apply_edit_script(source, ops)
        assert replayed.replay_errors


def test_scenario_bench_6505_precondition_and_blocked_paths(
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6505-PROVENANCE: failed runtime gates are terminal."""

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert mod._infer_quantization(None, "Q4_K_M") == "missing"
    assert mod._infer_quantization("/tmp/model-custom.gguf", "Q4_K_M") == "Q4_K_M"
    assert len(mod.select_development_sources(REPO, count=4)) == 4

    monkeypatch.setattr(mod, "_nvidia_smi_rows", lambda: [])
    monkeypatch.setattr(
        mod,
        "llama_cpp_status",
        lambda: {
            "import_ok": False,
            "version": "unavailable",
            "supports_gpu_offload": False,
            "system_info": "",
        },
    )
    monkeypatch.setattr(mod, "cached_sota_pair", lambda: None)
    monkeypatch.setattr(
        mod,
        "_disk_receipt",
        lambda _root: {"total_bytes": 1, "used_bytes": 1, "free_bytes": 1},
    )
    failed_specs = [dict(row, model_file_exists=False) for row in artifact["model_specs"]]
    failed = mod.preconditions_checked(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        run_date="20260822",
        gate={"passed": False, "observed_value": 0.0},
        model_specs=failed_specs,
        protected_before={},
    )
    assert failed["preconditions_ready"] is False
    assert {row["check"] for row in failed["failed_precondition_checks"]} >= {
        "exp6504_gate",
        "model_cache",
        "llama_cpp_import",
        "llama_cpp_cuda_offload",
        "gpu_inventory",
        "disk_free_bytes",
    }

    blocked = mod.blocked_generation_result(mod.MODEL_SPECS[0], [{"request_id": "r"}], b"boom")
    assert blocked["model_runtime_receipt"]["terminal_disposition"] == "blocked"
    assert blocked["rows"][0]["terminal_disposition"] == "runtime_blocked"

    monkeypatch.setattr(mod, "preconditions_checked", lambda **_kwargs: failed)
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    blocked_artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
        generation_runner=FakeMutationRunner(),
    )
    assert blocked_artifact["status"] == "blocked_formal_challenge_mutation_generation"
    assert blocked_artifact["challenge_generation_complete_score"] == 1.0
    assert blocked_artifact["challenge_pool_ready_score"] == 0.0

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced error"])
    with pytest.raises(ValueError, match="forced error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "bad.json",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260822",
            generation_runner=FakeMutationRunner(),
        )


def test_scenario_bench_6505_admission_failure_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6505-ADMISSION: exact failure modes are explicit."""

    source = mod.select_development_sources(REPO, count=1)[0]
    request = {
        "request_id": "req-fail",
        "source_instance_id": source["instance_id"],
        "source_raw_instance_hash": source["raw_instance_hash"],
        "model_id": mod.MODEL_SPECS[0]["hf_id"],
        "model_family": mod.MODEL_SPECS[0]["family"],
        "seed": 6505999,
        "truncated": False,
    }

    monkeypatch.setattr(mod, "_mutated_label_row", lambda _source, _mutated: {})
    ambiguous = mod.admit_response(
        request_row=request,
        source=source,
        response_bytes=b"BEGIN_MUTATION\nADD_CLAUSE -1\nEND_MUTATION\n",
        seen_mutation_hashes=set(),
    )
    assert ambiguous["quarantine_reason"] == "label_ambiguous"

    monkeypatch.setattr(
        mod,
        "_mutated_label_row",
        lambda _source, _mutated: {
            "accepted": True,
            "exact_label": "sat",
            "model_or_proof_valid": False,
            "proof_receipt": {},
            "backend_receipts": [],
        },
    )
    invalid = mod.admit_response(
        request_row=request,
        source=source,
        response_bytes=b"BEGIN_MUTATION\nADD_CLAUSE -1\nEND_MUTATION\n",
        seen_mutation_hashes=set(),
    )
    assert invalid["quarantine_reason"] == "model_or_proof_invalid"

    monkeypatch.setattr(
        mod,
        "_mutated_label_row",
        lambda _source, _mutated: {
            "accepted": False,
            "exact_label": "sat",
            "model_or_proof_valid": True,
            "proof_receipt": {},
            "backend_receipts": [],
        },
    )
    failed = mod.admit_response(
        request_row=request,
        source=source,
        response_bytes=b"BEGIN_MUTATION\nADD_CLAUSE -1\nEND_MUTATION\n",
        seen_mutation_hashes=set(),
    )
    assert failed["quarantine_reason"] == "exact_admission_failed"

    parse_failed = mod.admit_response(
        request_row=request,
        source=source,
        response_bytes=b"ADD_CLAUSE -1\n",
        seen_mutation_hashes=set(),
    )
    assert parse_failed["quarantine_reason"] == "parse_failed"

    edit_failed = mod.admit_response(
        request_row=request,
        source=source,
        response_bytes=b"BEGIN_MUTATION\nDROP_EDGE 1 2\nEND_MUTATION\n",
        seen_mutation_hashes=set(),
    )
    assert edit_failed["quarantine_reason"] == "edit_replay_failed"

    blocked = mod._status_verdict(
        {"challenge_generation_complete_score_from_rows": 0.0},
        {"checks": {"exp6504_gate": {"passed": False}}, "blocked_reason": "blocked_gate"},
    )
    assert blocked[0] == "blocked_formal_challenge_mutation_generation"
    complete_null = mod._status_verdict(
        {
            "challenge_generation_complete_score_from_rows": 1.0,
            "challenge_pool_ready_score_from_rows": 0.0,
            "runtime_blocked_count": 0,
        },
        {"checks": {"exp6504_gate": {"passed": True}}, "blocked_reason": ""},
    )
    assert complete_null[0] == "complete_null_formal_challenge_mutation_accounting"


def test_scenario_bench_6505_artifact_rows_scores_and_bytes(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6505-ONE-SHOT/SCORES/ARTIFACT: rows recompute."""

    result_path = Path(artifact["preconditions_checked"]["result_path"])
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_formal_challenge_mutation_accounting"
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_formal_challenge_mutations:")
    assert artifact["upstream_gate_receipt"]["observed_value"] == 1.0
    assert artifact["upstream_gate_receipt"]["passed"] is True
    assert artifact["challenge_generation_complete_score"] == 1.0
    assert artifact["challenge_pool_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    selected_sources = artifact["source_selection_receipt"]["selected_sources"]
    assert len(selected_sources) == mod.DEFAULT_SOURCE_COUNT
    assert all(row["split"] == "development" for row in selected_sources)
    assert artifact["source_selection_receipt"]["selected_before_generation"] is True

    expected_requests = mod.DEFAULT_SOURCE_COUNT * len(mod.MODEL_SPECS)
    assert len(artifact["raw_request_response_receipts"]) == expected_requests
    assert len(artifact["rows"]) == expected_requests
    assert len(artifact["exact_admission_rows"]) == expected_requests
    assert len(artifact["per_unit_rows"]) == expected_requests * 2
    assert all(row["request_sha256"].startswith("sha256:") for row in artifact["raw_request_response_receipts"])
    assert all(row["response_sha256"].startswith("sha256:") for row in artifact["raw_request_response_receipts"])
    assert all(row["parser_invoked_after_raw_persist"] is True for row in artifact["rows"])
    assert all(row["retry_count"] == 0 for row in artifact["rows"])

    accepted = [row for row in artifact["exact_admission_rows"] if row["accepted"]]
    rejected = [row for row in artifact["exact_admission_rows"] if not row["accepted"]]
    assert len(accepted) == 1
    assert rejected
    assert {row["quarantine_reason"] for row in rejected} >= {
        "duplicate_mutation",
        "prohibited_output",
    }
    assert all(row["no_prohibited_output"] is True for row in accepted)

    family_rows = artifact["model_family_results"]
    assert {row["model_id"] for row in family_rows} == {row["hf_id"] for row in mod.MODEL_SPECS}
    assert sum(row["request_count"] for row in family_rows) == expected_requests
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"]
    )
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["preconditions_checked"]["model_cache"]["all_cached"] is True
    assert artifact["preconditions_checked"]["llama_cpp"]["import_ok"] in {True, False}


def test_scenario_bench_6505_validation_mutations_fail_closed(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6505-ARTIFACT: validator rejects unsafe mutations."""

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        ("verdict_class outside closed enum", lambda item: item.__setitem__("verdict_class", "bad")),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "bad"),
        ),
        (
            "verifier_is_oracle must be true for exact admission only",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        (
            "challenge_generation_complete_score mismatch",
            lambda item: item.__setitem__("challenge_generation_complete_score", 0.0),
        ),
        (
            "challenge_pool_ready_score mismatch",
            lambda item: item.__setitem__("challenge_pool_ready_score", 0.0),
        ),
        (
            "aggregate_row_recomputation mismatch",
            lambda item: item["aggregate_row_recomputation"].__setitem__("request_count", -1),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "accepted row contains prohibited output",
            lambda item: item["exact_admission_rows"][0].__setitem__("no_prohibited_output", False),
        ),
        (
            "model_specs must list all mandated models",
            lambda item: item.__setitem__("model_specs", item["model_specs"][:2]),
        ),
        (
            "model_specs order mismatch",
            lambda item: item["model_specs"].reverse(),
        ),
        (
            "model_family_results must cover all families",
            lambda item: item.__setitem__("model_family_results", item["model_family_results"][:2]),
        ),
        (
            "prohibited_output_attack_matrix false accepts",
            lambda item: item["prohibited_output_attack_matrix"].__setitem__(
                "false_accept_count",
                1,
            ),
        ),
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "not terminal"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)

    assert mod.validate_artifact(Path("/tmp/carnot-exp6505-missing-artifact.json"))


def test_scenario_bench_6505_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6505-ARTIFACT: CLI writes and validates the artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert (
        mod.main(
            [
                "--date",
                "20260822",
                "--result-path",
                str(result_path),
                "--fixture-mode",
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["challenge_generation_complete_score"] == 1.0
    assert mod.main(["--validate", "--result-path", str(tmp_path / "missing.json")]) == 1
