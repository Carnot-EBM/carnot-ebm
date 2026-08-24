"""Tests for the Exp6570 independent proof-obligation audit.

Spec refs: REQ-REPORT-6570, SCENARIO-REPORT-6570-RAW,
SCENARIO-REPORT-6570-SPAN, SCENARIO-REPORT-6570-COMPILER,
SCENARIO-REPORT-6570-PAIRS, SCENARIO-REPORT-6570-ATTACKS,
SCENARIO-REPORT-6570-MISSING, SCENARIO-REPORT-6570-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot import experiment_6566_proof_obligation_and_graph_potts_method_contract as compiler
from carnot import experiment_6570_proof_obligation_independent_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6570", "exit_code": 0}]


def _write(path: Path, data: bytes) -> dict[str, Any]:
    path.write_bytes(data)
    return {
        "path": str(path),
        "sha256": mod.sha256_file(path),
        "byte_size": len(data),
    }


def _clean_inputs(tmp_path: Path) -> dict[str, Path]:
    model_specs: list[dict[str, Any]] = []
    model_files: dict[str, Path] = {}
    for index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"GGUF fixture {hf_id}".encode())
        model_files[hf_id] = path
        model_specs.append(
            {
                "hf_id": hf_id,
                "model_path": str(path),
                "model_file_sha256": mod.sha256_file(path),
            }
        )

    raw_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    compiler_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for model_index, hf_id in enumerate(mod.MANDATED_HF_IDS):
        for family_index, family in enumerate(("logic_grid", "scheduling")):
            for unit_index in range(2):
                unit_id = f"u-{model_index}-{family_index}-{unit_index}"
                left = 9 + unit_index
                right = 3
                source = f"Item {left} is greater than item {right}.".encode()
                prompt = f"Find the relation in {unit_id}.".encode()
                raw = f"{left} greater_than {right}".encode()
                source_ref = _write(tmp_path / f"{unit_id}.source", source)
                prompt_ref = _write(tmp_path / f"{unit_id}.prompt", prompt)
                raw_ref = _write(tmp_path / f"{unit_id}.raw", raw)
                token_ids = [left, right, model_index + 20]
                raw_rows.append(
                    {
                        "unit_id": unit_id,
                        "model_hf_id": hf_id,
                        "family": family,
                        "seed": 657000 + model_index,
                        "source_path": source_ref["path"],
                        "source_sha256": source_ref["sha256"],
                        "prompt_path": prompt_ref["path"],
                        "prompt_sha256": prompt_ref["sha256"],
                        "raw_response_path": raw_ref["path"],
                        "raw_output_sha256": raw_ref["sha256"],
                        "model_path": str(model_files[hf_id]),
                        "model_file_sha256": mod.sha256_file(model_files[hf_id]),
                        "command": f"llama-cli --model {model_files[hf_id]}",
                        "os_command": f"llama-cli --model {model_files[hf_id]}",
                        "pid": 8100 + model_index,
                        "os_pid": 8100 + model_index,
                        "parent_pid": 8000,
                        "start_monotonic_s": 10.0,
                        "end_monotonic_s": 11.0,
                        "receipt_captured_during_run": True,
                        "stale_receipt": False,
                        "gpu_samples": [
                            {"stage": "before", "pid": 0, "used_mb": 100},
                            {
                                "stage": "during",
                                "pid": 8100 + model_index,
                                "used_mb": 500,
                            },
                            {"stage": "after", "pid": 0, "used_mb": 110},
                        ],
                        "token_ids": token_ids,
                        "token_sha256": mod.sha256_json(token_ids),
                        "exit_code": 0,
                        "timed_out": False,
                        "unloaded": True,
                    }
                )

                claim_id = f"claim-{unit_id}"
                source_text = source.decode()
                claim = {
                    "claim_id": claim_id,
                    "unit_id": unit_id,
                    "model_hf_id": hf_id,
                    "family": family,
                    "seed": 657000 + model_index,
                    "source_path": source_ref["path"],
                    "source_sha256": source_ref["sha256"],
                    "source_start": 0,
                    "source_end": len(source),
                    "span_text": source_text,
                    "source_span_text_sha256": mod.sha256_bytes(source),
                    "typed_variables": {"left": "item_number", "right": "item_number"},
                    "bindings": {"left": left, "right": right},
                    "relation": "greater_than",
                    "operands": {"left": left, "right": right},
                    "overlap_allowed": False,
                }
                claim_rows.append(claim)
                compiled = compiler.compile_claim(
                    {
                        "unit_id": unit_id,
                        "source_text": source_text,
                        "span_text": source_text,
                        "source_start": 0,
                        "source_end": len(source_text),
                        "typed_variables": claim["typed_variables"],
                        "relation": claim["relation"],
                        "operands": claim["operands"],
                    }
                )
                compiler_rows.append(
                    {
                        "claim_id": claim_id,
                        "normalized_obligation": mod.canonical_json(compiled),
                        "normalized_obligation_sha256": mod.sha256_json(compiled),
                        "executable_obligation_hash": compiled["executable_obligation_hash"],
                        "compiler_name": compiler.COMPILER_NAME,
                        "compiler_version": compiler.COMPILER_VERSION,
                        "exact_result": compiled["exact_result"],
                        "witness": compiled.get("witness"),
                        "counterexample": compiled["counterexample"],
                        "release_action": compiled["release_action"],
                        "abstention": compiled["abstention"],
                    }
                )

                for arm in ("control", "proof_carrying"):
                    released = arm == "proof_carrying" or unit_index == 0
                    row = {
                        "unit_id": unit_id,
                        "model_hf_id": hf_id,
                        "family": family,
                        "seed": 657000 + model_index,
                        "arm_id": arm,
                        "target_supported": True,
                        "released": released,
                        "abstention": not released,
                        "exact_correct": released,
                        "false_accept": False,
                        "false_reject": not released,
                        "unsafe_release": False,
                        "prompt_tokens": 12,
                        "output_tokens": 4 if arm == "proof_carrying" else 2,
                        "retries": 0,
                        "solver_calls": 1 if arm == "proof_carrying" else 0,
                        "wall_time_s": 0.02 if arm == "proof_carrying" else 0.01,
                        "censored": False,
                        "charged_cost": 18.02 if arm == "proof_carrying" else 14.01,
                    }
                    unit_rows.append(row)
                    cost_rows.append(
                        {
                            key: row[key]
                            for key in (
                                "unit_id",
                                "model_hf_id",
                                "family",
                                "seed",
                                "arm_id",
                                "prompt_tokens",
                                "output_tokens",
                                "retries",
                                "solver_calls",
                                "wall_time_s",
                                "censored",
                                "charged_cost",
                            )
                        }
                    )

    exp6568 = {
        "status": "complete_immutable_live_claim_stream",
        "honest_verdict": "complete_stream",
        "verdict_class": None,
        "immutable_live_claim_stream_ready_score": 1.0,
        "MODEL_SPECS": model_specs,
        "source_prompt_and_raw_response_rows": raw_rows,
        "live_model_gpu_token_receipts": raw_rows,
        "independent_exact_target_rows": [],
        "charged_cost_rows": cost_rows,
        "per_unit_rows": raw_rows,
        "shard_journal_and_restart_receipts": {"shards": [], "restart_exact": True},
    }
    exp6568_path = tmp_path / "exp6568.json"
    exp6568_path.write_text(json.dumps(exp6568), encoding="utf-8")

    exp6569 = {
        "status": "complete_proof_carrying_extractor_candidate",
        "honest_verdict": "complete_candidate",
        "verdict_class": "positive",
        "proof_carrying_extractor_execution_ready_score": 1.0,
        "proof_carrying_extractor_candidate_score": 1.0,
        "source_span_claim_rows": claim_rows,
        "compiler_and_exact_obligation_rows": compiler_rows,
        "per_unit_rows": unit_rows,
        "charged_cost_rows": cost_rows,
        "unsafe_release_and_abstention_rows": [],
        "shortcut_attack_matrix": [],
        "aggregate_row_recomputation": {"candidate_score": 1.0},
    }
    exp6569_path = tmp_path / "exp6569.json"
    exp6569_path.write_text(json.dumps(exp6569), encoding="utf-8")
    return {"exp6568_artifact": exp6568_path, "exp6569_artifact": exp6569_path}


def test_req_report_6570_spec_declares_the_full_audit_contract() -> None:
    """REQ-REPORT-6570: OpenSpec owns all audit fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6570") : text.index("REQ-REPORT-6566")]
    normalized = " ".join(section.split())
    for marker in (
        "SCENARIO-REPORT-6570-RAW",
        "SCENARIO-REPORT-6570-SPAN",
        "SCENARIO-REPORT-6570-COMPILER",
        "SCENARIO-REPORT-6570-PAIRS",
        "SCENARIO-REPORT-6570-ATTACKS",
        "SCENARIO-REPORT-6570-MISSING",
        "SCENARIO-REPORT-6570-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenarios_report_6570_clean_replay_confirms_promotion(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6570-RAW/SPAN/COMPILER/PAIRS/ATTACKS/ATOMIC: close."""

    output = tmp_path / "audit.json"
    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=output,
        input_paths=_clean_inputs(tmp_path),
        write=True,
        duration_s=2.0,
        tests_run=TESTS_RUN,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_proof_obligation_independent_audit"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] is None
    assert artifact["proof_carrying_extractor_audit_ready_score"] == 1.0
    assert artifact["proof_carrying_extractor_promotion_score"] == 1.0
    assert all(row["provenance_valid"] for row in artifact["independent_live_provenance_rows"])
    assert all(row["span_valid"] for row in artifact["independent_source_span_rows"])
    assert all(
        row["compiler_and_exact_match"]
        for row in artifact["independent_compiler_and_exact_replay_rows"]
    )
    assert all(row["passed"] for row in artifact["shortcut_attack_matrix"])
    assert artifact["harmful_release_and_cost_audit"]["unsafe_release_count"] == 0
    assert artifact["gate_check_summary"]["failed_checks"] == []


def test_scenario_report_6570_missing_or_gate_blocked_input_closes_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6570-MISSING: absent and unusable inputs close terminally."""

    span_root = tmp_path / "bad-span"
    span_root.mkdir()
    paths = _clean_inputs(span_root)
    paths["exp6569_artifact"] = tmp_path / "missing.json"
    missing = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "missing-audit.json",
        input_paths=paths,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )
    assert missing["status"] == "blocked_proof_obligation_independent_audit_missing_inputs"
    assert missing["verdict_class"] == "blocked"
    assert missing["proof_carrying_extractor_audit_ready_score"] == 0.0
    assert missing["proof_carrying_extractor_promotion_score"] == 0.0
    assert "inputs_usable" in missing["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(missing) == []

    compiler_root = tmp_path / "bad-compiler"
    compiler_root.mkdir()
    paths = _clean_inputs(compiler_root)
    exp6568 = json.loads(paths["exp6568_artifact"].read_text(encoding="utf-8"))
    exp6568["status"] = "blocked_gate_check_failed"
    exp6568["immutable_live_claim_stream_ready_score"] = 0.0
    paths["exp6568_artifact"].write_text(json.dumps(exp6568), encoding="utf-8")
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked-audit.json",
        input_paths=paths,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )
    assert blocked["verdict_class"] == "blocked"
    assert "upstream_terminal_evidence" in blocked["gate_check_summary"]["failed_checks"]


def test_scenarios_report_6570_invalid_span_compiler_or_release_disqualifies(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6570-SPAN/COMPILER/ATTACKS: invalid evidence fails closed."""

    cases: list[tuple[str, Any]] = []

    span_root = tmp_path / "invalid-span"
    span_root.mkdir()
    paths = _clean_inputs(span_root)
    exp6569 = json.loads(paths["exp6569_artifact"].read_text(encoding="utf-8"))
    source_path = Path(exp6569["source_span_claim_rows"][0]["source_path"])
    source_path.write_bytes("é is greater than 1.".encode())
    claim = exp6569["source_span_claim_rows"][0]
    claim["source_sha256"] = mod.sha256_file(source_path)
    claim["source_start"] = 1
    claim["source_end"] = 4
    paths["exp6569_artifact"].write_text(json.dumps(exp6569), encoding="utf-8")
    cases.append(("source_spans_valid", paths))

    compiler_root = tmp_path / "invalid-compiler"
    compiler_root.mkdir()
    paths = _clean_inputs(compiler_root)
    exp6569 = json.loads(paths["exp6569_artifact"].read_text(encoding="utf-8"))
    exp6569["compiler_and_exact_obligation_rows"][0]["normalized_obligation"] = "{}"
    paths["exp6569_artifact"].write_text(json.dumps(exp6569), encoding="utf-8")
    cases.append(("compiler_exact_replay", paths))

    release_root = tmp_path / "invalid-release"
    release_root.mkdir()
    paths = _clean_inputs(release_root)
    exp6569 = json.loads(paths["exp6569_artifact"].read_text(encoding="utf-8"))
    exp6569["per_unit_rows"][0]["unsafe_release"] = True
    paths["exp6569_artifact"].write_text(json.dumps(exp6569), encoding="utf-8")
    cases.append(("no_harmful_release", paths))

    for expected_failure, case_paths in cases:
        artifact = mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / f"{expected_failure}.json",
            input_paths=case_paths,
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
        )
        assert artifact["status"] == "disqualified_proof_obligation_independent_audit"
        assert artifact["verdict_class"] == "disqualified"
        assert expected_failure in artifact["gate_check_summary"]["failed_checks"]
        assert artifact["proof_carrying_extractor_audit_ready_score"] == 0.0
        assert artifact["proof_carrying_extractor_promotion_score"] == 0.0
        assert mod.validate_artifact(artifact) == []


def test_scenario_report_6570_validation_and_helper_edges(tmp_path: Path, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-6570-ATOMIC: helpers and terminal validation fail closed."""

    assert mod.sha256_file(None) == "missing"
    missing = tmp_path / "absent"
    assert mod.read_json(missing) == {}
    missing.write_text("{bad", encoding="utf-8")
    assert mod.read_json(missing) == {}
    assert mod.default_input_paths(REPO)["exp6569_artifact"].name.startswith("experiment_6569")
    assert mod._role_for_key("shard_path") == "immutable_shard"
    assert mod._role_for_key("fixture_path") == "referenced_input"

    with monkeypatch.context() as context:
        context.setattr(
            mod.Path,
            "read_text",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fixture")),
        )
        assert mod._resource_receipt(REPO)["ram"] == {
            "total_kib": None,
            "available_kib": None,
        }

    paths_for_spans = _clean_inputs(tmp_path)
    exp6569_for_spans = json.loads(paths_for_spans["exp6569_artifact"].read_text(encoding="utf-8"))
    exp6568_for_spans = json.loads(paths_for_spans["exp6568_artifact"].read_text(encoding="utf-8"))
    base_claim = exp6569_for_spans["source_span_claim_rows"][0]
    invalid_offset = deepcopy(base_claim)
    invalid_offset["claim_id"] = "invalid-offset"
    invalid_offset["source_start"] = None
    missing_source = deepcopy(base_claim)
    missing_source["claim_id"] = "missing-source"
    missing_source["source_path"] = str(tmp_path / "missing-source")
    overlapping = deepcopy(base_claim)
    overlapping["claim_id"] = "overlapping"
    span_edges = mod.independent_source_span_rows(
        [base_claim, invalid_offset, missing_source, overlapping],
        exp6568_for_spans["source_prompt_and_raw_response_rows"],
    )
    assert all(not row["span_valid"] for row in span_edges)

    with monkeypatch.context() as context:
        context.setattr(
            mod.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0, stdout="{bad", stderr="fixture"
            ),
        )
        bad_compile = mod._compile_payload_clean_process("{}", 99)
    assert bad_compile["compiled"] == {}

    assert mod._exact_relation("equals", {"left": "a", "right": "a"})[0] == "certified_true"
    assert mod._exact_relation("not_equals", {"left": "a", "right": "a"})[0] == "counterexample"
    assert mod._exact_relation("subset_of", {"left": [1], "right": [1, 2]})[0] == "certified_true"
    assert mod._exact_relation("disjoint_from", {"left": [1], "right": [2]})[0] == "certified_true"
    assert mod._exact_relation("smuggled", {"left": 1, "right": 2}) == (
        "unsupported_relation",
        None,
    )
    partial_status = mod._status_and_verdict(
        {
            "checks": {
                "required_inputs_exist": True,
                "upstream_terminal_evidence": True,
                "live_provenance_recomputable": True,
                "source_spans_valid": True,
                "compiler_exact_replay": True,
                "no_harmful_release": True,
            },
            "audit_ready_from_rows": False,
        }
    )
    assert partial_status[2] == "partial"

    clean = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "clean.json",
        input_paths=_clean_inputs(tmp_path),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
    )
    cases: list[tuple[dict[str, Any], str]] = []
    absent_field = deepcopy(clean)
    del absent_field["status"]
    cases.append((absent_field, "required field set mismatch"))
    bad_status = deepcopy(clean)
    bad_status["status"] = "ready"
    cases.append((bad_status, "status lacks terminal prefix"))
    bad_verdict = deepcopy(clean)
    bad_verdict["honest_verdict"] = "ready"
    cases.append((bad_verdict, "honest_verdict lacks terminal prefix"))
    bad_class = deepcopy(clean)
    bad_class["verdict_class"] = "positive"
    cases.append((bad_class, "verdict_class outside Exp6570 enum"))
    bad_substrate = deepcopy(clean)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    cases.append((bad_substrate, "inference_substrate mismatch"))
    bad_oracle = deepcopy(clean)
    bad_oracle["verifier_is_oracle"] = False
    cases.append((bad_oracle, "verifier_is_oracle must be true"))
    bad_seed = deepcopy(clean)
    bad_seed["random_seed"] = 0
    cases.append((bad_seed, "random_seed mismatch"))
    bad_score = deepcopy(clean)
    bad_score["proof_carrying_extractor_audit_ready_score"] = 0.0
    cases.append((bad_score, "audit ready score mismatch"))
    bad_promotion = deepcopy(clean)
    bad_promotion["proof_carrying_extractor_promotion_score"] = 0.0
    cases.append((bad_promotion, "promotion score mismatch"))
    bad_principle = deepcopy(clean)
    bad_principle["field_provenance"]["status"]["principle"] = "bad"
    cases.append((bad_principle, "field provenance principle mismatch"))
    bad_provenance = deepcopy(clean)
    bad_provenance["field_provenance"] = {}
    cases.append((bad_provenance, "field provenance must cover required fields"))
    bad_row = deepcopy(clean)
    bad_row["per_unit_rows"][0]["row_hash"] = "sha256:bad"
    cases.append((bad_row, "per_unit_rows row_hash mismatch"))
    bad_protected = deepcopy(clean)
    bad_protected["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((bad_protected, "protected files changed"))

    for payload, expected in cases:
        if expected != "required field set mismatch":
            payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
        assert expected in mod.validate_artifact(payload)

    checksum_bad = deepcopy(clean)
    checksum_bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(checksum_bad)
