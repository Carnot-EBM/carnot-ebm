"""Tests for the Exp6577 independent source-stream audit.

Spec refs: REQ-REPORT-6577, SCENARIO-REPORT-6577-MISSING,
SCENARIO-REPORT-6577-COVERAGE, SCENARIO-REPORT-6577-REPLAY,
SCENARIO-REPORT-6577-ATTACKS, SCENARIO-REPORT-6577-ATOMIC.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import zlib

import pytest

from carnot import experiment_6577_flagship_source_stream_independent_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TEST_RECEIPTS = [{"command": "focused-exp6577", "exit_code": 0}]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fixture_repo(tmp_path: Path) -> Path:
    _write(tmp_path / mod.PROTECTED_RELATIVE_PATHS[0], "roadmap\n")
    _write(tmp_path / mod.PROTECTED_RELATIVE_PATHS[1], "conductor\n")
    for relative in mod.AUDIT_TOOL_RELATIVE_PATHS:
        _write(tmp_path / relative, f"tool:{relative}\n")
    _write(
        tmp_path / mod.EXP6575_RELATIVE_PATH,
        json.dumps(
            {
                "schema": "carnot.exp6575.v1",
                "status": "complete_v571_qualification",
                "v571_flagship_evidence_ready_score": 1.0,
            }
        ),
    )
    return tmp_path


def _content_fields(stem: str, payload: bytes) -> dict[str, object]:
    return {
        f"{stem}_bytes_b64": base64.b64encode(payload).decode("ascii"),
        f"{stem}_sha256": mod.sha256_bytes(payload),
    }


def _make_upstream(repo_root: Path) -> tuple[Path, dict[str, object]]:
    models: dict[str, tuple[Path, str]] = {}
    for index, (repository, _) in enumerate(mod.MANDATED_MODELS.items()):
        model_path = repo_root / "models" / f"model-{index}.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(f"GGUF fixture {repository}".encode())
        models[repository] = (model_path, mod.sha256_file(model_path))

    work: list[tuple[str, str, bytes, dict[str, bool], int]] = []
    repositories = list(mod.MANDATED_MODELS)
    for index, repository in enumerate(repositories):
        work.append(
            (
                repository,
                "shared-source",
                f"claim response {index}".encode(),
                {},
                0,
            )
        )
    work.extend(
        [
            (repositories[0], "timeout-source", b"timeout receipt", {"timeout": True}, 0),
            (
                repositories[0],
                "malformed-source",
                b"{",
                {"malformed_output": True},
                0,
            ),
            (repositories[0], "refusal-source", b"refusal receipt", {"refusal": True}, 0),
            (repositories[0], "empty-source", b"", {}, 0),
            (
                repositories[0],
                "process-source",
                b"process failure receipt",
                {},
                9,
            ),
        ]
    )

    manifest: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for order_index, (repository, source_id, response, flags, exit_code) in enumerate(work):
        family = mod.MANDATED_MODELS[repository]
        source = b"same source bytes" if source_id == "shared-source" else source_id.encode()
        prompt = f"frozen prompt {family} {source_id}".encode()
        model_path, model_hash = models[repository]
        unit_id = f"unit-{order_index}"
        seed = 657700 + order_index
        expected = {
            "unit_id": unit_id,
            "source_id": source_id,
            "family": family,
            "model_repository": repository,
            "model_revision": "rev-pinned-20260824",
            "model_path": str(model_path),
            "gguf_sha256": model_hash,
            "source_sha256": mod.sha256_bytes(source),
            "prompt_sha256": mod.sha256_bytes(prompt),
            "seed": seed,
            "order_index": order_index,
            "attempt_index": 0,
        }
        manifest.append(expected)
        prompt_tokens = order_index + 10
        response_tokens = len(response.split())
        latency_s = round(0.2 + order_index / 100, 6)
        components = [
            {"name": "tokens", "quantity": prompt_tokens + response_tokens, "unit_cost": 0.001},
            {"name": "latency", "quantity": latency_s, "unit_cost": 0.01},
        ]
        charged_cost = round(
            sum(float(item["quantity"]) * float(item["unit_cost"]) for item in components),
            12,
        )
        row = {
            **expected,
            **_content_fields("source", source),
            **_content_fields("prompt", prompt),
            **_content_fields("raw_response", response),
            "corpus_commit": "pending",
            "process_receipt": {
                "pid": 5000 + order_index,
                "exit_code": exit_code,
                "started_monotonic_ns": 1000 + order_index * 100,
                "ended_monotonic_ns": 1050 + order_index * 100,
            },
            "stop_reason": "timeout" if flags.get("timeout") else "stop",
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens,
            "latency_s": latency_s,
            "charged_cost_components": components,
            "charged_cost": charged_cost,
            "raw_response_recorded_monotonic_ns": 1060 + order_index * 100,
            "parser_started_monotonic_ns": 1070 + order_index * 100,
            "timeout": flags.get("timeout", False),
            "malformed_output": flags.get("malformed_output", False),
            "refusal": flags.get("refusal", False),
            "claim_bearing": order_index < 3,
            "retry_count": 0,
        }
        rows.append(row)

    corpus_commit = mod.recompute_corpus_commit(manifest)
    for row in rows:
        row["corpus_commit"] = corpus_commit
    totals = mod.recompute_raw_totals(rows)
    upstream: dict[str, object] = {
        "schema": mod.EXPECTED_UPSTREAM_SCHEMA,
        "status": "complete_immutable_flagship_source_stream",
        "honest_verdict": "complete_immutable_flagship_source_stream",
        "verdict_class": None,
        "expected_source_family_units": manifest,
        "rows": rows,
        "corpus_commit": corpus_commit,
        "immutable_claim_stream_ready_score": 1.0,
        "row_count": len(rows),
        "family_coverage": sorted(mod.MANDATED_MODELS.values()),
        **totals,
    }
    upstream_path = repo_root / mod.UPSTREAM_RELATIVE_PATH
    _write(upstream_path, json.dumps(upstream))
    return upstream_path, upstream


def _build(repo_root: Path, upstream_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.build_artifact(
        repo_root=repo_root,
        result_path=repo_root / "out.json",
        upstream_path=upstream_path,
        exp6575_path=repo_root / mod.EXP6575_RELATIVE_PATH,
        write=write,
        duration_s=0.25,
        tests_run=TEST_RECEIPTS,
    )


def test_req_report_6577_spec_declares_the_full_independent_replay_contract() -> None:
    """REQ-REPORT-6577: OpenSpec owns all fields, attacks, and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6577") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-6577-MISSING",
        "SCENARIO-REPORT-6577-COVERAGE",
        "SCENARIO-REPORT-6577-REPLAY",
        "SCENARIO-REPORT-6577-ATTACKS",
        "SCENARIO-REPORT-6577-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6577_missing_input_writes_exact_block_diagnosis(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6577-MISSING: a missing Exp6576 terminates honestly."""

    repo_root = _fixture_repo(tmp_path)
    upstream_path = repo_root / mod.UPSTREAM_RELATIVE_PATH
    result_path = repo_root / "blocked.json"
    artifact = mod.build_artifact(
        repo_root=repo_root,
        result_path=result_path,
        upstream_path=upstream_path,
        exp6575_path=repo_root / mod.EXP6575_RELATIVE_PATH,
        write=True,
        duration_s=0.1,
        tests_run=TEST_RECEIPTS,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"].startswith("blocked_")
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["claim_stream_audit_ready_score"] == 0.0
    assert artifact["rows"] == []
    assert artifact["upstream_artifact_receipt"]["path"] == str(upstream_path)
    assert artifact["upstream_artifact_receipt"]["exists"] is False
    assert str(upstream_path) in artifact["gate_check_summary"]["first_failure"]["field"]
    assert artifact["gate_check_summary"]["first_failure"]["observed"] == "missing"
    assert artifact["preconditions_checked"]["raw_response_recovery"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenarios_report_6577_clean_rows_recompute_all_families_cost_and_failures(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6577-COVERAGE/REPLAY/ATOMIC: clean raw rows replay."""

    repo_root = _fixture_repo(tmp_path)
    upstream_path, upstream = _make_upstream(repo_root)
    artifact = _build(repo_root, upstream_path, write=True)
    aggregate = artifact["aggregate_row_recomputation"]

    assert artifact["claim_stream_audit_ready_score"] == 1.0
    assert artifact["verdict_class"] is None
    assert artifact["status"].startswith("complete_")
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["rows"]) == len(upstream["expected_source_family_units"])
    assert all(row["row_replay_passed"] for row in artifact["rows"])
    assert aggregate["family_coverage"] == sorted(mod.MANDATED_MODELS.values())
    assert aggregate["failure_row_count"] == 5
    assert aggregate["prompt_tokens"] == upstream["prompt_tokens"]
    assert aggregate["response_tokens"] == upstream["response_tokens"]
    assert aggregate["total_tokens"] == upstream["total_tokens"]
    assert aggregate["latency_s"] == upstream["latency_s"]
    assert aggregate["charged_cost"] == upstream["charged_cost"]
    assert {row["failure_class"] for row in artifact["failure_retention_rows"]} == set(
        mod.FAILURE_CLASSES
    )
    assert all(row["passed"] for row in artifact["failure_retention_rows"])
    assert [row["attack"] for row in artifact["duplicate_and_drift_attack_rows"]] == list(
        mod.REQUIRED_ATTACKS
    )
    assert all(row["passed"] for row in artifact["duplicate_and_drift_attack_rows"])
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6577_content_paths_and_raw_before_parser_are_verified(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6577-REPLAY/RAW-FIRST: content paths and receipt order bind bytes."""

    repo_root = _fixture_repo(tmp_path)
    upstream_path, upstream = _make_upstream(repo_root)
    row = upstream["rows"][0]
    response = base64.b64decode(row.pop("raw_response_bytes_b64"))
    content_path = repo_root / "content" / f"{row['raw_response_sha256']}.bin"
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_bytes(response)
    row["raw_response_content_path"] = str(content_path)
    _write(upstream_path, json.dumps(upstream))

    artifact = _build(repo_root, upstream_path)
    assert artifact["rows"][0]["raw_response_recovery"] == "content_path"
    assert artifact["rows"][0]["raw_before_parser"] is True
    assert artifact["claim_stream_audit_ready_score"] == 1.0


@pytest.mark.parametrize("attack", mod.REQUIRED_ATTACKS)
def test_scenario_report_6577_each_attack_mutation_fails_closed(
    tmp_path: Path,
    attack: str,
) -> None:
    """SCENARIO-REPORT-6577-ATTACKS: every named mutation blocks readiness."""

    repo_root = _fixture_repo(tmp_path)
    upstream_path, upstream = _make_upstream(repo_root)
    manifest = upstream["expected_source_family_units"]
    rows = upstream["rows"]

    if attack == "source_alias":
        manifest[1]["source_id"] = "shared-source-alias"
        rows[1]["source_id"] = "shared-source-alias"
    elif attack == "duplicate_unit":
        rows.append(deepcopy(rows[0]))
    elif attack == "copied_response_across_models":
        rows[1]["raw_response_bytes_b64"] = rows[0]["raw_response_bytes_b64"]
        rows[1]["raw_response_sha256"] = rows[0]["raw_response_sha256"]
    elif attack == "selective_retry":
        rows[0]["retry_count"] = 1
    elif attack == "hidden_row_drop":
        rows.pop()
    elif attack == "legacy_model_substitution":
        rows[0]["model_repository"] = "Qwen/Qwen3.5-0.8B-GGUF"
    elif attack == "inconsistent_family_label":
        rows[0]["family"] = mod.MANDATED_MODELS[list(mod.MANDATED_MODELS)[1]]
    elif attack == "post_outcome_prompt_change":
        changed = b"changed after outcome"
        rows[0].update(_content_fields("prompt", changed))
    elif attack == "null_only_row":
        for stem in ("source", "prompt", "raw_response"):
            rows[0].pop(f"{stem}_bytes_b64", None)
            rows[0].pop(f"{stem}_content_path", None)
    elif attack == "missing_content_path":
        rows[0].pop("raw_response_bytes_b64")
        rows[0]["raw_response_content_path"] = str(repo_root / "missing-response.bin")
    elif attack == "aggregate_row_contradiction":
        upstream["row_count"] = 999
    else:  # pragma: no cover - the parametrization is the frozen attack list.
        raise AssertionError(attack)

    _write(upstream_path, json.dumps(upstream))
    artifact = _build(repo_root, upstream_path)
    attacks = {row["attack"]: row for row in artifact["duplicate_and_drift_attack_rows"]}

    assert attacks[attack]["passed"] is False
    assert artifact["claim_stream_audit_ready_score"] == 0.0
    assert artifact["verdict_class"] == "disqualified"
    assert mod.validate_artifact(artifact) == []


def test_req_report_6577_helpers_and_validator_reject_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-6577-REDUCER/ATOMIC: hashes, recovery, and validation fail closed."""

    repo_root = _fixture_repo(tmp_path)
    upstream_path, _ = _make_upstream(repo_root)
    artifact = _build(repo_root, upstream_path)
    payload_path = repo_root / "payload.bin"
    payload_path.write_bytes(b"payload")

    assert mod.sha256_file(repo_root / "absent") == "missing"
    assert mod.sha256_file(payload_path) == mod.sha256_bytes(b"payload")
    assert mod.recover_content_bytes({"x_text": "text"}, "x", repo_root) == (b"text", "text")
    assert mod.recover_content_bytes(
        {"x_bytes_b64": base64.b64encode(b"bytes").decode()}, "x", repo_root
    ) == (b"bytes", "inline_base64")
    assert mod.recover_content_bytes({"x_content_path": str(payload_path)}, "x", repo_root) == (
        b"payload",
        "content_path",
    )
    assert mod.recover_content_bytes({"x_content_path": "missing"}, "x", repo_root) == (
        None,
        "missing_content_path",
    )
    assert mod.recover_content_bytes({}, "x", repo_root) == (None, "missing")
    assert mod.recover_content_bytes({"x_bytes_b64": "%%%"}, "x", repo_root) == (
        None,
        "invalid_base64",
    )
    payload_b64 = base64.b64encode(b"payload bytes").decode()
    compressed_b64 = base64.b64encode(zlib.compress(b"compressed bytes")).decode()
    assert mod.recover_content_bytes(
        {"x_payload": {"encoding": "base64", "bytes_b64": payload_b64}},
        "x",
        repo_root,
    ) == (b"payload bytes", "inline_payload")
    assert mod.recover_content_bytes(
        {
            "x_payload": {
                "encoding": "base64",
                "compression": "zlib",
                "bytes_b64": compressed_b64,
            }
        },
        "x",
        repo_root,
    ) == (b"compressed bytes", "inline_payload")
    assert mod.recover_content_bytes(
        {"x_payload": {"encoding": "hex", "bytes_b64": payload_b64}}, "x", repo_root
    ) == (None, "invalid_payload_encoding")
    assert mod.recover_content_bytes(
        {"x_payload": {"encoding": "base64", "compression": "zlib", "bytes_b64": "%%%"}},
        "x",
        repo_root,
    ) == (None, "invalid_payload")

    assert (
        mod.recompute_raw_totals(
            [
                {"charged_cost_components": []},
                {"charged_cost_components": ["bad"]},
                {"charged_cost_components": [{"quantity": -1, "unit_cost": 1}]},
                {"raw_response_bytes_b64": "%%%"},
            ]
        )["charged_cost"]
        == 0.0
    )

    invalid_json = repo_root / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert mod._read_json(invalid_json)[1] == "unreadable:JSONDecodeError"
    list_json = repo_root / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(list_json) == (None, "root_not_object")

    mutations: list[tuple[str, object]] = [
        ("inference_substrate", "live_llm_inference"),
        ("verifier_is_oracle", False),
        ("verdict_class", "positive"),
        ("claim_stream_audit_ready_score", 0.0),
        ("reproducibility_checksum", "sha256:bad"),
    ]
    for field, value in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert mod.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed.pop("status")
    assert mod.validate_artifact(changed) == ["required field set mismatch"]

    changed = deepcopy(artifact)
    changed["rows"][0]["row_hash"] = "sha256:bad"
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "rows row_hash mismatch" in mod.validate_artifact(changed)


def test_req_report_6577_validator_and_atomic_failure_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6577-ATOMIC: invalid structures and replace failures are explicit."""

    repo_root = _fixture_repo(tmp_path)
    upstream_path, _ = _make_upstream(repo_root)
    artifact = _build(repo_root, upstream_path)

    cases: list[tuple[dict[str, object], str]] = []
    changed = deepcopy(artifact)
    changed["status"] = "bad"
    cases.append((changed, "status lacks terminal prefix"))
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "bad"
    cases.append((changed, "honest_verdict lacks terminal prefix"))
    changed = deepcopy(artifact)
    changed["field_provenance"] = {}
    cases.append((changed, "field_provenance must cover every required field"))
    changed = deepcopy(artifact)
    changed["field_provenance"]["status"]["principle"] = "wrong"
    cases.append((changed, "field_provenance principle mismatch"))
    changed = deepcopy(artifact)
    changed["aggregate_row_recomputation"] = "bad"
    cases.append((changed, "aggregate_row_recomputation must be a mapping"))
    changed = deepcopy(artifact)
    changed["verdict_class"] = "partial"
    changed["aggregate_row_recomputation"]["verdict_class_from_rows"] = "partial"
    changed["aggregate_row_recomputation"]["row_hash"] = mod.row_hash(
        changed["aggregate_row_recomputation"]
    )
    cases.append((changed, "ready score requires clean null audit"))
    changed = deepcopy(artifact)
    changed["duplicate_and_drift_attack_rows"].pop()
    cases.append((changed, "required attack set or order mismatch"))
    changed = deepcopy(artifact)
    changed["failure_retention_rows"].pop()
    cases.append((changed, "failure retention class set or order mismatch"))
    changed = deepcopy(artifact)
    changed["rows"] = "bad"
    cases.append((changed, "rows must be a list"))
    changed = deepcopy(artifact)
    changed["gate_check_summary"] = "bad"
    cases.append((changed, "gate_check_summary row_hash mismatch"))
    changed = deepcopy(artifact)
    changed["duration_s"] = -1
    cases.append((changed, "duration_s must be finite and nonnegative"))
    changed = deepcopy(artifact)
    changed["tests_run"] = [{"command": "bad", "exit_code": -1}]
    cases.append((changed, "tests_run must name commands and nonnegative exits"))

    for changed, expected in cases:
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)

    target = repo_root / "atomic.json"

    def fail_replace(_source: str, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        mod._atomic_write_json(target, {"ok": True})
    assert not list(repo_root.glob(".atomic.json.*"))

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced invalid"])
    with pytest.raises(ValueError, match="forced invalid"):
        _build(repo_root, upstream_path)


def test_scenario_report_6577_cli_writes_and_validates_only_requested_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-6577-ATOMIC: CLI run and validation use the requested file."""

    repo_root = _fixture_repo(tmp_path)
    upstream_path, _ = _make_upstream(repo_root)
    result_path = repo_root / "cli.json"

    assert (
        mod.main(
            [
                "--date",
                mod.RUN_DATE,
                "--repo-root",
                str(repo_root),
                "--upstream-path",
                str(upstream_path),
                "--exp6575-path",
                str(repo_root / mod.EXP6575_RELATIVE_PATH),
                "--result-path",
                str(result_path),
            ]
        )
        == 0
    )
    assert result_path.is_file()
    assert "claim_stream_audit_ready_score=1.0" in capsys.readouterr().out
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    assert "validated" in capsys.readouterr().out

    damaged = json.loads(result_path.read_text(encoding="utf-8"))
    damaged["reproducibility_checksum"] = "bad"
    _write(result_path, json.dumps(damaged))
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 1
    assert "reproducibility_checksum mismatch" in capsys.readouterr().out

    assert mod.main(["--date", "wrong", "--result-path", str(result_path)]) == 2
    assert "planning date must be" in capsys.readouterr().out
