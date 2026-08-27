"""Tests for the installed Torx CPU typed-factor parity contract.

Spec: REQ-SAMPLER-6684, REQ-REPORT-6684,
SCENARIO-SAMPLER-6684-EXACT-PARITY,
SCENARIO-SAMPLER-6684-FAIL-CLOSED,
SCENARIO-SAMPLER-6684-ADVERSARIAL-MAPPING,
SCENARIO-REPORT-6684-READY,
SCENARIO-REPORT-6684-BLOCKED,
SCENARIO-REPORT-6684-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_6684_torx_typed_factor_parity as mod


def _passing_test_rows() -> list[dict]:
    rows = []
    for definition in mod.VERIFICATION_DEFINITIONS:
        coverage = 100.0 if definition["check_id"] == "scoped_coverage" else None
        rows.append(
            mod.make_test_receipt(
                definition,
                exit_code=0,
                duration_s=0.01,
                summary="passed",
                output_sha256=mod.sha256_bytes(b"passed"),
                coverage_percent=coverage,
            )
        )
    return rows


@pytest.fixture(scope="module")
def upstream() -> dict:
    return mod.load_json(mod.REPO_ROOT / mod.UPSTREAM_PATH)


@pytest.fixture(scope="module")
def runtime() -> mod.TorxRuntime:
    return mod.load_torx_runtime()


@pytest.fixture(scope="module")
def parity(upstream: dict, runtime: mod.TorxRuntime) -> dict:
    return mod.replay_parity(upstream, runtime)


@pytest.fixture(scope="module")
def ready_artifact(upstream: dict, runtime: mod.TorxRuntime, parity: dict) -> dict:
    rejections = mod.build_rejection_rows(runtime)
    attacks = mod.build_attack_rows(upstream, runtime, parity, rejections)
    before = mod.protected_hashes()
    preconditions = mod.collect_preconditions(
        mod.REPO_ROOT,
        upstream=upstream,
        runtime=runtime,
    )
    return mod.build_artifact(
        date=mod.RUN_DATE,
        duration_s=0.5,
        upstream=upstream,
        runtime=runtime,
        parity=parity,
        rejection_rows=rejections,
        attack_rows=attacks,
        tests_run=_passing_test_rows(),
        preconditions=preconditions,
        protected=mod.protected_files_receipt(mod.REPO_ROOT, before),
    )


def test_req_sampler_6684_specs_precede_implementation() -> None:
    sampler = (mod.REPO_ROOT / mod.SAMPLER_SPEC_PATH).read_text(encoding="utf-8")
    report = (mod.REPO_ROOT / mod.REPORT_SPEC_PATH).read_text(encoding="utf-8")

    assert {
        "REQ-SAMPLER-6684-WIRES",
        "REQ-SAMPLER-6684-FACTORS",
        "REQ-SAMPLER-6684-TEMPERATURE",
        "REQ-SAMPLER-6684-PRECISION",
        "REQ-SAMPLER-6684-COMPOSITION",
        "REQ-SAMPLER-6684-PARITY",
        "REQ-SAMPLER-6684-REJECTION",
        "REQ-SAMPLER-6684-API",
        "REQ-SAMPLER-6684-BOUNDARY",
        "SCENARIO-SAMPLER-6684-EXACT-PARITY",
        "SCENARIO-SAMPLER-6684-FAIL-CLOSED",
        "SCENARIO-SAMPLER-6684-ADVERSARIAL-MAPPING",
    } <= set(mod.spec_anchors(sampler))
    assert {
        "REQ-REPORT-6684",
        "SCENARIO-REPORT-6684-READY",
        "SCENARIO-REPORT-6684-BLOCKED",
        "SCENARIO-REPORT-6684-ATOMIC-PROVENANCE",
    } <= set(mod.spec_anchors(report))


def test_req_sampler_6684_runtime_and_mapping_are_typed(
    upstream: dict, runtime: mod.TorxRuntime
) -> None:
    assert runtime.version == "0.0.1"
    assert runtime.distribution_name == "extro-torx"
    assert runtime.backend == "cpu"
    assert runtime.x64_enabled is True
    assert runtime.package_sha256.startswith("sha256:")
    assert set(mod.REQUIRED_API_SYMBOLS) <= set(runtime.symbols)

    fixture = mod.supported_instances(upstream)[0]
    factors = mod.map_fixture(fixture, runtime)
    assert len(factors) == fixture.n_spins + len(fixture.edges)
    assert factors[0].factor_type == "bias"
    assert factors[0].wire_dims == (2, 2)
    assert factors[0].pinned_auxiliary_wire == 0
    assert mod.spin_to_bit(-1) == 0
    assert mod.spin_to_bit(1) == 1
    with pytest.raises(mod.UnsupportedTorxInput, match="binary spin"):
        mod.spin_to_bit(0)


def test_scenario_sampler_6684_every_factor_and_state_matches_exact_reference(
    upstream: dict, parity: dict
) -> None:
    supported = [row for row in upstream["frozen_fixture_manifest"] if row["expected_supported"]]
    expected_factor_count = sum(
        row["graph"]["n_spins"] + len(row["graph"]["edges"]) for row in supported
    )
    expected_state_count = sum(2 ** row["graph"]["n_spins"] for row in supported)

    assert len(parity["factor_rows"]) == expected_factor_count == 97
    assert len(parity["state_parity_rows"]) == expected_state_count == 294
    assert all(row["valid"] for row in parity["factor_rows"])
    assert all(row["valid"] for row in parity["state_parity_rows"])
    assert all(
        len(row["state_energy_rows"])
        == 2
        ** next(
            item["graph"]["n_spins"]
            for item in supported
            if item["fixture_id"] == row["fixture_id"]
        )
        for row in parity["factor_rows"]
    )
    assert (
        parity["maximum_errors"]["factor_energy"]["absolute"]
        <= mod.TOLERANCES["factor_energy"]["absolute"]
    )
    assert (
        parity["maximum_errors"]["probability"]["relative"]
        <= mod.TOLERANCES["probability"]["relative"]
    )

    singleton = next(
        row
        for row in parity["state_parity_rows"]
        if row["fixture_id"] == "singleton_field" and row["state"] == [-1]
    )
    assert singleton["binary_state"] == [0]
    assert singleton["exact_total_energy"] == pytest.approx(0.31)
    assert singleton["torx_total_energy"] == pytest.approx(0.31)
    assert singleton["torx_node_marginals_plus"]["0"] == pytest.approx(
        singleton["exact_node_marginals_plus"]["0"]
    )


def test_scenario_sampler_6684_unsupported_inputs_and_attacks_fail_closed(
    upstream: dict, runtime: mod.TorxRuntime, parity: dict
) -> None:
    rejections = mod.build_rejection_rows(runtime)
    attacks = mod.build_attack_rows(upstream, runtime, parity, rejections)

    assert set(mod.REQUIRED_REJECTIONS) == {row["case_id"] for row in rejections}
    assert all(row["passed"] for row in rejections)
    assert set(mod.REQUIRED_ATTACKS) == {row["attack_id"] for row in attacks}
    assert all(row["passed"] for row in attacks)
    assert next(row for row in attacks if row["attack_id"] == "topology")["observed"][
        "disconnected_supported"
    ]
    assert (
        next(row for row in attacks if row["attack_id"] == "fallback")["observed"]["fallback_used"]
        is False
    )

    factor = mod.map_fixture(mod.supported_instances(upstream)[1], runtime)[0]
    with pytest.raises(mod.UnsupportedTorxInput, match="state width"):
        mod.factor_energy(factor, (-1,), runtime)


def test_req_report_6684_ready_artifact_recomputes_and_preserves_scope(
    ready_artifact: dict,
) -> None:
    assert mod.validate_artifact(ready_artifact) == []
    assert ready_artifact["status"] == "complete_ready"
    assert ready_artifact["honest_verdict"].startswith("complete:")
    assert ready_artifact["verdict_class"] == "circular_positive"
    assert ready_artifact["torx_factor_parity_ready"] is True
    assert ready_artifact["claim_scope"] == "installed_torx_cpu_software_only"
    assert ready_artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert ready_artifact["verifier_is_oracle"] is True
    assert ready_artifact["gate_check_summary"] == []
    assert ready_artifact["aggregate_row_recomputation"]["ready"] is True
    assert len(ready_artifact["per_unit_rows"]) == (
        len(ready_artifact["factor_rows"])
        + len(ready_artifact["state_parity_rows"])
        + len(ready_artifact["rejection_rows"])
        + len(ready_artifact["attack_rows"])
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(ready_artifact["field_provenance"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda value: value.update(claim_scope="hardware"), "claim_scope_mismatch"),
        (
            lambda value: value["factor_rows"][0].update(valid=False),
            "factor_row_hash_mismatch",
        ),
        (
            lambda value: value["state_parity_rows"].pop(),
            "state_row_count_mismatch",
        ),
        (
            lambda value: value["rejection_rows"][0].update(passed=False),
            "rejection_row_hash_mismatch",
        ),
        (
            lambda value: value["attack_rows"][0].update(passed=False),
            "attack_row_hash_mismatch",
        ),
        (
            lambda value: value.update(per_unit_rows=[]),
            "per_unit_rows_mismatch",
        ),
        (
            lambda value: value.update(field_provenance={}),
            "field_provenance_invalid",
        ),
        (
            lambda value: value["protected_files_unchanged"].update(unchanged=False),
            "protected_files_changed",
        ),
        (
            lambda value: value.update(torx_factor_parity_ready=False),
            "readiness_mismatch",
        ),
        (
            lambda value: value.update(reproducibility_checksum="sha256:bad"),
            "reproducibility_checksum_mismatch",
        ),
        (
            lambda value: value.update(inference_substrate="changed"),
            "inference_substrate_mismatch",
        ),
        (
            lambda value: value.update(verifier_is_oracle=False),
            "verifier_is_oracle_mismatch",
        ),
        (
            lambda value: value["state_parity_rows"][0].update(valid=False),
            "state_row_hash_mismatch",
        ),
        (
            lambda value: value["frozen_mapping_contract"].update(contract_sha256="sha256:changed"),
            "mapping_contract_hash_mismatch",
        ),
        (
            lambda value: value["frozen_mapping_contract"].update(expected_factor_count=0),
            "factor_row_count_mismatch",
        ),
        (
            lambda value: value.update(status="complete_but_wrong"),
            "ready_terminal_state_mismatch",
        ),
        (
            lambda value: value["torx_runtime_receipt"].update(backend="gpu"),
            "runtime_receipt_mismatch",
        ),
        (
            lambda value: value.update(duration_s=None),
            "duration_invalid",
        ),
    ],
)
def test_scenario_report_6684_mutations_fail_validation(
    ready_artifact: dict, mutation: object, expected_error: str
) -> None:
    changed = deepcopy(ready_artifact)
    mutation(changed)  # type: ignore[operator]
    errors = mod.validate_artifact(changed)
    assert expected_error in errors


def test_scenario_report_6684_blocked_gate_and_missing_api_are_terminal(
    tmp_path: Path, upstream: dict
) -> None:
    blocked_upstream = deepcopy(upstream)
    blocked_upstream["ising_reference_ready"] = False
    blocked_upstream["gate_check_summary"] = [{"reason": "owned test failed"}]
    output = tmp_path / "gate.json"
    result = mod.run(
        date=mod.RUN_DATE,
        output_path=output,
        upstream_payload=blocked_upstream,
        tests_run=_passing_test_rows(),
    )
    assert result["status"] == "blocked_upstream_gate"
    assert result["torx_factor_parity_ready"] is False
    assert result["gate_check_summary"][0]["check"] == "upstream"
    assert mod.validate_artifact(result) == []

    api_output = tmp_path / "api.json"

    def missing_runtime() -> mod.TorxRuntime:
        raise mod.TorxApiError("missing declared Torx API symbol: torx.psc.PISING")

    result = mod.run(
        date=mod.RUN_DATE,
        output_path=api_output,
        upstream_payload=upstream,
        runtime_loader=missing_runtime,
        tests_run=_passing_test_rows(),
    )
    assert result["status"] == "blocked_api"
    assert result["gate_check_summary"][0]["check"] == "api"
    assert mod.validate_artifact(result) == []

    result["status"] = "complete_wrong"
    assert "blocked_terminal_state_mismatch" in mod.validate_artifact(result)


def test_req_sampler_6684_missing_declared_symbol_has_no_fallback() -> None:
    fake = SimpleNamespace(__version__="0.0.1", psc=SimpleNamespace())
    with pytest.raises(mod.TorxApiError, match="PISING"):
        mod.load_torx_runtime(importer=lambda _name: fake)


def test_req_sampler_6684_runtime_gate_covers_every_declared_precondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingEnergy:
        get_generator = staticmethod(lambda: None)

    class MissingGenerator:
        _energies = staticmethod(lambda _theta: None)

    with pytest.raises(mod.TorxApiError, match="_energies"):
        mod.require_torx_api(SimpleNamespace(psc=SimpleNamespace(PISING=MissingEnergy)))
    with pytest.raises(mod.TorxApiError, match="get_generator"):
        mod.require_torx_api(SimpleNamespace(psc=SimpleNamespace(PISING=MissingGenerator)))
    with pytest.raises(mod.TorxApiError, match="no hashable"):
        mod._package_hash(SimpleNamespace(files=[]))

    class GoodPising:
        _energies = staticmethod(lambda _theta: [0.0, 0.0, 0.0, 0.0])
        get_generator = staticmethod(lambda: None)

    def torx_module(version: str) -> SimpleNamespace:
        return SimpleNamespace(
            __version__=version,
            psc=SimpleNamespace(PISING=GoodPising),
            __file__=__file__,
        )

    with pytest.raises(mod.TorxApiError, match="expected Torx"):
        mod.load_torx_runtime(importer=lambda _name: torx_module("9.9.9"))

    class Config:
        x64_enabled = True

        @staticmethod
        def update(_name: str, _value: bool) -> None:
            return None

    def importer_with_jax(backend: str, x64_enabled: bool):
        config = Config()
        config.x64_enabled = x64_enabled
        modules = {
            "torx": torx_module(mod.TORX_VERSION),
            "jax": SimpleNamespace(config=config, default_backend=lambda: backend),
            "jax.numpy": SimpleNamespace(),
        }
        return lambda name: modules[name]

    with pytest.raises(mod.TorxApiError, match="backend cpu"):
        mod.load_torx_runtime(importer=importer_with_jax("gpu", True))
    with pytest.raises(mod.TorxApiError, match="binary64"):
        mod.load_torx_runtime(importer=importer_with_jax("cpu", False))

    monkeypatch.setattr(
        mod,
        "Path",
        lambda _value: SimpleNamespace(is_file=lambda: False),
    )
    monkeypatch.setattr(mod.platform, "processor", lambda: "")
    monkeypatch.setattr(mod.platform, "machine", lambda: "fallback-cpu")
    assert mod._cpu_name() == "fallback-cpu"


def test_req_sampler_6684_gate_and_torx_table_drift_are_rejected(
    upstream: dict,
    runtime: mod.TorxRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_gate = deepcopy(upstream)
    failed_gate["ising_reference_ready"] = False
    with pytest.raises(mod.UpstreamGateError, match="gate failed"):
        mod.supported_instances(failed_gate)

    with monkeypatch.context() as context:
        context.setattr(
            mod,
            "_upstream_gate_receipt",
            lambda _payload: {"passed": True},
        )
        changed_set = deepcopy(upstream)
        removed_id = next(
            row["fixture_id"]
            for row in changed_set["frozen_fixture_manifest"]
            if row["expected_supported"]
        )
        changed_set["frozen_fixture_manifest"] = [
            row for row in changed_set["frozen_fixture_manifest"] if row["fixture_id"] != removed_id
        ]
        with pytest.raises(mod.UpstreamGateError, match="fixture set changed"):
            mod.supported_instances(changed_set)

        changed_hash = deepcopy(upstream)
        supported = next(
            row for row in changed_hash["frozen_fixture_manifest"] if row["expected_supported"]
        )
        supported["source_fixture_sha256"] = "sha256:changed"
        with pytest.raises(mod.UpstreamGateError, match="fixture hash changed"):
            mod.supported_instances(changed_hash)

    fixture = mod.supported_instances(upstream)[0]

    class BadGate:
        def __init__(self, *, sites: list[int]) -> None:
            assert sites == [0, 1]

        @staticmethod
        def _energies(_theta: object) -> list[float]:
            return [0.0]

    bad_runtime = replace(runtime, pising_class=BadGate)
    with pytest.raises(mod.UnsupportedTorxInput, match="invalid energy table"):
        mod.map_fixture(fixture, bad_runtime)

    with monkeypatch.context() as context:
        context.setattr(mod, "supported_instances", lambda _payload: (fixture,))
        with pytest.raises(mod.UpstreamGateError, match="state count changed"):
            mod.replay_parity({"exact_probability_rows": []}, runtime)

    assert mod._observe_failure(lambda: None) == "unexpectedly accepted"


def test_req_report_6684_verification_receipts_keep_failures_visible(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(command: list[str], cwd: Path) -> dict:
        assert cwd == tmp_path
        rendered = " ".join(command)
        calls.append(rendered)
        if "coverage report" in rendered:
            output = "Name Stmts Miss Cover\nTOTAL 400 0 100%\n"
        else:
            output = "passed\n"
        exit_code = 1 if "ruff check" in rendered else 0
        return {
            "exit_code": exit_code,
            "duration_s": 0.1,
            "summary": output.strip().splitlines()[-1],
            "output": output,
            "output_sha256": mod.sha256_bytes(output.encode()),
        }

    rows = mod.run_verification(tmp_path, command_runner=runner)
    assert len(rows) == len(mod.VERIFICATION_DEFINITIONS)
    assert len(calls) == len(rows)
    assert (
        next(row for row in rows if row["check_id"] == "scoped_coverage")["coverage_percent"]
        == 100.0
    )
    assert next(row for row in rows if row["check_id"] == "ruff_check")["passed"] is False
    failures, summary = mod.reduce_test_rows(rows)
    assert failures[0]["check"] == "test"
    assert summary["ready"] is False

    empty_failures, empty_summary = mod.reduce_test_rows([])
    assert empty_failures[0]["check"] == "test"
    assert empty_summary["command_count"] == 0

    receipt = mod.default_command_runner(
        [sys.executable, "-c", "print('subprocess receipt')"], tmp_path
    )
    assert receipt["exit_code"] == 0
    assert receipt["summary"] == "subprocess receipt"

    duplicate = empty_failures[0]
    aggregate = mod.recompute_aggregate(
        mapping_contract={},
        factor_rows=(),
        state_rows=(),
        rejection_rows=(),
        attack_rows=(),
        tests_run=(),
        protected_unchanged=True,
        gate_failures=(duplicate,),
    )
    assert aggregate["failures"].count(duplicate) == 1

    aggregate_without_duplicate = mod.recompute_aggregate(
        mapping_contract={},
        factor_rows=(),
        state_rows=(),
        rejection_rows=(),
        attack_rows=(),
        tests_run=(),
        protected_unchanged=True,
        gate_failures=(),
    )
    assert duplicate in aggregate_without_duplicate["failures"]


def test_scenario_report_6684_atomic_write_cli_and_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ready_artifact: dict,
    upstream: dict,
) -> None:
    output = tmp_path / "ready.json"
    receipt = mod.write_json_atomic(output, ready_artifact)
    assert receipt["atomic_replace"] is True
    assert mod.main(["--validate", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert mod.main(["--validate", "--output", str(malformed)]) == 1
    assert mod.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1
    capsys.readouterr()

    original_replace = mod.os.replace

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        mod.write_json_atomic(tmp_path / "failed.json", ready_artifact)
    assert not (tmp_path / "failed.json.tmp").exists()
    monkeypatch.setattr(mod.os, "replace", original_replace)

    run_output = tmp_path / "run.json"
    result = mod.run(
        date=mod.RUN_DATE,
        output_path=run_output,
        upstream_payload=upstream,
        tests_run=_passing_test_rows(),
    )
    assert result["torx_factor_parity_ready"] is True
    assert mod.load_json(run_output) == result

    monkeypatch.setattr(mod, "run", lambda **_kwargs: ready_artifact)
    assert mod.main(["--date", mod.RUN_DATE, "--output", str(run_output)]) == 0
    blocked = deepcopy(ready_artifact)
    blocked["torx_factor_parity_ready"] = False
    monkeypatch.setattr(mod, "run", lambda **_kwargs: blocked)
    assert mod.main(["--date", mod.RUN_DATE, "--output", str(run_output)]) == 2


def test_req_report_6684_helpers_reject_nonfinite_and_nonobjects(tmp_path: Path) -> None:
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod.relative_error(0.0, 0.0) == 0.0
    assert mod.relative_error(2.0, 1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="nonfinite"):
        mod.canonical_json({"bad": float("nan")})
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        mod.load_json(path)


def test_req_report_6684_validator_covers_schema_and_nonfinite_receipts(
    ready_artifact: dict,
) -> None:
    assert mod.validate_artifact({}) == ["missing_required_fields"]
    changed = deepcopy(ready_artifact)
    changed["duration_s"] = float("nan")
    assert mod.validate_artifact(changed) == ["nonfinite_artifact"]


def test_scenario_report_6684_run_failures_are_terminal_and_validated(
    tmp_path: Path,
    upstream: dict,
    runtime: mod.TorxRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_root = tmp_path / "missing-root"
    missing_root.mkdir()
    missing_result = mod.run(
        date=mod.RUN_DATE,
        root=missing_root,
        output_path=tmp_path / "missing-upstream.json",
        tests_run=_passing_test_rows(),
    )
    assert missing_result["status"] == "blocked_upstream_gate"

    with monkeypatch.context() as context:
        context.setattr(mod, "validate_artifact", lambda _payload: ["forced"])
        with pytest.raises(ValueError, match="invalid blocked upstream"):
            mod.run(
                date=mod.RUN_DATE,
                root=missing_root,
                output_path=tmp_path / "invalid-missing.json",
                tests_run=_passing_test_rows(),
            )

    blocked_upstream = deepcopy(upstream)
    blocked_upstream["ising_reference_ready"] = False
    with monkeypatch.context() as context:
        context.setattr(mod, "validate_artifact", lambda _payload: ["forced"])
        with pytest.raises(ValueError, match="invalid blocked upstream"):
            mod.run(
                date=mod.RUN_DATE,
                output_path=tmp_path / "invalid-gate.json",
                upstream_payload=blocked_upstream,
                tests_run=_passing_test_rows(),
            )

    with monkeypatch.context() as context:
        context.setattr(mod, "validate_artifact", lambda _payload: ["forced"])
        with pytest.raises(ValueError, match="invalid blocked API"):
            mod.run(
                date=mod.RUN_DATE,
                output_path=tmp_path / "invalid-api.json",
                upstream_payload=upstream,
                runtime_loader=lambda: (_ for _ in ()).throw(mod.TorxApiError("missing")),
                tests_run=_passing_test_rows(),
            )

    with monkeypatch.context() as context:
        context.setattr(mod, "replay_parity", lambda *_args: {})
        context.setattr(mod, "build_rejection_rows", lambda *_args: [])
        context.setattr(mod, "build_attack_rows", lambda *_args: [])
        context.setattr(mod, "collect_preconditions", lambda *_args, **_kwargs: {})
        context.setattr(mod, "build_artifact", lambda **_kwargs: {})
        context.setattr(mod, "validate_artifact", lambda _payload: ["forced"])
        with pytest.raises(ValueError, match="invalid Exp6684"):
            mod.run(
                date=mod.RUN_DATE,
                output_path=tmp_path / "invalid-ready.json",
                upstream_payload=upstream,
                runtime_loader=lambda: runtime,
                tests_run=_passing_test_rows(),
            )
