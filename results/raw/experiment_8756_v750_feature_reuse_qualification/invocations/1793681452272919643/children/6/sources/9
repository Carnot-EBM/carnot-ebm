"""REQ-VERIFY-8743: request ownership protects exact scoring and evidence."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from carnot.inference import resident_batch as b
from carnot.inference import resident_energy as n
from carnot.inference import resident_protocol as p


@pytest.fixture
def head():
    return dict(
        parameters=[0.01] * 109,
        decay_scale=1.0,
        temperature=1.0,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
    )


@pytest.fixture
def row():
    return dict(
        unit_id="item",
        q=0.4,
        public=dict(
            family_id="fixture",
            source_bytes=b"There are 12 apples.".hex(),
            answer_bytes=b"There are 12 apples.".hex(),
        ),
    )


def scientific(response):
    """Real clocks change between calls; only declared decision bytes are compared."""
    return json.dumps(
        {k: v for k, v in response.items() if k != "spans"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@pytest.mark.parametrize("size", [1, 4, 16])
def test_exact_outputs_and_counts(head, row, size, monkeypatch):
    """SCENARIO-REPORT-8743-INSTALLED: same-arm bytes retain their original meaning."""
    calls = []
    extract = n.extract
    monkeypatch.setattr(n, "extract", lambda r: (calls.append(r), extract(r))[1])
    rows = [dict(row, unit_id=str(i)) for i in range(size)]
    frame = dict(p.envelope("one", "hash", {}), operation="batch", rows=rows)
    baseline = p.decide(frame, dict(head=head, max_batch_size=16), "hash", None, set())
    assert baseline["status"] == "ok" and len(calls) == 2 * size
    calls.clear()
    reused = p.decide(
        frame, dict(head=head, max_batch_size=16, reuse_request_features=True), "hash", None, set()
    )
    assert reused["status"] == "ok" and len(calls) == size
    assert scientific(baseline) == scientific(reused)


def test_immutable_ownership(head, row):
    """SCENARIO-VERIFY-8743-OWNERSHIP: no stale or foreign vector can reach arithmetic."""
    from carnot.inference.resident_features_8743 import prepare

    owned = prepare(row)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        owned.values = (9.0,) * 8
    assert isinstance(owned.values, tuple)
    with pytest.raises(TypeError):
        owned.values[0] = 3.0
    assert owned.values_for(row) is owned.values
    with pytest.raises(ValueError, match="feature_ownership"):
        owned.values_for(deepcopy(row))
    for field, value in [("q", 0.5), ("unit_id", "changed"), ("public", {})]:
        modified = deepcopy(row)
        prepared = prepare(modified)
        modified[field] = value
        with pytest.raises(ValueError, match="feature_ownership"):
            n.batch(head, [modified], prepared=(prepared,))
    altered = replace(owned, values=(9.0,) * 8)
    with pytest.raises(ValueError, match="feature_digest"):
        altered.values_for(row)
    with pytest.raises(ValueError, match="feature_count"):
        n.batch(head, [row], prepared=())


@pytest.mark.parametrize(
    "case",
    [
        "ids",
        "mixed",
        "public",
        "q_nan",
        "q_inf",
        "too_many",
        "empty",
        "fields",
        "config",
        "deadline",
        "operation",
    ],
)
def test_fresh_identical_errors(head, row, case):
    """REQ-VERIFY-8743: a failed request cannot carry an earlier success payload."""
    rows = [row]
    frame = dict(p.envelope("bad", "hash", {}), operation="batch", rows=rows)
    if case == "ids":
        frame["rows"] = [row, row]
    elif case == "mixed":
        frame["rows"] = [row, dict(row, unit_id="bad", q=-1)]
    elif case == "public":
        frame["rows"] = [row, dict(row, unit_id="bad", public={})]
    elif case == "q_nan":
        frame["rows"] = [dict(row, q=math.nan)]
    elif case == "q_inf":
        frame["rows"] = [dict(row, q=math.inf)]
    elif case == "too_many":
        frame["rows"] = [row] * 17
    elif case == "empty":
        frame["rows"] = []
    elif case == "fields":
        frame["rows"] = [dict(row, extra=1)]
    elif case == "config":
        frame["config_hash"] = "changed"
    elif case == "deadline":
        frame["deadline_ns"] = 1
    else:
        frame["operation"] = "unknown"
    responses = []
    for reuse in [False, True]:
        config = dict(head=head, max_batch_size=16, reuse_request_features=reuse)
        seen = set()
        good = dict(p.envelope("good", "hash", {}), operation="batch", rows=[row])
        assert p.decide(good, config, "hash", None, seen)["status"] == "ok"
        response = p.decide(frame, config, "hash", None, seen)
        assert response["status"] == "error"
        assert not {"response", "responses", "feature_hashes"} & response.keys()
        responses.append(response)
    assert responses[0] == responses[1]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_features(head, row, monkeypatch, value):
    """REQ-VERIFY-8743: both providers reject nonfinite prepared numerical operands."""
    extract = n.extract

    def changed(public):
        feature = extract(public)
        feature["values"][0] = value
        feature["feature_hash"] = n.canonical_hash(
            {k: v for k, v in feature.items() if k != "feature_hash"}
        )
        return feature

    monkeypatch.setattr(n, "extract", changed)
    for reuse in [False, True]:
        frame = dict(p.envelope("bad", "hash", {}), operation="batch", rows=[row])
        response = p.decide(
            frame,
            dict(head=head, max_batch_size=16, reuse_request_features=reuse),
            "hash",
            None,
            set(),
        )
        assert response["error"] == "request_features"
        assert not {"response", "responses", "feature_hashes"} & response.keys()


def test_abstention_preserved(head, row):
    """REQ-VERIFY-8743: complete extraction retains abstention instead of truncation."""
    row["public"]["source_bytes"] = b" ".hex()
    errors = []
    for reuse in [False, True]:
        with pytest.raises(ValueError) as error:
            b.evaluate(head, [row], None, 16, reuse_features=reuse)
        errors.append(str(error.value))
    assert errors == ["request_features"] * 2


def test_authenticated_inputs_and_roster(tmp_path, monkeypatch):
    """REQ-REPORT-8743: immutable original producers authorize all twelve cells."""
    from carnot.reporting import single_extraction_8743 as a

    inputs = a.inputs(a.ROOT)
    assert inputs["passed"], inputs["gate_check_summary"]
    assert len(inputs["requests"]) == 64
    assert inputs["duplicate_fraction"]["clock_domain"] == "child_serve"
    assert inputs["duplicate_fraction"]["estimate"] == 0.45628422029939814
    assert len(a.roster()) == 12
    assert not a.inputs(tmp_path)["passed"]
    monkeypatch.setattr(a, "PINS", [(a.PINS[0][0], "sha256:changed", a.PINS[0][2])])
    assert not a.inputs(a.ROOT)["passed"]


def test_terminal_schema_and_failed_gates(tmp_path):
    """REQ-REPORT-8743: readiness never survives missing parity or owned validation."""
    from carnot.reporting import single_extraction_8743 as a

    raw = tmp_path / "raw"
    raw.mkdir()
    a.atomic_json(raw / "work.json", {})
    measured = dict(
        rows=[dict(r, status="completed", passed=True) for r in a.roster()],
        passed=True,
        feature_extraction_count_rows=[],
        wire_parity_rows=[],
        invalid_input_rows=[],
        ownership_mutation_rows=[],
        original_import_contract_rows=[],
    )
    work = dict(
        measurement=measured,
        validation=dict(
            checks=[dict(passed=True)], receipts=[], coverage=dict(complete=True), children=[]
        ),
        inputs=dict(gate_check_summary=[], references=[]),
        sources=[],
        raw=str(raw),
        invocation_id="test",
        preconditions={},
        owned_error=None,
        duration_s=1.0,
        started_at_utc="now",
        ended_at_utc="now",
        phase_spans=[],
    )
    result = a.build(work, tmp_path / (a.NAME + ".json"))
    assert result["feature_reuse_ready_score"] == 1
    assert result["MODEL_SPECS"] == [] and result["independent_count"] == 0
    work["validation"]["coverage"]["complete"] = False
    assert a.build(work, tmp_path / "output")["verdict_class"] == "disqualified"
    work["validation"]["coverage"]["complete"] = True
    work["inputs"]["gate_check_summary"] = [dict(field="external")]
    assert a.build(work, tmp_path / "output")["verdict_class"] == "blocked"
    work["inputs"]["gate_check_summary"] = []
    work["measurement"]["passed"] = False
    assert a.build(work, tmp_path / "output")["feature_reuse_ready_score"] == 0


def test_cli_paths_and_replay(tmp_path):
    """SCENARIO-VERIFY-8743-CUSTODY: private publication retains true terminal bytes."""
    from carnot.reporting import single_extraction_8743 as a
    from carnot.reporting import single_extraction_runner_8743 as r
    from carnot.reporting.v709_execution import child
    import sys
    from pathlib import Path

    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--private-output", str(a.ROOT / "unsafe.json")]) == 2
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    output = tmp_path / (a.NAME + ".json")
    receipt = child(
        "blocked",
        [sys.executable, "-u", a.CLI, "--root", str(tmp_path), "--private-output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 12
    authority = value["primitive_reference"]
    assert r.replay(output, authority) and not r.replay(output, None)
    assert (
        r.main(
            ["--cold-replay", str(output), "--authority", authority["path"], authority["sha256"]]
        )
        == 0
    )
    value["completed_count"] += 1
    a.atomic_json(output, value)
    assert not r.replay(output, authority)


def test_installed_probe_and_failed_extraction(tmp_path, monkeypatch, row):
    """REQ-VERIFY-8743: instrumentation counts successful extraction without changing errors."""
    from types import SimpleNamespace
    from carnot.inference import resident_reuse_probe_8743 as probe

    monkeypatch.setattr(probe, "sys", SimpleNamespace(path=["unchanged"], modules={}))

    def serve(*args):
        n.extract(row["public"])
        with pytest.raises(ValueError):
            n.extract({})
        return 0

    monkeypatch.setattr(probe.wire, "serve", serve)
    output = tmp_path / "counts.json"
    assert probe.main(["config", "digest", "python", "--counts-output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert len(value["extractions"]) == 2
    assert sum(r["successful"] for r in value["extractions"]) == 1
    assert not value["forbidden_imports"]


def test_changed_configuration(head, row, tmp_path, monkeypatch):
    """REQ-VERIFY-8743: head, config and extension identities fail before stale use."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(head=head, max_batch_size=16)))
    digest = p.sha256_file(path)
    with pytest.raises(ValueError, match="immutable_config"):
        p.configuration(path, "changed", "python")
    config = json.loads(path.read_text())
    config["head"]["temperature"] = 2.0
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="immutable_config"):
        p.configuration(path, digest, "python")
    extension = tmp_path / "extension.so"
    extension.write_bytes(b"invalid extension")
    config["extension"] = dict(path=str(extension), sha256="changed")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="extension_hash"):
        p.configuration(path, p.sha256_file(path), "native")
    digest = p.sha256_file(path)

    def read(*args):
        path.write_text("changed after load")
        return b"{}"

    monkeypatch.setattr(p, "read_frame", read)
    responses = []
    monkeypatch.setattr(p, "write_frame", lambda fd, value, *args: responses.append(value))
    assert p.serve(path, digest, "python") == 1
    assert len(responses) == 1 and responses[0]["status"] == "ready"


@pytest.fixture(scope="module")
def installed_study(tmp_path_factory):
    """SCENARIO-REPORT-8743-INSTALLED: a real wheel and compiled extension own the evidence."""
    from carnot.reporting import single_extraction_8743 as a
    from carnot.reporting import single_extraction_measure_8743 as m

    root = tmp_path_factory.mktemp("single-extraction")
    scratch = root / "scratch"
    scratch.mkdir()
    inputs = a.inputs(a.ROOT)
    result = m.measure(root / "raw", scratch, inputs)
    assert result["passed"], result
    return root, inputs, result


def test_actual_installed_panel(installed_study):
    """REQ-REPORT-8743: B1/4/16 same-arm bytes and extraction denominators are real."""
    from carnot.reporting import single_extraction_measure_8743 as m

    root, inputs, result = installed_study
    assert len(result["rows"]) == 12
    assert len(result["wire_parity_rows"]) == 384
    assert all(
        r["successful_extractions"] == (128 if r["mode"] == "baseline" else 64)
        for r in result["feature_extraction_count_rows"]
    )
    assert result["original_import_contract_rows"]
    assert result["ownership_mutation_rows"]
    frozen = result["frozen_8744_workload"]
    from carnot.reporting import single_extraction_8743 as a

    assert a.load(frozen["head"]) == inputs["head"]
    native = a.load(frozen["native_build_reference"])
    assert native["actual_loaded"] and len(native["sources"]) == 3
    assert native["sha256"] == frozen["extension"]["sha256"]
    assert all(ref in frozen["source_references"] for ref in native["sources"])
    assert len(result["invalid_input_rows"]) >= 60
    assert m.reduce(result["cell_primitives"], inputs) == {
        k: result[k] for k in ["rows", "feature_extraction_count_rows", "wire_parity_rows"]
    }
    broken = deepcopy(result["cell_primitives"])
    broken[0]["counts_reference"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        m.reduce(broken, inputs)


def test_runner_validation_and_dispatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8743: failed children and missing new statements deny coverage."""
    from carnot.reporting import single_extraction_runner_8743 as r
    from types import SimpleNamespace

    base = dict(
        coverage=dict(
            complete=True, files=[], combined=dict(combined_reference=dict(path="unused"))
        ),
        checks=[],
        receipts=[],
    )
    monkeypatch.setattr(r.runtime, "historical_command", lambda *args: ["bwrap", "--"])

    def historical(*args):
        argv = r.runtime.historical_command({}, tmp_path, "original")
        assert "--ro-bind" in argv and argv[-1] == "--"
        return deepcopy(base)

    monkeypatch.setattr(r, "BASE_VALIDATE", historical)
    monkeypatch.setattr(
        r,
        "Coverage",
        lambda **kw: SimpleNamespace(
            load=lambda: None, analysis2=lambda path: (None, [1], [], [], "")
        ),
    )
    monkeypatch.setattr(r.e, "added_lines", lambda name: {1})
    monkeypatch.setattr(r, "child", lambda *args, **kw: dict(passed=True))
    monkeypatch.setattr(r.runtime, "private_namespace", lambda prefix, private: prefix)
    assert r.validate(tmp_path, tmp_path, [], [])["coverage"]["complete"]
    monkeypatch.setattr(
        r,
        "Coverage",
        lambda **kw: SimpleNamespace(
            load=lambda: None, analysis2=lambda path: (None, [1], [], [1], "")
        ),
    )
    other = tmp_path / "second"
    other.mkdir()
    assert not r.validate(other, other, [], [])["coverage"]["complete"]
    monkeypatch.setattr(r.runtime, "run", lambda *args, **kw: 0)
    assert r.main([]) == 0


def test_sealed_measured_replay(installed_study, tmp_path):
    """SCENARIO-VERIFY-8743-CUSTODY: raw observations reconstruct the sealed report."""
    from carnot.reporting import single_extraction_8743 as a
    from carnot.reporting import single_extraction_runner_8743 as r

    root, inputs, measured = installed_study
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / (a.NAME + ".json")
    work = dict(
        raw=str(raw),
        output=str(output),
        invocation_id="control",
        inputs=inputs,
        sources=r.source_hashes(),
        preconditions={},
        measurement=measured,
        validation=dict(
            checks=[dict(name="private_control", passed=True)],
            receipts=[],
            coverage=dict(complete=True),
            children=[],
        ),
        owned_error=None,
        duration_s=1.0,
        started_at_utc="now",
        ended_at_utc="now",
        phase_spans=[],
    )
    a.atomic_json(raw / "seal.json", work)
    work["seal_reference"] = a.reference(raw / "seal.json")
    a.atomic_json(raw / "work.json", work)
    value = a.build(work, output)
    a.atomic_json(output, value)
    authority = value["primitive_reference"]
    assert r.replay(output, authority)
    changed = deepcopy(work)
    changed["measurement"]["rows"][0]["passed"] = False
    a.atomic_json(raw / "seal.json", {k: v for k, v in changed.items() if k != "seal_reference"})
    changed["seal_reference"] = a.reference(raw / "seal.json")
    a.atomic_json(raw / "work.json", changed)
    a.atomic_json(output, a.build(changed, output))
    assert not r.replay(output, a.reference(raw / "work.json"))


def test_input_seals_and_missing_operands(monkeypatch):
    """REQ-REPORT-8743: rejected seals and missing installed operands cannot authorize work."""
    from carnot.reporting import single_extraction_8743 as a
    from pathlib import Path

    original_load, original_reference = a.load, a.reference
    with monkeypatch.context() as control:
        control.setattr(a, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
        assert not a.inputs(a.ROOT)["passed"]
    with monkeypatch.context() as control:
        control.setattr(
            a, "load", lambda ref: {} if "seal" in Path(ref["path"]).name else original_load(ref)
        )
        assert not a.inputs(a.ROOT)["passed"]
    inputs = a.inputs(a.ROOT)
    target = inputs["original_wheel"]["path"]
    with monkeypatch.context() as control:
        control.setattr(
            a,
            "reference",
            lambda path: (
                dict(path=str(path), sha256="changed")
                if str(path) == target
                else original_reference(path)
            ),
        )
        checked = a.inputs(a.ROOT)
        assert not checked["passed"]
        assert checked["gate_check_summary"][0]["observed_state"] == "changed_bytes"
    assert all(a.added_lines(name) for name in a.MODIFIED)


def test_source_probe_cli_help(tmp_path):
    """SCENARIO-VERIFY-8743-CUSTODY: the installed probe retains a real module entry point."""
    from carnot.reporting.v709_execution import child
    import sys

    receipt = child(
        "probe_help",
        [sys.executable, "-m", "carnot.inference.resident_reuse_probe_8743", "--help"],
        tmp_path,
    )
    assert receipt["passed"]


@pytest.mark.parametrize("mode", ["identity", "count", "operands", "numeric", "panel", "bytes"])
def test_reducer_rejects_changed_observations(installed_study, tmp_path, mode):
    """SCENARIO-VERIFY-8743-CUSTODY: raw failures cannot be hidden by paired aggregates."""
    from carnot.reporting import single_extraction_8743 as a
    from carnot.reporting import single_extraction_measure_8743 as m

    root, inputs, result = installed_study
    cells = deepcopy(result["cell_primitives"])
    index = 1 if mode == "bytes" else 0
    key = "counts_reference" if mode == "count" else "session_reference"
    value = a.load(cells[index][key])
    if mode == "identity":
        value["actual_exit"] = 1
    elif mode == "count":
        value["extractions"].pop()
    elif mode == "operands":
        value["observations"][0]["request"]["rows"] = []
    elif mode == "numeric":
        value["observations"][0]["response"]["responses"][0]["probability"] += 0.1
    elif mode == "panel":
        value["observations"].pop(-2)
    else:
        value["observations"][0]["response"]["responses"][0]["probability"] += 1e-14
    path = tmp_path / "changed.json"
    a.atomic_json(path, value)
    cells[index][key] = a.reference(path)
    with pytest.raises(ValueError):
        m.reduce(cells, inputs)


def test_failed_mutations_and_admission(tmp_path, monkeypatch):
    """REQ-VERIFY-8743: failed owned children and late work deny qualification."""
    from carnot.reporting import single_extraction_8743 as a
    from carnot.reporting import single_extraction_measure_8743 as m

    monkeypatch.setattr(m, "child", lambda *args, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="installed_mutation_controls"):
        m.installed_controls("unused", tmp_path, tmp_path)
    inputs = a.inputs(a.ROOT)
    monkeypatch.setattr(m.build_native, "extension", lambda *args: (None, inputs["extension"]))

    def prepare(raw, scratch, operands):
        config = tmp_path / "config.json"
        a.atomic_json(
            config, dict(head=inputs["head"], extension=inputs["extension"], max_batch_size=16)
        )
        return ["unused"], config, [], {}

    monkeypatch.setattr(m.installed, "prepare", prepare)
    with pytest.raises(TimeoutError, match="prototype_stop_deadline"):
        m.measure(tmp_path / "raw", tmp_path, inputs, deadline_ns=1)


def test_replay_seal_sources_and_children(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8743-CUSTODY: independent seals and original child custody are required."""
    from carnot.reporting import single_extraction_8743 as a
    from carnot.reporting import single_extraction_runner_8743 as r

    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / (a.NAME + ".json")
    base = dict(
        raw=str(raw),
        output=str(output),
        invocation_id="test",
        sources=[],
        inputs=dict(gate_check_summary=[], references=[]),
        preconditions={},
        measurement={},
        validation=dict(
            checks=[dict(name="private_control", passed=True)],
            receipts=[],
            coverage=dict(complete=False),
            children=[],
        ),
        owned_error=None,
        duration_s=1.0,
        started_at_utc="now",
        ended_at_utc="now",
        phase_spans=[],
    )

    def seal(work, mismatched=False):
        work = deepcopy(work)
        a.atomic_json(raw / "seal.json", dict(work, duration_s=2.0) if mismatched else work)
        work["seal_reference"] = a.reference(raw / "seal.json")
        a.atomic_json(raw / "work.json", work)
        value = a.build(work, output)
        a.atomic_json(output, value)
        return value["primitive_reference"]

    authority = seal(base)
    assert r.replay(output, authority)
    assert not r.replay(output, dict(authority, sha256="changed"))
    assert not r.replay(output, seal(base, True))
    changed = deepcopy(base)
    changed["sources"] = [dict(path=str(a.ROOT / a.CLI), sha256="changed")]
    assert not r.replay(output, seal(changed))
    child = raw / "child.json"
    a.atomic_json(child, dict(node_ids=["test"]))
    changed = deepcopy(base)
    changed["validation"]["children"] = [a.reference(child)]
    monkeypatch.setattr(r.retained, "aggregate", lambda *args: dict(passed=False))
    assert not r.replay(output, seal(changed))
    changed["validation"]["checks"].append(dict(name="exact_disjoint_children", passed=False))
    assert r.replay(output, seal(changed))
    changed["validation"]["checks"].pop()
    monkeypatch.setattr(r.retained, "aggregate", lambda *args: dict(passed=True))
    assert r.replay(output, seal(changed))


def test_historical_fraction_cannot_change(monkeypatch):
    """REQ-REPORT-8743: the child-service denominator and historical fraction stay fixed."""
    from carnot.reporting import single_extraction_8743 as a

    authenticate = a.original.authenticate

    def altered(ref, fields):
        checked = deepcopy(authenticate(ref, fields))
        if checked["passed"] and fields.get("experiment_id") == 8733:
            for row in checked["value"]["component_fraction_intervals"]:
                if (
                    row["arm"] == "native"
                    and row["operation"] == "duplicate_feature_digest_extraction"
                ):
                    row["estimate"] = 0.1
        return checked

    monkeypatch.setattr(a.original, "authenticate", altered)
    with pytest.raises(ValueError, match="historical_fraction"):
        a.inputs(a.ROOT)


def test_e2e003_004_loaded_extension_and_serialized_head(installed_study):
    """REQ-VERIFY-8743: E2E-003/004 cross real PyO3 and a serialized parameter round trip."""
    from pathlib import Path

    root, inputs, result = installed_study
    extension = result["loaded_extension_reference"]
    native = n.load_module(Path(extension["path"]), "_nine_input_8533")
    assert native.__file__ == extension["path"]
    restored = json.loads(json.dumps(inputs["head"], allow_nan=False))
    original, _ = n.batch(inputs["head"], inputs["requests"], native)
    reopened, _ = n.batch(restored, inputs["requests"], native)
    assert [r["response"] for r in original] == [r["response"] for r in reopened]
    python, _ = n.batch(restored, inputs["requests"])
    for rust, reference in zip(reopened, python, strict=True):
        rust, reference = json.loads(rust["response"]), json.loads(reference["response"])
        assert rust["decision"] == reference["decision"]
        assert abs(rust["probability"] - reference["probability"]) <= 1e-12
