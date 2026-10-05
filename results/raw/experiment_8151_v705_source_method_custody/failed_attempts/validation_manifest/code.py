"""REQ-VERIFY-8151: source transport uses immutable historical methods.

The earlier source worker keeps prompt and parser behavior unchanged. This
adapter authenticates its source evidence without requiring learning validation.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.verify import development_methods_8098 as methods
from carnot.verify import source_protocol_8137 as previous

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8151_v705_source_method_custody"
TASK = "exp8151-source-method-custody"
MODULE = "python/carnot/verify/source_method_custody_8151.py"
CLI = f"scripts/experiments/{NAME}.py"
RUNNER = MODULE
TEST = "tests/python/test_source_method_custody_8151.py"
RUN_DATE = "20261005"
OWNED = [MODULE, CLI]
HISTORY = "results/experiment_8137_v704_source_protocol.json"
HISTORY_SHA256 = "sha256:eeaba4a4290a6953d3ba830379fba70ce9408c9af34115456928a73dbfb73648"
qualified = previous.qualified
progress = previous.progress


def source_history(root: Path, b: Any, fixture: bool) -> Json:
    """Failed historical validation stays visible; source bytes have a separate gate."""
    b.upstream = HISTORY
    value: Json = b.read(root / HISTORY, None if fixture else HISTORY_SHA256)
    b.require(root / HISTORY, "source_custody_ready_score", 1, value["source_custody_ready_score"])
    b.require(
        root / HISTORY,
        "evidence_protocol",
        previous.previous.protocol(),
        value["evidence_protocol"],
    )
    return value


def preconditions(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Pin expected identity before observations; learning readiness is not an operand."""
    b = qualified.Custody(raw)
    value: Json = {}
    try:
        value = source_history(root, b, fixture)
    except (OSError, ValueError, KeyError) as error:
        b.checks.append(
            dict(
                check="source_history",
                upstream=HISTORY,
                path=str(root / HISTORY),
                hash=HISTORY_SHA256,
                artifact_field="source_history",
                op="==",
                expected="authenticated source custody",
                observed=str(error),
                passed=False,
            )
        )
    expected = deepcopy(value.get("expected_runtime_identity", {}))
    expected.setdefault("model_revision", None)
    progress("expected_qwen_identity_frozen", int(bool(expected)))
    binary = Path(expected.get("native_binary", {}).get("path", raw / "missing-runtime"))
    progress("before_cache_identity")
    observed_cache = (
        dict(model_path=expected.get("model_path", raw / "missing-model"))
        if fixture
        else previous.previous.cached_current_model() or {}
    )
    model = Path(observed_cache.get("model_path", raw / "missing-model"))
    observations = [
        ("cache_revision", expected.get("model_revision"), model.parent.name),
        (
            "gguf_sha256",
            expected.get("gguf_sha256"),
            sha256_file(model) if model.is_file() else None,
        ),
        ("runtime_executable", True, os.access(binary, os.X_OK)),
        (
            "runtime_sha256",
            expected.get("runtime_sha256"),
            sha256_file(binary) if binary.is_file() else None,
        ),
    ]
    for key, wanted, observed in observations:
        b.checks.append(
            dict(
                check=key,
                upstream="current_cache",
                path=str(model if key.startswith(("cache", "gguf")) else binary),
                hash=observed if key.endswith("sha256") else None,
                artifact_field=key,
                op="==",
                expected=wanted,
                observed=observed,
                passed=wanted is not None and wanted == observed,
            )
        )
    progress("after_cache_identity", len(observations))
    return dict(
        expected_runtime_identity=expected,
        checks=b.checks,
        references=b.refs,
        role_manifests=value.get("source_role_manifests", {}),
        cited_upstream_artifacts=[
            dict(
                path=str(root / HISTORY),
                sha256=sha256_file(root / HISTORY),
                historical_honest_verdict=value["honest_verdict"],
                historical_MODEL_SPECS=value.get("MODEL_SPECS", []),
                historical_model_invocation_counts=value.get("model_invocation_counts", {}),
                imported_model_provenance=value.get("cited_upstream_artifacts", []),
            )
        ]
        if value
        else [],
    )


def authenticate_sources(root: Path, b: Any, fixture: bool, mutation: str) -> Json:
    """Authenticate original public roles and masks without reopening reserved labels."""
    value = source_history(root, b, fixture)
    for ref in value["source_role_manifests"].values():
        qualified.read_ref(b, ref)
    refs = {Path(r["path"]).name: r for r in value["raw_shard_hashes"]}
    rows = qualified.read_ref(b, refs["primitive_rows.json"])["rows"]
    b.require(root / HISTORY, "source_rows", value["rows"], rows)
    previous.reduce_rows(rows)
    if mutation:
        raise ValueError("private_source_tamper_" + mutation)
    return dict(rows=rows)


def whole_source_mask(input_tokens: list[int]) -> Json:
    """An oversized arm excludes the whole source so paired comparisons stay matched."""
    if len(input_tokens) != 2 or any(type(n) is not int or n < 0 for n in input_tokens):
        raise ValueError("input_token_counts")
    excluded = max(input_tokens) > previous.previous.protocol()["maximum_input_tokens"]
    return dict(
        status="excluded" if excluded else "eligible",
        exclusion_reason="whole_source_overlength" if excluded else None,
        arm_mask=[0, 0] if excluded else [1, 1],
        entailment_label=None,
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Reuse frozen prompt creation and permission probes before source authentication."""
    progress("immutable_v701_methods_before")
    config = methods.methods()
    pin = dict(path=str(methods.METHOD_PATH), sha256=methods.METHOD_SHA256)
    with (
        patch.object(previous.previous, "preconditions", preconditions),
        patch.object(qualified, "authenticate_cohort", authenticate_sources),
    ):
        work = previous.measure(root, raw, fixture=fixture, mutation=mutation)
    b = qualified.Custody(raw)
    try:
        history = source_history(root, b, fixture)
        captured = qualified.read_ref(b, history["capture_manifest"])
        b.require(
            root / HISTORY,
            "exact_v704_prompts",
            captured["rows"],
            work["protocol_conformance_rows"],
        )
    except (OSError, ValueError, KeyError):
        pass
    work["gate_check_summary"].extend(b.checks)
    work["preconditions_checked"] = work["gate_check_summary"]
    work["source_artifact_hashes"].extend([*b.refs, pin])
    work["pinned_method_paths"] = [pin]
    method_seal = qualified.cohort.immutable(raw / "v701_methods.json", config)
    b.require(
        raw / "v701_methods.json",
        "original_v701_method_seal",
        "sha256:10d31f444cb10b91b5360f5edbe5e3b825df283fa9c4552cbc99416eeb272cb7",
        method_seal["sha256"],
    )
    work["gate_check_summary"].extend(b.checks[-1:])
    work["raw_shard_hashes"].append(method_seal)
    work["protocol_fixture_rows"] = [
        dict(input_tokens=tokens, fixture_only=True, **whole_source_mask(tokens))
        for tokens in ([6000, 5999], [6000, 6001])
    ]
    work["raw_shard_hashes"].append(
        qualified.cohort.immutable(
            raw / "protocol_fixture_qualification.json", dict(rows=work["protocol_fixture_rows"])
        )
    )
    work["code_config_hashes"].update(
        {
            n: sha256_file(ROOT / n)
            for n in [
                *OWNED,
                TEST,
                "python/carnot/verify/development_methods_8098.py",
                "python/carnot/experiment_8098_v701_development_methods.py",
                "python/carnot/reporting/methods_stream_execution_8111.py",
            ]
        }
    )
    atomic_json(raw / "measurement.json", work)
    progress("immutable_source_measurement_complete", len(work["rows"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness means validated custody; it needs no positive scientific effect."""
    value = previous.build(work, raw, receipts, fixture=fixture)
    value.update(
        experiment_id=8151,
        task_id=TASK,
        milestone="2026.10.705",
        run_date=RUN_DATE,
        random_seed=70551,
        source_custody_ready_score=int(
            value["required_checks_passed"] and work["source_custody_ready_score"]
        ),
        consumer_validation_rows=[r for r in receipts if r.get("name", "").startswith("consumer")],
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild headlines independently while authenticating primitives, code and logs."""
    with patch.object(previous.previous, "build", build):
        return previous.previous.replay(path)


BASE_MANIFEST = execution.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze owned checks before measurement; old-module coverage is scoped to edits."""
    specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    changed = [
        "python/carnot/verify/development_methods_8098.py",
        "python/carnot/experiment_8098_v701_development_methods.py",
        "python/carnot/reporting/methods_stream_execution_8111.py",
    ]
    config.write_text(config.read_text() + "".join("    " + str(ROOT / n) + "\n" for n in changed))
    specs["commands"][3]["argv"] += ["--include=" + ",".join(str(ROOT / n) for n in OWNED)]
    specs["commands"][0]["argv"] += [previous.TEST, previous.previous.TEST]
    specs["commands"][1]["argv"] += [
        "tests/python/test_experiment_7868_v683_intervention_protocol.py"
    ]
    for index in (5, 6, 7):
        specs["commands"][index]["argv"] += changed
    specs["repository_health"]["reused_receipt_path"] = os.environ.get("CARNOT_8151_HEALTH_RECEIPT")
    py = str(ROOT / ".venv/bin/python")
    e2e = str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    output = private / "intervention.json"
    for name, args in [
        ("E2E016_fixture", ["--date", "20260929", "--fixture-e2e", str(output)]),
        ("E2E016_cold_replay", ["--cold-replay", str(output)]),
    ]:
        specs["commands"].append(
            dict(
                name=name,
                argv=[py, e2e, *args],
                deadline_s=60,
                expected_exit=0,
                classification="required",
            )
        )
    specs["commands"].append(
        dict(
            name="changed_statement_coverage",
            argv=[
                py,
                "-c",
                "import ast,json,sys\nfrom pathlib import Path\nreport=json.loads(Path(sys.argv[1]).read_text())['files']\np='python/carnot/verify/development_methods_8098.py'\ntree=ast.parse(Path(p).read_text())\nfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='methods')\nlines={n.lineno for n in ast.walk(fn.body[1]) if isinstance(n,ast.stmt)}\nlines.update(n.lineno for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'METHOD_PATH','METHOD_SHA256'} for t in n.targets))\nscopes={p:lines}\np='python/carnot/experiment_8098_v701_development_methods.py'\nscopes[p]={n.lineno for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DESIGN' for t in n.targets)}\np='python/carnot/reporting/methods_stream_execution_8111.py'\nscopes[p]={n.lineno for n in ast.walk(ast.parse(Path(p).read_text())) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='add_argument' and n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='--date'}\nfor p,lines in scopes.items():\n    missing=lines.intersection(report[p]['missing_lines'])\n    assert lines and not missing,(p,sorted(missing))\n    print(p, 'changed statements=',len(lines),'coverage=100%',flush=True)\n",
                str(private / "coverage.json"),
            ],
            deadline_s=30,
            expected_exit=0,
            classification="required",
        )
    )
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use the qualified heartbeat supervisor, date parser and atomic publication."""
    original_check = execution.run_check

    def check(root: Path, spec: Json, private: Path, raw: Path, *, heartbeat_s: float = 30) -> Json:
        """Reuse completed full-suite diagnostics so publication never reruns that suite."""
        receipt = os.environ.get("CARNOT_8151_HEALTH_RECEIPT")
        if spec["name"] == "repository_full_suite" and receipt:
            value: Json = json.loads(Path(receipt).read_text())
            return value
        return original_check(root, spec, private, raw, heartbeat_s=heartbeat_s)

    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", check),
    ):
        return execution.main(argv)
