"""REQ-REPORT-8990: retain the established supervised runtime and checked publication."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import patch

from carnot.inference import graph_corpus_runtime_8961 as runtime
from carnot.inference.sota_models import gguf_tokenizer_loadable
from carnot.reporting import graph_corpus_a_8990 as d
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import spline_basis_methods_8986 as basis
from carnot.reporting import graph_corpus_runner_8961 as runner
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
_freeze, _plan, _preflight = runner.freeze, runner.validation_plan, runtime.preflight
MODEL_PIN = "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"


def preflight(raw: Path) -> Json:
    """Check resources, network and embedded tokenizer before loading model weights."""
    source.progress("before_resource_preflight", 0, 512)
    evidence: Json = dict(
        checks=[],
        failures=[],
        method_source_rows=[
            dict(
                url="https://huggingface.co",
                path=str(d.ROOT / d.CLI),
                source_id="model_host_connectivity",
            )
        ],
    )
    resources = basis.preflight(raw, evidence)
    atomic_json(
        raw / "resource_preflight.json",
        dict(resources=resources, checks=evidence["checks"], failures=evidence["failures"]),
    )
    if evidence["failures"]:
        raise RuntimeError(
            "resource_preconditions:" + json.dumps(evidence["failures"], sort_keys=True)
        )
    try:
        result: Json = _preflight(raw)
        if result["model"]["sha256"] != MODEL_PIN:
            raise RuntimeError("model_cache_hash:" + result["model"]["sha256"])
        source.progress("before_tokenizer_preflight", 0, 512)
        ok, detail = gguf_tokenizer_loadable(result["model"]["model_path"])
        source.progress("after_tokenizer_preflight", int(ok), 512)
        if not ok:
            raise RuntimeError("embedded_tokenizer:" + detail)
        result.update(
            resources=resources,
            embedded_tokenizer=dict(passed=ok, detail=detail),
            server_binary=d.reference(
                Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
            ),
        )
        atomic_json(raw / "preflight.json", result)
        source.progress("after_resource_preflight", 1, 512)
        return result
    except ImportError as error:
        raise RuntimeError("runtime_dependency:" + str(error)) from error


def validation_plan(private: Path) -> Any:
    """Keep pytest selectors in pytest and include only newly owned statements."""
    return [
        replace(
            c,
            argv=(
                *c.argv[: c.argv.index("pytest") + 1],
                "--noconftest",
                "-s",
                *c.argv[c.argv.index("pytest") + 1 :],
            ),
        )
        if "pytest" in c.argv
        else replace(c, argv=(c.argv[0], "--noconftest", *c.argv[1:]))
        if c.name in {"e2e018", "e2e021"}
        else c
        for c in _plan(private, producer=d)
    ]


def freeze(
    root: Path, raw: Path, fixture: bool, *, producer: Any = None, commands: Any = None
) -> Json:
    """Freeze validation argv and inputs before any model action occurs."""
    private = Path(tempfile.mkdtemp(prefix="carnot-8990-plan-"))
    atomic_json(
        raw / "validation_plan.json",
        dict(
            commands=[
                dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s, scope=c.scope)
                for c in (commands or validation_plan(private))
            ]
        ),
    )
    input_root = Path(os.environ.get("CARNOT_8990_INPUT_ROOT", str(root)))
    work: Json = _freeze(input_root, raw, fixture, producer=d)
    work["raw_references"].append(d.reference(raw / "validation_plan.json"))
    return work


def main(argv: list[str] | None = None) -> int:
    """Reuse the direct CLI's private fixture, replay, validators and atomic writer."""
    commands = validation_plan(Path(tempfile.mkdtemp(prefix="carnot-8990-validation-")))
    with (
        patch.object(runner, "freeze", lambda *a, **kw: freeze(*a, **kw, commands=commands)),
        patch.object(runner, "validation_plan", lambda private, **_: commands),
        patch.object(runtime, "preflight", preflight),
    ):
        return int(runner.main(argv, producer=d))
