"""REQ-VERIFY-8634: inspect an installed wheel without adding checkout paths.

Run this file directly in a private environment. Each mode starts fresh so
cached imports cannot conceal missing dependencies or lazy export failures.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any


def probe(mode: str) -> dict[str, Any]:
    """Return primitive observations only after the actual public calls succeed."""
    if mode in {"absent", "errors"}:

        class Block(importlib.abc.MetaPathFinder):
            """Expose real import errors before optional dependencies are cached."""

            def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
                if mode == "absent" and (fullname == "jax" or fullname.startswith("jax.")):
                    raise ModuleNotFoundError("private absent JAX", name="jax")
                if mode == "errors" and fullname == "carnot.models":
                    raise ImportError("private model import failure")

        sys.meta_path.insert(0, Block())
    import carnot
    from carnot.inference import resident_protocol

    paths = [Path(carnot.__file__), Path(resident_protocol.__file__)]
    assert all(p.resolve().is_relative_to(Path(sys.prefix).resolve()) for p in paths)
    assert not any(
        n.startswith(("carnot.training", "carnot.reporting", "jax")) for n in sys.modules
    )
    rows = []
    if mode == "absent":
        assert carnot.__all__ == ["__version__", "RUST_AVAILABLE", "JAX_AVAILABLE"]
        assert carnot.JAX_AVAILABLE is False and carnot.RUST_AVAILABLE is False
        scope: dict[str, Any] = {}
        exec("from carnot import *", scope)
        for name in ["IsingModel", "jnp", "unknown_export"]:
            try:
                getattr(carnot, name)
            except AttributeError:
                rows.append(dict(name=name, rejected=True))
        assert len(rows) == 3
    elif mode == "errors":
        try:
            _ = carnot.IsingModel
        except ImportError as error:
            assert str(error) == "private model import failure"
            rows.append(dict(name="IsingModel", propagated_error=str(error)))
        assert rows
    elif mode == "exports":
        scope = {}
        exec("from carnot import *", scope)
        assert set(scope) - {"__builtins__"} == set(carnot.__all__)
        for name, module in carnot._EXPORTS.items():
            explicit: dict[str, Any] = {}
            exec("from carnot import " + name, explicit)
            expected = getattr(importlib.import_module("carnot." + module), name)
            assert scope[name] is explicit[name] is getattr(carnot, name) is expected
            assert getattr(carnot, name) is getattr(carnot, name)
            rows.append(dict(name=name, module=module, passed=True))
    else:
        from carnot.mcp import server as mcp
        from carnot.pipeline import abstention_http_rest as rest
        from carnot.pipeline.second_pair_detector import LabeledDetectorExample
        from carnot.verify.python_types import build_code_energy, code_to_embedding

        source = "def add(a, b):\n    return a + b\n"
        energy = build_code_energy(source, "add", [((1, 2), 3)], expected_type=int)
        result = energy.verify(code_to_embedding(source))
        assert result.total_energy == 0
        rows.append(dict(surface="python", module="carnot.verify.python_types", passed=True))
        assert mcp.health_check()["status"] == "ok"
        assert mcp.verify_code(source, "add", [dict(args=[1, 2], expected=3)])["energy"] == 0
        assert mcp.verify_code(source, "add", [dict(args=[1, 2], expected=4)])["energy"] > 0
        rows.append(dict(surface="mcp", module="carnot.mcp.server", passed=True))
        code_path = Path.cwd() / "add.py"
        code_path.write_text(source)
        for args in [
            ["--help"],
            ["verify", str(code_path), "--func", "add", "--type", "int", "--test", "(1,2):3"],
        ]:
            print("[8634] before installed CLI", file=sys.stderr, flush=True)
            cli = subprocess.run(
                [sys.executable, "-u", "-m", "carnot.cli", *args],
                capture_output=True,
                text=True,
                timeout=45,
            )
            print("[8634] after installed CLI", file=sys.stderr, flush=True)
            assert cli.returncode == 0, cli.stderr
            rows.append(
                dict(
                    surface="cli",
                    module="carnot.cli",
                    argv=args,
                    stdout=cli.stdout,
                    stderr=cli.stderr,
                    exit_code=cli.returncode,
                    passed=True,
                )
            )
        examples = [
            LabeledDetectorExample(
                domain="math",
                label=int(i < 40),
                ensemble_energy=0.95 - i * 0.01,
                confidence_error=0.9 - i * 0.01,
                example_id=str(i),
            )
            for i in range(80)
        ]
        server = rest.make_server(("127.0.0.1", 0), root=Path.cwd(), examples=examples)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        statuses = []
        try:
            for payload in [
                {"candidates": []},
                {
                    "domain": "math",
                    "candidates": [
                        dict(
                            candidate_id="existing",
                            domain="math",
                            text="8 + 5 = 13",
                            confidence_error=0.2,
                            ensemble_energy=0.2,
                        )
                    ],
                },
            ]:
                connection = http.client.HTTPConnection(*server.server_address, timeout=10)
                connection.request(
                    "POST",
                    rest.POST_PATH,
                    json.dumps(payload),
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                body = json.loads(response.read())
                statuses.append(response.status)
                connection.close()
            assert statuses == [400, 200]
            assert body["abstention_mode_enabled"] is False
            assert body["batch"]["n_candidates"] == 1
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        rows.append(
            dict(
                surface="http",
                module="carnot.pipeline.abstention_http_rest",
                path=rest.POST_PATH,
                statuses=statuses,
                detector_fixture_examples=80,
                detector_fixture_fit=True,
                health_contract="No GET health route; real server startup and existing POST checked",
                passed=True,
            )
        )
    return dict(
        mode=mode,
        passed=True,
        rows=rows,
        prefix=sys.prefix,
        modules=[
            dict(path=str(p), sha256="sha256:" + hashlib.sha256(p.read_bytes()).hexdigest())
            for p in paths
        ],
    )


if __name__ == "__main__":
    print(json.dumps(probe(sys.argv[1]), sort_keys=True), flush=True)
