"""REQ-REPORT-8965: score JSON operands while oracle files and imports are denied."""

from __future__ import annotations

import importlib
import importlib.abc
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

from carnot.autoresearch import relational_graph_encoder as g

Json = dict[str, Any]
FORBIDDEN = ("bfs", "travers", "shortest", "networkx", "carnot.reporting", "carnot.verify")


class DenyTruth(importlib.abc.MetaPathFinder):
    """Block dynamic loading of labeling code before it can inspect any graph."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if any(word in fullname for word in FORBIDDEN):
            raise PermissionError("labeling_import_denied")
        return None


def probabilities(service: Json, rows: list[Json], condition: str) -> Json:
    """Remove each context from both message passing and the final pooling operands."""
    packed = g.pack(rows)
    if condition == "adjacency_removed":
        packed["a"] = jnp.zeros_like(packed["a"])
        packed["x"] = packed["x"].at[:, :, 3].set(0)
    if condition == "source_marker_removed":
        packed["s"] = jnp.zeros_like(packed["s"])
        packed["x"] = packed["x"].at[:, :, 0].set(0)
    if condition == "answer_marker_removed":
        packed["q"] = jnp.zeros_like(packed["q"])
        packed["x"] = packed["x"].at[:, :, 1].set(0)
    result = {}
    for arm, states in service["states"].items():
        values = []
        for state in states:
            data = dict(packed)
            data["c"] = jnp.asarray(g.b._standardize(np.asarray(data["c"]), state["scaling"]))
            params = dict(
                encoder={k: jnp.asarray(v) for k, v in state["params"]["encoder"].items()},
                head=tuple(jnp.asarray(v) for v in state["params"]["head"]),
            )
            z = g.logits(params, data, state["relational"], state["energy"])
            values.append(g.b._sigmoid(z / service["temperatures"][arm]).tolist())
        result[arm] = dict(seeds=values, mean=np.mean(values, axis=0).tolist())
    result["prevalence"] = dict(seeds=[], mean=[service["prevalence_probability"]] * len(rows))
    return result


def run(request: Json) -> Json:
    """Warm library internals on synthetic inputs, then deny files for all real scoring."""
    g.b.progress("8965_before_frozen_weight_read", 0, 12)
    service = request["service"]
    synthetic = dict(
        nodes=["a"],
        edges=[],
        query_node="a",
        proposed_node=None,
        features={k: 0.0 for k in g.b.FEATURES},
    )
    probabilities(service, [synthetic], "baseline")
    g.b.progress("8965_after_frozen_weight_read", 12, 0)
    active = [True]

    def audit(event: str, args: Any) -> None:
        if active[0] and (
            event in {"open", "os.listdir", "os.scandir", "subprocess.Popen", "socket.connect"}
            or event == "import"
            and any(w in str(args[0]) for w in FORBIDDEN)
        ):
            raise PermissionError("scoring_access_denied")

    finder = DenyTruth()
    sys.meta_path.insert(0, finder)
    # CPython needs this opt-in to measure statements inside audit callbacks.
    audit.__dict__["__cantrace__"] = True
    sys.addaudithook(audit)
    try:
        probes = {}
        for name in request["denial_paths"]:
            try:
                Path(name).read_text()
            except PermissionError:
                probes[name] = True
        for name in ("carnot.reporting.graph_source_8931", "networkx.algorithms.shortest_paths"):
            try:
                importlib.import_module(name)
            except PermissionError:
                probes[name] = True
        result = {}
        for i, (condition, rows) in enumerate(request["conditions"].items()):
            g.b.progress("8965_before_score_" + condition, i, len(request["conditions"]) - i)
            result[condition] = probabilities(service, rows, condition)
            g.b.progress("8965_after_score_" + condition, i + 1, len(request["conditions"]) - i - 1)
        return dict(
            predictions=result,
            denial_probes=probes,
            labeling_helper_absent=not any(n.startswith("carnot.reporting") for n in sys.modules),
        )
    finally:
        active[0] = False
        sys.meta_path.remove(finder)


def main() -> None:
    """Only the envelope is read before scoring; result bytes are written after denial ends."""
    request = json.loads(Path(sys.argv[1]).read_text())
    result = run(request)
    Path(sys.argv[2]).write_text(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
