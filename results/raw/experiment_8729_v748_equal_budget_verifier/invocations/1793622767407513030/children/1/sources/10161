"""CAPTURE the REAL lp85 L2 reinduction prompt + the model's RAW completion, to diagnose why
code-gen fails on it (where the synthetic prompt is 3/3 reliable).

Mechanism: tee every llama-server /completion request body (the prompt) + response (the raw content,
stop_type, tokens_predicted) to results/l2_capture.jsonl, then run the lp85 agent until it reaches
L1 and fires the L2 level-up reinduction. The L2 reinduction call is the one whose prompt contains
the WIN STATE exemplar block AND the code-only directive. An external poller watches the JSONL for
that line and analyzes it (parse? prose leakage? stop-cut? missing def?).

Single game, warm Qwen :8920 reused. inference_substrate=live_llm_inference;
solve_provenance=development_proxy; verifier_is_oracle=false.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.pop("CARNOT_ARC_CODEONLY_INDUCE", None)  # default-ON path under test

CAP_PATH = REPO / "results" / "l2_capture.jsonl"
BUDGET = int(os.environ.get("CAPTURE_BUDGET", "400"))


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --- install the teeing urlopen BEFORE the agent runs ---
_real_urlopen = urllib.request.urlopen
_cap = open(CAP_PATH, "w")  # fresh file


class _Resp:
    """Re-serves an already-read response body so the caller's json.load(r) still works."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def read(self, *_a: object) -> bytes:
        return self._raw


def _tee_urlopen(req, timeout=None):  # noqa: ANN001
    body = getattr(req, "data", None)
    resp = _real_urlopen(req, timeout=timeout)
    raw = resp.read()
    if body:
        try:
            bj = json.loads(body.decode())
            if "prompt" in bj:
                rj = json.loads(raw)
                rec = {
                    "ts": time.strftime("%H:%M:%S"),
                    "prompt": bj["prompt"],
                    "stop": bj.get("stop"),
                    "n_predict": bj.get("n_predict"),
                    "content": rj.get("content", ""),
                    "stop_type": rj.get("stop_type"),
                    "tokens_predicted": rj.get("tokens_predicted"),
                }
                _cap.write(json.dumps(rec) + "\n")
                _cap.flush()
        except Exception:
            pass
    return _Resp(raw)


urllib.request.urlopen = _tee_urlopen


def main() -> int:
    t0 = time.time()
    # precondition
    try:
        with _real_urlopen("http://127.0.0.1:8920/health", timeout=4) as r:
            ok = json.load(r).get("status") == "ok"
    except Exception as ex:
        log(f"BLOCKED: server health failed: {ex}")
        return 0
    if not ok:
        log("BLOCKED: server not healthy")
        return 0

    import carnot.agentic.arc_solver_kit as kit  # noqa: E402
    from scripts.experiments.proto_graded_goal_bias_ab import _run_arm  # noqa: E402

    arc = kit.offline_arcade()
    log(f"Running lp85 agent (budget={BUDGET}) with /completion tee -> {CAP_PATH} ...")
    log("Watching for the L2 reinduction call (prompt with WIN STATE + code-only directive).")
    arm = _run_arm(arc, "lp85", graded=False, port=8920, budget=BUDGET)
    log(f"agent done: max_depth={arm.get('max_depth_reached')} wall_s={arm.get('wall_s')} "
        f"n_induce={arm.get('n_induction_attempts')}")
    log(f"DONE in {round(time.time() - t0, 1)}s. Captures in {CAP_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
