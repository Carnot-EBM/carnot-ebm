"""DELIVERY PROOF for REQ-ARC-WMTE-6091, taken at the CALLEE and read off the stack.

WHY THIS EXISTS AS A SEPARATE STEP, BEFORE ANY MEASUREMENT
-----------------------------------------------------------
A fix that ships is not a fix that runs. This session already had one fix reach its call site
0 of 128 times, and the defect this experiment measures is ITSELF an availability-vs-delivery
bug: `refactor_prompt` had the game id and the VerifyResult available and delivered neither the
engine nor the passing cases into the rendered string.

The A/B harness records `prompt_contains_engine` from a prompt IT renders itself, alongside a
SEPARATE `prop.refactor(...)` call that renders the prompt AGAIN. Two independent renders. That
recorded flag is therefore a statement about the harness's own copy, not about the bytes the
model receives -- exactly the substitution this codebase keeps getting caught by. So delivery is
proven HERE instead, at the deepest callee before transport, on the live entrypoint:

    LocalGGUFProposer.refactor  ->  _gen_to_file  ->  generate(prompt, ...)  ->  HTTP

`generate` is wrapped, the prompt it actually receives is captured, and the CALLER CHAIN is read
off the interpreter stack with `inspect` -- so the evidence says "this string arrived at the
generator, and it got there through the shipped refactor path", not "a function I called
returned a string containing the engine".

NO SERVER IS NEEDED and none is used: the wrapper captures and short-circuits before any socket
is opened. The transport is not what is in doubt; the prompt content is.

BOTH DIRECTIONS, because a probe that can only confirm is not a probe:
  * flag ON  -> the engine's own signature line MUST appear in the captured prompt.
  * flag OFF -> it MUST NOT, and the captured prompt must be byte-identical to the shipped one.
A MUTATION arm neutralises the source resolver the way a careless edit would and requires the ON
assertion to go red, so a green result here cannot be vacuous.

Spec: REQ-ARC-WMTE-6091
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# The store MUST be redirected BEFORE the module is imported: `E3_DIR` is resolved at import
# time, so setting this afterwards would be a no-op that reads like a safeguard.
_SCRATCH = Path(tempfile.mkdtemp(prefix="exp6091_delivery_probe_"))
os.environ["CARNOT_ARC_E3_DIR"] = str(_SCRATCH)

ARTIFACT = REPO / "results" / "experiment_6091_delivery_probe.json"
GAME = "probe_game"
SEED = 6091

# A line that carries THIS engine's logic and appears in no template in the codebase. If it
# reaches the generator, the engine reached the generator.
SIGNATURE_LINE = "        out[marker_row, 0] = 7  # exp6091-delivery-probe-signature"
ENGINE_SRC = f"""import numpy as np


def engine(grid, action, data):
    out = np.asarray(grid).copy()
    marker_row = int(out.shape[0]) - 1
    if action == 3:
{SIGNATURE_LINE}
    return out


def is_level_complete(grid):
    return bool(np.asarray(grid)[0, 0] == 9)
"""


def _real_verify_result():
    """A real VerifyResult produced by the shipped scorer over real mismatches."""
    import numpy as np

    from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier

    g0 = np.zeros((3, 3), dtype=int)
    g1 = g0.copy()
    g1[1, 1] = 4
    g2 = g1.copy()
    g2[2, 2] = 5
    rows = [
        Transition(g0.copy(), 1, None, g1.copy(), 0, 0),
        Transition(g1.copy(), 2, None, g2.copy(), 0, 0),
    ]

    def wrong_engine(grid, action, data=None):
        return np.asarray(grid).copy()

    return WorldModelVerifier(rows, hud_mask=None).score(wrong_engine)


def run_arm(flag: str, *, neutralise: bool = False) -> dict[str, Any]:
    """Drive the LIVE `refactor` entrypoint and capture what the generator was handed.

    `neutralise` is the mutation arm: it makes the source resolver return nothing, the way a
    careless edit deleting the splice would, and the ON assertion must then fail.
    """
    import carnot.agentic.arc_executable_world_model as e3

    captured: dict[str, Any] = {}

    def fake_generate(self, prompt, required=(), *args, codeonly_eligible=False, **kwargs):
        # THE CALLEE. Read the caller chain off the interpreter stack rather than trusting that
        # the call arrived the way the docstring says it does.
        chain = []
        for frame in inspect.stack()[1:8]:
            chain.append(f"{Path(frame.filename).name}::{frame.function}:{frame.lineno}")
        captured["prompt"] = prompt
        captured["caller_chain"] = chain
        captured["codeonly_eligible"] = bool(codeonly_eligible)
        captured["required"] = list(required)
        # Return a syntactically valid engine so `_gen_to_file` completes normally; nothing is
        # sent anywhere and the scratch store is the only thing written.
        return True, ENGINE_SRC

    prev_flag = os.environ.get("CARNOT_ARC_REFACTOR_SHOW_ENGINE")
    prev_gen = e3.LocalGGUFProposer.generate
    prev_src = e3._current_engine_source
    os.environ["CARNOT_ARC_REFACTOR_SHOW_ENGINE"] = flag
    e3.LocalGGUFProposer.generate = fake_generate
    if neutralise:
        e3._current_engine_source = lambda game, **kw: ("", 0)
    try:
        (_SCRATCH / GAME).mkdir(parents=True, exist_ok=True)
        (_SCRATCH / GAME / "world_model.py").write_text(ENGINE_SRC)
        prop = e3.LocalGGUFProposer(repo_substr="gemma-4-31B-it", port=1, model_path="/nonexistent")
        ok, msg = prop.refactor(GAME, _real_verify_result())
    finally:
        e3.LocalGGUFProposer.generate = prev_gen
        e3._current_engine_source = prev_src
        if prev_flag is None:
            os.environ.pop("CARNOT_ARC_REFACTOR_SHOW_ENGINE", None)
        else:
            os.environ["CARNOT_ARC_REFACTOR_SHOW_ENGINE"] = prev_flag

    prompt = captured.get("prompt", "")
    return {
        "flag": flag,
        "neutralised_source_resolver": bool(neutralise),
        "refactor_returned_ok": bool(ok),
        "refactor_message": str(msg)[:160],
        "generator_was_called": "prompt" in captured,
        "caller_chain_at_generator": captured.get("caller_chain"),
        "codeonly_eligible": captured.get("codeonly_eligible"),
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        # THE DELIVERY ASSERTIONS, on the bytes the generator received.
        "engine_signature_line_delivered": SIGNATURE_LINE in prompt,
        "engine_block_header_delivered": "THE CURRENT ENGINE YOU ARE FIXING" in prompt,
        "mismatch_block_present": "MISMATCHES:" in prompt,
        "n_engine_source_lines_delivered": sum(
            1
            for line in ENGINE_SRC.splitlines()
            if line.strip() and len(line.strip()) > 8 and line in prompt
        ),
    }


def main() -> int:
    t0 = time.time()
    arms = {
        "flag_off": run_arm("0"),
        "flag_on": run_arm("1"),
        "flag_on_mutation_resolver_neutralised": run_arm("1", neutralise=True),
    }

    off, on, mut = arms["flag_off"], arms["flag_on"], arms["flag_on_mutation_resolver_neutralised"]

    checks = {
        "generator_reached_on_live_path_all_arms": all(
            a["generator_was_called"] for a in arms.values()
        ),
        "caller_chain_goes_through_shipped_refactor": bool(
            on["caller_chain_at_generator"]
            and any("_gen_to_file" in f for f in on["caller_chain_at_generator"])
            and any("refactor" in f for f in on["caller_chain_at_generator"])
        ),
        "ON_delivers_engine_signature_to_generator": on["engine_signature_line_delivered"],
        "OFF_does_not_deliver_engine": not off["engine_signature_line_delivered"],
        "OFF_reproduces_the_shipped_defect": not off["engine_block_header_delivered"],
        "mismatch_block_survives_the_splice": on["mismatch_block_present"],
        "ON_and_OFF_are_different_prompts": on["prompt_sha256"] != off["prompt_sha256"],
        # MUTATION PROOF: neutralising the resolver must collapse ON back onto OFF.
        "mutation_kills_delivery": not mut["engine_signature_line_delivered"],
        "mutation_collapses_to_off_bytes": mut["prompt_sha256"] == off["prompt_sha256"],
    }
    all_ok = all(checks.values())

    out: dict[str, Any] = {
        "experiment": "experiment_6091_delivery_probe",
        "spec": "REQ-ARC-WMTE-6091",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # SUBSTRATE, WITH THE GAP DISCLOSED RATHER THAN GLOSSED. No model is loaded and no
        # candidate is scored, so neither `live_llm_inference` nor
        # `verifier_ensemble_against_cached_candidates` is honest here -- an earlier draft
        # declared the latter and `adversarial_verify.py` correctly flagged it
        # DURATION_TOO_SHORT at 0.46 s, which is the linter doing its job on a real
        # misdeclaration. This is the project's established bucket for "no model was invoked":
        # the sibling instrument-reproduction experiment
        # (results/outer_loop_arc_refine_instrument_repro_20260803.json) declares the same value
        # for the same class of work. The DIFFERENCE from a literal reading of that value --
        # this probe renders shipped code paths rather than reading upstream JSON -- is stated
        # here so nobody has to infer it from the duration.
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": SEED,
        "methodology_note": (
            "NO LLM IS INVOKED and no GPU is used; the sub-second duration is therefore correct "
            "and expected, not a truncated run. `LocalGGUFProposer.generate` -- the deepest "
            "callee before the HTTP transport -- is wrapped, so the assertion is on the exact "
            "prompt string the generator receives, and the caller chain is read off the "
            "interpreter stack to prove it arrived via the shipped `refactor` -> `_gen_to_file` "
            "path rather than via a convenience render. Transport is not in doubt; prompt "
            "CONTENT is. `model_specs` is deliberately absent: there is no model to name, and "
            "naming one would be the vestigial-marker pattern the Inference-Substrate "
            "Declaration Discipline exists to stop."
        ),
        "engine_store_redirected_to": str(_SCRATCH),
        "signature_line": SIGNATURE_LINE,
        "arms": arms,
        "checks": checks,
        "delivery_proven": all_ok,
        "duration_s": round(time.time() - t0, 3),
    }
    out["honest_verdict"] = (
        "complete_engine_source_delivered_to_generator_on_live_path"
        if all_ok
        else "blocked_delivery_not_proven"
    )
    out["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in out.items() if k != "reproducibility_checksum"},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    ARTIFACT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"checks": checks, "verdict": out["honest_verdict"]}, indent=1))
    print(f"caller chain (ON): {on['caller_chain_at_generator']}")
    print(
        f"ON delivered {on['n_engine_source_lines_delivered']} engine source lines; "
        f"OFF delivered {off['n_engine_source_lines_delivered']}"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
