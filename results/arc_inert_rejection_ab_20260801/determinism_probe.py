"""Why did two arms with the SAME prompt and the SAME seed draw different completions?

THE OBSERVATION THAT PROMPTED THIS. The A/B's first three (game, replicate) pairs each produced a
different engine in the two arms at byte-identical `prompt_sha256`, identical
`CARNOT_ARC_GENERATOR_SEED`, and exactly one completion call per arm. The treatment cannot
explain that -- it acts AFTER a completion arrives -- so the generator is not reproducing.

This matters beyond one experiment. `LocalGGUFProposer.sampling_seed`'s docstring says an A/A arm
"should come back byte-identical, which is a cheap positive control on the determinism itself",
and three A/B designs on this path have now been built on that promise. If it is false, every one
of them is a randomized comparison rather than a matched one, and their power estimates are wrong
in the optimistic direction.

FOUR CONDITIONS, chosen so the result identifies WHICH factor breaks it rather than just
confirming that something does. Each is the previous one with exactly one thing changed:

  A. short prompt, cache_prompt=True, back to back        -- the probe that already matched
  B. long (real induce) prompt, cache_prompt=True, back to back
  C. long prompt, cache_prompt=True, with a DIFFERENT long prompt in between
     (this is the A/B's actual access pattern: off(game), on(game), off(next game), ...)
  D. long prompt, cache_prompt=FALSE, with a different long prompt in between

If B matches and C does not, the cause is prefix reuse across a changed cache state. If D matches
where C does not, `cache_prompt` is the mechanism and disabling it buys determinism at the cost of
prefill time. If none match, the nondeterminism is in the long-prompt prefill itself and no
harness-side setting fixes it.

RUN THIS AFTER COLLECTION, AGAINST THE SAME SERVER, BEFORE TEARING IT DOWN. Running it during
collection would inject requests into the very cache whose reuse is under suspicion, changing the
experiment it is trying to explain.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 8100000
N_PREDICT = 256  # far below the 4096 cap: this asks about determinism, not about a full engine


def complete(port: int, prompt: str, *, cache_prompt: bool) -> tuple[str, str]:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": N_PREDICT,
            "temperature": 0.2,
            "cache_prompt": cache_prompt,
            "seed": SEED,
            "stop": ["```"],
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r).get("content", "")
    return hashlib.sha256(text.encode()).hexdigest()[:16], text


def main() -> int:
    port = json.loads((OUT / "server_witness.json").read_text())["port_actual"]
    rows = json.loads((OUT / "rows.json").read_text())

    # Two REAL induce prompts, rebuilt from the roster so this probe uses the same prompt shape
    # the experiment used rather than a synthetic stand-in.
    sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")
    import pickle

    from carnot.agentic import arc_executable_world_model as e3

    games = []
    for g in dict.fromkeys(r["game"] for r in rows):
        wp = OUT / "windows" / f"{g}.pkl"
        if wp.exists():
            games.append((g, pickle.loads(wp.read_bytes())))
        if len(games) == 2:
            break
    if len(games) < 2:
        print("need two window pickles; run this from the collection directory")
        return 1

    def induce_text(g, w) -> str:
        base = e3.induce_prompt(g, w["shown"], w["cell"], k=e3._induce_transitions_k())
        tail = (
            "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n"
        )
        return e3._L2_CODEONLY_DIRECTIVE + base + tail + "\n```python\n"

    (g1, w1), (g2, w2) = games
    long1, long2 = induce_text(g1, w1), induce_text(g2, w2)
    short = "Write one short Python function that adds two numbers.\n```python\n"

    results = {}

    h1, _ = complete(port, short, cache_prompt=True)
    h2, _ = complete(port, short, cache_prompt=True)
    results["A_short_cached_back_to_back"] = {"a": h1, "b": h2, "identical": h1 == h2}

    h1, _ = complete(port, long1, cache_prompt=True)
    h2, _ = complete(port, long1, cache_prompt=True)
    results["B_long_cached_back_to_back"] = {"a": h1, "b": h2, "identical": h1 == h2}

    h1, _ = complete(port, long1, cache_prompt=True)
    complete(port, long2, cache_prompt=True)
    h2, _ = complete(port, long1, cache_prompt=True)
    results["C_long_cached_with_another_prompt_between"] = {
        "a": h1,
        "b": h2,
        "identical": h1 == h2,
    }

    h1, _ = complete(port, long1, cache_prompt=False)
    complete(port, long2, cache_prompt=False)
    h2, _ = complete(port, long1, cache_prompt=False)
    results["D_long_uncached_with_another_prompt_between"] = {
        "a": h1,
        "b": h2,
        "identical": h1 == h2,
    }

    verdict = []
    if results["A_short_cached_back_to_back"]["identical"]:
        verdict.append("the seed IS honoured (A reproduces)")
    if not results["B_long_cached_back_to_back"]["identical"]:
        verdict.append("long prompts do not reproduce even back to back")
    elif not results["C_long_cached_with_another_prompt_between"]["identical"]:
        verdict.append("prefix reuse across a changed cache state is what breaks it (B ok, C not)")
        if results["D_long_uncached_with_another_prompt_between"]["identical"]:
            verdict.append("cache_prompt=False RESTORES determinism (D reproduces)")
        else:
            verdict.append("cache_prompt=False does NOT restore it (D also differs)")

    payload = {
        "what_this_is": __doc__.strip().splitlines()[0],
        "port": port,
        "seed": SEED,
        "n_predict": N_PREDICT,
        "prompt_games": [g1, g2],
        "prompt_chars": {"short": len(short), "long1": len(long1), "long2": len(long2)},
        "conditions": results,
        "verdict": verdict or ["inconclusive"],
        "caveat": "One replicate per condition. A single match is weak evidence of determinism "
        "(two draws can agree by chance when the distribution is peaked); a single "
        "MISMATCH is strong evidence of nondeterminism, because identical inputs cannot "
        "legitimately differ. Read the mismatches, not the matches.",
    }
    (OUT / "determinism_probe.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
