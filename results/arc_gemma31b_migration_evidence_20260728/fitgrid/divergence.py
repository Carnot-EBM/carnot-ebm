"""KV-CACHE FIDELITY: f16 vs q8_0, measured as GREEDY-DECODE DIVERGENCE.

WHY NOT induce_ok. The obvious quality metric -- does the generated world_model.py import and
score on held-out transitions -- turned out to be unmeasurable on this corpus at any affordable
budget: gemma-4-31B-it is a reasoning model, and on REAL 64x64 induce prompts it does not finish
thinking within 12288 tokens (3x the production budget), so BOTH arms emit zero code. Comparing
two empty outputs and reporting "no quality difference" would be a degenerate result dressed up
as a finding.

WHAT THIS MEASURES INSTEAD, and why it is strictly sharper for the question asked. Both arms
decode GREEDILY (temperature 0.0) from a byte-identical prompt with the same seed on the same
model file and the same card. The ONLY difference is the KV cache dtype. Under exact arithmetic
the two token streams would therefore be IDENTICAL forever. Any divergence is caused by
q8_0 quantization error in the attention keys/values, full stop -- there is no other free
variable. So:

  * identical output            -> q8_0 is LOSSLESS on this workload, not merely "near-lossless"
  * divergence at token N       -> N is a direct, quantitative fidelity budget: the model
                                   produces bit-identical reasoning for N tokens before the
                                   quantization error is large enough to flip a single argmax

This also avoids the trap of declaring "near-lossless" from an aggregate score that could hide a
large early divergence behind similar-looking summary statistics.

REPORTED HONESTLY: a late divergence is good news but is NOT proof of equal task quality --
after the first flipped token the two continuations are different texts and are no longer
comparable token-by-token. The divergence INDEX is the measurement; anything past it is
commentary.
"""

import json
import os

SCRATCH = os.path.dirname(os.path.abspath(__file__))
A_TAG = "QC_egpu_24576_f16_chat"
B_TAG = "QC_egpu_24576_q8_chat"


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main() -> None:
    da, db = os.path.join(SCRATCH, "gen", A_TAG), os.path.join(SCRATCH, "gen", B_TAG)
    if not (os.path.isdir(da) and os.path.isdir(db)):
        print(json.dumps({"error": "one or both arms missing", "f16": os.path.isdir(da),
                          "q8": os.path.isdir(db)}))
        return
    games = sorted(set(os.listdir(da)) & set(os.listdir(db)))
    rows = []
    for fn in games:
        a = open(os.path.join(da, fn)).read()
        b = open(os.path.join(db, fn)).read()
        cp = common_prefix_len(a, b)
        rows.append({
            "game": fn[:-4],
            "f16_chars": len(a),
            "q8_chars": len(b),
            "identical": a == b,
            "common_prefix_chars": cp,
            "prefix_fraction_of_shorter": round(cp / max(1, min(len(a), len(b))), 6),
            # the first place they differ, with a little context, so a reader can see WHAT flipped
            "first_divergence_context": {
                "f16": a[max(0, cp - 60):cp + 60],
                "q8": b[max(0, cp - 60):cp + 60],
            } if a != b else None,
        })
    n_ident = sum(r["identical"] for r in rows)
    out = {
        "arms": {"f16": A_TAG, "q8_0": B_TAG},
        "n_games": len(rows),
        "n_byte_identical": n_ident,
        "verdict": (
            "q8_0 KV is BIT-LOSSLESS vs f16 on this workload (all greedy streams byte-identical)"
            if n_ident == len(rows) and rows else
            "q8_0 KV DIVERGES from f16 -- see per-game common_prefix_chars for the fidelity budget"
        ),
        "per_game": rows,
    }
    json.dump(out, open(os.path.join(SCRATCH, "kv_divergence.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_game"}, indent=1))
    for r in rows:
        print(f"  {r['game']:6s} identical={r['identical']} "
              f"common_prefix={r['common_prefix_chars']}/{min(r['f16_chars'], r['q8_chars'])}")


if __name__ == "__main__":
    main()
