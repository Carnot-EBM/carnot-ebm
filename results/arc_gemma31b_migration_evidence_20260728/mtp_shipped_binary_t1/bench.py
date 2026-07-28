"""Matched MTP on/off throughput bench against a llama-server /completion endpoint.

WHY this shape: an ARC induction is ~95-97% decode, so decode tok/s is the
end-to-end proxy. We hold prompt, n_predict, temperature and seed IDENTICAL
across the two server configurations so the ONLY difference is speculation.
We read the server's own `timings` block rather than wall-clock so that HTTP
overhead does not contaminate the decode rate.
"""
import json, sys, time, urllib.request

PORT = sys.argv[1]
LABEL = sys.argv[2]
OUT = sys.argv[3]

# Representative of the live workload: structured code emission for a grid
# transform, which is what ARC dynamics-induction actually asks the model for.
PROMPT = (
    "You are an expert Python programmer working on ARC-style grid puzzles.\n"
    "Write a single self-contained Python function `transform(grid)` that takes a "
    "list of lists of small integers and returns a new grid. The transform should: "
    "(1) find every connected region of non-zero cells using 4-connectivity, "
    "(2) compute each region's bounding box, "
    "(3) recolor each region by the count of cells it contains, "
    "(4) leave background zeros untouched.\n"
    "Include a docstring and inline comments explaining each step in plain language. "
    "Write complete, runnable code.\n\n```python\n"
)

REQ = {
    "prompt": PROMPT,
    "n_predict": 512,
    "temperature": 0.0,
    "top_k": 1,
    "seed": 1234,
    "cache_prompt": False,
}

def one_call():
    data = json.dumps(REQ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0

runs = []
for i in range(3):
    body, wall = one_call()
    t = body.get("timings", {})
    runs.append({
        "iter": i,
        "wall_s": round(wall, 3),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "predicted_per_second": t.get("predicted_per_second"),
        "prompt_n": t.get("prompt_n"),
        "draft_n": t.get("draft_n"),
        "draft_n_accepted": t.get("draft_n_accepted"),
        "content_sha_prefix": body.get("content", "")[:60],
        "content_len": len(body.get("content", "")),
    })
    print(json.dumps(runs[-1]), flush=True)

tps = [r["predicted_per_second"] for r in runs if r["predicted_per_second"]]
summary = {
    "label": LABEL,
    "port": PORT,
    "runs": runs,
    "median_tok_s": sorted(tps)[len(tps)//2] if tps else None,
    "mean_tok_s": sum(tps)/len(tps) if tps else None,
    "content_identical_across_runs": len({r["content_sha_prefix"] for r in runs}) == 1,
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "runs"}))
