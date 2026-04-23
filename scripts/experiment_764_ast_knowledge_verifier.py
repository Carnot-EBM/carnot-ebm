#!/usr/bin/env python3
"""Experiment 764: AST Knowledge Verifier — Execution-Free Hallucination Detection.

**Researcher summary (arXiv 2601.19106):**
    LLMs frequently hallucinate Python API calls — they generate method names that
    sound plausible but do not exist in the library (e.g., ``json.parse()`` instead
    of ``json.loads()``).  These are Knowledge Conflicting Hallucinations (KCHs).
    Static AST analysis can detect them with 100% precision and 87.6% recall, with
    NO code execution required.  This experiment validates that ASTKnowledgeVerifier
    achieves the same precision/recall on a curated 50-snippet corpus and deploys it
    as a Tier 0d pre-filter in the ThreeTierPipeline.

**Why this matters for Carnot:**
    - Tier 0d fires BEFORE the Ising verifier (which is more expensive).
    - 100% precision means every detection is a confirmed error — zero wasted repair cycles.
    - Execution-free: safe to call on untrusted LLM output without sandbox overhead.
    - Complements the EBM energy approach: AST catches structural KCHs; EBM catches
      semantic/reasoning failures.

**Experiment design:**
    25 CORRECT snippets: standard Python code using real APIs from os, sys, json, re,
    math, random, collections, itertools, functools.  These must all pass (no violations).

    25 HALLUCINATED snippets: plausible-sounding but non-existent API calls from the
    same modules.  Each must be flagged as a violation.

    Metrics:
      precision = TP / (TP + FP)  — must be 1.0 (no false positives)
      recall    = TP / (TP + FN)  — target >= 0.75
      f1        = 2 * P * R / (P + R)

Spec: REQ-EXTRACT-035, REQ-EXTRACT-036, SCENARIO-EXTRACT-070, SCENARIO-EXTRACT-071,
      SCENARIO-EXTRACT-072
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allows running from repo root without pip install
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.extraction.ast_knowledge_verifier import (  # noqa: E402
    ASTKnowledgeVerifier,
    KnowledgeBase,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

# 25 CORRECT snippets — each uses a real Python standard-library API.
# Labels: True = correct (no KCH), False = hallucinated (KCH present).
CORRECT_SNIPPETS: list[tuple[str, bool]] = [
    # json
    ("import json\ndata = json.loads(text)", True),
    ("import json\ns = json.dumps(obj)", True),
    ("import json\nwith open(f) as fh:\n    data = json.load(fh)", True),
    ("import json\nwith open(f, 'w') as fh:\n    json.dump(obj, fh)", True),
    # os
    ("import os\nresult = os.path.join(a, b)", True),
    ("import os\nexists = os.path.exists(path)", True),
    ("import os\nos.makedirs(path, exist_ok=True)", True),
    ("import os\nfiles = os.listdir(path)", True),
    # sys
    ("import sys\nargs = sys.argv", True),
    ("import sys\nsys.exit(0)", True),
    # re
    ("import re\nm = re.match(pattern, text)", True),
    ("import re\nmatches = re.findall(pattern, text)", True),
    ("import re\ns = re.sub(pattern, repl, text)", True),
    # math
    ("import math\nresult = math.sqrt(x)", True),
    ("import math\nresult = math.floor(x)", True),
    ("import math\nresult = math.log(x)", True),
    # random
    ("import random\nitem = random.choice(lst)", True),
    ("import random\nrandom.shuffle(lst)", True),
    ("import random\nn = random.randint(1, 10)", True),
    # collections
    ("from collections import Counter\nc = Counter(lst)", True),
    ("from collections import defaultdict\nd = defaultdict(list)", True),
    # itertools
    ("import itertools\nchain = itertools.chain(a, b)", True),
    ("import itertools\nproduct = list(itertools.product(a, b))", True),
    # functools
    ("import functools\nresult = functools.reduce(fn, lst)", True),
    ("import functools\n@functools.lru_cache(maxsize=128)\ndef f(x): return x", True),
]

# 25 HALLUCINATED snippets — each has a plausible-sounding but non-existent API.
HALLUCINATED_SNIPPETS: list[tuple[str, bool]] = [
    # json hallucinations
    ("import json\ndata = json.parse(text)", False),
    ("import json\ns = json.stringify(obj)", False),
    ("import json\njson.write_file(obj, path)", False),
    ("import json\ndata = json.decode(text)", False),
    # os hallucinations
    ("import os\ndata = os.read_file(path)", False),
    ("import os\nos.delete(path)", False),
    ("import os\nos.copy_file(src, dst)", False),
    ("import os\nfiles = os.walk_files(path)", False),
    # sys hallucinations
    ("import sys\nsys.terminate(0)", False),
    ("import sys\npath = sys.get_path()", False),
    # re hallucinations
    ("import re\nmatches = re.findall_named(pattern, text)", False),
    ("import re\nm = re.search_all(pattern, text)", False),
    ("import re\ns = re.replace(pattern, repl, text)", False),
    # math hallucinations
    ("import math\nresult = math.relu(x)", False),
    ("import math\nresult = math.sigmoid(x)", False),
    ("import math\nresult = math.power(x, n)", False),
    # random hallucinations
    ("import random\nitem = random.choice_weighted(lst, weights)", False),
    ("import random\nn = random.sample_one(lst)", False),
    ("import random\nrandom.seed_from_entropy()", False),
    # collections hallucinations
    ("import collections\nc = collections.count(lst)", False),
    ("import collections\nd = collections.sorted_dict()", False),
    # itertools hallucinations
    ("import itertools\nresult = itertools.flatten(lst)", False),
    ("import itertools\npairs = itertools.zip_longest_strict(a, b)", False),
    # functools hallucinations
    ("import functools\nresult = functools.compose(f, g)", False),
    ("import functools\nf2 = functools.memoize(f)", False),
]

CORPUS: list[tuple[str, bool]] = CORRECT_SNIPPETS + HALLUCINATED_SNIPPETS

# Modules to introspect for the KnowledgeBase.
MODULES_TO_INTROSPECT = [
    "os", "sys", "json", "re", "math", "random", "collections", "itertools", "functools",
]


def run_evaluation() -> dict:
    """Build KB, run verifier on all 50 snippets, compute precision/recall/f1.

    Returns a dict with all experiment metrics suitable for tmpl.build_result().
    """
    # Build knowledge base by introspecting standard library modules.
    kb = KnowledgeBase.build_from_modules(MODULES_TO_INTROSPECT)
    verifier = ASTKnowledgeVerifier(kb)

    n_correct = sum(1 for _, label in CORPUS if label)
    n_hallucinated = sum(1 for _, label in CORPUS if not label)

    if n_correct + n_hallucinated < 20:
        return {
            "honest_verdict": "corpus_construction_failed",
            "n_correct_snippets": n_correct,
            "n_hallucinated_snippets": n_hallucinated,
        }

    tp = 0   # hallucinated snippet correctly flagged
    fp = 0   # correct snippet incorrectly flagged (false positive — must be 0!)
    fn = 0   # hallucinated snippet not flagged (false negative)

    violations_per_snippet: list[dict] = []

    for code, is_correct in CORPUS:
        violations = verifier.verify(code)
        violation_detected = len(violations) > 0

        violations_per_snippet.append({
            "code_preview": code[:60].replace("\n", " "),
            "label": "correct" if is_correct else "hallucinated",
            "violations": [
                {"module": v.module, "attr": v.attr, "violation_type": v.violation_type}
                for v in violations
            ],
            "violation_detected": violation_detected,
        })

        if is_correct and violation_detected:
            fp += 1   # false positive: correct code wrongly flagged
        elif not is_correct and violation_detected:
            tp += 1   # true positive: hallucination correctly caught
        elif not is_correct and not violation_detected:
            fn += 1   # false negative: hallucination missed

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Honest verdict as specified in the task.
    if n_correct + n_hallucinated < 20:
        honest_verdict = "corpus_construction_failed"
    elif precision < 1.0:
        honest_verdict = "false_positives_detected"  # unsafe — do NOT deploy
    elif recall >= 0.75:
        honest_verdict = "tier0d_viable"
    else:
        honest_verdict = "tier0d_partial"            # safe but incomplete

    tier0d_deployed = precision == 1.0 and recall >= 0.75

    return {
        "n_correct_snippets": n_correct,
        "n_hallucinated_snippets": n_hallucinated,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "violations_per_snippet": violations_per_snippet,
        "tier0d_deployed": tier0d_deployed,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=764,
        title="AST Knowledge Verifier — Execution-Free Hallucination Detection (arXiv 2601.19106)",
        deliverable="results/experiment_764_ast_knowledge_verifier.json",
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(
        experiment_id=764,
        timeout_minutes=30,
        result_path=str(_REPO / "results" / "experiment_764_ast_knowledge_verifier.json"),
    ):
        metrics = run_evaluation()

        artifact = tmpl.build_result(
            metrics,
            status="success",
            decision_class="detect",
            cost_usd=0.0,
        )

        out_path = _REPO / "results" / "experiment_764_ast_knowledge_verifier.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(artifact, fh, indent=2)

        print(f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}  honest_verdict={metrics['honest_verdict']}")
        print(f"Artifact written: {out_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
