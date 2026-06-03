"""
Anomaly-Escalation (Deep Think P3 upgrade) — defuse the Verification Trap.

The autonomous loop auto-reconciles EVERY negative result as a dead-end. DT P3's
finding: that rigor is exactly what would silently prune a frame-breaking breadcrumb
(it buried the Thesis-A INFRA false-negative this session — a "bounded" verdict written
when the EBT trained for ZERO steps). This module distinguishes:

  - CLEAN bounded negative  -> auto-reconcile is fine (the method genuinely ran and
    produced an honest negative; e.g. EBT trained 16k steps and lost to AR fairly).
  - FRAME-VIOLATING anomaly  -> HALT silent pruning, ESCALATE to a human queue
    (ops/anomaly-escalations.md). The human triages "dead-end vs breadcrumb."

It does NOT edit verifiers or auto-fix anything (per the project's operator-only
discipline) and does NOT modify the conductor — it is a post-hoc scanner. It reuses
adversarial_verify for the fabrication/false-negative flags and adds the
"negative-verdict-but-the-method-didn't-actually-run" and invariant-regression checks
that the manual catches this session relied on.

CLI:
  python scripts/anomaly_escalation.py --scan --since-hours 24      # scan recent results
  python scripts/anomaly_escalation.py --classify results/x.json    # one artifact
"""
import os
import sys
import json
import glob
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

try:
    import adversarial_verify as av
except Exception:
    av = None

NEG_TOKENS = ("bounded", "negative", "blocked", "inconclusive", "rejected",
              "not_met", "not met", "no_improvement", "failed", "fail_", "halt", "stop")
# fields whose value of 0 / null means "the method may not have genuinely run"
DID_NOT_RUN_FIELDS = ("cumulative_steps_trained", "n_eval", "n_train", "flip_count",
                      "n_changed", "constraint_coverage", "steps", "n_samples")


def _is_negative(verdict: str) -> bool:
    v = (verdict or "").lower()
    return any(t in v for t in NEG_TOKENS)


def _method_did_not_run(d: dict):
    """Return a reason string if a load-bearing 'did the method run' field is 0/empty."""
    reasons = []
    for f in DID_NOT_RUN_FIELDS:
        if f in d and d[f] in (0, 0.0):
            # coverage==0 / steps==0 / n_eval==0 etc. on a negative claim is the red flag
            reasons.append(f"{f}=0")
    # blocked_* verdicts that nonetheless claim a finding
    pc = d.get("preconditions_checked")
    if isinstance(pc, dict) and any(v is False for v in pc.values()):
        reasons.append("a precondition was False (method may have been infra-blocked)")
    return reasons


def _invariant_regression(d: dict):
    reasons = []
    if d.get("paper_ready") is False or d.get("paper_ready_preserved") is False:
        reasons.append("paper_ready regressed to False")
    fh = d.get("frozen_headline_unchanged")
    if fh is False:
        reasons.append("frozen headline changed")
    # a frozen-headline-adjacent numeric that drifted off 0.9131
    for k, v in d.items():
        if "frozen" in k.lower() and "auroc" in k.lower():
            try:
                if abs(float(v) - 0.9131) > 0.02:
                    reasons.append(f"{k}={v} drifted from frozen 0.9131")
            except Exception:
                pass
    return reasons


def classify(path: str):
    """-> (category, reasons[]). category in:
       clean_negative | frame_violating_anomaly | success_or_other | unreadable"""
    try:
        d = json.load(open(path))
    except Exception as e:
        return "frame_violating_anomaly", [f"UNREADABLE/corrupt artifact ({e}) — breaks the gate"]
    if not isinstance(d, dict):
        return "success_or_other", ["not a verdict artifact (non-object JSON) — skipped"]
    verdict = d.get("honest_verdict", "")
    if isinstance(verdict, dict):                 # principle-annotated {value, principle}
        verdict = str(verdict.get("value", verdict))
    verdict = str(verdict)
    reasons = []

    # adversarial flags. TAUTOLOGY is the lowest-precision critical flag (often a
    # benign by-construction duplicate field, e.g. matched_ar==arV) — it must NOT
    # escalate on its own, or every clean result with a derived field false-fires.
    # Only the fabrication-class critical flags trigger escalation.
    BENIGN_CRITICAL = {"TAUTOLOGY"}
    is_aggregation = str(d.get("inference_substrate", "")).startswith("aggregation")
    crit, fnr = [], []
    if av is not None:
        try:
            from pathlib import Path
            res = av.verify_artifact(Path(path))
            for fl in res.get("flags", []):
                kind = fl.get("kind", ""); sev = (fl.get("severity") or "").lower()
                # aggregation artifacts are sub-second by design — DURATION_TOO_SHORT is benign there
                if kind == "DURATION_TOO_SHORT" and is_aggregation:
                    continue
                if sev == "critical" and kind not in BENIGN_CRITICAL:
                    crit.append(kind)
                if "FALSE_NEGATIVE_RISK" in kind or "CEILING" in kind:
                    fnr.append(kind)
        except Exception as e:
            reasons.append(f"adversarial_verify error: {e}")

    neg = _is_negative(verdict)
    didnt_run = _method_did_not_run(d) if neg else []
    inv = _invariant_regression(d)

    if inv:
        return "frame_violating_anomaly", [f"INVARIANT regression: {'; '.join(inv)}"] + reasons
    if neg and (crit or didnt_run):
        why = []
        if crit:
            why.append(f"CRITICAL adversarial flag(s) {crit} on a negative verdict (possible infra/fabrication artifact masquerading as a finding)")
        if didnt_run:
            why.append(f"method may not have genuinely run ({', '.join(didnt_run)}) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)")
        return "frame_violating_anomaly", why + reasons
    if neg and fnr:
        return "frame_violating_anomaly", [f"FALSE_NEGATIVE_RISK flag {fnr} on a null/negative verdict — verify a positive control ran"] + reasons
    if neg:
        return "clean_negative", ["honest negative; method appears to have run; auto-reconcile is fine"] + reasons
    if crit:
        return "frame_violating_anomaly", [f"CRITICAL adversarial flag {crit} on a non-negative verdict"] + reasons
    return "success_or_other", reasons


def scan(since_hours: float, apply: bool):
    cutoff = time.time() - since_hours * 3600
    paths = [p for p in glob.glob(os.path.join(PROJECT_ROOT, "results", "*.json"))
             if os.path.getmtime(p) >= cutoff]
    escalations = []
    for p in sorted(paths, key=os.path.getmtime):
        cat, reasons = classify(p)
        if cat == "frame_violating_anomaly":
            escalations.append((p, reasons))
    # de-dup: don't re-escalate artifacts already in the queue (the janitor re-runs)
    qpath = os.path.join(PROJECT_ROOT, "ops", "anomaly-escalations.md")
    already = set()
    if os.path.exists(qpath):
        qtext = open(qpath).read()
        already = {os.path.basename(p) for p, _ in escalations if os.path.basename(p) in qtext}
    escalations = [(p, r) for p, r in escalations if os.path.basename(p) not in already]
    print(f"scanned {len(paths)} artifacts (last {since_hours}h); "
          f"{len(escalations)} NEW frame-violating anomalies ({len(already)} already queued)")
    for p, reasons in escalations:
        print(f"  ESCALATE {os.path.basename(p)}: {reasons[0]}")
    if apply and escalations:
        with open(qpath, "a") as f:
            for p, reasons in escalations:
                f.write(f"\n## {os.path.basename(p)}\n")
                f.write("- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: "
                        "dead-end or breadcrumb?)\n")
                for r in reasons:
                    f.write(f"  - {r}\n")
                f.write(f"- artifact: {os.path.relpath(p, PROJECT_ROOT)}\n")
        print(f"-> appended {len(escalations)} to {os.path.relpath(qpath, PROJECT_ROOT)}")
    return escalations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--classify", type=str)
    ap.add_argument("--since-hours", type=float, default=24)
    ap.add_argument("--apply", action="store_true", help="write escalations to ops/anomaly-escalations.md")
    a = ap.parse_args()
    if a.classify:
        cat, reasons = classify(a.classify)
        print(json.dumps({"artifact": a.classify, "category": cat, "reasons": reasons}, indent=2))
    else:
        scan(a.since_hours, a.apply)


if __name__ == "__main__":
    main()
