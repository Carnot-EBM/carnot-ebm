# Contradiction escalation

Cheap detectors for rows that disagree with THEMSELVES, escalated to an adversarial
reviewer when they fire. Never edits, never blocks. Detections are facts; verdicts
downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on evidence the reviewer
could not have read -- do NOT act on those.

Inspected 1, found 1.

## h2h_shard_qwen38_27b.jsonl#0

_Claim:_ This row reports success (induce_ok=True) while `cell_recall` is exactly 0.0. Explain how both can be true. Check specifically whether the produced artifact is STRUCTURALLY incompatible with the scorer -- wrong function arity, wrong argument order, wrong return type -- such that every scored item raises and is skipped, leaving the metric computed over an empty set. Quote the scorer's call site and the artifact's signature.

_(not escalated)_
