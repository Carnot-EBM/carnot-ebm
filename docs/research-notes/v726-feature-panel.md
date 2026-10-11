# V726 frozen feature panel — 2026-10-11

Exp8430 reconstructs 900 released answers in 300 question clusters. It inherits
Exp8416's exact 150 fit, 50 tune and 100 test question IDs through Exp8429's
pinned protocol. Each question keeps its three generator answers and all four
released prompt conditions together. No partition is recomputed or reseeded.

The nine inputs are GPT-5-mini's released p2 hallucination probability, clipped
to [0.0001, 0.9999], followed by the eight public lexical measurements from
`evidence_features_7980.extract`. Answer text and `ref_str` come from reviewed
original CSV bytes. A reference can be incomplete or wrong. Word agreement
measures reference faithfulness; it does not establish independent factuality.
Labels never supply source text or replace missing observations.

Dataset and generator IDs are reporting strata only. Feature rows contain no
human targets. The trainer view contains fit and tune targets; the separate
evaluator custody file contains test targets. Public labels were previously
exposed, so this separation prevents accidental trainer leakage during this
run but does not undo historical exposure. Old question overlap remains unknown.

Usable support requires all three siblings in a question to have source, q and
primary target. Floors remain 128/32/80 questions, plus eight questions carrying
each class in tune and test. Missing inputs retain their intended row and
per-question exclusion. Insufficient support blocks later numeric fitting.

A fixed sample recomputes q and lexical reductions with separate arithmetic.
Cold replay reconstructs rows from independent protocol and release pins.
Private E2E-018/E2E-021 and unchanged publication, adversarial and strict-row
consumers qualify input custody. Readiness is reproducible input, not semantic
benefit. Both generalization scores remain zero. There are no model loads or
current LLM calls, no changed generator weights and no external publication.
