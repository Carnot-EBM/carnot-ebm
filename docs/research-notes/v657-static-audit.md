# V657 static audit

Run date: 20260922
Verdict: `complete_disqualified_required_validation` (`disqualified`)

The audit independently reproduced the held-out static rows. It did not load a model.
The evidence is exploratory because the source corpus had prior exposure.

## Qualified outcomes

- Complete accounting: 1
- Scientific rows qualified: 0
- Static probability value: 0
- Selective decision value: 0

Both value scores are separate. A failure of benefit does not invalidate the completed audit.

## Independent reduction

- Source groups: 116
- Probability Holm family: 2
- Decision Holm family: 9
- Probability contrasts: `{"brier": {"identical_ten_feature_logistic": {"ci95": [-0.009082628152956082, 0.020036630236617548], "delta": 0.004996420432914351, "draws": 2000, "group_count": 116, "holm_adjusted_p": 1.0, "holm_alpha": 0.025, "holm_rank": 1, "holm_upper": 0.020036630236617548, "one_sided_p": 0.760119940029985, "seed": 657007}, "whole_only_gibbs": {"ci95": [0.005309722554895263, 0.03433459012302738], "delta": 0.018786007519536043, "draws": 2000, "group_count": 116, "holm_adjusted_p": 1.0, "holm_alpha": 0.05, "holm_rank": 2, "holm_upper": 0.031598554037311204, "one_sided_p": 0.9965017491254373, "seed": 657007}}, "log_loss": {"ci95": [-0.07837670159302937, 0.11547182314534493], "control": "raw_whole_expectation", "delta": 0.01816710641354497, "draws": 2000, "group_count": 116, "one_sided_p": 0.6276861569215393, "seed": 657007}}`
- Decision cells recomputed: 9

Private mutations covered label, option order, source, seed, escalation, checkpoint,
multiplicity, and descriptive-scope corruption. Corrupted fixtures are not published.

## Gate disposition

The ledger preserves failed gates rather than repairing upstream JSON or changing thresholds.
Failed gates: `[{"category": "validity", "check": "required_current_validation", "expected": true, "field_path": "validation_receipts", "observed": false, "op": "eq", "upstream": "current_audit"}, {"category": "readiness", "check": "scientific_claims_qualified", "expected": 1, "field_path": "static_claims_qualified_score", "observed": 0, "op": "eq", "upstream": "Exp7504/7505/7507"}, {"category": "benefit", "check": "qualified_probability_value", "expected": 1, "field_path": "qualified_static_probability_value_score", "observed": 0, "op": "eq", "upstream": "Exp7507 raw evaluation rows"}, {"category": "benefit", "check": "qualified_selective_decision_value", "expected": 1, "field_path": "qualified_selective_decision_value_score", "observed": 0, "op": "eq", "upstream": "Exp7507 raw policy rows"}]`
