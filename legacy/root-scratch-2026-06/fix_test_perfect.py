import re

with open("tests/python/test_experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.py") as f:
    content = f.read()

# 1. Fix smoke test required fields
content = content.replace('"honest_verdict", "inference_substrate", "n_steps", "fresh_corpus_used",', '"honest_verdict", "inference_substrate", "n_steps", "initial_true_accuracy", "nondegenerate_corpus_gate_passed",')
content = content.replace('"honest_verdict", "inference_substrate", "n_steps",\n            "conservative_default_beta"', '"honest_verdict", "inference_substrate", "n_steps", "initial_true_accuracy", "nondegenerate_corpus_gate_passed",\n            "conservative_default_beta"')

# 2. Fix test_quality_not_maintained_gives_tuning_verdict
content = content.replace('assert "over_regularizes" in res["honest_verdict"]', 'assert "degrades_real_quality" in res["honest_verdict"]')

# 3. Fix test_deploy_prevents_collapse_control_collapses expected verdict
content = content.replace(
    'assert res["honest_verdict"] == (\n            "complete: conservative_default_beta_deploys_end_to_end_prevents_collapse_"\n            "to_N200_quality_maintained"\n        )',
    'assert res["honest_verdict"] == (\n            "complete: conservative_default_beta_deploys_on_nondegenerate_corpus_prevents_collapse_"\n            "to_N200_real_quality_maintained"\n        )'
)

# 4. Make sure test_deploy_prevents_collapse_control_collapses actually passes the quality gate
content = content.replace('"final_true_accuracy": 0.12,', '"final_true_accuracy": 0.38,')

with open("tests/python/test_experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.py", "w") as f:
    f.write(content)

