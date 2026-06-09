import re

with open("tests/python/test_experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.py") as f:
    content = f.read()

# Replace mock initial_true_accuracy values
content = content.replace('"initial_true_accuracy": 0.1,', '"initial_true_accuracy": 0.4,')
content = content.replace('"initial_true_accuracy": 0.2,', '"initial_true_accuracy": 0.4,')
content = content.replace('"initial_true_accuracy": 0.5,', '"initial_true_accuracy": 0.5,') # no change, but just in case
# Ensure final true accuracy meets the quality drop condition for tuning test
# In test_quality_not_maintained_gives_tuning_verdict: final_true_accuracy was 0.01
# Since initial is 0.4, 0.01 is well below 0.9 * 0.4 = 0.36

# Also need to fix n_correct/n_wrong in traces generator for the tests
content = content.replace("n_correct: int = 3, n_wrong: int = 7", "n_correct: int = 4, n_wrong: int = 6")
content = content.replace("n_correct=1, n_wrong=9", "n_correct=4, n_wrong=6")

with open("tests/python/test_experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.py", "w") as f:
    f.write(content)

