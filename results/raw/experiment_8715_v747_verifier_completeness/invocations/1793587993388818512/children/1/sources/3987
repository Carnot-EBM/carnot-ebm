from carnot.pipeline.verification_learning import VerificationLearningProxy
from carnot.paths import results_path

# Initialize with sample constraint
constraints = [{"type": "must_contain", "value": "test"}]
proxy = VerificationLearningProxy(constraints=constraints)

# Unlabelled dummy data
unlabelled_data = [
    {"id": "gen_1", "text": "This is a test generation"},
    {"id": "gen_2", "text": "This fails the requirement"},
    {"id": "gen_3", "text": "Another test case"},
]

# Write to the specified results directory
# Resolved via the central resolver, not hardcoded -- see python/carnot/paths.py.
result_path = str(results_path("experiment_1854_vl_proxy.json"))
proxy.run_experiment_and_save(unlabelled_data, result_path)

print(f"Successfully wrote results to {result_path}")
