import sys; sys.path.insert(0,'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline
pipeline = VerifyRepairPipeline()
pipeline._model = True  # Mock has_model
pipeline._tokenizer = True
pipeline._generate = lambda prompt, **kwargs: "10 + 5 = 15."
result = pipeline.verify_and_repair(question="What is 10 + 5?", response="10 + 5 = 16.", domain="arithmetic")
print("Verified:", result.verified, "Repaired:", result.repaired, "Final:", result.final_response)
