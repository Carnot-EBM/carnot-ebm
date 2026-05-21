import sys; sys.path.insert(0,'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline
pipeline = VerifyRepairPipeline()
res = pipeline.verify("What is 10+5?", "10 + 5 = 16.")
print("Verified:", res.verified, "Energy:", res.energy)
