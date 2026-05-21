import sys; sys.path.insert(0,'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline
p = VerifyRepairPipeline()
res = p.verify("What is 10+5?", "10 + 5 = 16.")
print("Constraints:", res.constraints)
print("Violations:", res.violations)
