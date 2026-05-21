import sys; sys.path.insert(0, 'python')
from carnot.pipeline.verify_repair import VerifyRepairPipeline
pipeline = VerifyRepairPipeline()
res = pipeline.verify("What is 2+2?", "4")
print(type(res), res)
