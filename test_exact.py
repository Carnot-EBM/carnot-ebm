from carnot.pipeline.verify_repair import VerifyRepairPipeline
pipeline = VerifyRepairPipeline()
print(pipeline.repair("foo", "bar"))
