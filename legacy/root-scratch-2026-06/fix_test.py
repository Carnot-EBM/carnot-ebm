import re

with open("tests/python/test_experiment_3533_fr11_conservative_default_deploy_closed_loop.py") as f:
    content = f.read()

content = content.replace("3533", "3544")
content = content.replace("v1", "v2")
content = content.replace("closed_loop_v2", "nondegenerate_corpus_v2")
content = content.replace("QUALITY_DEGRADATION_TOLERANCE", "QUALITY_DEGRADATION_MULTIPLIER")

with open("tests/python/test_experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.py", "w") as f:
    f.write(content)

