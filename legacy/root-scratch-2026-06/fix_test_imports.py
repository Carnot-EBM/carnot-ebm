import re

with open("tests/python/test_experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.py") as f:
    content = f.read()

content = content.replace("run_conservative_default_deploy_closed_loop", "run_conservative_default_deploy_nondegenerate_corpus_v2")

with open("tests/python/test_experiment_3544_fr11_conservative_default_deploy_nondegenerate_corpus_v2.py", "w") as f:
    f.write(content)

