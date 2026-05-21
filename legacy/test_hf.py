from huggingface_hub import HfApi
api = HfApi()
try:
    repos = api.list_repos(author="Carnot-EBM")
    for r in repos:
        print(r.id)
except Exception as e:
    print(e)
