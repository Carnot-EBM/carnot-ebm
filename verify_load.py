
import sys
from huggingface_hub import hf_hub_download
import torch

try:
    path = hf_hub_download(repo_id="Carnot-EBM/ThinkPRM-v2", filename="prmv2_fover_1508_checkpoint.pt")
    print(f"Downloaded to {path}")
    # Verify load
    data = torch.load(path, map_location="cpu", weights_only=False)
    print("Successfully loaded model checkpoint")
    sys.exit(0)
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)
