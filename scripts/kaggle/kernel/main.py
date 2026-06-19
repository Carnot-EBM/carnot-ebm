# Carnot ARC submission — Kaggle env probe (settles GPU/VRAM/CUDA before the MTP binary build)
import json, subprocess, os
def run(c): 
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e}"
rep = {}
rep["nvidia_smi"] = run("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader")
rep["n_gpus"] = run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
rep["cuda_version_nvcc"] = run("nvcc --version 2>/dev/null | grep release || echo no-nvcc")
rep["cuda_toolkit"] = run("ls -d /usr/local/cuda* 2>/dev/null; cat /usr/local/cuda/version.json 2>/dev/null | head -c 200")
rep["cmake"] = run("cmake --version 2>/dev/null | head -1 || echo no-cmake")
rep["llama_cpp_python_preinstalled"] = run("python -c 'import llama_cpp; print(llama_cpp.__version__)' 2>/dev/null || echo not-preinstalled")
rep["python"] = run("python --version")
rep["disk_free"] = run("df -h /kaggle/working | tail -1")
# verdict
smi = rep["nvidia_smi"].splitlines()[0] if rep["nvidia_smi"] else ""
total_mb = 0
try: total_mb = int(smi.split(", ")[1].replace(" MiB","").strip())
except Exception: pass
rep["VERDICT"] = {
    "gpu_is_L4_24GB": total_mb > 20000,
    "gpu_is_T4_16GB": 14000 < total_mb < 18000,
    "total_vram_MB": total_mb,
    "qwen35_9b_mtp_fits_with_q8kv": True if total_mb >= 15000 else "TIGHT",  # validated footprint ~11.5GB
}
os.makedirs("/kaggle/working", exist_ok=True)
open("/kaggle/working/kaggle_env_probe.json","w").write(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
