import json
import os
import subprocess

def check_rocminfo():
    """Run rocminfo to get hardware state."""
    try:
        # First try to run rocminfo from PATH
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e1:
        try:
            # Fallback for rocm typical path
            result = subprocess.run(["/opt/rocm/bin/rocminfo"], capture_output=True, text=True, check=True)
            return result.stdout
        except Exception as e2:
            return f"error: {str(e2)}"

import re

def map_memory_limits(output):
    """Extract memory limits from rocminfo output."""
    limits = []
    lines = output.split('\n')
    current_agent = None
    
    for line in lines:
        if "Agent " in line:
            current_agent = line.strip()
        elif "Name:" in line and current_agent:
            current_agent += f" - {line.strip()}"
        elif "Size:" in line and "KB" in line:
            match = re.search(r'Size:\s+(\d+)', line)
            if match:
                size_kb = int(match.group(1))
                limits.append({"agent": current_agent, "size_kb": size_kb})
                
    if not limits:
        return {"raw_output": output}
    
    return {"parsed_pools": limits}

def main():
    rocminfo_output = check_rocminfo()
    memory_limits = map_memory_limits(rocminfo_output)
    
    if "error:" in rocminfo_output.lower() or "not found" in rocminfo_output.lower():
        verdict = "hardware missing but probe works"
    else:
        verdict = "hardware_probe_success"
        
    data = {
        "schema": "carnot.hardware.v1",
        "experiment": 2008,
        "honest_verdict": verdict,
        "rocminfo_output": rocminfo_output,
        "memory_limits": memory_limits
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2008_rocm_probe.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
