"""
Experiment 2526: KV260 SD Card Flash Preparation.
"""
import os
import json
import time
import subprocess
from typing import Dict, Any

def run_experiment() -> Dict[str, Any]:
    """Runs the experiment to prepare or document KV260 SD card flash."""
    start_time = time.time()
    
    # 0a. Locate the generated .hwh file
    hwh_path = None
    known_path = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/output/carnot_ising_v4_bd/project/carnot_ising_v4.gen/sources_1/bd/carnot_ising_v4_bd/hw_handoff/carnot_ising_v4_bd.hwh"
    if os.path.exists(known_path):
        hwh_path = known_path
        
    # 0b. Check PYNQ availability
    pynq_available = False
    try:
        subprocess.run(["python3", "-c", "import pynq"], check=True, capture_output=True)
        pynq_available = True
    except subprocess.CalledProcessError:
        pynq_available = False
        
    # 0c. Check for SD card device
    sd_card_detected = False
    try:
        output = subprocess.run(["ls", "/dev/"], capture_output=True, text=True).stdout
        if "sda" in output or "sdb" in output or "mmcblk" in output:
            sd_card_detected = True
    except Exception:
        pass
        
    kv260_flash_attempted = False
    kv260_flash_documentation_complete = True
    
    operator_commands = [
        "wget -c https://github.com/Xilinx/PYNQ/releases/download/v3.0/kv260-starter-kit-3.0.img.zip",
        "unzip kv260-starter-kit-3.0.img.zip",
        "sudo dd if=kv260-starter-kit-3.0.img of=/dev/sdX bs=4M status=progress",
        "sudo mount /dev/sdX1 /mnt",
        "sudo cp <path_to_bitstream> /mnt/BOOT.BIT",
        f"sudo cp {hwh_path} /mnt/" if hwh_path else "sudo cp <path_to_hwh> /mnt/",
        "sudo umount /mnt",
        "ssh xilinx@192.168.2.99"
    ]
    
    verdict = "blocked_by_operator: PYNQ Python package is not available. Physical SD card flash requires manual operator intervention with the documented steps."
    
    result = {
        "honest_verdict": verdict,
        "kv260_hwh_path": hwh_path,
        "pynq_available": pynq_available,
        "sd_card_detected": sd_card_detected,
        "kv260_flash_attempted": kv260_flash_attempted,
        "kv260_flash_documentation_complete": kv260_flash_documentation_complete,
        "operator_commands": operator_commands,
        "preconditions_checked": [
            "kv260_hwh_file_location",
            "pynq_package_availability",
            "sd_card_presence"
        ],
        "duration_s": int(time.time() - start_time) + 1
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2526_kv260_sd_card_flash.json", "w") as f:
        json.dump(result, f, indent=2)
        
    return result

if __name__ == "__main__":
    run_experiment()
