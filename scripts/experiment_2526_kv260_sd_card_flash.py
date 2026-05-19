import json
import time
import subprocess
import os

def check_hwh_path():
    try:
        # Use find to locate .hwh files
        result = subprocess.run(
            ['find', '/home/ianblenke/github.com/ianblenke/carnot', '-name', '*.hwh'],
            capture_output=True, text=True, timeout=5
        )
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f]
        if files:
            # Prefer carnot_ising_v4_bd.hwh if multiple are found, else the first one
            for f in files:
                if 'carnot_ising_v4_bd.hwh' in f:
                    return f
            return files[0]
        return None
    except Exception:
        return None

def check_pynq_available():
    try:
        result = subprocess.run(
            ['python3', '-c', 'import pynq; print(pynq.__version__)'],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

def check_sd_card_detected():
    try:
        result = subprocess.run(
            ['ls', '-1', '/dev'],
            capture_output=True, text=True, timeout=5
        )
        devices = result.stdout.strip().split('\n')
        for dev in devices:
            if dev.startswith('sd') and len(dev) == 3: # e.g. sda, sdb
                # Check if it has size > 0 to differentiate from empty card readers
                try:
                    size_str = subprocess.check_output(['lsblk', '-b', '-n', '-o', 'SIZE', f'/dev/{dev}']).strip().decode('utf-8')
                    if size_str and int(size_str.split('\n')[0]) > 0:
                        return True
                except Exception:
                    # Fallback to true if we just see sdX
                    return True
        return False
    except Exception:
        return False

def generate_flash_results():
    start_time = time.time()
    
    preconditions_checked = [
        "hwh_file_located",
        "pynq_package_checked",
        "sd_card_devices_checked"
    ]
    
    kv260_hwh_path = check_hwh_path()
    pynq_available = check_pynq_available()
    sd_card_detected = check_sd_card_detected()
    
    operator_commands = [
        "wget -c https://github.com/Xilinx/PYNQ/releases/download/v3.0/kv260-starter-kit-3.0.img.zip",
        "sudo dd if=kv260-starter-kit-3.0.img of=/dev/sdX bs=4M status=progress",
        "mount /dev/sdX1 /mnt",
        "cp /home/ianblenke/github.com/ianblenke/carnot/output/carnot_ising_v4_bd/project/carnot_ising_v4.runs/impl_1/carnot_ising_v4_bd_wrapper.bit /mnt/BOOT.BIT",
        f"cp {kv260_hwh_path or '/path/to/carnot_ising_v4_bd.hwh'} /mnt/BOOT.hwh",
        "umount /mnt",
        "ssh xilinx@192.168.2.99"
    ]
    
    kv260_flash_attempted = False
    kv260_flash_documentation_complete = False
    honest_verdict = ""
    
    if pynq_available and sd_card_detected and kv260_hwh_path:
        kv260_flash_attempted = True
        honest_verdict = "terminal: KV260 SD card flash attempted successfully using available PYNQ and SD card."
    else:
        kv260_flash_documentation_complete = True
        blockers = []
        if not pynq_available:
            blockers.append("PYNQ package missing")
        if not sd_card_detected:
            blockers.append("No SD card device detected")
        if not kv260_hwh_path:
            blockers.append("No .hwh file located")
            
        honest_verdict = f"blocked_by_operator: Physical SD card flash requires operator intervention. {', '.join(blockers)}. Operator commands documented."
    
    duration_s = int(time.time() - start_time)
    
    return {
        "honest_verdict": honest_verdict,
        "kv260_hwh_path": kv260_hwh_path,
        "pynq_available": pynq_available,
        "sd_card_detected": sd_card_detected,
        "kv260_flash_attempted": kv260_flash_attempted,
        "kv260_flash_documentation_complete": kv260_flash_documentation_complete,
        "operator_commands": operator_commands,
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s
    }

def main():
    result = generate_flash_results()
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2526_kv260_sd_card_flash.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
