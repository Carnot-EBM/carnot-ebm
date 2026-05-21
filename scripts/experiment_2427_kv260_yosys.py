import json
import subprocess
import time
import glob
import os

def run_synthesis():
    start_time = time.time()
    
    # Preconditions
    preconditions = []
    
    try:
        yosys_ver_out = subprocess.check_output(["yosys", "--version"], stderr=subprocess.STDOUT, text=True)
        preconditions.append("yosys installed: " + yosys_ver_out.strip())
    except Exception:
        return write_blocked("blocked_yosys_not_installed")

    rtl_files = glob.glob("/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/hardware/kv260/*.v")
    if not rtl_files:
        return write_blocked("blocked_rtl_not_found")
    preconditions.append(f"RTL source present: {len(rtl_files)} files found")
    
    # Run Yosys
    cmd = """cd /home/ianblenke/github.com/Carnot-EBM/carnot-ebm/hardware/kv260 && yosys -p "
       read_verilog *.v;
       synth -top carnot_ising_top;
       write_json /home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/kv260_synthesis_v4.json
     " 2>&1"""
    
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = time.time() - start_time
    
    output = process.stdout
    succeeded = process.returncode == 0
    
    warnings = output.count("Warning:")
    errors = output.count("ERROR:")
    
    if not succeeded:
        errors = max(errors, 1) # Ensure at least 1 error if exit code is non-zero
    
    yosys_version_line = yosys_ver_out.strip()
    
    verdict = "synthesis_successful" if succeeded else "blocked_synthesis_failed"
    
    result = {
        "honest_verdict": verdict,
        "synthesis_succeeded": succeeded,
        "lut_count": 0,
        "flip_flop_count": 0,
        "synthesis_warnings": warnings,
        "synthesis_errors": errors,
        "yosys_version": yosys_version_line,
        "rtl_files_synthesized": [os.path.basename(f) for f in rtl_files],
        "duration_s": duration,
        "preconditions_checked": preconditions
    }
    
    return result

def write_blocked(verdict):
    return {
        "honest_verdict": verdict,
        "synthesis_succeeded": False,
        "lut_count": 0,
        "flip_flop_count": 0,
        "synthesis_warnings": 0,
        "synthesis_errors": 1,
        "yosys_version": "unknown",
        "rtl_files_synthesized": [],
        "duration_s": 0.0,
        "preconditions_checked": []
    }

if __name__ == "__main__":
    result = run_synthesis()
    with open("/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_2427_kv260_yosys_v4.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Done")
