import json
import os
import subprocess

def check_rocminfo():
    try:
        result = subprocess.run(["/opt/rocm/bin/rocminfo"], capture_output=True, text=True, timeout=10)
        return result.stdout
    except Exception as e:
        try:
            result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
            return result.stdout
        except Exception as e:
            return str(e)

def check_jax():
    try:
        result = subprocess.run(
            [".venv/bin/python", "-c", "import jax; print(jax.devices())"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return str(e)

def main():
    rocminfo_output = check_rocminfo()
    jax_output = check_jax()

    # We know gfx1100 is not in rocminfo in current state based on previous manual check
    is_egpu_detected = "gfx1100" in rocminfo_output

    if is_egpu_detected:
        honest_verdict = "egpu_detected_successfully"
    else:
        honest_verdict = "hardware_not_detected_egpu_missing"

    artifact = {
        "schema": "carnot.hardware.v1",
        "experiment": 1991,
        "honest_verdict": honest_verdict,
        "rocminfo_contains_gfx1100": is_egpu_detected,
        "jax_devices": jax_output
    }

    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1991_egpu_rocm.json", "w") as f:
        json.dump(artifact, f, indent=2)

if __name__ == "__main__":
    main()
