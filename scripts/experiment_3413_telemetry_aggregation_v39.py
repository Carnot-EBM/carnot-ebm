import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from carnot.reporting.telemetry_aggregation_3413 import generate_telemetry_aggregation_v39

def main():
    print("Running Exp 3413 Telemetry Aggregation v39...")
    data = generate_telemetry_aggregation_v39()
    print("Deliverable written successfully.")
    print("Data:", data)

if __name__ == "__main__":
    main()
