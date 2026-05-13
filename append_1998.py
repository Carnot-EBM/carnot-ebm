import os

with open("openspec/capabilities/verification/spec.md", "a") as f:
    f.write("\n### REQ-VERIFY-1998: Live IT Baselines with GSM8K\n\n")
    f.write("The repository shall establish real baselines with instruction-tuned models on GSM8K using the new SMT extractor.\n")
    f.write("- Runs 200 GSM8K questions.\n")
    f.write("- Uses `inference_mode=\"live_gpu\"` in all results.\n")
    f.write("- Calculates TP and FP rates.\n")
    f.write("- Writes `results/experiment_1998_live_it_baselines_gsm8k.json` artifact.\n\n")
    f.write("### SCENARIO-VERIFY-1998: Run GSM8K Baseline\n")
    f.write("Given 200 GSM8K questions and the new SMT extractor,\n")
    f.write("When the models run inference with `live_gpu` mode,\n")
    f.write("Then the baselines TP and FP rates are calculated and saved to the JSON artifact.\n")
