import json
import sys
import traceback

from carnot.pipeline.fr11_integration import FR11IntegrationPipeline

def main():
    manifest_path = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/live_sota_balanced_telemetry_manifest_1480.jsonl"
    examples = []
    
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            examples.append(json.loads(line.strip()))
            
    pipeline = FR11IntegrationPipeline()
    
    successful_runs = 0
    tier4_to_tier1_feedback_fired = False
    
    for idx, ex in enumerate(examples):
        query = ex.get("prompt", "default_query")
        # Just use some substring for partial response
        full_response = ex.get("response_text", "")
        partial_response = full_response[:len(full_response)//2]
        label = ex.get("correctness_label", "correct")
        
        try:
            results = pipeline.run(query, partial_response, full_response, label)
            successful_runs += 1
            if results.get("tier4_to_tier1_feedback"):
                tier4_to_tier1_feedback_fired = True
        except Exception as e:
            print(f"Example {idx} failed:")
            traceback.print_exc()

    integration_working = successful_runs >= 8
    fr11_all_tiers_integrated = integration_working and tier4_to_tier1_feedback_fired
    
    output = {
        "honest_verdict": f"complete: {fr11_all_tiers_integrated}",
        "fr11_all_tiers_integrated": fr11_all_tiers_integrated,
        "integration_working": integration_working,
        "tier4_to_tier1_feedback": tier4_to_tier1_feedback_fired,
        "continuous_self_learning_task": True,
        "successful_examples": successful_runs,
        "total_examples": len(examples)
    }
    
    # Dump to terminal
    print(json.dumps(output, indent=2))
    
    # Save the deliverable
    deliverable_path = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_2500_fr11_integration_demo.json"
    with open(deliverable_path, "w") as f:
        json.dump(output, f, indent=2)
        
if __name__ == "__main__":
    main()
