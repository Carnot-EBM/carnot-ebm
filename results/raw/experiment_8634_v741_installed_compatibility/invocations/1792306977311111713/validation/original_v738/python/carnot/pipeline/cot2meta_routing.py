import random
from typing import Callable, Dict, Any, List
from carnot.pipeline.fast_slow_variant import VerifyResult

def evaluate_odar_action(
    energy_expand: float, 
    energy_prune: float, 
    energy_repair: float, 
    energy_stop: float, 
    energy_fallback: float,
    risk_sensitivity: float = 1.0
) -> str:
    """
    Pick action via free-energy-principled fusion (ODAR rule):
    action = argmax over {expand, prune, repair, stop, fallback}
    with risk-sensitive weighting.
    """
    energies = {
        "expand": energy_expand,
        "prune": energy_prune,
        "repair": energy_repair,
        "stop": energy_stop,
        "fallback": energy_fallback * risk_sensitivity
    }
    return max(energies, key=energies.get)

def cot2meta_state_machine(
    prompt: str,
    base_llm: Callable[[str], str],
    verifier_ensemble: Callable[[str], VerifyResult],
    odar_router: Callable[[str, List[str], str], str],
    max_iters: int = 10,
) -> Dict[str, Any]:
    """
    CoT2-Meta routing framework on top of Fast-Slow Variant.
    Actions: expand, prune, repair, stop, fallback.
    """
    context_buffer: List[str] = []
    response = ""
    action_sequence = []
    fallback_triggered = False
    passed = False
    
    for i in range(max_iters):
        current_prompt = prompt
        if context_buffer:
            current_prompt += "\n\nFeedback from previous attempts:\n" + "\n".join(context_buffer)
            
        # 1. Decide action based on current state
        action = odar_router(response, context_buffer, current_prompt)
        action_sequence.append(action)
        
        if action == "stop":
            # ODAR-style risk-sensitive acceptance
            break
        elif action == "fallback":
            # escalate to k=16 ensemble disagreement check OR mark as "needs human review"
            fallback_triggered = True
            break
        elif action == "prune":
            # verifier rejection
            context_buffer.clear()
            response = base_llm(prompt)
        elif action in ("expand", "repair"):
            # generate next iteration's candidate OR repair with context
            response = base_llm(current_prompt)
            
        # 2. Verify response
        if action in ["expand", "repair", "prune"]:
            verify_result = verifier_ensemble(response)
            if verify_result.is_correct:
                passed = True
                break
            else:
                context_buffer.append(f"- {verify_result.failure_summary}")

    return {
        "response": response, 
        "iters": len(action_sequence), 
        "passed": passed, 
        "action_sequence": action_sequence,
        "final_action": action_sequence[-1] if action_sequence else "stop",
        "fallback_triggered": fallback_triggered
    }
