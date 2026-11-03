"""Multi-turn code/constraint repair ladder.

Spec: REQ-VERIFY-3355, SCENARIO-VERIFY-3355
"""

from dataclasses import dataclass
from typing import Callable, Any

from carnot.pipeline.verify_repair import VerifyRepairPipeline, VerificationResult


@dataclass
class RepairLadderResult:
    """Result of a multi-turn repair ladder process."""
    initial_response: str
    final_response: str
    repaired: bool
    iterations: int
    history: list[VerificationResult]
    satisfiable_drift: float


class RepairLadder:
    """Multi-turn generation-verification repair loop where exact solver
    counterexamples are fed back into the model prompt.
    """
    def __init__(
        self,
        pipeline: VerifyRepairPipeline,
        max_iterations: int = 3,
        llm_caller: Callable[[str], str] | None = None
    ) -> None:
        """Initialize the repair ladder.
        
        Args:
            pipeline: Verification pipeline to use.
            max_iterations: Maximum number of repair turns.
            llm_caller: Optional callable(prompt) -> str for LLM repair.
                If None, and pipeline has a model, pipeline._generate is used.
        """
        self.pipeline = pipeline
        self.max_iterations = max_iterations
        self._llm_caller = llm_caller

    def _generate(self, prompt: str) -> str:
        """Call the LLM to generate a response."""
        if self._llm_caller is not None:
            return self._llm_caller(prompt)
        if self.pipeline.has_model:
            # pylint: disable=protected-access
            return self.pipeline._generate(prompt)
        return "[no repair — CI mode]"

    def repair(self, question: str, initial_response: str, domain: str | None = None) -> RepairLadderResult:
        """Run the repair ladder loop up to max_iterations.
        
        Args:
            question: The original question or task description.
            initial_response: The model's baseline response.
            domain: The verification domain (e.g., 'math', 'code').
        
        Returns:
            RepairLadderResult capturing the repair trace and outcome.
        """
        history = []
        current_response = initial_response
        repaired = False
        satisfiable_drift = 0.0

        for i in range(self.max_iterations + 1):
            ver_result = self.pipeline.verify(question, current_response, domain=domain)
            history.append(ver_result)
            
            if ver_result.verified:
                if i > 0:
                    repaired = True
                break

            if i == self.max_iterations:
                break
                
            # Feed exact solver counterexamples back
            counterexamples = [
                f"- Constraint {v.constraint_type}: {v.description} (Satisfied: {v.metadata.get('satisfied', False)})"
                for v in ver_result.violations
            ]
            
            prompt = (
                f"Question: {question}\n\n"
                f"Your previous answer: {current_response}\n\n"
                f"The exact solver counterexamples found are:\n"
                f"{chr(10).join(counterexamples)}\n\n"
                f"Please retry generation and fix the constraints."
            )
            
            new_response = self._generate(prompt)
            if new_response == current_response or new_response == "[no repair — CI mode]":
                break
                
            current_response = new_response
            satisfiable_drift += 0.1  # Arbitrary measurement for testing drift

        return RepairLadderResult(
            initial_response=initial_response,
            final_response=current_response,
            repaired=repaired,
            iterations=len(history) - 1,
            history=history,
            satisfiable_drift=satisfiable_drift
        )
