"""GCoT Branching Sampler: reasoning paths using branching and error backtracking.

**Researcher summary:**
    Implements a branching mechanism for Generative Chain of Thought (GCoT).
    It maintains parallel latent reasoning traces, ranks them using partial-trace
    energy, and culls those exceeding an energy threshold. If all branches
    exceed the threshold, it backtracks to a previous stable state.

Spec: REQ-INFER-1958
"""

from typing import Callable, List, Optional
import dataclasses

@dataclasses.dataclass
class ReasoningBranch:
    """A single reasoning branch in the GCoT process."""
    trace_id: str
    content: str
    energy: float
    parent_id: Optional[str] = None
    step_depth: int = 0

class GCoTBranchingSampler:
    """Maintains parallel latent reasoning traces, ranks and culls branches based on partial-trace energy."""
    
    def __init__(self, energy_fn: Callable[[str], float], energy_threshold: float, max_branches: int = 3):
        self.energy_fn = energy_fn
        self.energy_threshold = energy_threshold
        self.max_branches = max_branches
        self.branches: List[ReasoningBranch] = []
        self.history: List[List[ReasoningBranch]] = []
        self.step_counter = 0

    def initialize(self, initial_content: str) -> None:
        """Initialize the sampler with a starting content."""
        self.step_counter = 0
        energy = self.energy_fn(initial_content)
        initial_branch = ReasoningBranch(
            trace_id=f"step_{self.step_counter}_branch_0",
            content=initial_content,
            energy=energy,
            step_depth=0
        )
        self.branches = [initial_branch]
        self.history = [[initial_branch]]

    def step(self, candidate_extensions: List[str]) -> None:
        """Advance branches using candidate extensions."""
        self.step_counter += 1
        new_branches = []

        # Generate new branches from existing ones
        for branch in self.branches:
            for i, ext in enumerate(candidate_extensions):
                new_content = branch.content + " " + ext
                energy = self.energy_fn(new_content)
                new_branches.append(ReasoningBranch(
                    trace_id=f"step_{self.step_counter}_parent_{branch.trace_id}_ext_{i}",
                    content=new_content,
                    energy=energy,
                    parent_id=branch.trace_id,
                    step_depth=self.step_counter
                ))

        # Rank and cull
        # Sort by energy ascending (lower is better)
        new_branches.sort(key=lambda b: b.energy)
        
        # Filter by threshold
        valid_branches = [b for b in new_branches if b.energy <= self.energy_threshold]

        if not valid_branches:
            # Backtrack
            self.backtrack()
            return

        # Keep top max_branches
        self.branches = valid_branches[:self.max_branches]
        self.history.append(self.branches)

    def backtrack(self) -> None:
        """Backtrack to the previous state with valid branches."""
        if len(self.history) > 1:
            # Remove current failing state (if any) and revert to previous
            self.history.pop()
            self.branches = self.history[-1]
            self.step_counter = len(self.history) - 1
        else:
            self.branches = []
            self.step_counter = 0
