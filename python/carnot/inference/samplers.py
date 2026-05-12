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

class NISampler:
    """Neural Indicator (NI) Sampling for discrete diffusion.
    
    Optimizes token resolution order by using an energy-based indicator step
    to dynamically determine which tokens to denoise first, accelerating
    discrete diffusion convergence.
    """
    
    def __init__(self, indicator_fn: Callable[[List[int], int], float]):
        """
        Args:
            indicator_fn: Function that computes the indicator (energy/loss proxy) 
                          for denoising a specific token at `token_idx` given `current_sequence`.
                          Lower indicator values mean higher priority to denoise.
        """
        self.indicator_fn = indicator_fn
        
    def determine_order(self, current_sequence: List[int], mask: List[bool]) -> List[int]:
        """
        Determine the order in which to denoise tokens based on the indicator function.
        
        Args:
            current_sequence: The current token sequence.
            mask: Boolean mask indicating which tokens are currently noised (True = noised).
            
        Returns:
            List of indices sorted by priority (first to denoise).
        """
        priorities = []
        for idx, is_noised in enumerate(mask):
            if is_noised:
                # Compute indicator for this specific token
                indicator_val = self.indicator_fn(current_sequence, idx)
                priorities.append((idx, indicator_val))
                
        # Sort indices by indicator value (ascending, lower energy/loss first)
        priorities.sort(key=lambda x: x[1])
        return [idx for idx, _ in priorities]
        
    def sample(self, initial_sequence: List[int], denoise_fn: Callable[[List[int], int], int]) -> List[int]:
        """
        Denoise the sequence using the Neural Indicator order.
        
        Args:
            initial_sequence: The initial fully or partially noised sequence.
            denoise_fn: Function to denoise a single token at a given index.
            
        Returns:
            The fully denoised sequence.
        """
        current_sequence = list(initial_sequence)
        # Assume 0 is the noise token for this mockup
        mask = [token == 0 for token in current_sequence]
        
        # Determine optimal token order using NI
        order = self.determine_order(current_sequence, mask)
        
        for idx in order:
            current_sequence[idx] = denoise_fn(current_sequence, idx)
            mask[idx] = False
            
        return current_sequence

class RandomDiscreteDiffusionSampler:
    """Baseline Random Order Discrete Diffusion."""
    
    def sample(self, initial_sequence: List[int], denoise_fn: Callable[[List[int], int], int]) -> List[int]:
        import random
        current_sequence = list(initial_sequence)
        mask = [token == 0 for token in current_sequence]
        order = [idx for idx, is_noised in enumerate(mask) if is_noised]
        random.shuffle(order)
        
        for idx in order:
            current_sequence[idx] = denoise_fn(current_sequence, idx)
            mask[idx] = False
            
        return current_sequence

