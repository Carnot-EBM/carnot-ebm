class ZeroForgettingGate:
    """
    A strict promotion gate that requires zero forgetting before a newly learned policy is retained.
    """
    
    def evaluate(self, pre_failures: set, post_failures: set) -> bool:
        """
        Runs pre/post tests on replay buffer.
        Blocks update if any prior constraint is violated (i.e. introduced new failures).
        """
        new_failures = post_failures - pre_failures
        return len(new_failures) == 0
