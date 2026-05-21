from typing import Dict, List

class TruncProofLL1Parser:
    def __init__(self, grammar: Dict[str, List[List[str]]], start_symbol: str, max_budget: int):
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.max_budget = max_budget
        self.stack = [start_symbol]
        self.consumed_tokens = 0
        self._min_lengths = self._compute_min_lengths()

    def _compute_min_lengths(self) -> Dict[str, int]:
        lengths = {'e': 0}
        for rhs_list in self.grammar.values():
            for rhs in rhs_list:
                for sym in rhs:
                    if sym not in self.grammar and sym != 'e':
                        lengths[sym] = 1

        changed = True
        while changed:
            changed = False
            for nt, rhs_list in self.grammar.items():
                min_l = lengths.get(nt, float('inf'))
                for rhs in rhs_list:
                    l = sum(lengths.get(sym, float('inf')) for sym in rhs)
                    if l < min_l:
                        min_l = l
                        changed = True
                if min_l < lengths.get(nt, float('inf')):
                    lengths[nt] = min_l
        return lengths

    def min_tokens_needed(self, stack: List[str]) -> int:
        return sum(self._min_lengths.get(sym, float('inf')) for sym in stack)

    def is_terminal(self, sym: str) -> bool:
        return sym not in self.grammar and sym != 'e'

    def _try_consume(self, stack: List[str], token: str) -> List[str]:
        if not stack:
            return None
        
        top = stack[-1]
        rest = stack[:-1]

        if self.is_terminal(top):
            if token == top:
                return rest
            return None

        # Try all productions, pick first that works
        for rhs in self.grammar[top]:
            new_stack = rest + [s for s in reversed(rhs) if s != 'e']
            resulting_stack = self._try_consume(new_stack, token)
            if resulting_stack is not None:
                if self.consumed_tokens + 1 + self.min_tokens_needed(resulting_stack) <= self.max_budget:
                    return resulting_stack
        return None

    def consume(self, token: str) -> bool:
        if self.consumed_tokens >= self.max_budget:
            return False
            
        new_stack = self._try_consume(self.stack, token)
        if new_stack is not None:
            self.stack = new_stack
            self.consumed_tokens += 1
            return True
        return False

    def force_closing_tokens(self) -> List[str]:
        tokens = []
        while self.stack:
            top = self.stack.pop()
            if self.is_terminal(top):
                tokens.append(top)
                self.consumed_tokens += 1
            else:
                best_rhs = min(self.grammar[top], key=lambda rhs: sum(self._min_lengths[sym] for sym in rhs))
                self.stack.extend([s for s in reversed(best_rhs) if s != 'e'])
        return tokens

grammar = {
    'S': [['{', 'M', '}']],
    'M': [['e'], ['PAIR', 'M_TAIL']],
    'M_TAIL': [['e'], [',', 'PAIR', 'M_TAIL']],
    'PAIR': [['KEY', ':', 'VAL']],
    'KEY': [['"k"']],
    'VAL': [['"v"'], ['{', 'M', '}']]
}

parser = TruncProofLL1Parser(grammar, 'S', max_budget=7)
print("consume {", parser.consume('{'))
print("consume k", parser.consume('"k"'))
print("consume :", parser.consume(':'))
print("consume v", parser.consume('"v"'))
print("stack is:", parser.stack)
print("budget needed:", parser.min_tokens_needed(parser.stack))
print("closing:", parser.force_closing_tokens())
