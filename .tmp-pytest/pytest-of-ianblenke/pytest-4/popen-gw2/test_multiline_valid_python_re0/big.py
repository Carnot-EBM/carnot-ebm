from dataclasses import dataclass

@dataclass
class Foo:
    x: int
    y: float

def compute(a: int, b: int) -> int:
    return a + b
