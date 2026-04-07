"""Example functions for testing Carnot CLI verification."""


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclid's algorithm."""
    while b:
        a, b = b, a % b
    return a


def factorial(n: int) -> int:
    """Compute n! iteratively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def fibonacci(n: int) -> int:
    """Compute nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def buggy_add(a: int, b: int) -> int:
    """Deliberately buggy: off by one."""
    return a + b + 1
