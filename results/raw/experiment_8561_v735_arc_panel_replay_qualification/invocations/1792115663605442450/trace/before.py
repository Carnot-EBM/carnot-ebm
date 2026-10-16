def probe(value):
    if (
        value
        or False
    ):
        return 1
    return 0
assert probe(True) == 1
assert probe(False) == 0
