def probe(value):
    rejected = (
        value
        or False
    )
    if rejected:
        return 1
    return 0
assert probe(True) == 1
assert probe(False) == 0
