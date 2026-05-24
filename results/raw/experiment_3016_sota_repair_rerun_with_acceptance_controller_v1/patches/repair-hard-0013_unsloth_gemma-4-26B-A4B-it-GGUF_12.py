def first_non_none(values, default=None):
    for val in values:
        if val is not None:
            return val
    return default
