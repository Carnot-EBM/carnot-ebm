def first_non_none(values, default=None):
    for v in values:
        if v is not None:
            return v
    return default
