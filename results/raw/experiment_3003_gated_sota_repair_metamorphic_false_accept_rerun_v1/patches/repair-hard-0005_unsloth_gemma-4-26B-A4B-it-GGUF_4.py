def flatten_once(items):
    out = []
    for item in items:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out
