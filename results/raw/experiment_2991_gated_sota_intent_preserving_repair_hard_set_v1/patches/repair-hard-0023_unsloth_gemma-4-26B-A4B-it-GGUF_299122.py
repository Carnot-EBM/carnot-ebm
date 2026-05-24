def parse_kv_pairs(text):
    out = {}
    for part in text.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            out[key.strip()] = value.strip()
    return out
