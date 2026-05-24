def normalize_whitespace(text):
    import re
    # Replace all sequences of whitespace characters with a single space
    collapsed = re.sub(r'\s+', ' ', text)
    # Trim leading and trailing whitespace
    return collapsed.strip()
