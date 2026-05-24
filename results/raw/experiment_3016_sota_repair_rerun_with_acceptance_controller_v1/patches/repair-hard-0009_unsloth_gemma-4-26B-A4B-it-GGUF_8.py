def chunked(seq, n):
    return [list(seq[i:i+n]) for i in range(0, len(seq), n)]
