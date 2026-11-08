def strong(payload: Json) -> str:
    """Enumerate shared assignments over signed records; no absent atom becomes a false fact."""
    original.parse(payload)
    signs: dict[tuple[str, ...], set[bool]] = {}
    for s, r, o, sign in payload['facts']:
        _charge('record_visits')
        (_charge('lookup_attempts'), signs.setdefault((s, r, o), set()))[1].add(sign)
    if any(((_charge('fact_comparisons'), len(v) > 1)[1] for v in signs.values() if _charge('other_iterations'))):
        return ('insufficient', None)
    variables = sorted({t for q in payload['query'] if _charge('query_visits') for t in q if _charge('other_iterations') if t.startswith('?')})
    domain = sorted({row[i] for kind in ('facts', 'query') if _charge('record_visits') for row in payload[kind] if _charge('other_iterations') for i in (0, 2) if _charge('other_iterations') if not row[i].startswith('?')})
    possible = False
    for assignment in product(domain, repeat=len(variables)):
        _charge('candidate_bindings')
        binding = dict(zip(variables, assignment, strict=True))
        states = [(_charge('lookup_attempts'), signs.get(tuple(((_charge('lookup_attempts'), binding.get(t, t))[1] for t in q if _charge('other_iterations'))), set()))[1] for q in payload['query'] if _charge('query_visits')]
        if all(((_charge('fact_comparisons'), v == {True})[1] for v in states if _charge('other_iterations'))):
            return ('support', binding)
        possible |= not any(((_charge('fact_comparisons'), v == {False})[1] for v in states if _charge('other_iterations')))
    return ('insufficient' if possible else 'refute', None)
