from carnot.reporting.count_semantics_8566 import corpus,classify
rows=classify(corpus())
assert all(not r['old_critical'] for r in rows if r['expected']=='valid_discrete_identity')
