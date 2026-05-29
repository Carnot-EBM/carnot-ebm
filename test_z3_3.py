import z3

solver = z3.Solver()
x = z3.Int('x')
y = z3.Int('y')
solver.add(x > 0)
solver.add(y < 0)

solver.push()
solver.add(x < 0)
solver.check()
print("1:", solver.statistics().get_key_value('rlimit count'))
solver.pop()

solver.add(z3.Not(x < 0))

solver.push()
solver.add(x < 0)
solver.check()
print("2:", solver.statistics().get_key_value('rlimit count'))
solver.pop()
