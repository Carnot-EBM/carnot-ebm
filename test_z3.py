import z3

solver = z3.Solver()
x = z3.Int('x')
y = z3.Int('y')
solver.add(x > 0)
solver.add(y < 0)

solver.push()
solver.add(x < 0)
print(solver.check())
stats = solver.statistics()
print(list(stats.keys()))
solver.pop()
