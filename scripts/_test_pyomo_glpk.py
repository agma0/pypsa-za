import pyomo.environ as pyo
from pyomo.opt import SolverFactory

model = pyo.ConcreteModel()

model.x = pyo.Var(within=pyo.NonNegativeReals)
model.y = pyo.Var(within=pyo.NonNegativeReals)

model.obj = pyo.Objective(expr=2*model.x + 3*model.y)

model.con1 = pyo.Constraint(expr=3*model.x + 4*model.y >= 1)
model.con2 = pyo.Constraint(expr=2*model.x + model.y >= 1)

solver = SolverFactory('cbc')

solver.solve(model)

print('x = ', pyo.value(model.x))
print('y = ', pyo.value(model.y))