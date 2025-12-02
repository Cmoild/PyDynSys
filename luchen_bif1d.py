import numpy as np
from matplotlib import pyplot as plt
import dynsys
import time

dt = 0.01
initial = [0.1, 0.3, -0.6]

dsl = """
x' = a * (y - x)
y' = (1 - z) * x + c * y + u
z' = x * y - b * z
"""

params = {"a": 36.0, "b": 3.0, "c": 20.0, "u": 0}

mod = dynsys.model.Model(dt=dt, initial=initial, params=params, code=dsl)

bif = mod.bifurcation1d(
    steps=10000,
    integrator_type="rk4",
    num_transition_points=5000,
    constant_name="a",
    variable_name="x",
    min_max_dt=(0.0, 100., .25),
)

plt.figure(figsize=(10, 6))
plt.scatter(bif[:, 0], bif[:, 1], s=1)
plt.xlabel("Constant")
plt.ylabel("Variable_max")
plt.show()