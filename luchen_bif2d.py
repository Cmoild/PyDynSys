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

bif2d = mod.bifurcation2d(
    steps=10000,
    integrator_type="rk4",
    num_transition_points=6000,
    variable_name="x",
    constants_dict={"a": (0.0, 100.0, .25), "b": (0.0, 100.0, .25)},
)

bif2d_2d = bif2d[:, :, 0]

plt.figure(figsize=(8, 6))
plt.imshow(
    bif2d_2d,
    extent=(0.0, 100.0, 0.0, 100.0),
    origin="lower",
    aspect="auto",
    cmap="plasma",
)
plt.colorbar(label="Number of clusters")
plt.xlabel("Parameter 1")
plt.ylabel("Parameter 2")
plt.title("2D bifurcation diagram")
plt.show()