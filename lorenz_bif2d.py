import numpy as np
from matplotlib import pyplot as plt
import dynsys
import time

dt = 0.01
initial = [1.0, 1.0, 1.0]

dsl = """
x' = sigma * (y - x)
y' = x * (rho - z) - y
z' = x * y - beta * z
"""

params = {"sigma": 10.0, "rho": 20.0, "beta": 8.0 / 3.0}

mod = dynsys.model.Model(dt=dt, initial=initial, params=params, code=dsl)

bif2d = mod.bifurcation2d(
    steps=40000,
    integrator_type="rk4",
    num_transition_points=20000,
    variable_name="x",
    constants_dict={"sigma": (0.0, 100.0, .25), "rho": (0.0, 100.0, .25)},
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