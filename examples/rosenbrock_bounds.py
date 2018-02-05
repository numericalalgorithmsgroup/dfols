# DFO-LS example: minimize the Rosenbrock function
from __future__ import print_function
import numpy as np
import dfols

# Define the objective function
def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

# Define the starting point
x0 = np.array([-1.2, 1.0])

# Define bound constraints (lower <= x <= upper)
lower = np.array([-10.0, -10.0])
upper = np.array([0.9, 0.85])

# Set random seed (for reproducibility)
np.random.seed(0)

# For optional extra output details
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Call DFO-LS
soln = dfols.solve(rosenbrock, x0, bounds=(lower, upper))

# Display output
print(soln)

