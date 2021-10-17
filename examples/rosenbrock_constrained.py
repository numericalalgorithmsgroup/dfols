# DFO-LS example: minimize the Rosenbrock function with arbitrary convex constraints
from __future__ import print_function
import numpy as np
import dfols

# Define the objective function
def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

# Define the starting point
x0 = np.array([-1.2, 1])

# Define the projection functions
def pball(x):
    c = np.array([0.7,1.5]) # ball centre
    r = 0.4 # ball radius
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)

def pbox(x):
    l = np.array([-2, 1.1]) # lower bound
    u = np.array([0.9, 3]) # upper bound
    return np.minimum(np.maximum(x,l), u)

# For optional extra output details
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Call DFO-LS
soln = dfols.solve(rosenbrock, x0, projections=[pball,pbox], user_params={'dykstra.d_tol': 1e-50})

# Display output
print(soln)
