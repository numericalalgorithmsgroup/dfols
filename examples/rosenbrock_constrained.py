'''
DFO-LS example: minimize the Rosenbrock function with arbitrary convex constraints

This example defines two functions pball(x) and pbox(x) that project onto ball and
box constraint sets respectively. It then passes both these functions to the DFO-LS
solver so that it can find a constrained minimizer to the Rosenbrock function.
Such a minimizer must lie in the intersection of constraint sets corresponding to
projection functions pball(x) and pbox(x). The description of the problem is as follows:

    min rosenbrock(x)
    s.t.
        -2 <= x[0] <= 1.1,
        1.1 <= x[1] <= 3,
        norm(x-c) <= 0.4

where c = [0.7, 1.5] is the centre of the ball.
'''
from __future__ import print_function
import numpy as np
import dfols

# Define the objective function
def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

# Define the starting point
x0 = np.array([-1.2, 1])

'''
Define ball projection function
Projects the input x onto a ball with
centre point (0.7,1.5) and radius 0.4.
'''
def pball(x):
    c = np.array([0.7,1.5]) # ball centre
    r = 0.4 # ball radius
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)

'''
Define box projection function
Projects the input x onto a box
such that -2 <= x[0] <= 0.9 and
1.1 <= x[1] <= 3.

Note: One could equivalently add bound
constraints as a separate input to the solver
instead.
'''
def pbox(x):
    l = np.array([-2, 1.1]) # lower bound
    u = np.array([0.9, 3]) # upper bound
    return np.minimum(np.maximum(x,l), u)

# For optional extra output details
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Call DFO-LS
soln = dfols.solve(rosenbrock, x0, projections=[pball,pbox])

# Display output
print(soln)
