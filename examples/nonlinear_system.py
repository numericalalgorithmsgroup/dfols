# DFO-LS example: Solving a nonlinear system of equations
# Originally from:
# http://support.sas.com/documentation/cdl/en/imlug/66112/HTML/default/viewer.htm#imlug_genstatexpls_sect004.htm

from __future__ import print_function
from math import exp
import numpy as np
import dfols

# Want to solve:
#   x1 + x2 - x1*x2 + 2 = 0
#   x1 * exp(-x2) - 1   = 0
def nonlinear_system(x):
    return np.array([x[0] + x[1] - x[0]*x[1] + 2,
                     x[0] * exp(-x[1]) - 1.0])

# Warning: if there are multiple solutions, which one
#          DFO-LS returns will likely depend on x0!
x0 = np.array([0.1, -2.0])

# Set random seed (for reproducibility)
np.random.seed(0)

# Call DFO-LS
soln = dfols.solve(nonlinear_system, x0)

# Display output
print(soln)

