import numpy as np
import math
import dfols
from scipy.optimize import minimize

x0 = np.array([0.0, 0.0])
shift = np.array([1, 1])
objfun = lambda x: x-shift
proj = lambda x: pball(x, np.array([0, 0]), 1)
h = lambda x: np.linalg.norm(x-shift, 1)
maxhessian = 1e2
lh = math.sqrt(3)

def pball(x,c,r):
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)

# NOTE: Takes a few minutes to run
def prox_uh(xopt, u, d):
    # prox_uh(d) = min_{s} ||s-d||^2 / 2 + uh(xopt+s) 
    # When h is 1-norm, we have the explicit solution
    n = d.shape[0]
    rtn = np.zeros(d.shape)
    for i in range(n):
        if d[i] > u - xopt[i]:
            rtn[i] = d[i] - u
        elif d[i] < -u-xopt[i]:
            rtn[i] = d[i] + u
        else:
            rtn[i] = d[i]
    return rtn

soln = dfols.solve(objfun, h, x0, maxhessian, lh, prox_uh, projections = [proj])
print(soln.x)