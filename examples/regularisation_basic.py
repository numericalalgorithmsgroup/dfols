import numpy as np
import math
import dfols
from scipy.optimize import minimize

x0 = np.array([2, 0.5])
shift = np.array([1, 1])
objfun = lambda x: x-shift
proj = lambda x: pball(x, np.array([0, 0]), 1)
h = lambda x: np.linalg.norm(x-shift, 1)
maxhessian = 1e2
lh = math.sqrt(3)

def pball(x,c,r):
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)

# NOTE: Takes a few minutes to run
def prox_uh(u, h, xopt, d):
    # Find prox_{uh} using Nelder–Mead method
    func = lambda s: u*h(xopt+s) + np.linalg.norm(s-d, 2)**2 / 2
    res = minimize(func, d, method='Nelder-Mead', tol=1e-8)
    return res.x

soln = dfols.solve(objfun, h, x0, maxhessian, lh, prox_uh, projections = [proj])
print(soln.x)