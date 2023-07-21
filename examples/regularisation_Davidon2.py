import numpy as np
import math
import dfols
import time
from scipy import optimize

def f(i, x):
    ti = 0.2*i
    return (x[0] + x[1]*ti - math.exp(ti))**2 + (x[2] + x[3]*math.sin(ti) - math.cos(ti))**2

def h(x):
    n = 20
    rtn_list = np.zeros([n,1])
    for i in range(n):
        rtn_list[i] = abs(f(i, x))
    return max(rtn_list)

def prox_uh(x, u, argsprox = h):
    prox = lambda s: np.linalg.norm(s-x, 2) ** 2 / 2 + u * h(s)
    rtn = optimize.minimize(prox, np.array([0.0, 0.0, 0.0, 0.0]), method='Nelder-Mead')
    return rtn.x

x0 = np.array([1.0, 1.0, 1.0, 1.0])
objfun = lambda x: np.zeros(x.shape)
lh = 10

time_start = time.time()
soln = dfols.solve(objfun, x0, h, lh, prox_uh, projections=[])
print(soln)
time_end = time.time()
time_taken = time_end - time_start
print("taken time: ", time_taken)