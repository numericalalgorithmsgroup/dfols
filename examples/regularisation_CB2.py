import numpy as np
import math
import dfols
import time
from scipy import optimize

def f1(x):
    return x[0]**2 + x[1]**4

def f2(x):
    return (2-x[0])**2 + (2-x[1])**2

def f3(x):
    return 2*math.exp(x[1] - x[0])

def h(x):
    return max([f1(x), f2(x), f3(x)])

def prox_uh(x, u, argsprox = h):
    prox = lambda s: np.linalg.norm(s-x, 2) ** 2 / 2 + u * h(s)
    rtn = optimize.minimize(prox, np.array([0.0, 0.0]), method='Nelder-Mead')
    return rtn.x

x0 = np.array([2.0, 2.0])
objfun = lambda x: np.zeros(x.shape)
lh = 10

time_start = time.time()
soln = dfols.solve(objfun, x0, h, lh, prox_uh, projections=[])
print(soln)
time_end = time.time()
time_taken = time_end - time_start
print("taken time: ", time_taken)