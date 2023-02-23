import numpy as np
import math
import dfols
import time

# objective function: x + ||x||_1. 
# Expected solution: x = [0, 0]
# unconstrained

x0 = 10*np.zeros(5)
# shift = np.array([1, 1])
objfun = lambda x: 6*(x-np.ones(5))
# proj = lambda x: pball(x, np.array([0, 0]), 3)
h = lambda x: 2*np.linalg.norm(x-np.ones(5), 1)
lh = math.sqrt(2)

def pball(x,c,r):
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)

def prox_uh(x, u):
    # prox_uh(d) = min_{s} ||s-d||^2 / 2u + h(s) 
    # When h is 1-norm, we have the explicit solution
    rtn = np.zeros(x.shape)
    for i in range(x.shape[0]):
        rtn[i] = np.sign(x[i])*max(abs(x[i])-u, 0)
    return rtn

time_start = time.time()
soln = dfols.solve(objfun, x0, h, lh, prox_uh)
print(soln)
time_end = time.time()
time_taken = time_end - time_start
print("taken time: ", time_taken)