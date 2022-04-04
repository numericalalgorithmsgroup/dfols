from math import sqrt
import numpy as np
import time
import pylab

from dfols.trust_region import ctrsbox_sfista
from sqlalchemy import func


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

# Same as TestUncInternalCDFO in test_regu_trust_region.py
n = 3
g = np.array([1.0, 0.0, 1.0])
H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
h = lambda d: np.linalg.norm(d, 1)
L_h = sqrt(3)
delta = 2.0
xopt = np.ones((n,))  # trying nonzero (since bounds inactive)
n = 6
func_tol_array = np.array([10**i for i in range(-n,0)])[::-1]
time_array = np.zeros(func_tol_array.size)
for j in range(n):
    time_start = time.time()
    d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox_uh, func_tol=func_tol_array[j])
    time_end = time.time()
    time_taken = time_end- time_start
    print("func_tol", func_tol_array[j])
    print("time_taken", time_taken)
    time_array[j] = time_taken
pylab.loglog(func_tol_array, time_array)
pylab.show()
    