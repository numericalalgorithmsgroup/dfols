"""
Demonstration of using database of existing evaluations to speed up DFO-LS

Test problem is the 'Watson function': for details, see
J. J. More, B. S. Garbow, K. E. Hillstrom. Testing Unconstrained Optimization Software.
ACM Transactions on Mathematical Software, 7:1 (1981), pp. 17-41.
"""
import numpy as np
import dfols

# Define the objective function
def watson(x):
    n = len(x)
    m = 31
    fvec = np.zeros((m,), dtype=float)

    for i in range(1, 30):  # i=1,...,29
        div = float(i) / 29.0
        s1 = 0.0
        dx = 1.0
        for j in range(2, n + 1):  # j = 2,...,n
            s1 = s1 + (j - 1) * dx * x[j - 1]
            dx = div * dx
        s2 = 0.0
        dx = 1.0
        for j in range(1, n + 1):  # j = 1,...,n
            s2 = s2 + dx * x[j - 1]
            dx = div * dx
        fvec[i - 1] = s1 - s2 ** 2 - 1.0

    fvec[29] = x[0]
    fvec[30] = x[1] - x[0] ** 2 - 1.0

    return fvec

# Define the starting point
n = 6
x0 = 0.5 * np.ones((n,), dtype=float)

# When n=6, we expect f(x0) ~ 16.4308 and f(xmin) ~ 0.00228767 at xmin ~ [ -0.0157, 1.0124, 1.2604, -1.5137, 0.992996]

# For optional extra output details
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Now build a database of evaluations
eval_db = dfols.EvaluationDatabase()
eval_db.append(x0, watson(x0), make_starting_eval=True)  # make x0 the starting point

# Note: x0, x1 and x2 are colinear, so at least one of x1 and x2 shouldn't be included in the initial model
x1 = np.ones((n,), dtype=float)
x2 = np.zeros((n,), dtype=float)
x3 = np.arange(n).astype(float)
eval_db.append(x1, watson(x1))
eval_db.append(x2, watson(x2))
eval_db.append(x3, watson(x3))

soln = dfols.solve(watson, eval_db)  # replace x0 with eval_db

print(soln)