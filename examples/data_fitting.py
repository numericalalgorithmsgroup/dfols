# DFO-LS example: data fitting problem
# Originally from:
# https://uk.mathworks.com/help/optim/ug/lsqcurvefit.html
from __future__ import print_function
import numpy as np
import dfols

# Observations
tdata = np.array([0.9, 1.5, 13.8, 19.8, 24.1, 28.2, 35.2,
                  60.3, 74.6, 81.3])
ydata = np.array([455.2, 428.6, 124.1, 67.3, 43.2, 28.1, 13.1,
                  -0.4, -1.3, -1.5])

# Model is y(t) = x[0] * exp(x[1] * t)
def prediction_error(x):
    return ydata - x[0] * np.exp(x[1] * tdata)

# Define the starting point
x0 = np.array([100.0, -1.0])

# Set random seed (for reproducibility)
np.random.seed(0)

# We expect exponential decay: set upper bound x[1] <= 0
upper = np.array([1e20, 0.0])

# Call DFO-LS
soln = dfols.solve(prediction_error, x0, bounds=(None, upper))

# Display output
print(soln)

# Plot calibrated model vs. observations
ts = np.linspace(0.0, 90.0)
ys = soln.x[0] * np.exp(soln.x[1] * ts)

import matplotlib.pyplot as plt
plt.figure(1)
ax = plt.gca()  # current axes
ax.plot(ts, ys, 'k-', label='Model')
ax.plot(tdata, ydata, 'bo', label='Data')
ax.set_xlabel('t')
ax.set_ylabel('y(t)')
ax.legend(loc='upper right')
ax.grid()
plt.show()

