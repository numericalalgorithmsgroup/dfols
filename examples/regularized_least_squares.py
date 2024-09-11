# DFO-LS example: regularized least-squares regression
import numpy as np
import dfols

# Solve a LASSO problem (linear least-squares + L1 regularizer)
#     min_{x} ||Ax-b||_2^2 + lda*||x||_1
# Larger values of lda>0 encourage greater sparsity in the solution x
# (i.e. more entries of x should be zero)

n = 5  # dimension of x
m = 10  # number of residuals, i.e. dimension of b

# Generate some artificial data for A and b
A = np.arange(m*n).reshape((m,n))
b = np.sqrt(np.arange(m))
objfun = lambda x: A @ x - b  # vector of residuals r(x)=Ax-b

# L1 regularizer: h(x) = lda*||x||_1 for some lda>0
lda = 1.0
h = lambda x: lda * np.linalg.norm(x, 1)
Lh = lda * np.sqrt(n)  # Lipschitz constant of h(x)
prox_uh = lambda x, u: np.sign(x) * np.maximum(np.abs(x) - lda*u, 0.0)  # proximal operator, prox_{uh}(x)


x0 = np.zeros((n,))  # arbitrary starting point

# Call DFO-LS
soln = dfols.solve(objfun, x0, h=h, lh=Lh, prox_uh=prox_uh)

# Display output
print(soln)
