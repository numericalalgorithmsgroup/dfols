Overview
========

When to use DFO-LS
------------------
DFO-LS is designed to solve the nonlinear least-squares minimization problem (with optional bound constraints)

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
   \text{s.t.} &\quad  a \leq x \leq b

We call :math:`f(x)` the objective function and :math:`r_i(x)` the residual functions (or simply residuals).

DFO-LS is a *derivative-free* optimization algorithm, which means it does not require the user to provide the derivatives of :math:`f(x)` or :math:`r_i(x)`, nor does it attempt to estimate them internally (by using finite differencing, for instance). 

There are two main situations when using a derivative-free algorithm (such as DFO-LS) is preferable to a derivative-based algorithm (which is the vast majority of least-squares solvers).

If **the residuals are noisy**, then calculating or even estimating their derivatives may be impossible (or at least very inaccurate). By noisy, we mean that if we evaluate :math:`r_i(x)` multiple times at the same value of :math:`x`, we get different results. This may happen when a Monte Carlo simulation is used, for instance, or :math:`r_i(x)` involves performing a physical experiment. 

If **the residuals are expensive to evaluate**, then estimating derivatives (which requires :math:`n` evaluations of each :math:`r_i(x)` for every point of interest :math:`x`) may be prohibitively expensive. Derivative-free methods are designed to solve the problem with the fewest number of evaluations of the objective as possible.

**However, if you have provide (or a solver can estimate) derivatives** of :math:`r_i(x)`, then it is probably a good idea to use one of the many derivative-based solvers (such as `one from the SciPy library <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_).

Parameter Fitting
-----------------
A very common problem in many quantitative disciplines is fitting parameters to observed data. Typically, this means that we have developed a model for some proccess, which takes a vector of (known) inputs :math:`\mathrm{obs}\in\mathbb{R}^N` and some model parameters :math:`x=(x_1, \ldots, x_n)\in\mathbb{R}^n`, and computes a (predicted) quantity of interest :math:`y\in\mathbb{R}`:

.. math::

   y = \mathrm{model}(\mathrm{obs}, x)

For this model to be useful, we need to determine a suitable choice for the parameters :math:`x`, which typically cannot be directly observed. A common way of doing this is to calibrate from observed relationships.

Suppose we have some observations of the input-to-output relationship. That is, we have data

.. math::

   (\mathrm{obs}_1, y_1), \ldots, (\mathrm{obs}_m, y_m)

Then, we try to find the parameters :math:`x` which produce the best possible fit to these observations by minimizing the sum-of-squares of the prediction errors:

.. math::

   \min_{x\in\mathbb{R}^n}  \quad  f(x) := \sum_{i=1}^{m}(y_i - \mathrm{model}(\mathrm{obs}_i, x))^2

which is in the least-squares form required by DFO-LS.

As described above, DFO-LS is a particularly good choice for parameter fitting when the model has noise (e.g. Monte Carlo simulation) or is expensive to evaluate.

Solving Nonlinear Systems of Equations
--------------------------------------
Suppose we wish to solve the system of nonlinear equations: find :math:`x\in\mathbb{R}^n` satisfying

.. math::

   r_1(x) &= 0 \\
   r_2(x) &= 0 \\
   &\vdots \\
   r_m(x) &= 0

Such problems can have no solutions, one solution, or many solutions (possibly infinitely many). Often, but certainly not always, the number of solutions depends on whether there are more equations or unknowns: if :math:`m<n` we say the system is underdetermined (and there are often multiple solutions), if :math:`m=n` we say the system is square (and there is often only one solution), and if :math:`m>n` we say the system is overdetermined (and there are often no solutions).

This is not always true -- there is no solution to the underdetermined system when :math:`m=1` and :math:`n=2` and we choose :math:`r_1(x)=\sin(x_1+x_2)-2`, for example.
Similarly, if we take :math:`n=1` and :math:`r_i(x)=i (x-1)(x-2)`, we can make :math:`m` as large as we like while keeping :math:`x=1` and :math:`x=2` as solutions (to the overdetermined system).

If no solution exists, it makes sense to instead search for an :math:`x` which approximately satisfies each equation. A common way to do this is to minimize the sum-of-squares of the left-hand-sides:

.. math::

   \min_{x\in\mathbb{R}^n}  \quad  f(x) := \sum_{i=1}^{m}r_i(x)^2

which is the form required by DFO-LS.

If a solution does exist, then this formulation will also find this (where we will get :math:`f=0` at the solution).

**Which solution?** DFO-LS, and most similar software, will only find one solution to a set of nonlinear equations. Which one it finds is very difficult to predict, and depends very strongly on the point where the solver is started from. Often it finds the closest solution, but there are no guarantees this will be the case. If you need to find all/multiple solutions for your problem, consider techniques such as `deflation <http://www.sciencedirect.com/science/article/pii/0022247X83900550>`_.

Details of the DFO-LS Algorithm
-------------------------------
DFO-LS is a type of *trust-region* method, a common category of optimization algorithms for nonconvex problems. Given a current estimate of the solution :math:`x_k`, we compute a model which approximates the objective :math:`m_k(s)\approx f(x_k+s)` (for small steps :math:`s`), and maintain a value :math:`\Delta_k>0` (called the *trust region radius*) which measures the size of :math:`s` for which the approximation is good.

At each step, we compute a trial step :math:`s_k` designed to make our approximation :math:`m_k(s)` small (this task is called the *trust region subproblem*). We evaluate the objective at this new point, and if this provided a good decrease in the objective, we take the step (:math:`x_{k+1}=x_k+s_k`), otherwise we stay put (:math:`x_{k+1}=x_k`). Based on this information, we choose a new value :math:`\Delta_{k+1}`, and repeat the process.

In DFO-LS, we construct our approximation :math:`m_k(s)` by interpolating a linear approximation for each residual :math:`r_i(x)` at several points close to :math:`x_k`. To make sure our interpolated model is accurate, we need to regularly check that the points are well-spaced, and move them if they aren't (i.e. improve the geometry of our interpolation points).

A complete description of the DFO-LS algorithm is given in our paper [CFMR2018]_.

References
----------

.. [CFMR2018]   
   C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://arxiv.org/abs/1804.00154>`_, technical report, University of Oxford, (2018).

