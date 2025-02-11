Using DFO-LS
============
This section describes the main interface to DFO-LS and how to use it.

Nonlinear Least-Squares Minimization
------------------------------------
DFO-LS is designed to solve the local optimization problem

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 + h(x) \\
   \text{s.t.} &\quad  a \leq x \leq b\\
               &\quad x \in C := C_1 \cap \cdots \cap C_n, \quad \text{all $C_i$ convex}

where the bound constraints :math:`a \leq x \leq b` and general convex constraints :math:`x\in C` are optional. The upper and lower bounds on the variables are non-relaxable (i.e. DFO-LS will never ask to evaluate a point outside the bounds). The general convex constraints are also non-relaxable, but they may be slightly violated at some points from rounding errors. 
The function :math:`h(x)` is an optional regularizer, often used to avoid overfitting, and must be Lipschitz continuous and convex (but possibly non-differentiable). 
A common choice is :math:`h(x)=\lambda \|x\|_1` (called L1 regularization or LASSO) for :math:`\lambda>0`. 
Note that in the case of Tikhonov regularization/ridge regression, :math:`h(x)=\lambda\|x\|_2^2` is not Lipschitz continuous, so should instead be incorporated by adding an extra term into the least-squares sum, :math:`r_{m+1}(x)=\sqrt{\lambda} \|x\|_2`.

DFO-LS iteratively constructs an interpolation-based model for the objective, and determines a step using a trust-region framework.
For an in-depth technical description of the algorithm see the papers [CFMR2018]_, [HR2022]_ and [LLR2024]_.

How to use DFO-LS
-----------------
The main interface to DFO-LS is via the function :code:`solve`

  .. code-block:: python
  
      soln = dfols.solve(objfun, x0)

The input :code:`objfun` is a Python function which takes an input :math:`x\in\mathbb{R}^n` and returns the vector of residuals :math:`[r_1(x)\: \cdots \: r_m(x)]\in\mathbb{R}^m`. Both the input and output of :code:`objfun` must be one-dimensional NumPy arrays (i.e. with :code:`x.shape == (n,)` and :code:`objfun(x).shape == (m,)`).

The input :code:`x0` is the starting point for the solver, and (where possible) should be set to be the best available estimate of the true solution :math:`x_{min}\in\mathbb{R}^n`. It should be specified as a one-dimensional NumPy array (i.e. with :code:`x0.shape == (n,)`).
As DFO-LS is a local solver, providing different values for :code:`x0` may cause it to return different solutions, with possibly different objective values.

The output of :code:`dfols.solve` is an object containing:

* :code:`soln.x` - an estimate of the solution, :math:`x_{min}\in\mathbb{R}^n`, a one-dimensional NumPy array.
* :code:`soln.resid` - the vector of residuals at the calculated solution, :math:`[r_1(x_{min})\:\cdots\: r_m(x_{min})]`, a one-dimensional NumPy array.
* :code:`soln.f` - the objective value at the calculated solution, :math:`f(x_{min})`, a Float.
* :code:`soln.jacobian` - an estimate of the Jacobian matrix of first derivatives of the residuals, :math:`J_{i,j} \approx \partial r_i(x_{min})/\partial x_j`, a NumPy array of size :math:`m\times n`.
* :code:`soln.nf` - the number of evaluations of :code:`objfun` that the algorithm needed, an Integer.
* :code:`soln.nx` - the number of points :math:`x` at which :code:`objfun` was evaluated, an Integer. This may be different to :code:`soln.nf` if sample averaging is used.
* :code:`soln.nruns` - the number of runs performed by DFO-LS (more than 1 if using multiple restarts), an Integer.
* :code:`soln.flag` - an exit flag, which can take one of several values (listed below), an Integer.
* :code:`soln.msg` - a description of why the algorithm finished, a String.
* :code:`soln.diagnostic_info` - a table of diagnostic information showing the progress of the solver, a Pandas DataFrame.
* :code:`soln.xmin_eval_num` - an integer representing which evaluation point (i.e. same as :code:`soln.nx`) gave the solution :code:`soln.x`. Evaluation counts are 1-indexed, to match the logging information. 
* :code:`soln.jacmin_eval_nums` - a NumPy integer array of length :code:`npt` with the evaluation point numbers (i.e. same as :code:`soln.nx`) used to build :code:`soln.jacobian` via linear interpolation to the residual values at these points. Evaluation counts are 1-indexed, to match the logging information. This array will usually, but not always, include :code:`soln.xmin_eval_num`.

The possible values of :code:`soln.flag` are defined by the following variables:

* :code:`soln.EXIT_SUCCESS` - DFO-LS terminated successfully (the objective value or trust region radius are sufficiently small).
* :code:`soln.EXIT_MAXFUN_WARNING` - maximum allowed objective evaluations reached. This is the most likely return value when using multiple restarts.
* :code:`soln.EXIT_SLOW_WARNING` - maximum number of slow iterations reached.
* :code:`soln.EXIT_FALSE_SUCCESS_WARNING` - DFO-LS reached the maximum number of restarts which decreased the objective, but to a worse value than was found in a previous run.
* :code:`soln.EXIT_TR_INCREASE_WARNING` - model increase when solving the trust region subproblem with multiple arbitrary constraints.
* :code:`soln.EXIT_INPUT_ERROR` - error in the inputs.
* :code:`soln.EXIT_TR_INCREASE_ERROR` - error occurred when solving the trust region subproblem.
* :code:`soln.EXIT_LINALG_ERROR` - linear algebra error, e.g. the interpolation points produced a singular linear system.
* :code:`soln.EXIT_EVAL_ERROR` - the objective function returned a NaN value when evaluating at a new trial point.

These variables are defined in the :code:`soln` object, so can be accessed with, for example

  .. code-block:: python
  
      if soln.flag == soln.EXIT_SUCCESS:
          print("Success!")

In newer versions DFO-LS (v1.5.4 onwards), the results object can be converted to, or loaded from, a serialized Python dictionary. This allows the results to be saved as a JSON file.
For example, to save the results to a JSON file, you may use

  .. code-block:: python
  
      import json
      soln_dict = soln.to_dict()  # convert soln to serializable dict object
      with open("dfols_results.json", 'w') as f:
          json.dump(soln_dict, f, indent=2)

The :code:`to_dict()` function takes an optional boolean, :code:`to_dict(replace_nan=True)`. If :code:`replace_nan` is :code:`True`, any NaN values in the results object are converted to :code:`None`.

To load results from a JSON file and convert to a solution object, you may use

  .. code-block:: python
  
      import json
      soln_dict = None
      with open("dfols_results.json") as f:
          soln_dict = json.load(f)  # read JSON into dict
      soln = dfols.OptimResults.from_dict(soln_dict)  # convert to DFO-LS results object
      print(soln)


Optional Arguments
------------------
The :code:`solve` function has several optional arguments which the user may provide:

  .. code-block:: python
  
      dfols.solve(objfun, x0, 
                  h=None, lh=None, prox_uh=None, 
                  argsf=(), argsh=(), argsprox=(), 
                  bounds=None, projections=[], npt=None, rhobeg=None, 
                  rhoend=1e-8, maxfun=None, nsamples=None, 
                  user_params=None, objfun_has_noise=False, 
                  scaling_within_bounds=False,
                  do_logging=True, print_progress=False)

These arguments are:

* :code:`h` - the regularizer function which takes an input :math:`x\in\mathbb{R}^n` and returns :math:`h(x)`. 
* :code:`lh` - the `Lipschitz constant <https://en.wikipedia.org/wiki/Lipschitz_continuity>`_ (with respect to the Euclidean norm on :math:`\mathbb{R}^n`) of :math:`h(x)`, a positive number if :code:`h` given. For example, if :math:`h(x)=\lambda \|x\|_1` for :math:`\lambda>0`, then :math:`L_h=\lambda \sqrt{n}`.
* :code:`prox_uh` - the `proximal operator <https://en.wikipedia.org/wiki/Proximal_operator>`_ of :math:`h(x)`. This function has the form :code:`prox_uh(x, u)`, where :math:`x\in \mathbb{R}^n` and :math:`u>0`, and returns :math:`\operatorname{prox}_{uh}(x)`. For example, if :math:`h(x)=\lambda \|x\|_1` for :math:`\lambda>0`, then :code:`prox_uh(x, u) = np.sign(x) * np.maximum(np.abs(x) - lambda*u, 0)`. More examples of proximal operators may be found on `this page <https://proximity-operator.net/>`_. 
* :code:`argsf` - a tuple of extra arguments passed to the objective function :code:`objfun(x, *argsf)`.
* :code:`argsh` - a tuple of extra arguments passed to the regularizer :code:`h(x, *argsh)`. 
* :code:`argsprox` - a tuple of extra arguments passed to the proximal operator :code:`prox_uh(x, u, *argsprox)`. 
* :code:`bounds` - a tuple :code:`(lower, upper)` with the vectors :math:`a` and :math:`b` of lower and upper bounds on :math:`x` (default is :math:`a_i=-10^{20}` and :math:`b_i=10^{20}`). To set bounds for either :code:`lower` or :code:`upper`, but not both, pass a tuple :code:`(lower, None)` or :code:`(None, upper)`.
* :code:`projections` - a list :code:`[f1,f2,...,fn]` of functions that each take as input a point :code:`x` and return a new point :code:`y`. The new point :code:`y` should be given by the projection of :code:`x` onto a closed convex set. The intersection of all sets corresponding to a function must be non-empty.
* :code:`npt` - the number of interpolation points to use (default is :code:`len(x0)+1`). If using restarts, this is the number of points to use in the first run of the solver, before any restarts (and may be optionally increased via settings in :code:`user_params`).
* :code:`rhobeg` - the initial value of the trust region radius (default is :math:`0.1\max(\|x_0\|_{\infty}, 1)`, or 0.1 if :code:`scaling_within_bounds`).
* :code:`rhoend` - minimum allowed value of trust region radius, which determines when a successful termination occurs (default is :math:`10^{-8}`).
* :code:`maxfun` - the maximum number of objective evaluations the algorithm may request (default is :math:`\min(100(n+1),1000)`).
* :code:`nsamples` - a Python function :code:`nsamples(delta, rho, iter, nrestarts)` which returns the number of times to evaluate :code:`objfun` at a given point. This is only applicable for objectives with stochastic noise, when averaging multiple evaluations at the same point produces a more accurate value. The input parameters are the trust region radius (:code:`delta`), the lower bound on the trust region radius (:code:`rho`), how many iterations the algorithm has been running for (:code:`iter`), and how many restarts have been performed (:code:`nrestarts`). Default is no averaging (i.e. :code:`nsamples(delta, rho, iter, nrestarts)=1`).
* :code:`user_params` - a Python dictionary :code:`{'param1': val1, 'param2':val2, ...}` of optional parameters. A full list of available options is given in the next section :doc:`advanced`.
* :code:`objfun_has_noise` - a flag to indicate whether or not :code:`objfun` has stochastic noise; i.e. will calling :code:`objfun(x)` multiple times at the same value of :code:`x` give different results? This is used to set some sensible default parameters (including using multiple restarts), all of which can be overridden by the values provided in :code:`user_params`.
* :code:`scaling_within_bounds` - a flag to indicate whether the algorithm should internally shift and scale the entries of :code:`x` so that the bounds become :math:`0 \leq x \leq 1`. This is useful is you are setting :code:`bounds` and the bounds have different orders of magnitude. If :code:`scaling_within_bounds=True`, the values of :code:`rhobeg` and :code:`rhoend` apply to the *shifted* variables.
* :code:`do_logging` - a flag to indicate whether logging output should be produced. This is not automatically visible unless you use the Python `logging <https://docs.python.org/3/library/logging.html>`_ module (see below for simple usage).
* :code:`print_progress` - a flag to indicate whether to print a per-iteration progress log to terminal.

In general when using optimization software, it is good practice to scale your variables so that moving each by a given amount has approximately the same impact on the objective function.
The :code:`scaling_within_bounds` flag is designed to provide an easy way to achieve this, if you have set the bounds :code:`lower` and :code:`upper`.

A Simple Example
----------------
Suppose we wish to minimize the `Rosenbrock test function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_:

.. math::

   \min_{(x_1,x_2)\in\mathbb{R}^2}  &\quad  100(x_2-x_1^2)^2 + (1-x_1)^2 \\

This function has exactly one local minimum :math:`f(x_{min})=0` at :math:`x_{min}=(1,1)`. We can write this as a least-squares problem as:

.. math::

   \min_{(x_1,x_2)\in\mathbb{R}^2}  &\quad  [10(x_2-x_1^2)]^2 + [1-x_1]^2 \\

A commonly-used starting point for testing purposes is :math:`x_0=(-1.2,1)`. The following script shows how to solve this problem using DFO-LS:

  .. code-block:: python
  
      # DFO-LS example: minimize the Rosenbrock function
      from __future__ import print_function
      import numpy as np
      import dfols

      # Define the objective function
      def rosenbrock(x):
          return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
      
      # Define the starting point
      x0 = np.array([-1.2, 1.0])
      
      # Call DFO-LS
      soln = dfols.solve(rosenbrock, x0)
      
      # Display output
      print(soln)
      
Note that DFO-LS is a randomized algorithm: in its first phase, it builds an internal approximation to the objective function by sampling it along random directions. In the code above, we set NumPy's random seed for reproducibility over multiple runs, but this is not required. The output of this script, showing that DFO-LS finds the correct solution, is

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [1. 1.]
      Residual vector = [0. 0.]
      Objective value f(xmin) = 0
      Needed 33 objective evaluations (at 33 points)
      Approximate Jacobian = [[-1.9982000e+01  1.0000000e+01]
       [-1.0000000e+00  1.0079924e-14]]
      Solution xmin was evaluation point 33
      Approximate Jacobian formed using evaluation points [29 31 32]
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************

This and all following problems can be found in the `examples <https://github.com/numericalalgorithmsgroup/dfols/tree/master/examples>`_ directory on the DFO-LS Github page.

Adding Bounds and More Output
-----------------------------
We can extend the above script to add constraints. To add bound constraints alone, we can add the lines

  .. code-block:: python
  
      # Define bound constraints (lower <= x <= upper)
      lower = np.array([-10.0, -10.0])
      upper = np.array([0.9, 0.85])
      
      # Call DFO-LS (with bounds)
      soln = dfols.solve(rosenbrock, x0, bounds=(lower, upper))

DFO-LS correctly finds the solution to the constrained problem:

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [0.9  0.81]
      Residual vector = [0.  0.1]
      Objective value f(xmin) = 0.01
      Needed 56 objective evaluations (at 56 points)
      Approximate Jacobian = [[-1.79999999e+01  1.00000000e+01]
       [-1.00000000e+00 -5.15519307e-10]]
      Solution xmin was evaluation point 42
      Approximate Jacobian formed using evaluation points [55 42 54]
      Exit flag = 0
      Success: rho has reached rhoend
      ****************************


However, we also get a warning that our starting point was outside of the bounds:

  .. code-block:: none
  
      RuntimeWarning: x0 above upper bound, adjusting

DFO-LS automatically fixes this, and moves :math:`x_0` to a point within the bounds, in this case :math:`x_0=(-1.2,0.85)`.

We can also get DFO-LS to print out more detailed information about its progress using the `logging <https://docs.python.org/3/library/logging.html>`_ module. To do this, we need to add the following lines:

  .. code-block:: python
  
      import logging
      logging.basicConfig(level=logging.INFO, format='%(message)s')
      
      # ... (call dfols.solve)

And for the simple bounds example we can now see each evaluation of :code:`objfun`:

  .. code-block:: none
  
      Function eval 1 at point 1 has f = 39.65 at x = [-1.2   0.85]
      Initialising (coordinate directions)
      Function eval 2 at point 2 has f = 14.337296 at x = [-1.08  0.85]
      Function eval 3 at point 3 has f = 55.25 at x = [-1.2   0.73]
      ...
      Function eval 55 at point 55 has obj = 0.0100000000000225 at x = [0.9        0.80999998]
      Function eval 56 at point 56 has obj = 0.01 at x = [0.9  0.81]
      Did a total of 1 run(s)

If we wanted to save this output to a file, we could replace the above call to :code:`logging.basicConfig()` with

  .. code-block:: python
  
      logging.basicConfig(filename="myfile.log", level=logging.INFO, 
                          format='%(message)s', filemode='w')

If you have logging for some parts of your code and you want to deactivate all DFO-LS logging, you can use the optional argument :code:`do_logging=False` in :code:`dfols.solve()`.

An alternative option available is to get DFO-LS to print to terminal progress information every iteration, by setting the optional argument :code:`print_progress=True` in :code:`dfols.solve()`. If we do this for the above example, we get

  .. code-block:: none
  
       Run  Iter     Obj       Grad     Delta      rho     Evals 
        1     1    1.43e+01  1.61e+02  1.20e-01  1.20e-01    3   
        1     2    4.35e+00  3.77e+01  4.80e-01  1.20e-01    4   
        1     3    4.35e+00  3.77e+01  6.00e-02  1.20e-02    4 
      ...
        1    55    1.00e-02  2.00e-01  1.50e-08  1.00e-08   56   
        1    56    1.00e-02  2.00e-01  1.50e-08  1.00e-08   57

Adding General Convex Constraints
---------------------------------
We can also add more general convex constraints :math:`x \in C := C_1 \cap \cdots \cap C_n` to our problem, where
each :math:`C_i` is a convex set. To do this, we need to know the Euclidean projection operator for each :math:`C_i`:

.. math::

   \operatorname{proj}_{C_i}(x) := \operatorname{argmin}_{y\in C_i} \|y-x\|_2^2.

i.e. given a point :math:`x`, return the closest point to :math:`x` in the set :math:`C_i`.
There are many examples of simple convex sets :math:`C_i` for which this function has a known, simple form, such as:

* Bound constraints (but since DFO-LS supports this directly, it is better to give these explicitly via the :code:`bounds` input, as above)
* Euclidean ball constraints: :math:`\|x-c\|_2 \leq r`
* Unit simplex: :math:`x_i \geq 0` and :math:`\sum_{i=1}^{n} x_i \leq 1`
* Linear inequalities: :math:`a^T x \geq b`

Note the intersection of the user-provided convex sets must be non-empty.

In DFO-LS, set the input :code:`projections` to be a list of projection functions, one per :math:`C_i`.
Internally, DFO-LS computes the projection onto the intersection of these sets and the bound constraints
using `Dykstra's projection algorithm <https://en.wikipedia.org/wiki/Dykstra%27s_projection_algorithm>`_.

For the explicit expressions for the above projections, and more examples, see for example `this online database <https://proximity-operator.net/indicatorfunctions.html>`_
or Section 6.4.6 of the textbook [B2017]_.

As an example, let's minimize the above Rosenbrock function with different bounds, and with a Euclidean
ball constraint, namely :math:`(x_1-0.7)^2 + (x_2-1.5)^2 \leq 0.4^2`.

  .. code-block:: python
  
      import numpy as np
      import dfols
      
      # Define the objective function
      def rosenbrock(x):
          return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
      
      # Define the starting point
      x0 = np.array([-1.2, 1])
      
      # Define bound constraints (lower <= x <= upper)
      lower = np.array([-2.0, 1.1])
      upper = np.array([0.9, 3.0])
      
      # Define the ball constraint ||x-center|| <= radius, and its projection operator
      center = np.array([0.7, 1.5])
      radius = 0.4
      ball_proj = lambda x: center + (radius/max(np.linalg.norm(x-center), radius)) * (x-center)
      
      # Call DFO-LS (with bounds and projection operator)
      # Note: it is better to provide bounds explicitly, instead of using the corresponding
      #       projection function
      # Note: input 'projections' must be a list of projection functions
      soln = dfols.solve(rosenbrock, x0, bounds=(lower, upper), projections=[ball_proj])
      
      # Display output
      print(soln)

Note that for bound constraints one can choose to either implement them by defining a projection function as above, or by passing the bounds as input like in the example from the section on adding bound constraints.

DFO-LS correctly finds the solution to this constrained problem too. Note that we get a warning because the step computed in the trust region subproblem
gave an increase in the model. This is common in the case where multiple constraints are active at the optimal point.

  .. code-block:: none

      ****** DFO-LS Results ******
      Solution xmin = [0.9        1.15359245]
      Residual vector = [3.43592448 0.1       ]
      Objective value f(xmin) = 11.81557703
      Needed 10 objective evaluations (at 10 points)
      Approximate Jacobian = [[-1.79826221e+01  1.00004412e+01]
       [-1.00000000e+00 -1.81976605e-15]]
      Solution xmin was evaluation point 5
      Approximate Jacobian formed using evaluation points [8 5 9]
      Exit flag = 5
      Warning (trust region increase): Either multiple constraints are active or trust region step gave model increase
      ****************************

Just like for bound constraints, DFO-LS will automatically ensure the starting point is feasible with respect to all constraints (bounds and general convex constraints).

Adding a Regularizer
--------------------
We can add a convex, Lipschitz continuous, but potentially non-differentiable regularizer to our objective function, to encourage the solution :math:`x` to have certain properties.
This is most commonly used to avoid overfitting.
A very common choice of regularizer is :math:`h(x)=\lambda\|x\|_2^2` for :math:`\lambda>0` (called Tikhonov regularization or ridge regression), but this is not Lipschitz continuous. For this regularizer, you can add a new residual function :math:`r_{m+1}(x)=\sqrt{\lambda}\|x\|_2` to the objective.

A suitable and widely used regularizer is the L1 norm (i.e. L1 regularization or LASSO), :math:`h(x)=\lambda\|x\|_1` for :math:`\lambda>0`.
This encourages the solution :math:`x` to be sparse (i.e. many entries are zero).
To use :math:`h(x)` in DFO-LS, we need to know its `Lipschitz constant <https://en.wikipedia.org/wiki/Lipschitz_continuity>`_ and `proximal operator <https://en.wikipedia.org/wiki/Proximal_operator>`_.

In this case, the Lipschitz constant of :math:`h(x)` may be computed via

.. math::

   |h(x) - h(x)| = \lambda\|x\|_1 - \lambda\|y\|_1 \leq \lambda\|x-y\|_1 \leq \lambda\sqrt{n} \|x-y\|_2

using the reverse triangle inequality to get the first inequality. Hence the Lipschitz constant of :math:`h(x)` is :math:`\lambda\sqrt{n}`.

The proximal operator for :math:`h(x)` with a parameter :math:`u>0` is defined as 

.. math::

   \operatorname{prox}_{uh}(x) := \operatorname{argmin}_{y\in\mathbb{R}^n} h(y) + \frac{1}{2u}\|y-x\|_2^2

There are many regularizers with known proximal operators. See for example `this online database <https://proximity-operator.net/>`_
or Section 6.9 of the textbook [B2017]_.
In the case of :math:`h(x)=\lambda\|x\|_1`, the proximal operator is the soft-thresholding function, defined element-wise as

.. math::

   [\operatorname{prox}_{uh}(x)]_i = \max(|x_i|-\lambda u, 0) \operatorname{sign}(x_i)

We can use DFO-LS to solve a simple regularized linear least-squares problem (with artificially generated data) as follows:

  .. code-block:: python
  
      # DFO-LS example: regularized least-squares regression
      import numpy as np
      import dfols
      
      n = 5  # dimension of x
      m = 10  # number of residuals, i.e. dimension of b
      
      # Generate some artificial data for A and b
      A = np.arange(m*n).reshape((m,n))
      b = np.sqrt(np.arange(m))
      objfun = lambda x: A @ x - b
      
      # L1 regularizer: h(x) = lda*||x||_1 for some lda>0
      lda = 1.0
      h = lambda x: lda * np.linalg.norm(x, 1)
      Lh = lda * np.sqrt(n)  # Lipschitz constant of h(x)
      prox_uh = lambda x, u: np.sign(x) * np.maximum(np.abs(x) - lda*u, 0.0)
      
      
      x0 = np.zeros((n,))  # arbitrary starting point
      
      # Call DFO-LS
      soln = dfols.solve(objfun, x0, h=h, lh=Lh, prox_uh=prox_uh)
      
      # Display output
      print(soln)

The solution found by DFO-LS is:

  .. code-block:: none

      ****** DFO-LS Results ******
      Solution xmin = [-6.85049254e-02 -7.03534168e-11  1.19957812e-15  7.47953030e-11
        1.30074165e-01]
      Residual vector = [ 0.52029666 -0.17185715 -0.27822451 -0.28821556 -0.24831856 -0.17654034
       -0.08211591  0.02946872  0.1546391   0.29091242]
      Objective value f(xmin) = 0.8682829845
      Needed 34 objective evaluations (at 34 points)
      Approximate Jacobian = [[-1.75619848e-09  1.00000000e+00  2.00000000e+00  3.00000000e+00
         4.00000000e+00]
       ...
       [ 4.50000000e+01  4.60000000e+01  4.70000000e+01  4.80000000e+01
         4.90000000e+01]]
      Solution xmin was evaluation point 34
      Approximate Jacobian formed using evaluation points [30 32 29 31 33 27]
      Exit flag = 0
      Success: rho has reached rhoend
      ****************************

We can see that 3 of the 5 components of the solution are very close to zero.
Note that many LASSO-type algorithms can produce a solution with many entries being exactly zero, but DFO-LS can only make them very small (related to how it calculates a new point with trust-region constraints).

Example: Noisy Objective Evaluation
-----------------------------------
As described in :doc:`info`, derivative-free algorithms such as DFO-LS are particularly useful when :code:`objfun` has noise. Let's modify the previous example to include random noise in our objective evaluation, and compare it to SciPy's derivative-based solver (the below results came from using SciPy v1.13.0):

  .. code-block:: python
  
      # DFO-LS example: minimize the noisy Rosenbrock function
      from __future__ import print_function
      import numpy as np
      import dfols
      
      # Define the objective function
      def rosenbrock(x):
          return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
      
      # Modified objective function: add 1% Gaussian noise
      def rosenbrock_noisy(x):
          return rosenbrock(x) * (1.0 + 1e-2 * np.random.normal(size=(2,)))
      
      # Define the starting point
      x0 = np.array([-1.2, 1.0])
      
      # Set random seed (for reproducibility)
      np.random.seed(0)
      
      print("Demonstrate noise in function evaluation:")
      for i in range(5):
          print("objfun(x0) = %s" % str(rosenbrock_noisy(x0)))
      print("")
      
      # Call DFO-LS
      soln = dfols.solve(rosenbrock_noisy, x0)
      
      # Display output
      print(soln)
      
      # Compare with a derivative-based solver
      import scipy.optimize as opt
      soln = opt.least_squares(rosenbrock_noisy, x0)
      
      print("")
      print("** SciPy results **")
      print("Solution xmin = %s" % str(soln.x))
      print("Objective value f(xmin) = %.10g" % (2.0 * soln.cost))
      print("Needed %g objective evaluations" % soln.nfev)
      print("Exit flag = %g" % soln.status)
      print(soln.message)


The output of this is:

  .. code-block:: none
  
      Demonstrate noise in function evaluation:
      objfun(x0) = [-4.4776183   2.20880346]
      objfun(x0) = [-4.44306447  2.24929965]
      objfun(x0) = [-4.48217255  2.17849989]
      objfun(x0) = [-4.44180389  2.19667014]
      objfun(x0) = [-4.39545837  2.20903317]
      
      ****** DFO-LS Results ******
      Solution xmin = [1.00000001 1.00000002]
      Residual vector = [ 5.17481720e-09 -1.04150014e-08]
      Objective value f(xmin) = 1.352509879e-16
      Needed 35 objective evaluations (at 35 points)
      Approximate Jacobian = [[-1.98079840e+01  1.00105722e+01]
       [-9.93887907e-01 -3.06567570e-04]]
      Solution xmin was evaluation point 35
      Approximate Jacobian formed using evaluation points [30 33 34]
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************
      
      
      ** SciPy results **
      Solution xmin = [-1.2  1. ]
      Objective value f(xmin) = 23.83907501
      Needed 5 objective evaluations
      Exit flag = 3
      `xtol` termination condition is satisfied.

DFO-LS is able to find the solution, but SciPy's derivative-based solver, which has no trouble solving the noise-free problem, is unable to make any progress.

As noted above, DFO-LS has an input parameter :code:`objfun_has_noise` to indicate if :code:`objfun` has noise in it, which it does in this case. Therefore we can call DFO-LS with

  .. code-block:: python
  
      soln = dfols.solve(rosenbrock_noisy, x0, objfun_has_noise=True)

Using this setting, we find the correct solution faster:

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [1. 1.]
      Residual vector = [-6.56093684e-10 -1.17835345e-10]
      Objective value f(xmin) = 4.443440912e-19
      Needed 28 objective evaluations (at 28 points)
      Approximate Jacobian = [[-1.98649933e+01  9.93403044e+00]
       [-9.93112150e-01  5.78830812e-03]]
      Solution xmin was evaluation point 28
      Approximate Jacobian formed using evaluation points [27 25 26]
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************

Example: Parameter Estimation/Data Fitting
------------------------------------------
Next, we show a short example of using DFO-LS to solve a parameter estimation problem (taken from `here <https://uk.mathworks.com/help/optim/ug/lsqcurvefit.html#examples>`_). Given some observations :math:`(t_i,y_i)`, we wish to calibrate parameters :math:`x=(x_1,x_2)` in the exponential decay model

.. math::

   y(t) = x_1 \exp(x_2 t)

The code for this is:

  .. code-block:: python
  
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
      
      # We expect exponential decay: set upper bound x[1] <= 0
      upper = np.array([1e20, 0.0])

      # Call DFO-LS
      soln = dfols.solve(prediction_error, x0, bounds=(None, upper))

      # Display output
      print(soln)

The output of this is (noting that DFO-LS moves :math:`x_0` to be far away enough from the upper bound)

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [ 4.98830861e+02 -1.01256863e-01]
      Residual vector = [-0.1816709   0.06098396  0.76276296  0.11962351 -0.26589799 -0.59788816
       -1.02611898 -1.51235371 -1.56145452 -1.63266662]
      Objective value f(xmin) = 9.504886892
      Needed 111 objective evaluations (at 111 points)
      Approximate Jacobian = [[-9.12901055e-01 -4.09843504e+02]
       [-8.59087363e-01 -6.42808534e+02]
       [-2.47254068e-01 -1.70205403e+03]
       [-1.34676757e-01 -1.33017163e+03]
       [-8.71358948e-02 -1.04752831e+03]
       [-5.75309286e-02 -8.09280596e+02]
       [-2.83185935e-02 -4.97239504e+02]
       [-2.22997879e-03 -6.70749550e+01]
       [-5.24146460e-04 -1.95045170e+01]
       [-2.65964661e-04 -1.07858021e+01]]
      Solution xmin was evaluation point 111
      Approximate Jacobian formed using evaluation points [104 109 110]
      Exit flag = 0
      Success: rho has reached rhoend
      ****************************

This produces a good fit to the observations.

.. image:: data_fitting.png
   :width: 75%
   :alt: Data Fitting Results
   :align: center

To generate this plot, run:

  .. code-block:: python
  
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

Example: Solving a Nonlinear System of Equations
------------------------------------------------
Lastly, we give an example of using DFO-LS to solve a nonlinear system of equations (taken from `here <http://support.sas.com/documentation/cdl/en/imlug/66112/HTML/default/viewer.htm#imlug_genstatexpls_sect004.htm>`_). We wish to solve the following set of equations

.. math::

   x_1 + x_2 - x_1 x_2 + 2 &= 0, \\
   x_1 \exp(-x_2) - 1 &= 0.

The code for this is:

  .. code-block:: python
  
      # DFO-LS example: Solving a nonlinear system of equations
      # Originally from:
      # http://support.sas.com/documentation/cdl/en/imlug/66112/HTML/default/viewer.htm#imlug_genstatexpls_sect004.htm
      
      from __future__ import print_function
      from math import exp
      import numpy as np
      import dfols
      
      # Want to solve:
      #   x1 + x2 - x1*x2 + 2 = 0
      #   x1 * exp(-x2) - 1   = 0
      def nonlinear_system(x):
          return np.array([x[0] + x[1] - x[0]*x[1] + 2, 
                           x[0] * exp(-x[1]) - 1.0])
      
      # Warning: if there are multiple solutions, which one
      #          DFO-LS returns will likely depend on x0!
      x0 = np.array([0.1, -2.0])
      
      # Call DFO-LS
      soln = dfols.solve(nonlinear_system, x0)
      
      # Display output
      print(soln)


The output of this is

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [ 0.09777309 -2.32510588]
      Residual vector = [-1.38601752e-09 -1.70204653e-08]
      Objective value f(xmin) = 2.916172822e-16
      Needed 13 objective evaluations (at 13 points)
      Approximate Jacobian = [[ 3.32527052  0.90227531]
       [10.22943034 -0.99958226]]
      Solution xmin was evaluation point 13
      Approximate Jacobian formed using evaluation points [ 8 11 12]
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************

Here, we see that both entries of the residual vector are very small, so both equations have been solved to high accuracy.

References
----------

.. [CFMR2018]   
   Coralia Cartis, Jan Fiala, Benjamin Marteau and Lindon Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`preprint <https://arxiv.org/abs/1804.00154>`_] 

.. [HR2022]   
   Matthew Hough and Lindon Roberts, `Model-Based Derivative-Free Methods for Convex-Constrained Optimization <https://doi.org/10.1137/21M1460971>`_, *SIAM Journal on Optimization*, 21:4 (2022), pp. 2552-2579 [`preprint <https://arxiv.org/abs/2111.05443>`_].

.. [LLR2024]   
   Yanjun Liu, Kevin H. Lam and Lindon Roberts, `Black-box Optimization Algorithms for Regularized Least-squares Problems <http://arxiv.org/abs/2407.14915>`_, *arXiv preprint arXiv:2407.14915* (2024).

.. [B2017]
   Amir Beck, `First-Order Methods in Optimization <https://doi.org/10.1137/1.9781611974997>`_, SIAM (2017).
