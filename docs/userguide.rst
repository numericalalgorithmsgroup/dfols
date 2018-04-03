Using DFO-LS
============
This section describes the main interface to DFO-LS and how to use it.

Nonlinear Least-Squares Minimization
------------------------------------
DFO-LS is designed to solve the local optimization problem

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
   \text{s.t.} &\quad  a \leq x \leq b

where the bound constraints :math:`a \leq x \leq b` are optional.

DFO-LS iteratively constructs an interpolation-based model for the objective, and determines a step using a trust-region framework.
For an in-depth technical description of the algorithm see the paper [CFMR2018]_.

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

The possible values of :code:`soln.flag` are defined by the following variables:

* :code:`soln.EXIT_SUCCESS` - DFO-LS terminated successfully (the objective value or trust region radius are sufficiently small).
* :code:`soln.EXIT_MAXFUN_WARNING` - maximum allowed objective evaluations reached. This is the most likely return value when using multiple restarts.
* :code:`soln.EXIT_SLOW_WARNING` - maximum number of slow iterations reached.
* :code:`soln.EXIT_FALSE_SUCCESS_WARNING` - DFO-LS reached the maximum number of restarts which decreased the objective, but to a worse value than was found in a previous run.
* :code:`soln.EXIT_INPUT_ERROR` - error in the inputs.
* :code:`soln.EXIT_TR_INCREASE_ERROR` - error occurred when solving the trust region subproblem.
* :code:`soln.EXIT_LINALG_ERROR` - linear algebra error, e.g. the interpolation points produced a singular linear system.

These variables are defined in the :code:`soln` object, so can be accessed with, for example

  .. code-block:: python
  
      if soln.flag == soln.EXIT_SUCCESS:
          print("Success!")

Optional Arguments
------------------
The :code:`solve` function has several optional arguments which the user may provide:

  .. code-block:: python
  
      dfols.solve(objfun, x0, args=(), bounds=None, npt=None, rhobeg=None, 
                  rhoend=1e-8, maxfun=None, nsamples=None, 
                  user_params=None, objfun_has_noise=False, 
                  scaling_within_bounds=False)

These arguments are:

* :code:`args` - a tuple of extra arguments passed to the objective function. This feature is new, and not yet avaiable in the PyPI version of DFO-LS; instead, use Python's built-in function :code:`lambda`.
* :code:`bounds` - a tuple :code:`(lower, upper)` with the vectors :math:`a` and :math:`b` of lower and upper bounds on :math:`x` (default is :math:`a_i=-10^{20}` and :math:`b_i=10^{20}`). To set bounds for either :code:`lower` or :code:`upper`, but not both, pass a tuple :code:`(lower, None)` or :code:`(None, upper)`.
* :code:`npt` - the number of interpolation points to use (default is :code:`len(x0)+1`). If using restarts, this is the number of points to use in the first run of the solver, before any restarts (and may be optionally increased via settings in :code:`user_params`).
* :code:`rhobeg` - the initial value of the trust region radius (default is :math:`0.1\max(\|x_0\|_{\infty}, 1)`).
* :code:`rhoend` - minimum allowed value of trust region radius, which determines when a successful termination occurs (default is :math:`10^{-8}`).
* :code:`maxfun` - the maximum number of objective evaluations the algorithm may request (default is :math:`\min(100(n+1),1000)`).
* :code:`nsamples` - a Python function :code:`nsamples(delta, rho, iter, nrestarts)` which returns the number of times to evaluate :code:`objfun` at a given point. This is only applicable for objectives with stochastic noise, when averaging multiple evaluations at the same point produces a more accurate value. The input parameters are the trust region radius (:code:`delta`), the lower bound on the trust region radius (:code:`rho`), how many iterations the algorithm has been running for (:code:`iter`), and how many restarts have been performed (:code:`nrestarts`). Default is no averaging (i.e. :code:`nsamples(delta, rho, iter, nrestarts)=1`).
* :code:`user_params` - a Python dictionary :code:`{'param1': val1, 'param2':val2, ...}` of optional parameters. A full list of available options is given in the next section :doc:`advanced`.
* :code:`objfun_has_noise` - a flag to indicate whether or not :code:`objfun` has stochastic noise; i.e. will calling :code:`objfun(x)` multiple times at the same value of :code:`x` give different results? This is used to set some sensible default parameters (including using multiple restarts), all of which can be overridden by the values provided in :code:`user_params`.
* :code:`scaling_within_bounds` - a flag to indicate whether the algorithm should internally shift and scale the entries of :code:`x` so that the bounds become :math:`0 \leq x \leq 1`. This is useful is you are setting :code:`bounds` and the bounds have different orders of magnitude. If :code:`scaling_within_bounds=True`, the values of :code:`rhobeg` and :code:`rhoend` apply to the *shifted* variables.

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
      
      # Set random seed (for reproducibility)
      np.random.seed(0)
      
      # Call DFO-LS
      soln = dfols.solve(rosenbrock, x0)
      
      # Display output
      print(soln)
      
Note that DFO-LS is a randomized algorithm: in its first phase, it builds an internal approximation to the objective function by sampling it along random directions. In the code above, we set NumPy's random seed for reproducibility over multiple runs, but this is not required. The output of this script, showing that DFO-LS finds the correct solution, is

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [ 1.  1.]
      Residual vector = [ -2.22044605e-15   0.00000000e+00]
      Objective value f(xmin) = 4.930380658e-30
      Needed 36 objective evaluations (at 36 points)
      Approximate Jacobian = [[ -1.98957443e+01   1.00000000e+01]
       [ -1.00000000e+00   8.37285083e-16]]
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************

This and all following problems can be found in the `examples <https://github.com/numericalalgorithmsgroup/dfols/tree/master/examples>`_ directory on the DFO-LS Github page.

Adding Bounds and More Output
-----------------------------
We can extend the above script to add constraints. To do this, we can add the lines

  .. code-block:: python
  
      # Define bound constraints (lower <= x <= upper)
      lower = np.array([-10.0, -10.0])
      upper = np.array([0.9, 0.85])
      
      # Call DFO-LS (with bounds)
      soln = dfols.solve(rosenbrock, x0, bounds=(lower, upper))

DFO-LS correctly finds the solution to the constrained problem:

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [ 0.9   0.81]
      Residual vector = [ 0.   0.1]
      Objective value f(xmin) = 0.01
      Needed 65 objective evaluations (at 65 points)
      Approximate Jacobian = [[ -1.79999998e+01   9.99999990e+00]
       [ -9.99999998e-01  -2.53940698e-09]]
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

And we can now see each evaluation of :code:`objfun`:

  .. code-block:: none
  
      Function eval 1 at point 1 has f = 39.65 at x = [-1.2   0.85]
      Initialising (random directions)
      Function eval 2 at point 2 has f = 14.337296 at x = [-1.08  0.85]
      Function eval 3 at point 3 has f = 55.25 at x = [-1.2   0.73]
      ...
      Function eval 64 at point 64 has f = 0.0100000029949496 at x = [ 0.89999999  0.81      ]
      Function eval 65 at point 65 has f = 0.00999999999999993 at x = [ 0.9   0.81]
      Did a total of 1 run(s)

If we wanted to save this output to a file, we could replace the above call to :code:`logging.basicConfig()` with

  .. code-block:: python
  
      logging.basicConfig(filename="myfile.log", level=logging.INFO, 
                          format='%(message)s', filemode='w')

Example: Noisy Objective Evaluation
-----------------------------------
As described in :doc:`info`, derivative-free algorithms such as DFO-LS are particularly useful when :code:`objfun` has noise. Let's modify the previous example to include random noise in our objective evaluation, and compare it to a derivative-based solver:

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
      Solution xmin = [ 1.  1.]
      Residual vector = [  3.51006670e-08   2.00158313e-10]
      Objective value f(xmin) = 1.232096886e-15
      Needed 46 objective evaluations (at 46 points)
      Approximate Jacobian = [[ -2.04330578e+01   1.00296466e+01]
       [ -9.88260906e-01  -3.77364910e-03]]
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************
      
      
      ** SciPy results **
      Solution xmin = [-1.2  1. ]
      Objective value f(xmin) = 23.96809472
      Needed 5 objective evaluations
      Exit flag = 3
      `xtol` termination condition is satisfied.

DFO-LS is able to find the solution with only 10 more function evaluations than in the noise-free case. However SciPy's derivative-based solver, which has no trouble solving the noise-free problem, is unable to make any progress.

As noted above, DFO-LS has an input parameter :code:`objfun_has_noise` to indicate if :code:`objfun` has noise in it, which it does in this case. Therefore we can call DFO-LS with

  .. code-block:: python
  
      soln = dfols.solve(rosenbrock_noisy, x0, objfun_has_noise=True)

Using this setting, we find the correct solution faster:

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [ 1.  1.]
      Residual vector = [ -5.80172077e-09   2.10781076e-09]
      Objective value f(xmin) = 3.810283004e-17
      Needed 29 objective evaluations (at 29 points)
      Approximate Jacobian = [[ -1.96671666e+01   9.88784341e+00]
       [ -1.00451147e+00   1.43596001e-04]]
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
      
      # Set random seed (for reproducibility)
      np.random.seed(0)
      
      # We expect exponential decay: set upper bound x[1] <= 0
      upper = np.array([1e20, 0.0])

      # Call DFO-LS
      soln = dfols.solve(prediction_error, x0, bounds=(None, upper))

      # Display output
      print(soln)

The output of this is (noting that DFO-LS moves :math:`x_0` to be far away enough from the upper bound)

  .. code-block:: none
  
      RuntimeWarning: x0 too close to upper bound, adjusting
      ****** DFO-LS Results ******
      Solution xmin = [  4.98830860e+02  -1.01256863e-01]
      Residual vector = [-0.18167084  0.06098401  0.76276294  0.11962349 -0.265898   -0.59788818
       -1.02611899 -1.51235371 -1.56145452 -1.63266662]
      Objective value f(xmin) = 9.504886892
      Needed 99 objective evaluations (at 99 points)
      Approximate Jacobian = [[ -9.12901557e-01  -4.09843510e+02]
       [ -8.59085471e-01  -6.42808522e+02]
       [ -2.47253894e-01  -1.70205399e+03]
       [ -1.34675403e-01  -1.33017159e+03]
       [ -8.71359818e-02  -1.04752827e+03]
       [ -5.75305576e-02  -8.09280563e+02]
       [ -2.83185322e-02  -4.97239478e+02]
       [ -2.22993603e-03  -6.70749492e+01]
       [ -5.24135530e-04  -1.95045149e+01]
       [ -2.65977795e-04  -1.07858009e+01]]
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
      
      # Set random seed (for reproducibility)
      np.random.seed(0)
      
      # Call DFO-LS
      soln = dfols.solve(nonlinear_system, x0)
      
      # Display output
      print(soln)


The output of this is

  .. code-block:: none
  
      ****** DFO-LS Results ******
      Solution xmin = [ 0.09777309 -2.32510588]
      Residual vector = [  2.89990254e-13   3.31557004e-12]
      Objective value f(xmin) = 1.107709904e-23
      Needed 18 objective evaluations (at 18 points)
      Approximate Jacobian = [[  3.32510429   0.90222738]
       [ 10.22774647  -0.9999939 ]]
      Exit flag = 0
      Success: Objective is sufficiently small
      ****************************

Here, we see that both entries of the residual vector are very small, so both equations have been solved to high accuracy.

References
----------

.. [CFMR2018]   
   C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://arxiv.org/abs/1804.00154>`_, technical report, University of Oxford, (2018).

