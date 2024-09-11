.. DFO-LS documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DFO-LS: Derivative-Free Optimizer for Least-Squares Minimization
================================================================

**Release:** |version|

**Date:** |today|

**Author:** `Lindon Roberts <lindon.roberts@sydney.edu.au>`_

DFO-LS is a flexible package for finding local solutions to nonlinear least-squares minimization problems (with optional regularizer and constraints), without requiring any derivatives of the objective. DFO-LS stands for Derivative-Free Optimizer for Least-Squares.

That is, DFO-LS solves

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 + h(x) \\
   \text{s.t.} &\quad  a \leq x \leq b\\
               &\quad x \in C := C_1 \cap \cdots \cap C_n, \quad \text{all $C_i$ convex}\\

The optional regularizer :math:`h(x)` is a Lipschitz continuous and convex, but possibly non-differentiable function that is typically used to avoid overfitting. 
A common choice is :math:`h(x)=\lambda \|x\|_1` (called L1 regularization or LASSO) for :math:`\lambda>0`. 
Note that in the case of Tikhonov regularization/ridge regression, :math:`h(x)=\lambda\|x\|_2^2` is not Lipschitz continuous, so should instead be incorporated by adding an extra term into the least-squares sum, :math:`r_{m+1}(x)=\sqrt{\lambda} \|x\|_2`.
The (optional) constraint set :math:`C` is the intersection of multiple convex sets provided as input by the user. All constraints are non-relaxable (i.e. DFO-LS will never ask to evaluate a point that is not feasible), although the general constraints :math:`x\in C` may be slightly violated from rounding errors.

Full details of the DFO-LS algorithm are given in our papers: 

1. C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`preprint <https://arxiv.org/abs/1804.00154>`_] . 
2. M. Hough, and L. Roberts, `Model-Based Derivative-Free Methods for Convex-Constrained Optimization <https://doi.org/10.1137/21M1460971>`_, *SIAM Journal on Optimization*, 21:4 (2022), pp. 2552-2579 [`preprint <https://arxiv.org/abs/2111.05443>`_].
3. Y. Liu, K. H. Lam and L. Roberts, `Black-box Optimization Algorithms for Regularized Least-squares Problems <http://arxiv.org/abs/2407.14915>`_, *arXiv preprint arXiv:arXiv:2407.14915*, 2024.

DFO-LS is a more flexible version of `DFO-GN <https://github.com/numericalalgorithmsgroup/dfogn>`_.

If you are interested in solving general optimization problems (without a least-squares structure), you may wish to try `Py-BOBYQA <https://github.com/numericalalgorithmsgroup/pybobyqa>`_, which has many of the same features as DFO-LS.

DFO-LS is released under the GNU General Public License. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   info
   userguide
   advanced
   diagnostic
   history
   contributors

Acknowledgements
----------------
This software was initially developed under the supervision of `Coralia Cartis <https://www.maths.ox.ac.uk/people/coralia.cartis>`_, and was supported by the EPSRC Centre For Doctoral Training in `Industrially Focused Mathematical Modelling <https://www.maths.ox.ac.uk/study-here/postgraduate-study/industrially-focused-mathematical-modelling-epsrc-cdt>`_ (EP/L015803/1) in collaboration with the `Numerical Algorithms Group <http://www.nag.com/>`_. Development of DFO-LS has also been supported by the Australian Research Council (DE240100006).
