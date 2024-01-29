.. DFO-LS documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DFO-LS: Derivative-Free Optimizer for Least-Squares Minimization
================================================================

**Release:** |version|

**Date:** |today|

**Author:** `Lindon Roberts <lindon.roberts@sydney.edu.au>`_

DFO-LS is a flexible package for finding local solutions to nonlinear least-squares minimization problems (with optional constraints), without requiring any derivatives of the objective. DFO-LS stands for Derivative-Free Optimizer for Least-Squares.

That is, DFO-LS solves

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
   \text{s.t.} &\quad x \in C\\
               &\quad  a \leq x \leq b

The constraint set :math:`C` is the intersection of multiple convex sets provided as input by the user. All constraints are non-relaxable (i.e. DFO-LS will never ask to evaluate a point that is not feasible).

Full details of the DFO-LS algorithm are given in our papers: 

* C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`preprint <https://arxiv.org/abs/1804.00154>`_] . 
* Hough, M. and Roberts, L., `Model-Based Derivative-Free Methods for Convex-Constrained Optimization <https://arxiv.org/abs/2111.05443>`_, *arXiv preprint arXiv:2111.05443*, (2021).

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
This software was initially developed under the supervision of `Coralia Cartis <https://www.maths.ox.ac.uk/people/coralia.cartis>`_, and was supported by the EPSRC Centre For Doctoral Training in `Industrially Focused Mathematical Modelling <https://www.maths.ox.ac.uk/study-here/postgraduate-study/industrially-focused-mathematical-modelling-epsrc-cdt>`_ (EP/L015803/1) in collaboration with the `Numerical Algorithms Group <http://www.nag.com/>`_.

