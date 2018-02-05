.. DFO-LS documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DFO-LS: Derivative-Free Optimizer for Least-Squares Minimization
================================================================

**Release:** |version|

**Date:** |today|

**Author:** `Lindon Roberts <lindon.roberts@maths.ox.ac.uk>`_

DFO-LS is a flexible package for finding local solutions to nonlinear least-squares minimization problems (with optional bound constraints), without requiring any derivatives of the objective. DFO-LS stands for Derivative-Free Optimizer for Least-Squares.

That is, DFO-LS solves

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
   \text{s.t.} &\quad  a \leq x \leq b

Full details of the DFO-LS algorithm are given in our paper: C. Cartis, J. Fiala, B. Marteau and L. Roberts, Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers, technical report, University of Oxford, (2018).

DFO-LS is released under the GNU General Public License. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   info
   userguide
   advanced
   diagnostic

Acknowledgements
----------------
This software was developed under the supervision of `Coralia Cartis <https://www.maths.ox.ac.uk/people/coralia.cartis>`_, and was supported by the EPSRC Centre For Doctoral Training in `Industrially Focused Mathematical Modelling <https://www.maths.ox.ac.uk/study-here/postgraduate-study/industrially-focused-mathematical-modelling-epsrc-cdt>`_ (EP/L015803/1) in collaboration with the `Numerical Algorithms Group <http://www.nag.com/>`_.

