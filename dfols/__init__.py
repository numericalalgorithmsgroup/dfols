"""
DFO-LS
====

Derivative-Free Optimization for Least-Squares (DFO-LS) is a
nonlinear least-squares solver which only requires function values.

It solves the nonlinear least-squares problem:
    min_{x}  f(x) = r1(x)**2 + ... + rm(x)**2,
(optionally) subject to finitely many convex constraints,
where each function ri(x) is differentiable, possibly nonconvex.
Since the derivatives of ri(x) are never required or approximated,
the solver works when the evaluation of ri(x) is noisy.

----

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

The development of this software was sponsored by NAG Ltd. (http://www.nag.co.uk)
and the EPSRC Centre For Doctoral Training in Industrially Focused Mathematical
Modelling (EP/L015803/1) at the University of Oxford. Please contact NAG for
alternative licensing.


"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

# DFO-LS version
__version__ = '1.4.1'

# Main solver & exit flags
from .solver import *
__all__ = ['solve']

