"""

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

import numpy as np
import unittest

import dfols


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def rosenbrock_jacobian(x):
    return np.array([[-20.0*x[0], 10.0], [-1.0, 0.0]])


def p_box(x,l,u):
    return np.minimum(np.maximum(x,l), u)

def p_ball(x,c,r):
    return c + (r/np.max([np.linalg.norm(x-c),r]))*(x-c)


class TestNans(unittest.TestCase):
    # Generic objective that only returns NaNs (like optclim code)
    # Verify get a sensible termination
    def runTest(self):
        x0 = np.array([-1.2, 1.0])
        r_error = lambda x: np.array([np.nan, np.nan, np.nan])
        # First attempt: exit gracefully
        soln = dfols.solve(r_error, x0)
        self.assertEqual(soln.flag, soln.EXIT_LINALG_ERROR, "Wrong error message")
        # Second attempt: throw error when trying to interpolate
        with self.assertRaises(np.linalg.LinAlgError):
            soln = dfols.solve(r_error, x0, user_params={"interpolation.throw_error_on_nans": True})


class TestRosenbrockGeneric(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        np.random.seed(0)
        soln = dfols.solve(rosenbrock, x0)
        # print(soln.x)
        self.assertTrue(array_compare(soln.x, np.array([1.0, 1.0]), thresh=1e-4), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        # print(soln.jacobian, rosenbrock_jacobian(soln.x))
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=2e-1), "Wrong Jacobian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")


class TestRosenbrockBounds(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function, where x[1] hits the upper bound
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 0.7])  # standard start point does not satisfy bounds
        lower = np.array([-2.0, -2.0])
        upper = np.array([1.0, 0.9])
        xmin = np.array([0.9486, 0.9])  # approximate
        fmin = np.dot(rosenbrock(xmin), rosenbrock(xmin))
        soln = dfols.solve(rosenbrock, x0, bounds=(lower, upper))
        print(soln.x)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")


class TestRosenbrockBounds2(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function, where x[0] hits upper bound
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 0.7])  # standard start point too close to upper bounds
        lower = np.array([-2.0, -2.0])
        upper = np.array([0.9, 0.9])
        xmin = np.array([0.9, 0.81])  # approximate
        fmin = np.dot(rosenbrock(xmin), rosenbrock(xmin))
        jacmin = np.array([[-18.0, 10.0], [-1.0, 0.0]])
        soln = dfols.solve(rosenbrock, x0, bounds=(lower, upper))
        print(soln.x)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")
        self.assertTrue(array_compare(soln.jacobian, jacmin, thresh=2e-2), "Wrong jacobian")


class TestLinear(unittest.TestCase):
    # Solve min_x ||Ax-b||^2, for some random A and b
    def runTest(self):
        n, m = 2, 5
        np.random.seed(0)  # (fixing random seed)
        A = np.random.rand(m, n)
        b = np.random.rand(m)
        objfun = lambda x: np.dot(A, x) - b
        xmin = np.linalg.lstsq(A, b, rcond=None)[0]
        fmin = np.dot(objfun(xmin), objfun(xmin))
        x0 = np.zeros((n,))
        soln = dfols.solve(objfun, x0)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, objfun(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, A, thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")


class TestRosenbrockRegression(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        soln = dfols.solve(rosenbrock, x0, npt=5)
        self.assertTrue(array_compare(soln.x, np.array([1.0, 1.0]), thresh=1e-4), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=1.5e-1), "Wrong Jacobian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")


class TestRosenbrockGrowing(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        soln = dfols.solve(rosenbrock, x0, user_params={"growing.ndirs_initial": 1})
        self.assertTrue(array_compare(soln.x, np.array([1.0, 1.0]), thresh=1e-4), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        print(soln.jacobian)
        print(rosenbrock_jacobian(soln.x))
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=3e-0), "Wrong Jacobian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")


class TestInverseProblem(unittest.TestCase):
    # Minimise an inverse problem
    def runTest(self):
        # Simple problem with global minimum at the origin, for instance
        objfun = lambda x: np.array([np.sin(x[0])**2, np.sin(x[1])**2, np.sum(np.sin(x[2:]**2))])
        jac = lambda x: np.array([[2*np.sin(x[0])*np.cos(x[0]), 0, 0, 0, 0], 
                                  [0, 2*np.sin(x[1])*np.cos(x[1]), 0, 0, 0], 
                                  [0, 0, 2*np.sin(x[2])*np.cos(x[2]), 2*np.sin(x[3])*np.cos(x[3]), 2*np.sin(x[4])*np.cos(x[4])]])  # for n=5 only
        x0 = np.ones((5,))
        np.random.seed(0)
        soln = dfols.solve(objfun, x0)
        self.assertTrue(array_compare(soln.x, np.zeros((5,)), thresh=1e-3), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, objfun(soln.x), thresh=1e-10), "Wrong resid")
        print(soln.jacobian)
        print(jac(soln.x))
        self.assertTrue(array_compare(soln.jacobian, jac(soln.x), thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")


class TestInverseProblemGrowing(unittest.TestCase):
    # Minimise an inverse problem, with growing
    def runTest(self):
        # Simple problem with global minimum at the origin, for instance
        objfun = lambda x: np.array([np.sin(x[0])**2, np.sin(x[1])**2, np.sum(np.sin(x[2:]**2))])
        jac = lambda x: np.array([[2*np.sin(x[0])*np.cos(x[0]), 0, 0, 0, 0], 
                                  [0, 2*np.sin(x[1])*np.cos(x[1]), 0, 0, 0], 
                                  [0, 0, 2*np.sin(x[2])*np.cos(x[2]), 2*np.sin(x[3])*np.cos(x[3]), 2*np.sin(x[4])*np.cos(x[4])]])  # for n=5 only
        x0 = np.ones((5,))
        np.random.seed(0)
        soln = dfols.solve(objfun, x0, user_params={'growing.ndirs_initial':2})
        self.assertTrue(array_compare(soln.x, np.zeros((5,)), thresh=1e-3), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, objfun(soln.x), thresh=1e-10), "Wrong resid")
        print(soln.jacobian)
        print(jac(soln.x))
        self.assertTrue(array_compare(soln.jacobian, jac(soln.x), thresh=1e-1), "Wrong Jacobian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")


class TestRosenbrockBoxBall(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function, where x[1] hits the upper bound
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 0.7])  # standard start point does not satisfy bounds
        lower = np.array([0.7, -2.0])
        upper = np.array([1.0, 2])
        boxproj = lambda x: p_box(x,lower,upper)
        ballproj = lambda x: p_ball(x,np.array([0.5,1]),0.25)
        xmin = np.array([0.70424386, 0.85583188])  # approximate
        fmin = np.dot(rosenbrock(xmin), rosenbrock(xmin))
        soln = dfols.solve(rosenbrock, x0, projections=[boxproj,ballproj])
        print(soln.x)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")
