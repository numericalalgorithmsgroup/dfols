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

from dfols.util import *


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


class TestSumsq(unittest.TestCase):
    def runTest(self):
        n = 10
        x = np.sin(np.arange(n))
        normx = np.sum(x**2)
        self.assertAlmostEqual(normx, sumsq(x), msg='Wrong answer')


class TestEval(unittest.TestCase):
    def runTest(self):
        objfun = lambda x : np.array([10*(x[1]-x[0]**2), 1-x[0]])
        x = np.array([-1.2, 1.0])
        fvec, f = eval_least_squares_objective(objfun, x)
        self.assertTrue(np.all(fvec == objfun(x)), 'Residuals wrong')
        self.assertAlmostEqual(f, sumsq(fvec), msg='Sum of squares wrong')


class TestModelValue(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=float).reshape((n, n))
        H = np.sin(A + A.T)  # force symmetric
        vec = np.exp(np.arange(n, dtype=float))
        g = np.cos(3*np.arange(n, dtype=float) - 2.0)
        xopt = np.ones((n,))
        h = lambda d: np.linalg.norm(d, 1)
        mval = np.dot(g, vec) + 0.5 * np.dot(vec, np.dot(H, vec)) + h(xopt + vec)
        self.assertAlmostEqual(mval, model_value(g, H, h, xopt, vec), msg='Wrong value')


class TestRandom(unittest.TestCase):
    def runTest(self):
        n = 3
        lower = -10.0 * np.ones((n,))
        upper = 10.0 * np.ones((n,))
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        for i in range(n):
            self.assertTrue(array_compare(dirns[i, :], -dirns[n+i,:]), "Second set should be -ve first set")
        for i in range(2*n-1):
            self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First 2n directions should be orthog")


class TestRandomBox(unittest.TestCase):
    def runTest(self):
        n = 5
        lower = np.array([-10.0, -0.1, -0.5, 0.0, -1.0])
        upper = np.array([10.0, 10.0, 0.2, 10.0, 0.0])
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper)
        # print(dirns)
        for i in range(num_pts):
            # self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        # for i in range(n):
        #     self.assertTrue(array_compare(dirns[i, :], -dirns[n+i,:]), "Second set should be -ve first set")
        # for i in range(2*n-1):
        #     self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First 2n directions should be orthog")


class TestRandomShort(unittest.TestCase):
    def runTest(self):
        n = 3
        lower = -10.0 * np.ones((n,))
        upper = 10.0 * np.ones((n,))
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper, with_neg_dirns=False)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        for i in range(n-1):
            self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First n directions should be orthog")
        # print(dirns)
        # self.assertTrue(False, "bad")


class TestRandomBoxShort(unittest.TestCase):
    def runTest(self):
        n = 5
        lower = np.array([-10.0, -0.1, -0.5, 0.0, -1.0])
        upper = np.array([10.0, 10.0, 0.2, 10.0, 0.0])
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_orthog_directions_within_bounds(num_pts, delta, lower, upper, with_neg_dirns=False)
        # print(dirns)
        for i in range(num_pts):
            # self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        # for i in range(n):
        #     self.assertTrue(array_compare(dirns[i, :], -dirns[n+i,:]), "Second set should be -ve first set")
        # for i in range(2*n-1):
        #     self.assertTrue(abs(np.dot(dirns[i, :], dirns[i+1, :])) < 1e-10, "First 2n directions should be orthog")
        # self.assertTrue(False, "bad")


class TestRandomNotOrthog(unittest.TestCase):
    def runTest(self):
        n = 3
        lower = -10.0 * np.ones((n,))
        upper = 10.0 * np.ones((n,))
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_directions_within_bounds(num_pts, delta, lower, upper)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        # print(dirns)
        # self.assertTrue(False, "bad")


class TestRandomNotOrthogBox(unittest.TestCase):
    def runTest(self):
        n = 5
        lower = np.array([-10.0, -0.1, -0.5, 0.0, -1.0])
        upper = np.array([10.0, 10.0, 0.2, 10.0, 0.0])
        num_pts = 2 * n + 4
        delta = 1.0
        dirns = random_directions_within_bounds(num_pts, delta, lower, upper)
        # print(dirns)
        for i in range(num_pts):
            self.assertTrue(np.linalg.norm(dirns[i, :]) <= delta + 1e-10, "Unconstrained: dirn %i too long" % i)
            self.assertTrue(np.all(dirns[i, :] >= lower), "Direction %i below lower bound" % i)
            self.assertTrue(np.all(dirns[i, :] <= upper), "Direction %i above upper bound" % i)
        # self.assertTrue(False, "bad")

# Trivial case of full rank
class TestMatrixRankQR1(unittest.TestCase):
    def runTest(self):
        mr_tol = 1e-18
        A = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]])
        rank, D = qr_rank(A,mr_tol)
        self.assertTrue(np.all(D > mr_tol), "Incorrect diagonal matrix output")
        self.assertTrue(rank == 4, "Incorrect rank output")

# Full rank but QR has negative entries for diag(R)
class TestMatrixRankQR2(unittest.TestCase):
    def runTest(self):
        mr_tol = 1e-18
        A = np.array([
            [1,2,3,4],
            [0,6,7,8],
            [-1,-2,-2,-1],
            [4,2,2,1]])
        rank, D = qr_rank(A,mr_tol)
        self.assertTrue(np.all(D > mr_tol), "Incorrect diagonal matrix output")
        self.assertTrue(rank == 4, "Incorrect rank output")


# Not full rank
class TestMatrixRankQR3(unittest.TestCase):
    def runTest(self):
        mr_tol = 1e-18
        A = np.array([
            [1,2,3,4],
            [2,6,4,8],
            [0,0,0,0],
            [0,0,0,0]])
        rank, D = qr_rank(A,mr_tol)
        self.assertTrue(np.all(D[0:2] > mr_tol), "Incorrect diagonal matrix output (rows 1,2)")
        self.assertTrue(np.all(D[2:4] <= mr_tol), "Incorrect diagonal matrix output (rows 3,4)")
        self.assertTrue(rank == 2, "Incorrect rank output")


class TestDykstraBoxInt(unittest.TestCase):
    def runTest(self):
        x0 = np.array([0,0])
        lower = np.array([-0.01, -0.1])
        upper = np.array([0.01, 0.5])
        boxproj = lambda x: pbox(x,lower,upper)
        P = [boxproj]
        xproj = dykstra(P,x0)
        self.assertTrue(np.all(xproj == x0), "Incorrect point returned by Dykstra")


class TestDykstraBoxExt(unittest.TestCase):
    def runTest(self):
        x0 = np.array([-2,5])
        lower = np.array([-1, -1])
        upper = np.array([0.5, 0.9])
        boxproj = lambda x: pbox(x,lower,upper)
        P = [boxproj]
        xproj = dykstra(P,x0)
        xtrue = np.array([-1,0.9])
        self.assertTrue(np.allclose(xproj, xtrue), "Incorrect point returned by Dykstra")

class TestDykstraBallInt(unittest.TestCase):
    def runTest(self):
        x0 = np.array([0,0])
        ballproj = lambda x: pball(x,x0+1,2)
        P = [ballproj]
        xproj = dykstra(P,x0)
        self.assertTrue(np.all(xproj == x0), "Incorrect point returned by Dykstra")


class TestDykstraBallExt(unittest.TestCase):
    def runTest(self):
        x0 = np.array([-3,5])
        ballproj = lambda x: pball(x,np.array([-0.5,1]),1)
        P = [ballproj]
        xproj = dykstra(P,x0)
        xtrue = np.array([-1.02999894, 1.8479983])
        self.assertTrue(np.allclose(xproj, xtrue), "Incorrect point returned by Dykstra")


class TestDykstraBoxBallInt(unittest.TestCase):
    def runTest(self):
        x0 = np.array([0.72,1.1])
        lower = np.array([0.7, -2.0])
        upper = np.array([1.0, 2])
        boxproj = lambda x: pbox(x,lower,upper)
        ballproj = lambda x: pball(x,np.array([0.5,1]),0.25)
        P = [boxproj,ballproj]
        xproj = dykstra(P,x0)
        self.assertTrue(np.all(xproj == x0), "Incorrect point returned by Dykstra")

class TestDykstraBoxBallExt1(unittest.TestCase):
    def runTest(self):
        x0 = np.array([0,4])
        lower = np.array([0.7, -2.0])
        upper = np.array([1.0, 2])
        boxproj = lambda x: pbox(x,lower,upper)
        ballproj = lambda x: pball(x,np.array([0.5,1]),0.25)
        P = [boxproj,ballproj]
        xproj = dykstra(P,x0)
        xtrue = np.array([0.6940582, 1.1576116])
        self.assertTrue(np.allclose(xproj, xtrue), "Incorrect point returned by Dykstra")


class TestDykstraBoxBallExt2(unittest.TestCase):
    def runTest(self):
        x0 = np.array([0.8,-3])
        lower = np.array([0.7, -2.0])
        upper = np.array([1.0, 2])
        boxproj = lambda x: pbox(x,lower,upper)
        ballproj = lambda x: pball(x,np.array([0.5,1]),0.25)
        P = [boxproj,ballproj]
        xproj = dykstra(P,x0)
        xtrue = np.array([0.68976232, 0.8372417])
        self.assertTrue(np.allclose(xproj, xtrue), "Incorrect point returned by Dykstra")


class TestDykstraBoxBallBdry(unittest.TestCase):
    def runTest(self):
        x0 = np.array([0.7,0.85])
        lower = np.array([0.7, -2.0])
        upper = np.array([1.0, 2])
        boxproj = lambda x: pbox(x,lower,upper)
        ballproj = lambda x: pball(x,np.array([0.5,1]),0.25)
        P = [boxproj,ballproj]
        xproj = dykstra(P,x0)
        self.assertTrue(np.allclose(xproj, x0), "Incorrect point returned by Dykstra")
