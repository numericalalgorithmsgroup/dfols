# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt
import numpy as np
import unittest

from dfols.trust_region import ctrsbox_sfista
from dfols.util import model_value, pball, pbox

# NOTE: S-FISTA slow because of the implicit form of prox_uh
def prox_uh(x, u):
    # prox_uh(d) = min_{s} ||s-d||^2 / 2u + h(s) 
    # When h is 1-norm, we have the explicit solution
    #rtn = np.zeros(x.shape)
    #for i in range(x.shape[0]):
    #    rtn[i] = np.sign(x[i])*max(abs(x[i])-u, 0)
    rtn = np.sign(x) * np.maximum(np.abs(x)-u, 0.0)
    return rtn

class TestUncInternalCDFO(unittest.TestCase):
    def runTest(self):
        n = 3
        g = np.array([1.0, 0.0, 1.0])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        h = lambda d: np.linalg.norm(d, 1)
        L_h = sqrt(n)
        delta = 2.0
        xopt = np.ones((n,))  # trying nonzero (since bounds inactive)
        func_tol = 1e-4
        d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox_uh, func_tol=func_tol)
        for i in range(50):
            d_e = delta * np.ones(n) # initialize d_e
            while np.linalg.norm(d_e, 2) > delta:
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = model_value(g, H, d_k, xopt, h)
            func_d_e = model_value(g, H, d_e, xopt, h)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "sufficient decrease does not achieved!")

class TestUncBdryCDFO(unittest.TestCase):
    def runTest(self):
        n = 3
        g = np.array([1.0, 0.0, 1.0])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        h = lambda d: np.linalg.norm(d, 1)
        L_h = sqrt(n)
        delta = 5.0 / 12.0
        xopt = np.zeros((n,))
        func_tol = 1e-4
        d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox_uh, func_tol=func_tol)
        for i in range(50):
            d_e = delta * np.ones(n) # initialize d_e
            while np.linalg.norm(d_e, 2) > delta:
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = model_value(g, H, d_k, xopt, h)
            func_d_e = model_value(g, H, d_e, xopt, h)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "sufficient decrease does not achieved!")

class TestUncHardCDFO(unittest.TestCase):
    def runTest(self):
        n = 3
        g = np.array([0.0, 0.0, 1.0])
        H = np.array([[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        h = lambda d: np.linalg.norm(d, 1)
        L_h = sqrt(n)
        delta = sqrt(2.0)
        xopt = np.zeros((n,))
        func_tol = 1e-4
        d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox_uh, func_tol=func_tol)
        for i in range(50):
            d_e = delta * np.ones(n) # initialize d_e
            while np.linalg.norm(d_e, 2) > delta:
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = model_value(g, H, d_k, xopt, h)
            func_d_e = model_value(g, H, d_e, xopt, h)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "sufficient decrease does not achieved!")

class TestConInternalCDFO(unittest.TestCase):
    def runTest(self):
        n = 3
        g = np.array([1.0, 0.0, 1.0])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        h = lambda d: np.linalg.norm(d, 1)
        L_h = sqrt(n)
        delta = 2.0
        xopt = np.ones((n,))  # trying nonzero (since bounds inactive)
        sl = xopt + np.array([-0.5, -10.0, -10.0])
        su = xopt + np.array([10.0, 10.0, 10.0])
        proj = lambda x: pbox(x,sl,su)
        func_tol = 1e-4
        d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [proj], delta, h, L_h, prox_uh, func_tol=func_tol)
        for i in range(50):
            d_e = delta * np.ones(n) # initialize d_e
            while np.linalg.norm(d_e, 2) > delta or not np.allclose(proj(xopt+d_e), xopt+d_e):
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = model_value(g, H, d_k, xopt, h)
            func_d_e = model_value(g, H, d_e, xopt, h)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "sufficient decrease does not achieved!")

class TestConBdryCDFO(unittest.TestCase):
    def runTest(self):
        n = 3
        g = np.array([1.0, 0.0, 1.0])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        h = lambda d: np.linalg.norm(d, 1)
        L_h = sqrt(n)
        delta = 5.0 / 12.0
        xopt = np.zeros((n,))
        sl = xopt + np.array([-0.3, -0.01, -0.1])
        su = xopt + np.array([10.0, 1.0, 10.0])
        proj = lambda x: pbox(x,sl,su)
        func_tol = 1e-4
        d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [proj], delta, h, L_h, prox_uh, func_tol=func_tol)
        for i in range(50):
            d_e = delta * np.ones(n) # initialize d_e
            while np.linalg.norm(d_e, 2) > delta or not np.allclose(proj(xopt+d_e), xopt+d_e):
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = model_value(g, H, d_k, xopt, h)
            func_d_e = model_value(g, H, d_e, xopt, h)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "sufficient decrease does not achieved!")

class TestBoxBallInternalCDFO(unittest.TestCase):
    def runTest(self):
        n = 3
        g = np.array([1.0, 0.0, 1.0])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        h = lambda d: np.linalg.norm(d, 1)
        L_h = sqrt(n)
        delta = 2.0
        xopt = np.ones((n,))  # trying nonzero (since bounds inactive)
        sl = xopt + np.array([-0.5, -10.0, -10.0])
        su = xopt + np.array([10.0, 10.0, 10.0])
        boxproj = lambda x: pbox(x,sl,su)
        ballproj = lambda x: pball(x,xopt,5)
        func_tol = 1e-4
        d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [boxproj, ballproj], delta, h, L_h, prox_uh, func_tol=func_tol)
        for i in range(50):
            d_e = delta * np.ones(n) # initialize d_e
            while np.linalg.norm(d_e, 2) > delta or not np.allclose(boxproj(xopt+d_e), xopt+d_e) or not np.allclose(ballproj(xopt+d_e), xopt+d_e):
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = model_value(g, H, d_k, xopt, h)
            func_d_e = model_value(g, H, d_e, xopt, h)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "sufficient decrease does not achieved!")

class TestBoxBallBdryCDFO(unittest.TestCase):
    def runTest(self):
        n = 3
        g = np.array([1.0, 0.0, 1.0])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        h = lambda d: np.linalg.norm(d, 1)
        L_h = sqrt(n)
        delta = 5.0 / 12.0
        xopt = np.zeros((n,))
        sl = xopt + np.array([-0.3, -0.01, -0.1])
        su = xopt + np.array([10.0, 1.0, 10.0])
        boxproj = lambda x: pbox(x,sl,su)
        ballproj = lambda x: pball(x,xopt,0.25)
        func_tol = 1e-4
        d_k, gnew, crvmin = ctrsbox_sfista(xopt, g, H, [boxproj, ballproj], delta, h, L_h, prox_uh, func_tol=func_tol)
        for i in range(50):
            d_e = delta * np.ones(n) # initialize d_e
            while np.linalg.norm(d_e, 2) > delta or not np.allclose(boxproj(xopt+d_e), xopt+d_e) or not np.allclose(ballproj(xopt+d_e), xopt+d_e):
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = model_value(g, H, d_k, xopt, h)
            func_d_e = model_value(g, H, d_e, xopt, h)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "sufficient decrease does not achieved!")
