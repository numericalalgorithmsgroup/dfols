# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt, ceil
import numpy as np
import unittest

from dfols.trust_region import ctrsbox
from dfols.util import model_value
from scipy.optimize import minimize

def prox_uh(u, h, d):
    # Find prox_{uh} using Nelder–Mead method
    func = lambda x: u*h(x) + np.linalg.norm(x-d, 2)**2 / 2
    res = minimize(func, d, method='Nelder-Mead', tol=1e-8)
    return res.x

class TestSFISTA(unittest.TestCase):

    def test_smoooth(self):
        n = 3
        xopt = np.ones((n,))
        J_k = np.ones([n,n])
        H = J_k.T @ J_k # H is positive semidefinite
        d_opt_un = np.ones(n) # optimal solution without constraint
        g = -H @ d_opt_un
        h = lambda d: 0
        k_H = 1
        delta = 1
        L_h = 1
        func_tol = 1e-3
        d_k, gnew, crvmin = ctrsbox(xopt, g, H, h, [], k_H, L_h, prox_uh, delta, func_tol) 
        d_opt = d_opt_un / np.linalg.norm(d_opt_un, 2)
        # print("d_k", d_k)
        # print("d_opt", d_opt)
        self.assertTrue(np.allclose(d_k, d_opt), "unsuccessful when h=0!")
        for i in range(100):
            d_e = delta * np.ones(n)
            while np.linalg.norm(d_e, 2) > delta:
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = np.dot(g, d_k) + d_k.T @ H @ d_k / 2 + h(d_k)
            func_d_e = np.dot(g, d_e) + d_e.T @ H @ d_e / 2 + h(d_e)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "unsuccessful when h=0!")
        
    def test_nonsmooth(self):
        n = 3
        xopt = np.ones((n,))
        g = np.zeros(n)
        H = np.zeros([n,n])
        x = np.array([-0.2, 0.5, -0.1])
        h = lambda d: np.linalg.norm(x + d, 1)
        k_H = 1
        delta = 1
        L_h = sqrt(3)
        func_tol = 1e-3
        d_k, gnew, crvmin = ctrsbox(xopt, g, H, h, [], k_H, L_h, prox_uh, delta, func_tol) 
        d_opt = -x
        # print("d_k", d_k)
        # print("d_opt", d_opt)
        self.assertTrue(np.allclose(d_k, d_opt), "unsuccessful when g=0, H=0!")
        for i in range(100):
            d_e = delta * np.ones(n)
            while np.linalg.norm(d_e, 2) > delta:
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = np.dot(g, d_k) + d_k.T @ H @ d_k / 2 + h(d_k)
            func_d_e = np.dot(g, d_e) + d_e.T @ H @ d_e / 2 + h(d_e)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "unsuccessful when g=0, H=0!")
        
    def test_all(self):
        n = 3
        xopt = np.ones((n,))
        J_k = np.ones([n,n])
        H = J_k.T @ J_k
        g = -H @ np.ones(n)
        x = np.array([-0.2, 0.5, -0.1])
        h = lambda d: np.linalg.norm(x + d, 1)
        k_H = 1
        delta = 1
        L_h = sqrt(3)
        func_tol = 1e-3
        d_k, gnew, crvmin = ctrsbox(xopt, g, H, h, [], k_H, L_h, prox_uh, delta, func_tol) 
        # print("d_k",d_k)
        for i in range(100):
            d_e = delta * np.ones(n)
            while np.linalg.norm(d_e, 2) > delta:
                err = (np.random.rand(n) - 0.5)*1e-3
                d_e = d_k + err
            func_d_k = np.dot(g, d_k) + d_k.T @ H @ d_k / 2 + h(d_k)
            func_d_e = np.dot(g, d_e) + d_e.T @ H @ d_e / 2 + h(d_e)
            self.assertTrue(func_d_k <= func_d_e or (func_d_k > func_d_e and
                            func_d_k-func_d_e < func_tol), "unsuccessful when mixed!")

if __name__ == '__main__':
    unittest.main()
