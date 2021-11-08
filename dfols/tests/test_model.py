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

from math import sqrt
import numpy as np
import unittest

from dfols.model import Model
from dfols.util import sumsq


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


class TestAddValues(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        self.assertEqual(model.npt(), 1, 'Wrong npt after initialisation')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x0), 'Wrong xopt after initialisation')
        self.assertTrue(array_compare(model.ropt(), rosenbrock(x0)), 'Wrong ropt after initialisation')
        self.assertAlmostEqual(model.fopt(), sumsq(rosenbrock(x0)), 'Wrong fopt after initialisation')
        # Now add better point
        x1 = np.array([1.0, 0.9])
        rvec = rosenbrock(x1)
        model.change_point(1, x1 - model.xbase, rvec, allow_kopt_update=True)
        self.assertEqual(model.npt(), 2, 'Wrong npt after x1')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after x1')
        self.assertTrue(array_compare(model.ropt(), rosenbrock(x1)), 'Wrong ropt after x1')
        self.assertAlmostEqual(model.fopt(), sumsq(rosenbrock(x1)), 'Wrong fopt after x1')
        # Now add worse point
        x2 = np.array([2.0, 0.9])
        rvec = rosenbrock(x2)
        model.change_point(2, x2 - model.xbase, rvec, allow_kopt_update=True)
        self.assertEqual(model.npt(), 3, 'Wrong npt after x2')
        self.assertTrue(array_compare(model.xpt(0, abs_coordinates=True), x0), 'Wrong xpt(0) after x2')
        self.assertTrue(array_compare(model.xpt(1, abs_coordinates=True), x1), 'Wrong xpt(1) after x2')
        self.assertTrue(array_compare(model.xpt(2, abs_coordinates=True), x2), 'Wrong xpt(2) after x2')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after x2')
        self.assertTrue(array_compare(model.ropt(), rosenbrock(x1)), 'Wrong ropt after x2')
        self.assertAlmostEqual(model.fopt(), sumsq(rosenbrock(x1)), 'Wrong fopt after x2')
        # Now add best point (but don't update kopt)
        x3 = np.array([1.0, 1.0])
        rvec = rosenbrock(x3)
        model.change_point(0, x3 - model.xbase, rvec, allow_kopt_update=False)  # full: overwrite x0
        self.assertEqual(model.npt(), 3, 'Wrong npt after x3')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after x3')
        self.assertTrue(array_compare(model.ropt(), rosenbrock(x1)), 'Wrong ropt after x3')
        self.assertAlmostEqual(model.fopt(), sumsq(rosenbrock(x1)), 'Wrong fopt after x3')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), model.as_absolute_coordinates(model.xopt())),
                        'Comparison wrong after x3')
        dirns = model.xpt_directions(include_kopt=True)
        self.assertTrue(array_compare(x3 - x1, dirns[0, :]), 'Wrong dirn 0')
        self.assertTrue(array_compare(x1 - x1, dirns[1, :]), 'Wrong dirn 1')
        self.assertTrue(array_compare(x2 - x1, dirns[2, :]), 'Wrong dirn 2')
        dirns = model.xpt_directions(include_kopt=False)
        self.assertTrue(array_compare(x3 - x1, dirns[0, :]), 'Wrong dirn 0 (no kopt)')
        # self.assertTrue(array_compare(x1 - x1, dirns[1, :]), 'Wrong dirn 1')
        self.assertTrue(array_compare(x2 - x1, dirns[1, :]), 'Wrong dirn 1 (no kopt)')


class TestSwap(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        # Now add better point
        x1 = np.array([1.0, 0.9])
        rvec = rosenbrock(x1)
        model.change_point(1, x1 - model.xbase, rvec, allow_kopt_update=True)
        # Now add worse point
        x2 = np.array([2.0, 0.9])
        rvec = rosenbrock(x2)
        model.change_point(2, x2 - model.xbase, rvec, allow_kopt_update=True)
        model.swap_points(0, 2)
        self.assertTrue(array_compare(model.xpt(0, abs_coordinates=True), x2), 'Wrong xpt(0) after swap 1')
        self.assertTrue(array_compare(model.xpt(1, abs_coordinates=True), x1), 'Wrong xpt(1) after swap 1')
        self.assertTrue(array_compare(model.xpt(2, abs_coordinates=True), x0), 'Wrong xpt(2) after swap 1')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after swap 1')
        model.swap_points(1, 2)
        self.assertTrue(array_compare(model.xpt(0, abs_coordinates=True), x2), 'Wrong xpt(0) after swap 2')
        self.assertTrue(array_compare(model.xpt(1, abs_coordinates=True), x0), 'Wrong xpt(1) after swap 2')
        self.assertTrue(array_compare(model.xpt(2, abs_coordinates=True), x1), 'Wrong xpt(2) after swap 2')
        self.assertTrue(array_compare(model.xopt(abs_coordinates=True), x1), 'Wrong xopt after swap 2')

class TestBasicManipulation(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        self.assertTrue(array_compare(model.sl, xl - x0), 'Wrong sl after initialisation')
        self.assertTrue(array_compare(model.su, xu - x0), 'Wrong su after initialisation')
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        self.assertTrue(array_compare(model.as_absolute_coordinates(x1 - x0), x1), 'Wrong abs coords')
        self.assertTrue(array_compare(model.as_absolute_coordinates(np.array([-1e3, 1e3])-x0), np.array([-1e2, 1e2])),
                        'Bad abs coords with bounds')
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        sqdists = model.distances_to_xopt()
        self.assertAlmostEqual(sqdists[0], sumsq(x0 - x1), 'Wrong distance 0')
        self.assertAlmostEqual(sqdists[1], sumsq(x1 - x1), 'Wrong distance 1')
        self.assertAlmostEqual(sqdists[2], sumsq(x2 - x1), 'Wrong distance 2')
        model.add_new_sample(0, rosenbrock(x0))
        self.assertEqual(model.nsamples[0], 2, 'Wrong number of samples 0')
        self.assertEqual(model.nsamples[1], 1, 'Wrong number of samples 1')
        self.assertEqual(model.nsamples[2], 1, 'Wrong number of samples 2')
        for i in range(50):
            model.add_new_sample(0, np.array([0.0, 0.0]))
        self.assertEqual(model.kopt, 0, 'Wrong kopt after bad resampling')
        self.assertTrue(array_compare(model.ropt(), 2*rosenbrock(x0)/52), 'Wrong ropt after bad resampling')
        self.assertAlmostEqual(model.fopt(), sumsq(2 * rosenbrock(x0) / 52), 'Wrong fopt after bad resampling')
        d = np.array([10.0, 10.0])
        dirns_old = model.xpt_directions(include_kopt=True)
        model.shift_base(d)
        dirns_new = model.xpt_directions(include_kopt=True)
        self.assertTrue(array_compare(model.xbase, x0 + d), 'Wrong new base')
        self.assertEqual(model.kopt, 0, 'Wrong kopt after shift base')
        for i in range(3):
            self.assertTrue(array_compare(dirns_old[i, :], dirns_new[i, :]), 'Wrong dirn %i after shift base' % i)
        self.assertTrue(array_compare(model.sl, xl - x0 - d), 'Wrong sl after shift base')
        self.assertTrue(array_compare(model.su, xu - x0 - d), 'Wrong su after shift base')
        # save_point and get_final_results
        model.change_point(0, x0 - model.xbase, rosenbrock(x0))  # revert after resampling
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))  # revert after resampling
        x, rvec, f, jacmin, nsamples = model.get_final_results()
        self.assertTrue(array_compare(x, x1), 'Wrong final x')
        self.assertTrue(array_compare(rvec, rosenbrock(x1)), 'Wrong final rvec')
        self.assertAlmostEqual(sumsq(rosenbrock(x1)), f, 'Wrong final f')
        self.assertTrue(array_compare(np.zeros((2,2)), jacmin), 'Wrong final jacmin')
        self.assertEqual(1, nsamples, 'Wrong final nsamples')
        self.assertIsNone(model.xsave, 'xsave not none after initialisation')
        self.assertIsNone(model.rsave, 'rsave not none after initialisation')
        self.assertIsNone(model.fsave, 'fsave not none after initialisation')
        self.assertIsNone(model.nsamples_save, 'nsamples_save not none after initialisation')
        model.save_point(x0, rosenbrock(x0), 1, x_in_abs_coords=True)
        self.assertTrue(array_compare(model.xsave, x0), 'Wrong xsave after saving')
        self.assertTrue(array_compare(model.rsave, rosenbrock(x0)), 'Wrong rsave after saving')
        self.assertAlmostEqual(model.fsave, sumsq(rosenbrock(x0)), 'Wrong fsave after saving')
        self.assertEqual(model.nsamples_save, 1, 'Wrong nsamples_save after saving')
        x, rvec, f, jacmin, nsamples = model.get_final_results()
        self.assertTrue(array_compare(x, x1), 'Wrong final x after saving')
        self.assertTrue(array_compare(rvec, rosenbrock(x1)), 'Wrong final rvec after saving')
        self.assertAlmostEqual(sumsq(rosenbrock(x1)), f, 'Wrong final f after saving')
        self.assertEqual(1, nsamples, 'Wrong final nsamples after saving')
        model.save_point(x2 - model.xbase, np.array([0.0, 0.0]), 2, x_in_abs_coords=False)
        self.assertTrue(array_compare(model.xsave, x2), 'Wrong xsave after saving 2')
        self.assertTrue(array_compare(model.rsave, np.array([0.0, 0.0])), 'Wrong rsave after saving 2')
        self.assertAlmostEqual(model.fsave, 0.0, 'Wrong fsave after saving 2')
        self.assertEqual(model.nsamples_save, 2, 'Wrong nsamples_save after saving 2')
        x, rvec, f, jacmin, nsamples = model.get_final_results()
        self.assertTrue(array_compare(x, x2), 'Wrong final x after saving 2')
        self.assertTrue(array_compare(rvec, np.array([0.0, 0.0])), 'Wrong final rvec after saving 2')
        self.assertAlmostEqual(f, 0.0, 'Wrong final f after saving 2')
        self.assertEqual(2, nsamples, 'Wrong final nsamples after saving 2')
        model.save_point(x0, rosenbrock(x0), 3, x_in_abs_coords=True)  # try to re-save a worse value
        self.assertTrue(array_compare(model.xsave, x2), 'Wrong xsave after saving 3')
        self.assertTrue(array_compare(model.rsave, np.array([0.0, 0.0])), 'Wrong rsave after saving 3')
        self.assertAlmostEqual(model.fsave, 0.0, 'Wrong fsave after saving 3')
        self.assertEqual(model.nsamples_save, 2, 'Wrong nsamples_save after saving 3')


class TestAveraging(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([1.0, 1.0])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        self.assertEqual(model.kopt, 2, 'Wrong kopt before resampling')
        # Originally, x2 is the ideal point
        # Here, testing that kopt moves back to x1 after adding heaps of bad x2 samples
        for i in range(10):
            model.add_new_sample(2, np.array([5.0, 5.0]))
        self.assertEqual(model.kopt, 1, 'Wrong kopt after resampling')


class TestMinObjValue(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        self.assertAlmostEqual(model.min_objective_value(), 1e-12, 'Wrong min obj value')
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1, rel_tol=1e-2)
        self.assertAlmostEqual(model.min_objective_value(), 1e-2 * sumsq(rosenbrock(x0)), 'Wrong min obj value 2')
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1, abs_tol=1.0)
        self.assertAlmostEqual(model.min_objective_value(), 1.0, 'Wrong min obj value 3')
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1, abs_tol=1.0, rel_tol=1e-2)
        self.assertAlmostEqual(model.min_objective_value(), 1.0, 'Wrong min obj value 4')


class TestInterpMatrixSVD(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        model.add_new_sample(0, rosenbrock(x0))
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        A, left_scaling, right_scaling = model.interpolation_matrix()
        A_expect = np.ones((3, 3))
        A_expect[0, 1:] = x0 - x1
        A_expect[1, 1:] = x1 - x1  # x1 is xopt in this situation
        A_expect[2, 1:] = x2 - x1
        A_after_scaling = np.dot(np.diag(1.0 / left_scaling), np.dot(A, np.diag(1.0 / right_scaling)))
        self.assertTrue(array_compare(A_after_scaling, A_expect), 'Interp matrix 1')
        # For reference: model based around model.xbase
        interp_ok, interp_error, norm_J_error, linalg_resid, ls_interp_cond_num = model.interpolate_mini_models_svd()
        self.assertTrue(interp_ok, 'Interpolation failed')
        # print(x0, x1, x2)
        # print(rosenbrock(x0), rosenbrock(x1), rosenbrock(x2))
        # print("xbase =", model.xbase)
        # print("xopt =", model.xopt(abs_coordinates=True))
        # print(model.model_const, rosenbrock(model.xbase))
        self.assertTrue(array_compare(model.model_const, rosenbrock(model.xbase), thresh=1e-10), 'Wrong constant term')
        self.assertTrue(array_compare(model.model_value(x0 - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      rosenbrock(x0), thresh=1e-10), 'Wrong x0')  # allow some inexactness
        self.assertTrue(array_compare(model.model_value(x1 - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      rosenbrock(x1), thresh=1e-10), 'Wrong x1')  # allow some inexactness
        self.assertTrue(array_compare(model.model_value(x2 - model.xbase, d_based_at_xopt=False, with_const_term=True),
                                      rosenbrock(x2), thresh=1e-10), 'Wrong x2')
        # Test some other parameter settings for model.model_value()
        self.assertTrue(array_compare(model.model_value(x2 - x1, d_based_at_xopt=True, with_const_term=True),
                                      rosenbrock(x2), thresh=1e-10), 'Wrong x2 (from xopt)')
        self.assertTrue(array_compare(model.model_value(x2 - x1, d_based_at_xopt=True, with_const_term=False),
                                      rosenbrock(x2)-rosenbrock(model.xbase), thresh=1e-10), 'Wrong x2 (no constant)')
        self.assertTrue(array_compare(model.model_value(x2 - model.xbase, d_based_at_xopt=False, with_const_term=False),
                                rosenbrock(x2) - rosenbrock(model.xbase), thresh=1e-10), 'Wrong x2 (no constant v2)')
        g, H = model.build_full_model()
        r = rosenbrock(x1)
        J = model.model_jac
        self.assertTrue(array_compare(g, 2.0*np.dot(J.T, r), thresh=1e-10), 'Bad gradient')
        self.assertTrue(array_compare(H, 2.0*np.dot(J.T, J)), 'Bad Hessian')


class TestGeomSystem(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        # x0 = np.array([1.0, 2.9])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        model.add_new_sample(0, rosenbrock(x0))
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        c0, g0 = model.lagrange_gradient(0)
        lag_thresh = 1e-10
        xopt = model.xopt(abs_coordinates=True)
        # print(c0 + np.dot(g0, x0 - xopt), c0, g0)
        self.assertTrue(abs(c0 + np.dot(g0, x0 - xopt) - 1.0) < lag_thresh, 'Bad L0(x0)')
        self.assertTrue(abs(c0 + np.dot(g0, x1 - xopt) - 0.0) < lag_thresh, 'Bad L0(x1)')
        self.assertTrue(abs(c0 + np.dot(g0, x2 - xopt) - 0.0) < lag_thresh, 'Bad L0(x2)')
        c1, g1 = model.lagrange_gradient(1)
        # print(c1 + np.dot(g1, x0 - xopt), c1, g1)
        self.assertTrue(abs(c1 + np.dot(g1, x0 - xopt) - 0.0) < lag_thresh, 'Bad L1(x0)')
        self.assertTrue(abs(c1 + np.dot(g1, x1 - xopt) - 1.0) < lag_thresh, 'Bad L1(x1)')
        self.assertTrue(abs(c1 + np.dot(g1, x2 - xopt) - 0.0) < lag_thresh, 'Bad L1(x2)')
        c2, g2 = model.lagrange_gradient(2)
        self.assertTrue(abs(c2 + np.dot(g2, x0 - xopt) - 0.0) < lag_thresh, 'Bad L2(x0)')
        self.assertTrue(abs(c2 + np.dot(g2, x1 - xopt) - 0.0) < lag_thresh, 'Bad L2(x1)')
        self.assertTrue(abs(c2 + np.dot(g2, x2 - xopt) - 1.0) < lag_thresh, 'Bad L2(x2)')


class TestFullRankSVD(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        model.add_new_sample(0, rosenbrock(x0))
        x1 = np.array([-1.2, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        # Here, x0 is base
        # A = model.interpolation_matrix()
        # self.assertTrue(array_compare(A[1, :], delta * np.array([1.0, 0.0])), 'Bad padded interp matrix')
        old_jac = model.model_jac.copy()
        model.interpolate_mini_models_svd(make_full_rank=False)
        J = model.model_jac.copy()
        model.model_jac = old_jac  # reset
        sng_vals = np.linalg.svd(J)[1]
        model.interpolate_mini_models_svd(make_full_rank=True)
        J2 = model.model_jac.copy()
        sng_vals2 = np.linalg.svd(J2)[1]
        print("Original", sng_vals)
        print("After making full rank", sng_vals2)
        self.assertTrue(10*sng_vals[-1] < sng_vals2[-1], 'New singular value too small')


class TestPoisedness(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        delta = 0.5
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        model.add_new_sample(0, rosenbrock(x0))
        x1 = x0 + delta * np.array([1.0, 0.0])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = x0 + delta * np.array([0.0, 1.0])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        model.kopt = 0  # force this
        # Here (use delta=1), Lagrange polynomials are (1-x-y), 1-x and 1-y
        # Maximum value in ball is for (1-x-y) at (x,y)=(1/sqrt2, 1/sqrt2) --> max value = 1 + sqrt(2)
        self.assertTrue(abs(model.poisedness_constant(delta) - 1.0 - sqrt(2.0)) < 1e-6, 'Poisedness constant wrong')


class TestAddPoint(unittest.TestCase):
    def runTest(self):
        n, m = 2, 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, x0, rosenbrock(x0), xl, xu, [], 1)
        x1 = np.array([1.0, 0.9])
        model.change_point(1, x1 - model.xbase, rosenbrock(x1))
        x2 = np.array([2.0, 0.9])
        model.change_point(2, x2 - model.xbase, rosenbrock(x2))
        # Now add a new point
        x3 = np.array([1.0, 1.0])  # good point
        model.add_new_point(x3 - model.xbase, rosenbrock(x3))
        self.assertEqual(model.npt(), 4, "Wrong number of points after x3")
        self.assertTrue(array_compare(model.xpt(3, abs_coordinates=True), x3), "Wrong new point after x3")
        self.assertTrue(array_compare(model.rvec(3), rosenbrock(x3)), "Wrong rvec after x3")
        self.assertEqual(model.kopt, 3, "Wrong kopt after x3")
        self.assertEqual(len(model.nsamples), 4, "Wrong nsamples length after x3")
        self.assertEqual(model.nsamples[-1], 1, "Wrong nsample value after x3")
        x4 = np.array([-1.8, 1.8])  # bad point
        model.add_new_point(x4 - model.xbase, rosenbrock(x4))
        self.assertEqual(model.npt(), 5, "Wrong number of points after x4")
        self.assertTrue(array_compare(model.xpt(4, abs_coordinates=True), x4), "Wrong new point after x4")
        self.assertTrue(array_compare(model.rvec(4), rosenbrock(x4)), "Wrong rvec after x4")
        self.assertEqual(model.kopt, 3, "Wrong kopt after x4")


class TestRegression(unittest.TestCase):
    def runTest(self):
        # Test against np.linalg.lstsq
        # m = 1
        n, npt = 2, 6
        np.random.seed(0)  # (fixing random seed)
        A = np.random.rand(npt, n)
        b = np.random.rand(npt)

        # x0 = np.zeros((n,))
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        model = Model(npt, A[0,:], np.array([b[0]]), xl, xu, [], 1)

        for i in range(1,npt):
            xi = A[i,:]
            model.change_point(i, xi - model.xbase, b[i])

        xopt = model.xopt(abs_coordinates=True)
        A_for_interp = np.zeros((npt, n+1))  # there are actually npt+1 points
        A_for_interp[:,0] = 1.0  # constant terms
        for i in range(npt):
            A_for_interp[i, 1:] = A[i,:] - xopt

        A2, left_scaling, right_scaling = model.interpolation_matrix()
        A_after_scaling = np.dot(np.diag(1.0 / left_scaling), np.dot(A2, np.diag(1.0 / right_scaling)))  # undo scaling
        # print(A_for_interp)
        # print(A_after_scaling)
        self.assertTrue(array_compare(A_for_interp, A_after_scaling), 'Interp matrix 1')
        # For reference: model based around model.xbase
        interp_ok, interp_error, norm_J_error, linalg_resid, ls_interp_cond_num = model.interpolate_mini_models_svd()
        J_true = np.linalg.lstsq(A_for_interp, b, rcond=None)[0]
        self.assertTrue(interp_ok, 'Interpolation failed')
        # print(model.model_const, model.model_jac)
        # print(J_true[0] - np.dot(J_true[1:], model.xopt()), J_true[1:])
        # print(model.xopt(abs_coordinates=True), model.xbase)
        g = J_true[1:]
        c = J_true[0] - np.dot(g, model.xopt())  # centred at xbase
        self.assertTrue(array_compare(model.model_const, np.array([c])), 'Wrong constant term')
        self.assertTrue(array_compare(model.model_jac, g), 'Wrong Jacobian term')


class TestUnderdetermined(unittest.TestCase):
    def runTest(self):
        # Test against np.linalg.lstsq
        # m = 1
        n, npt = 5, 3
        np.random.seed(0)  # (fixing random seed)
        A = np.random.rand(npt, n)
        b = np.random.rand(npt)

        x0 = np.zeros((n,))
        xl = -1e2 * np.ones((n,))
        xu = 1e2 * np.ones((n,))
        # model = Model(n+1, x0, np.array([0.0]), xl, xu, 1)
        model = Model(n+1, A[0, :], np.array([b[0]]), xl, xu, [], 1)

        for i in range(1,npt):
            xi = A[i,:]
            model.change_point(i, xi - model.xbase, b[i])

        xopt = model.xopt(abs_coordinates=True)
        A_for_interp = np.zeros((npt, n + 1))  # there are actually npt+1 points
        A_for_interp[:, 0] = 1.0  # constant terms
        for i in range(npt):
            A_for_interp[i, 1:] = A[i, :] - xopt

        A2, left_scaling, right_scaling = model.interpolation_matrix()
        A_after_scaling = np.dot(np.diag(1.0 / left_scaling), np.dot(A2, np.diag(1.0 / right_scaling)))  # undo scaling
        self.assertTrue(array_compare(A_for_interp, A_after_scaling), 'Interp matrix 1')
        # For reference: model based around model.xbase
        interp_ok, interp_error, norm_J_error, linalg_resid, ls_interp_cond_num = model.interpolate_mini_models_svd()
        self.assertTrue(interp_ok, 'Interpolation failed')
        J_true = np.linalg.lstsq(A_for_interp, b, rcond=None)[0]
        g = J_true[1:]
        c = J_true[0] - np.dot(g, model.xopt())  # centred at xbase
        self.assertTrue(array_compare(model.model_const, np.array([c])), 'Wrong constant term')
        self.assertTrue(array_compare(model.model_jac, g), 'Wrong Jacobian term')

