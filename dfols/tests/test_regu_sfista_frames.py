# Regression tests for the coordinate frames used by the regularised S-FISTA subproblem
# (dfols/trust_region.py::ctrsbox_sfista and its two pbox call sites in dfols/controller.py).
#
# Defect A (scaled gradient): with scaling_changes set, h and prox_uh act on the original
#   (unscaled) coordinates while the iterate d lives in the internally scaled coordinates.
#   The smoothed gradient must therefore carry the chain-rule factor scale, and the
#   gradient's Lipschitz constant is ||H||_2 + max(scale**2)/u.
# Defect B (box frame): Model.sl/su are bounds *relative to Model.xbase*, but the S-FISTA
#   receives the *absolute* xopt and applies the box to xopt + d, so the box must be shifted
#   back into absolute coordinates.
#
# TestScaledGradient and TestCriticalityBoxFrame are the two defect tests; they fail on
#   89ac51b with the wrong gradient / the wrong box and pass on the patched code.
# TestScaledStepSize additionally pins the Lipschitz constant, so it still fails on a
#   partial fix that only corrects the gradient formula.
# TestTrustRegionBoxFrame covers the second pbox call site (controller.py:586), so the
#   suite also fails on a partial fix that only corrects evaluate_criticality_measure.
# TestScaledGradientUnscaledControl passes on every tree and guards against over-correcting
#   the unscaled path.
from __future__ import absolute_import, division, print_function, unicode_literals

import math
import unittest

import numpy as np

from dfols.trust_region import ctrsbox_sfista
from dfols.util import remove_scaling


def prox_l1_scaled(lam):
    return lambda x, u: np.sign(x) * np.maximum(np.abs(x) - lam * u, 0.0)


def moreau_l1(z, u, lam):
    """h_u(z) for h(z) = lam*||z||_1 (Moreau envelope)."""
    p = np.sign(z) * np.maximum(np.abs(z) - lam * u, 0.0)
    return lam * np.abs(p).sum() + np.sum((z - p) ** 2) / (2.0 * u)


def smooth_model(g, H, d, xopt, u, lam, scaling_changes=None):
    """F_u(d) = g'd + 0.5 d'Hd + h_u(remove_scaling(xopt + d))."""
    z = remove_scaling(xopt + d, scaling_changes)
    return float(g @ d + 0.5 * d @ H @ d + moreau_l1(z, u, lam))


def sfista_iters(delta, L_h, k_H, func_tol, max_iters, scale=2.0):
    it = math.ceil(scale * delta * (L_h + math.sqrt(L_h * L_h + 2.0 * k_H * func_tol)) / func_tol)
    return min(it, max_iters)


class TestScaledGradient(unittest.TestCase):
    """gnew must be the chain-rule gradient of F_u when scaling_changes is set."""

    def runTest(self):
        shift = np.array([1.0, -2.0])
        scale = np.array([2.0, 0.5])
        xopt = np.array([0.5, -0.3])
        g = np.array([-0.7, 0.4])
        H = np.array([[0.3, 0.1], [0.1, -0.2]])
        lam = 1.0
        L_h = lam * math.sqrt(2.0)
        delta = 1.0
        h = lambda x: lam * np.linalg.norm(x, 1)
        prox = prox_l1_scaled(lam)
        d, gnew, _ = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox,
                                    func_tol=1e-3, max_iters=500, sfista_iters_scale=2.0,
                                    scaling_changes=(shift, scale))
        N = sfista_iters(delta, L_h, np.linalg.norm(H, 2), 1e-3, 500)
        u = 2.0 * delta / (N * L_h)
        v = np.array([0.6, -0.8])
        eps = 1e-6
        fd = (smooth_model(g, H, d + eps * v, xopt, u, lam, (shift, scale)) -
              smooth_model(g, H, d - eps * v, xopt, u, lam, (shift, scale))) / (2 * eps)
        self.assertAlmostEqual(fd, float(gnew @ v), delta=2e-4,
                               msg="gnew ignores the chain-rule factor dx_raw/dx_scaled")


class TestScaledGradientUnscaledControl(unittest.TestCase):
    """Control: the same check without scaling must pass on the unpatched code."""

    def runTest(self):
        xopt = np.array([0.5, -0.3])
        g = np.array([-0.7, 0.4])
        H = np.array([[0.3, 0.1], [0.1, -0.2]])
        lam = 1.0
        L_h = lam * math.sqrt(2.0)
        delta = 1.0
        h = lambda x: lam * np.linalg.norm(x, 1)
        prox = prox_l1_scaled(lam)
        d, gnew, _ = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox,
                                    func_tol=1e-3, max_iters=500, sfista_iters_scale=2.0)
        N = sfista_iters(delta, L_h, np.linalg.norm(H, 2), 1e-3, 500)
        u = 2.0 * delta / (N * L_h)
        v = np.array([0.6, -0.8])
        eps = 1e-6
        fd = (smooth_model(g, H, d + eps * v, xopt, u, lam) -
              smooth_model(g, H, d - eps * v, xopt, u, lam)) / (2 * eps)
        self.assertAlmostEqual(fd, float(gnew @ v), delta=2e-4)


class TestScaledStepSize(unittest.TestCase):
    """A single S-FISTA step from d=0 must respect the scaled Lipschitz constant.

    With max_iters=1 the returned step is d_1 = proj(-grad F_u(0)/l).  The projection is
    non-expansive and fixes 0, and here it does not clip, so ||d_1|| * l == ||grad F_u(0)||.
    Using l = ||H||+1/u instead of ||H||+max(scale**2)/u makes the step max(scale**2) times
    too long and violates the bound; the ball then clips, so ||d_1|| * l > ||grad F_u(0)||.
    This test therefore fails if only the gradient formula is fixed.
    """

    def runTest(self):
        shift = np.array([0.0, 0.0])
        scale = np.array([20.0, 0.1])
        xopt = np.array([0.3, 0.06])
        g = np.array([-1.0, 0.5])
        H = np.zeros((2, 2))
        lam = 1.0
        L_h = lam * math.sqrt(2.0)
        delta = 1.0
        h = lambda x: lam * np.linalg.norm(x, 1)
        prox = prox_l1_scaled(lam)
        d, gnew, _ = ctrsbox_sfista(xopt, g, H, [], delta, h, L_h, prox,
                                    func_tol=1e-3, max_iters=1, sfista_iters_scale=2.0,
                                    scaling_changes=(shift, scale))
        u = 2.0 * delta / (1 * L_h)
        z0 = remove_scaling(xopt, (shift, scale))
        grad0 = g + scale * (z0 - prox(z0, u)) / u
        L = np.linalg.norm(H, 2) + np.max(scale ** 2) / u
        lhs = np.linalg.norm(d) * L
        rhs = np.linalg.norm(grad0)
        self.assertLessEqual(lhs, rhs * (1.0 + 1e-12),
                             "step is longer than grad/L allows: ||d||*L=%r > ||grad||=%r"
                             % (lhs, rhs))


class _FakeModel(object):
    projections = []

    def __init__(self, xopt, g, H):
        self._x = np.array(xopt, dtype=float)
        self._g = np.array(g, dtype=float)
        self._H = np.array(H, dtype=float)
        self.sl = -1e20 * np.ones(self._x.size)
        self.su = 1e20 * np.ones(self._x.size)
        self.xbase = np.zeros(self._x.size)

    def build_full_model(self):
        return self._g, self._H

    def xopt(self, abs_coordinates=False):
        return self._x


class _FakeController(object):
    """Duck-typed `self` for Controller.evaluate_criticality_measure."""

    def __init__(self, xopt, g, H, h, lh, prox):
        self.model = _FakeModel(xopt, g, H)
        self.h = h
        self.lh = lh
        self.prox_uh = prox
        self.argsh = ()
        self.argsprox = ()
        self.scaling_changes = None
        self.delta = 1.0


class TestCriticalityBoxFrame(unittest.TestCase):
    """The criticality subproblem must be solved over the TRUE box, not a shifted one.

    Analytic oracle (no reference to the implementation): n=1, xbase=0.5, sl/su=-0.5/0.5
    (true box [0,1]), xopt=0.8, g=3, H=0, h=2|x|, delta=1.
        m(d) = 3d + 2|0.8+d|,  d in [-0.8, 0.2] (true box)  ->  min at d=-0.8, m=-2.4,
        eta_1 = h(xopt) - min m = 1.6 - (-2.4) = 4.0.
    The shifted box [-0.5,0.5] instead allows d down to -1 (ball limited), m=-2.6, eta=4.2.
    """

    def runTest(self):
        from dfols.controller import Controller
        from dfols.params import ParameterList
        params = ParameterList(1, 2, 300)
        fake = _FakeController(np.array([0.8]), np.array([3.0]), np.zeros((1, 1)),
                               lambda x: 2.0 * np.abs(x).sum(), 2.0, prox_l1_scaled(2.0))
        fake.model.sl = np.array([-0.5])     # bounds relative to xbase, as Model stores them
        fake.model.su = np.array([0.5])
        fake.model.xbase = np.array([0.5])
        eta = Controller.evaluate_criticality_measure(fake, params)
        self.assertAlmostEqual(eta, 4.0, delta=1e-6,
                               msg="criticality subproblem uses a shifted box: eta=%r, expected 4.0" % eta)


class TestTrustRegionBoxFrame(unittest.TestCase):
    """The trust-region S-FISTA call (controller.py:586 site) must use the TRUE box."""

    def runTest(self):
        from dfols.controller import Controller
        from dfols.params import ParameterList
        params = ParameterList(1, 2, 300)
        fake = _FakeController(np.array([0.8]), np.array([3.0]), np.zeros((1, 1)),
                               lambda x: 2.0 * np.abs(x).sum(), 2.0, prox_l1_scaled(2.0))
        fake.model.sl = np.array([-0.5])
        fake.model.su = np.array([0.5])
        fake.model.xbase = np.array([0.5])
        d, gopt, H, gnew, crvmin = Controller.trust_region_step(fake, params, 4.0)
        x = np.array([0.8]) + np.asarray(d, dtype=float)
        # the true box is [sl+xbase, su+xbase] = [0, 1]; d=0 is feasible, so the returned
        # step must not leave it
        self.assertTrue(np.all(x >= -1e-12) and np.all(x <= 1.0 + 1e-12),
                        "trust-region step leaves the true box: x=%r" % x)
