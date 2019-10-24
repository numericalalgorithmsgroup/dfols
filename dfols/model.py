"""
Model
====

Maintain a class which represents an interpolating set, and its corresponding linear models
for each residual.
This class should calculate the various geometric quantities of interest to us.


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

import logging
from math import sqrt
import numpy as np
import scipy.linalg as LA

from .hessian import Hessian
from .trust_region import trsbox_geometry
from .util import sumsq
from .sketcher import sketch

__all__ = ['Model']

class Model(object):
    def __init__(self, npt, x0, r0, xl, xu, r0_nsamples, n=None, m=None, sub_sample_dim=None, abs_tol=1e-12, rel_tol=1e-20, precondition=True, sample_method=None):
        if n is None:
            n = len(x0)
        if m is None:
            m = len(r0)
        assert npt >= n + 1, "Require npt >= n+1 for linear models"
        assert x0.shape == (n,), "x0 has wrong shape (got %s, expect (%g,))" % (str(x0.shape), n)
        assert xl.shape == (n,), "xl has wrong shape (got %s, expect (%g,))" % (str(xl.shape), n)
        assert xu.shape == (n,), "xu has wrong shape (got %s, expect (%g,))" % (str(xu.shape), n)
        assert r0.shape == (m,), "r0 has wrong shape (got %s, expect (%g,))" % (str(r0.shape), m)
        self.dim = n
        self.resid_dim = m
        self.sub_sample_dim = sub_sample_dim
        self.num_pts = npt
        self.npt_so_far = 1  # number of points added so far (with function values)
        self._sample_method = sample_method

        # Initialise to blank some useful stuff
        # Interpolation points
        self.xbase = x0.copy()
        self.sl = xl - self.xbase  # lower bound w.r.t. xbase (require xpt >= sl)
        self.su = xu - self.xbase  # upper bound w.r.t. xbase (require xpt <= su)
        self.points = np.zeros((npt, n))  # interpolation points w.r.t. xbase

        # Function values
        self.fval_v = np.inf * np.ones((npt, m))  # residuals for each xpt
        self.fval_v[0, :] = r0.copy()
        self.fval = np.inf * np.ones((npt, ))  # overall objective value for each xpt
        self.fval[0] = sumsq(r0)
        self.kopt = 0  # index of current iterate (should be best value so far)
        self.nsamples = np.zeros((npt,), dtype=np.int)  # number of samples used to evaluate objective at each point
        self.nsamples[0] = r0_nsamples
        self.fbeg = self.fval[0]  # f(x0), saved to check for sufficient reduction

        # Termination criteria
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        # Model information
        sample_dim = sub_sample_dim if sub_sample_dim else m
        self.model_const = np.zeros((sample_dim, ))  # constant term for model m(s) = c + J*s
        self.model_jac = np.zeros((sample_dim, n))  # Jacobian term for model m(s) = c + J*s

        # Unsampled Model information
        self.unsampled_model_const = np.zeros((m, ))
        self.unsampled_model_jac = np.zeros((m, n))

        # Saved point (in absolute coordinates) - always check this value before quitting solver
        self.xsave = None
        self.rsave = None
        self.fsave = None
        self.jacsave = None
        self.nsamples_save = None

        # Factorisation of interpolation matrix
        self.factorisation_current = False
        self.Q = None
        self.R = None
        self.qr_of_transpose = False  # is QR for W (finished growing) or W.T (growing)?
        self.precondition = precondition
        self.left_scaling = None  # preconditioning
        self.right_scaling = None

    def n(self):
        return self.dim

    def m(self):
        return self.resid_dim

    def sample_size(self):
        return self.sub_sample_dim

    def sample_residuals(self, r):
        # sample each model differently
        return sketch(r.T, self.sample_size(), method=self.sample_method())

        # full_sample = range(self.m())
        # if self.sample_size() is None:
        #     return full_sample
        # else:
        #     return  np.sort(np.random.choice(full_sample, self.sub_sample_dim, replace=False))

    def sample_method(self):
        return self._sample_method

    def npt(self):
        return min(self.num_pts, self.npt_so_far)

    def xopt(self, abs_coordinates=False):
        return self.xpt(self.kopt, abs_coordinates=abs_coordinates)

    def ropt(self):
        return self.fval_v[self.kopt, :]  # residuals for current iterate

    def fopt(self):
        return self.fval[self.kopt]

    def xpt(self, k, abs_coordinates=False):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        if not abs_coordinates:
            return self.points[k, :].copy()
        else:
            # Apply bounds and convert back to absolute coordinates
            return self.xbase + np.minimum(np.maximum(self.sl, self.points[k, :]), self.su)

    def rvec(self, k):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        return self.fval_v[k, :]

    def fval(self, k):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        return self.fval[k]

    def as_absolute_coordinates(self, x):
        # If x were an interpolation point, get the absolute coordinates of x
        return self.xbase + np.minimum(np.maximum(self.sl, x), self.su)

    def xpt_directions(self, include_kopt=True):
        if include_kopt:
            ndirs = self.npt()
        else:
            ndirs = self.npt() - 1

        dirns = np.zeros((ndirs, self.n()))  # vector of directions xpt - xopt, excluding for xopt
        xopt = self.xopt()
        for k in range(self.npt()):
            if not include_kopt and k == self.kopt:
                continue  # skipt
            idx = k if include_kopt or k < self.kopt else k - 1
            dirns[idx, :] = self.xpt(k) - xopt
        return dirns

    def distances_to_xopt(self):
        sq_distances = np.zeros((self.npt(),))
        xopt = self.xopt()
        for k in range(self.npt()):
            sq_distances[k] = sumsq(self.points[k, :] - xopt)
        return sq_distances

    def change_point(self, k, x, rvec, allow_kopt_update=True):
        # Update point k to x (w.r.t. xbase), with residual values fvec
        if k >= self.npt_so_far and self.npt_so_far < self.num_pts:
            assert k == self.npt_so_far, "Growing: updating wrong point"
            self.npt_so_far += 1
        else:
            assert 0 <= k < self.npt(), "Invalid index %g" % k

        self.points[k, :] = x.copy()
        self.fval_v[k, :] = rvec.copy()
        self.fval[k] = sumsq(rvec)
        self.nsamples[k] = 1
        self.factorisation_current = False

        if allow_kopt_update and self.fval[k] < self.fopt():
            self.kopt = k
        return

    def swap_points(self, k1, k2):
        self.points[[k1, k2], :] = self.points[[k2, k1], :]
        self.fval_v[[k1, k2], :] = self.fval_v[[k2, k1], :]
        self.fval[[k1, k2]] = self.fval[[k2, k1]]
        if self.kopt == k1:
            self.kopt = k2
        elif self.kopt == k2:
            self.kopt = k1
        self.factorisation_current = False
        return

    def add_new_sample(self, k, rvec_extra):
        # We have resampled at xpt(k) - add this information (fval and fval_v are averages of all samples)
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        t = float(self.nsamples[k]) / float(self.nsamples[k] + 1)
        self.fval_v[k, :] = t * self.fval_v[k, :] + (1 - t) * rvec_extra
        self.fval[k] = sumsq(self.fval_v[k, :])
        self.nsamples[k] += 1

        self.kopt = np.argmin(self.fval[:self.npt()])  # make sure kopt is always the best value we have
        return

    def add_new_point(self, x, rvec):
        self.points = np.append(self.points, x.reshape((1, self.n())), axis=0)  # append row to xpt
        self.fval_v = np.append(self.fval_v, rvec.reshape((1, self.m())), axis=0)  # append row to fval_v
        f = np.dot(rvec, rvec)
        self.fval = np.append(self.fval, f)  # append entry to fval
        self.nsamples = np.append(self.nsamples, 1)  # add new sample number
        self.num_pts += 1  # make sure npt is updated
        self.npt_so_far += 1

        if f < self.fopt():
            self.kopt = self.npt() - 1

        self.factorisation_current = False
        return

    def shift_base(self, xbase_shift):
        # Shifting xbase -> xbase + xbase_shift
        for k in range(self.npt()):
            self.points[k, :] = self.points[k, :] - xbase_shift
        self.xbase += xbase_shift
        self.sl = self.sl - xbase_shift
        self.su = self.su - xbase_shift
        self.factorisation_current = False

        # Update model
        self.model_const += np.dot(self.model_jac, xbase_shift)
        return

    def save_point(self, x, rvec, nsamples, x_in_abs_coords=True):
        f = sumsq(rvec)
        if self.fsave is None or f <= self.fsave:
            self.xsave = x.copy() if x_in_abs_coords else self.as_absolute_coordinates(x)
            self.rsave = rvec.copy()
            self.fsave = f
            self.jacsave = self.model_jac.copy()
            self.nsamples_save = nsamples
            return True
        else:
            return False  # this value is worse than what we have already - didn't save

    def get_final_results(self):
        # Return x and fval for optimal point (either from xsave+fsave or kopt)
        if self.fsave is None or self.fopt() <= self.fsave:  # optimal has changed since xsave+fsave were last set
            return self.xopt(abs_coordinates=True).copy(), self.ropt().copy(), self.fopt(), self.model_jac.copy(), self.nsamples[self.kopt]
        else:
            return self.xsave.copy(), self.rsave.copy(), self.fsave, self.jacsave, self.nsamples_save

    def min_objective_value(self):
        # Get termination criterion for f small: f <= abs_tol or f <= rel_tol * f0
        return max(self.abs_tol, self.rel_tol * self.fbeg)

    def model_value(self, d, d_based_at_xopt=True, with_const_term=False):
        if d_based_at_xopt:
            Jd = np.dot(self.model_jac, d + self.xopt())
        else:  # d based at xbase
            Jd = np.dot(self.model_jac, d)  # J * d
        return Jd + (self.model_const if with_const_term else 0.0)

    # refactor this and above method
    def unsampled_model_value(self, d, d_based_at_xopt=True, with_const_term=False):
        if d_based_at_xopt:
            Jd = np.dot(self.unsampled_model_jac, d + self.xopt())
        else:  # d based at xbase
            Jd = np.dot(self.unsampled_model_jac, d)  # J * d
        return Jd + (self.unsampled_model_const if with_const_term else 0.0)

    def interpolation_matrix(self):
        W = np.zeros((self.npt(), self.n()+1))
        if self.precondition:
            approx_delta = sqrt(np.max(self.distances_to_xopt()))  # largest distance to xopt ~ delta
        else:
            approx_delta = 1.0

        W[:, 0] = 1.0
        W[:, 1:] = self.xpt_directions(include_kopt=True) / approx_delta  # rows are (yt-xk)

        left_scaling = np.ones((self.npt(),))  # no left scaling
        right_scaling = np.ones((self.n() + 1,))
        right_scaling[1:] = 1.0 / approx_delta
        return W, left_scaling, right_scaling

    def factorise_geom_system(self):
        if not self.factorisation_current:
            W, self.left_scaling, self.right_scaling = self.interpolation_matrix()
            p, n = W.shape  # npt = p + 1
            if p >= n:  # finished growing (npt >= n+1)
                self.Q, self.R = LA.qr(W, mode='economic')  # reduced QR (saves memory)
                self.qr_of_transpose = False
            else:  # growing (npt < n+1)
                self.Q, self.R = LA.qr(W.T, mode='economic')  # reduced QR (saves memory)
                self.qr_of_transpose = True
            self.factorisation_current = True
        return

    def solve_geom_system(self, rhs):
        if self.factorisation_current:
            if self.qr_of_transpose:
                # Growing case: solve underdetermined system W*x=rhs with W.T = Q*R
                # Golub & Van Loan (3rd edn), Algorithm 5.7.2
                Rb = LA.solve_triangular(self.R, rhs * self.left_scaling, trans='T')  # R.T \ rhs
                return np.dot(self.Q, Rb) * self.right_scaling  # minimal norm solution
            else:
                # Normal case: solve overdetermined system W*x=rhs with W=Q*R
                Qb = np.dot(self.Q.T, rhs * self.left_scaling)
                return LA.solve_triangular(self.R, Qb) * self.right_scaling
        else:
            logging.warning("model.solve_geom_system not using factorisation")
            W, left_scaling, right_scaling = self.interpolation_matrix()
            return LA.lstsq(W, rhs * left_scaling)[0] * right_scaling

    def interpolate_mini_models_svd(self, verbose=False, make_full_rank=False, min_sing_val=1e-6, sing_val_frac=1.0, max_jac_cond=1e8,
                                    get_chg_J=False, build_unsampled_model=False, ratio_factor=1):
        W, left_scaling, right_scaling = self.interpolation_matrix()
        self.factorise_geom_system()
        ls_interp_cond_num = np.linalg.cond(W) if verbose else 0.0  # scipy.linalg does not have condition number!

        # If not make_full_rank, Q is size (npt+n-1, npt+n), R is size (npt+n-1, n)
        # If make_full_rank, Q is size (2n, 2n), R is size (2n, n)
        xopt = self.xopt()
        ropt = self.ropt()
        fval_row_idx = np.arange(self.npt())  # indices of all rows
        norm_J_error = 0.0
        linalg_resid = 0.0


        if make_full_rank:
            # Remove old full-rank components of Jacobian
            Y = self.xpt_directions(include_kopt=False).T
            Qy, Ry = LA.qr(Y, mode='full')  # Qy is (n,n), Ry is (n,npt-1)=(n,p)
            Qhat = Qy[:, :Y.shape[1]]
            self.model_jac = np.dot(self.model_jac, np.dot(Qhat, Qhat.T))

        # sample each model differently
        r = self.fval_v[fval_row_idx]
        residuals = self.sample_residuals(r) if self.sample_size() is not None else r.T

        for index, rhs in enumerate(residuals):
            g_old = self.model_jac[index, :].copy()
            try:
                dg = self.solve_geom_system(rhs)
            except LA.LinAlgError:
                return False, None, None, None, None, None  # flag error
            except ValueError:
                return False, None, None, None, None, None  # flag error (e.g. inf or NaN encountered)

            if not np.all(np.isfinite(dg)):  # another check for inf or NaN
                return False, None, None, None, None, None  # flag error

            self.model_jac[index, :] = dg[1:]
            self.model_const[index] = dg[0] - np.dot(self.model_jac[index, :], xopt)  # shift base to xbase
            if verbose or get_chg_J:
                norm_J_error += sumsq(dg[1:] - g_old)
                linalg_resid += sumsq(np.dot(W, dg) - rhs)

        if make_full_rank:
            try:
                U, s, Vt = LA.svd(self.model_jac, full_matrices=False)  # U is (m,k), s has length k, Vt is (k,n), where k=min(m,n)
            except LA.LinAlgError:
                return False, None, None, None, None, None  # flag error
            k = min(self.n(), self.m())
            r = min(self.npt_so_far - 1, self.n(), self.m())  # current number of directions (i.e. rank of J)
            floor_val = max(s[0]/max_jac_cond, sing_val_frac * s[r-1], min_sing_val)
            s = np.maximum(s, floor_val)
            S = LA.diagsvd(s, k, k)  # s from vector to matrix of correct shape
            self.model_jac = np.dot(U, np.dot(S, Vt))  # reconstruct J from new svd

        interp_error = 0.0
        unsampled_interp_error = 0.0
        if verbose:
            if build_unsampled_model:
                # build unsampled model
                for m1 in range(self.m()):
                    rhs = self.fval_v[fval_row_idx, m1]  # length (npt)
                    try:
                        dg = self.solve_geom_system(rhs)
                    except LA.LinAlgError:
                        return False, None, None, None, None, None  # flag error
                    except ValueError:
                        return False, None, None, None, None, None  # flag error (e.g. inf or NaN encountered)

                    if not np.all(np.isfinite(dg)):  # another check for inf or NaN
                        return False, None, None, None, None, None  # flag error

                    self.unsampled_model_jac[m1, :] = dg[1:]
                    self.unsampled_model_const[m1] = dg[0] - np.dot(self.unsampled_model_jac[m1, :], xopt)  # shift base to xbase

                for k in range(self.npt()):
                    # Unsampled model interpolation error
                    unsampled_r_pred = self.unsampled_model_value(self.xpt(k), d_based_at_xopt=False, with_const_term=True)
                    # unsampled_interp_error += self.nsamples[k] * abs(sumsq(self.fval_v[k, :]) - sumsq(unsampled_r_pred))
                    unsampled_interp_error = max(abs(sumsq(self.fval_v[k, :]) - sumsq(unsampled_r_pred)), unsampled_interp_error)

                    # Sampled model interpolation error
                    sampled_r_pred = self.model_value(self.xpt(k), d_based_at_xopt=False, with_const_term=True)
                    # interp_error += self.nsamples[k] * abs(sumsq(unsampled_r_pred) - sumsq(sampled_r_pred * self.sample_size() / self.m()))
                    interp_error = max(abs(sumsq(self.fval_v[k, :]) - sumsq(sampled_r_pred)/ratio_factor), interp_error)
            else:
                for k in range(self.npt()):
                    # Sampled model interpolation error compared with actual objective
                    sampled_r_pred = self.model_value(self.xpt(k), d_based_at_xopt=False, with_const_term=True)
                    # interp_error += self.nsamples[k] * abs(sumsq(self.fval_v[k, :]) - sumsq(sampled_r_pred * self.m()/self.sample_size()))
                    interp_error = max(abs(sumsq(self.fval_v[k, :]) - sumsq(sampled_r_pred)/ratio_factor), interp_error)
        return True, interp_error, unsampled_interp_error, sqrt(norm_J_error), linalg_resid, ls_interp_cond_num  # flag ok

    def build_full_model(self):
        # Build full least squares objective model from mini-models
        # Centred around xopt
        r = self.model_const + np.dot(self.model_jac, self.xopt())  # constant term (for inexact interpolation)
        J = self.model_jac

        # Apply scaling based on convention for objective - this code uses sumsq(rvec) not 0.5*sumsq(rvec)
        g = 2.0 * np.dot(J.T, r)  # n-vector
        hess = Hessian(self.n(), vals=2.0 * np.dot(J.T, J))
        # print(J)
        return g, hess

    def lagrange_gradient(self, k, factorise_first=True):
        assert 0 <= k < self.npt(), "Invalid index %g" % k
        if factorise_first:
            self.factorise_geom_system()

        rhs = np.zeros((self.npt(),))
        rhs[k] = 1.0
        soln = self.solve_geom_system(rhs)
        c = soln[0]
        g = soln[1:]
        return c, g  # constant, gradient [all based at xopt]

    def poisedness_constant(self, delta, xbase=None, xbase_in_abs_coords=True):
        # Calculate the poisedness constant of the current interpolation set in B(xbase, delta)
        # if xbase is None, use self.xopt()
        overall_max = None
        if xbase is None:
            xbase = self.xopt()
        elif xbase_in_abs_coords:
            xbase = xbase - self.xbase  # shift to correct position
        for k in range(self.npt()):
            c, g = self.lagrange_gradient(k, factorise_first=True)
            newc = c + np.dot(g, xbase - self.xopt())  # based at xbase
            # Solve problem: bounds are sl <= x <= su, and ||x-xopt|| <= delta
            xmax = trsbox_geometry(xbase, newc, g, self.sl, self.su, delta)
            lmax = abs(c + np.dot(g, xmax - self.xopt()))  # evaluate Lagrange poly
            if overall_max is None or lmax > overall_max:
                overall_max = lmax
        return overall_max

