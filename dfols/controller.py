"""
Controller
====

A class containing all the actions we want to perform during the algorithm.
This allows the code for the actual solver to focus on the logic & flow of the
algorithm, not the specific implementation details.


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
from math import log, sqrt
import numpy as np
import scipy.linalg as LA

from .model import *
from .trust_region import *
from .util import *

__all__ = ['Controller', 'ExitInformation', 'EXIT_SLOW_WARNING', 'EXIT_MAXFUN_WARNING', 'EXIT_SUCCESS',
           'EXIT_INPUT_ERROR', 'EXIT_TR_INCREASE_ERROR', 'EXIT_LINALG_ERROR', 'EXIT_FALSE_SUCCESS_WARNING',
           'EXIT_AUTO_DETECT_RESTART_WARNING']

EXIT_AUTO_DETECT_RESTART_WARNING = 4  # warning, auto-detected restart criteria
EXIT_FALSE_SUCCESS_WARNING = 3  # warning, maximum fake successful steps reached
EXIT_SLOW_WARNING = 2  # warning, maximum number of slow (successful) iterations reached
EXIT_MAXFUN_WARNING = 1  # warning, reached max function evals
EXIT_SUCCESS = 0  # successful finish (rho=rhoend, sufficient objective reduction, or everything in noise level)
EXIT_INPUT_ERROR = -1  # error, bad inputs
EXIT_TR_INCREASE_ERROR = -2  # error, trust region step increased model value
EXIT_LINALG_ERROR = -3  # error, linalg error (singular matrix encountered)


class ExitInformation(object):
    def __init__(self, flag, msg_details):
        self.flag = flag
        self.msg = msg_details

    def flag(self):
        return self.flag

    def message(self, with_stem=True):
        if not with_stem:
            return self.msg
        elif self.flag == EXIT_SUCCESS:
            return "Success: " + self.msg
        elif self.flag == EXIT_SLOW_WARNING:
            return "Warning (slow progress): " + self.msg
        elif self.flag == EXIT_MAXFUN_WARNING:
            return "Warning (max evals): " + self.msg
        elif self.flag == EXIT_INPUT_ERROR:
            return "Error (bad input): " + self.msg
        elif self.flag == EXIT_TR_INCREASE_ERROR:
            return "Error (trust region increase): " + self.msg
        elif self.flag == EXIT_LINALG_ERROR:
            return "Error (linear algebra): " + self.msg
        elif self.flag == EXIT_FALSE_SUCCESS_WARNING:
            return "Warning (max false good steps): " + self.msg
        else:
            return "Unknown exit flag: " + self.msg

    def able_to_do_restart(self):
        if self.flag in [EXIT_TR_INCREASE_ERROR, EXIT_LINALG_ERROR, EXIT_SLOW_WARNING, EXIT_AUTO_DETECT_RESTART_WARNING]:
            return True
        elif self.flag in [EXIT_MAXFUN_WARNING, EXIT_INPUT_ERROR]:
            return False
        else:
            # Successful step (rho=rhoend, noise level termination, or value small)
            return "sufficiently small" not in self.msg  # restart for rho=rhoend and noise level termination


class Controller(object):
    def __init__(self, objfun, args, x0, r0, r0_nsamples, xl, xu, npt, rhobeg, rhoend, nf, nx, maxfun, params, scaling_changes, sub_sample_dim, sample_method):
        self.objfun = objfun
        self.args = args
        self.maxfun = maxfun
        self.model = Model(npt, x0, r0, xl, xu, r0_nsamples, sub_sample_dim=sub_sample_dim, precondition=params("interpolation.precondition"),
                           abs_tol = params("model.abs_tol"), rel_tol = params("model.rel_tol"), sample_method=sample_method)
        self.nf = nf
        self.nx = nx
        self.rhobeg = rhobeg
        self.delta = rhobeg
        self.rho = rhobeg
        self.rhoend = rhoend
        self.diffs = [0.0, 0.0, 0.0]
        self.last_iters_step_taken = []
        self.last_fopts_step_taken = []
        self.num_slow_iters = 0
        self.finished_growing = False
        self.finished_halfway_growing = False
        # For measuing slow iterations
        self.last_iters_step_taken = []
        self.last_fopts_step_taken = []
        self.num_slow_iters = 0  # how many consecutive slow iterations have we had so far
        # For determining when to reduce rho
        self.last_successful_iter = 0  # when ||d|| >= rho
        # For counting the number of soft restarts
        self.last_successful_run = 0
        self.last_run_fopt = sumsq(r0)
        self.scaling_changes = scaling_changes

    def n(self):
        return self.model.n()

    def m(self):
        return self.model.m()

    def npt(self):
        return self.model.npt()

    def initialise_coordinate_directions(self, number_of_samples, num_directions, params):
        logging.debug("Initialising with coordinate directions")
        # self.model already has x0 evaluated, so only need to initialise the other points
        # num_directions = params("growing.ndirs_initial")
        assert self.model.num_pts <= (self.n() + 1) * (self.n() + 2) // 2, "prelim: must have npt <= (n+1)(n+2)/2"
        assert 1 <= num_directions < self.model.num_pts, "Initialisation: must have 1 <= ndirs_initial < npt"

        at_lower_boundary = (self.model.sl > -0.01 * self.delta)  # sl = xl - x0, should be -ve, actually < -rhobeg
        at_upper_boundary = (self.model.su < 0.01 * self.delta)  # su = xu - x0, should be +ve, actually > rhobeg

        xpts_added = np.zeros((num_directions + 1, self.n()))
        for k in range(1, num_directions + 1):
            # k = 0 --> base point (xpt = 0)  [ not here]
            # k = 1, ..., 2n --> coordinate directions [1,...,n and n+1,...,2n]
            # k = 2n+1, ..., (n+1)(n+2)/2 --> off-diagonal directions
            if 1 <= k < self.n() + 1:  # first step along coord directions
                dirn = k - 1  # direction to move in (0,...,n-1)
                stepa = self.delta if not at_upper_boundary[dirn] else -self.delta
                stepb = None
                xpts_added[k, dirn] = stepa

            elif self.n() + 1 <= k < 2 * self.n() + 1:  # second step along coord directions
                dirn = k - self.n() - 1  # direction to move in (0,...,n-1)
                stepa = xpts_added[k - self.n(), dirn]
                stepb = -self.delta
                if at_lower_boundary[dirn]:
                    stepb = min(2.0 * self.delta, self.model.su[dirn])  # su = xu - x0, should be +ve
                if at_upper_boundary[dirn]:
                    stepb = max(-2.0 * self.delta, self.model.sl[dirn])  # sl = xl - x0, should be -ve
                xpts_added[k, dirn] = stepb

            else:  # k = 2n+1, ..., (n+1)(n+2)/2
                # p = (k - 1) % n + 1  # cycles through (1,...,n), starting at 2n+1 --> 1
                # l = (k - 2 * n - 1) / n + 1  # (1,...,1, 2, ..., 2, etc.) where each number appears n times
                # q = (p + l if p + l <= n else p + l - n)
                stepa = None
                stepb = None
                itemp = (k - self.n() - 1) // self.n()
                q = k - itemp * self.n() - self.n()
                p = q + itemp
                if p > self.n():
                    p, q = q, p - self.n()  # does swap correctly in Python

                xpts_added[k, p - 1] = xpts_added[p, p - 1]
                xpts_added[k, q - 1] = xpts_added[q, q - 1]

            # Evaluate objective at this new point
            x = self.model.as_absolute_coordinates(xpts_added[k, :])
            rvec_list, f_list, num_samples_run, exit_info = self.evaluate_objective(x, number_of_samples, params)

            # Handle exit conditions (f < min obj value or maxfun reached)
            if exit_info is not None:
                if num_samples_run > 0:
                    self.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run,
                                          x_in_abs_coords=True)
                return exit_info  # return & quit

            # Otherwise, add new results (increments model.npt_so_far)
            self.model.change_point(k, x - self.model.xbase, rvec_list[0, :])  # expect step, not absolute x
            for i in range(1, num_samples_run):
                self.model.add_new_sample(k, rvec_extra=rvec_list[i, :])

            # If k exceeds N+1, then the positions of the k-th and (k-N)-th interpolation
            # points may be switched, in order that the function value at the first of them
            # contributes to the off-diagonal second derivative terms of the initial quadratic model.
            # Note: this works because the steps for (k) and (k-n) points were in the same coordinate direction
            if self.n() + 1 <= k < 2 * self.n() + 1:
                # Only swap if steps were in different directions AND new pt has lower objective
                if stepa * stepb < 0.0 and self.model.fval[k] < self.model.fval[k - self.n()]:
                    xpts_added[[k, k-self.n()]] = xpts_added[[k-self.n(), k]]

        return None   # return & continue

    def initialise_random_directions(self, number_of_samples, num_directions, params):
        logging.debug("Initialising with random orthogonal directions")
        # self.model already has x0 evaluated, so only need to initialise the other points
        assert 1 <= num_directions < self.model.num_pts, "Initialisation: must have 1 <= ndirs_initial < npt"

        # Get ndirs_initial random orthogonal directions
        xopt = self.model.xopt()
        if params("init.random_directions_make_orthogonal"):
            dirns = random_orthog_directions_within_bounds(num_directions, self.delta, self.model.sl - xopt,
                                                           self.model.su - xopt)
        else:
            dirns = random_directions_within_bounds(num_directions, self.delta, self.model.sl - xopt,
                                                    self.model.su - xopt)
        # Now add the random directions
        if params("init.run_in_parallel"):
            # Run all results in parallel first, then process
            eval_obj_results = []
            for ndirns in range(num_directions):
                new_point = xopt + dirns[ndirns, :]  # alway base move around best value so far

                # Evaluate objective
                x = self.model.as_absolute_coordinates(new_point)
                eval_obj_results.append(self.evaluate_objective(x, number_of_samples, params))

            for ndirns in range(num_directions):
                new_point = xopt + dirns[ndirns, :]  # alway base move around best value so far
                x = self.model.as_absolute_coordinates(new_point)
                rvec_list, f_list, num_samples_run, exit_info = eval_obj_results[ndirns]
                # Handle exit conditions (f < min obj value or maxfun reached)
                if exit_info is not None:
                    if num_samples_run > 0:
                        self.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run,
                                              x_in_abs_coords=True)
                    return exit_info  # return & quit

                # Otherwise, add new results (increments model.npt_so_far)
                self.model.change_point(1 + ndirns, x - self.model.xbase,
                                        rvec_list[0, :])  # expect step, not absolute x
                for i in range(1, num_samples_run):
                    self.model.add_new_sample(1 + ndirns, rvec_extra=rvec_list[i, :])
        else:
            for ndirns in range(num_directions):
                new_point = xopt + dirns[ndirns, :]  # alway base move around best value so far

                # Evaluate objective
                x = self.model.as_absolute_coordinates(new_point)
                rvec_list, f_list, num_samples_run, exit_info = self.evaluate_objective(x, number_of_samples, params)

                # Handle exit conditions (f < min obj value or maxfun reached)
                if exit_info is not None:
                    if num_samples_run > 0:
                        self.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run,
                                              x_in_abs_coords=True)
                    return exit_info  # return & quit

                # Otherwise, add new results (increments model.npt_so_far)
                self.model.change_point(1 + ndirns, x - self.model.xbase, rvec_list[0, :])  # expect step, not absolute x
                for i in range(1, num_samples_run):
                    self.model.add_new_sample(1 + ndirns, rvec_extra=rvec_list[i, :])

        return None

    def add_new_direction_while_growing(self, number_of_samples, params, min_num_steps=0):
        num_steps = max(params('growing.num_new_dirns_each_iter'), min_num_steps)
        step_length = params('growing.delta_scale_new_dirns') * self.delta
        if num_steps < 1:  # not actually adding new directions
            return None

        # Step from xopt along a random direction orthogonal to other yt (or multiple mutually orthogonal steps)
        xopt = self.model.xopt()
        dirns = random_directions_within_bounds(num_steps, step_length, self.model.sl - xopt, self.model.su - xopt)
        # Make direction orthogonal
        Y = self.model.xpt_directions(include_kopt=False).T  # columns are the current set of directions
        Q, R = LA.qr(Y, mode='economic')  # columns of Q are orthonormal basis for current set of directions
        for k in range(Q.shape[1]):
            qk = Q[:, k]
            for j in range(dirns.shape[0]):
                dirns[j, :] = dirns[j, :] - np.dot(dirns[j, :], qk) * qk

        # Evaluate the points
        for j in range(num_steps):
            xnew = self.model.xopt() + (step_length / LA.norm(dirns[j, :])) * dirns[j, :]
            x = self.model.as_absolute_coordinates(xnew)
            rvec_list, f_list, num_samples_run, exit_info = self.evaluate_objective(x, number_of_samples, params)

            # Handle exit conditions (f < min obj value or maxfun reached)
            if exit_info is not None:
                if num_samples_run > 0:
                    self.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run,
                                          x_in_abs_coords=True)
                return exit_info  # return & quit

            if self.model.npt() < self.model.num_pts:  # still growing
                kmin = self.model.npt()
            else:  # full set - choose in usual way
                kmin, linalg_error = self.choose_point_to_replace(xnew - self.model.xopt(), skip_kopt=True)

                if linalg_error:
                    exit_info = ExitInformation(EXIT_LINALG_ERROR, "Singular matrix when finding kmin (in main loop)")
                    return exit_info  # return & quit

            # Otherwise, add new results
            self.model.change_point(kmin, xnew, rvec_list[0, :])  # expect step, not absolute x
            for i in range(1, num_samples_run):
                self.model.add_new_sample(kmin, rvec_extra=rvec_list[i, :])

        return None

    def get_new_direction_for_growing(self, step_length):
        # Step from xopt along a random direction orthogonal to other yt (or multiple mutually orthogonal steps)
        xopt = self.model.xopt()
        dirn = random_directions_within_bounds(1, step_length, self.model.sl - xopt, self.model.su - xopt)[0, :]
        # Make direction orthogonal
        Y = self.model.xpt_directions(include_kopt=False).T  # columns are the current set of directions
        Q, R = LA.qr(Y, mode='economic')  # columns of Q are orthonormal basis for current set of directions
        for k in range(Q.shape[1]):
            qk = Q[:, k]
            dirn = dirn - np.dot(dirn, qk) * qk

        return dirn * (step_length / LA.norm(dirn))

    def trust_region_step(self):
        # Build model for full least squares objectives
        gopt, hq = self.model.build_full_model()
        d, gnew, crvmin = trsbox(self.model.xopt(), gopt, hq, self.model.sl, self.model.su, self.delta)
        return d, gopt, hq, gnew, crvmin

    def geometry_step(self, knew, adelt, number_of_samples, params):
        logging.debug("Running geometry-fixing step")
        try:
            c, g = self.model.lagrange_gradient(knew)
            # c = 1.0 if knew == self.model.kopt else 0.0  # based at xopt, just like d
            # Solve problem: bounds are sl <= xnew <= su, and ||xnew-xopt|| <= adelt
            xnew = trsbox_geometry(self.model.xopt(), c, g, self.model.sl, self.model.su, adelt)
        except LA.LinAlgError:
            exit_info = ExitInformation(EXIT_LINALG_ERROR, "Singular matrix encountered in geometry step")
            return exit_info  # didn't fix geometry - return & quit

        gopt, hq = self.model.build_full_model()  # save here, to calculate predicted value from geometry step
        fopt = self.model.fopt()  # again, evaluate now, before model.change_point()
        d = xnew - self.model.xopt()
        x = self.model.as_absolute_coordinates(xnew)
        rvec_list, f_list, num_samples_run, exit_info = self.evaluate_objective(x, number_of_samples, params)

        # Handle exit conditions (f < min obj value or maxfun reached)
        if exit_info is not None:
            if num_samples_run > 0:
                self.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run,
                                      x_in_abs_coords=True)
            return exit_info  # didn't fix geometry - return & quit

        # Otherwise, add new results
        self.model.change_point(knew, xnew, rvec_list[0, :])  # expect step, not absolute x
        for i in range(1, num_samples_run):
            self.model.add_new_sample(knew, rvec_extra=rvec_list[i, :])

        # Estimate actual reduction to add to diffs vector
        f = sumsq(np.mean(rvec_list[:num_samples_run, :], axis=0))  # estimate actual objective value

        # pred_reduction = - calculate_model_value(gopt, hq, d)
        pred_reduction = - model_value(gopt, hq, d)
        actual_reduction = fopt - f
        self.diffs = [abs(pred_reduction - actual_reduction), self.diffs[0], self.diffs[1]]
        return None  # exit_info = None

    def check_and_fix_geometry(self, distsq_thresh, update_delta, number_of_samples, params):  # check and fix geometry
        # Find the point furthest from xopt
        sq_distances = self.model.distances_to_xopt()
        knew = np.argmax(sq_distances)
        distsq = sq_distances[knew]

        if distsq <= distsq_thresh:  # points close enough - don't do anything
            return False, None  # didn't fix geometry at all

        dist = sqrt(distsq)
        if update_delta:  # optional
            self.delta = max(min(0.1 * self.delta, 0.5 * dist), 1.5 * self.rho)

        adelt = max(min(0.1 * dist, self.delta), self.rho)
        dsq = adelt ** 2
        if dsq <= params("general.rounding_error_constant") * sumsq(self.model.xopt()):
            self.model.shift_base(self.model.xopt())

        exit_info = self.geometry_step(knew, adelt, number_of_samples, params)
        return True, exit_info  # didn't fix geometry - return & continue

    def evaluate_objective(self, x, number_of_samples, params):
        # Sample from objective function several times, keeping track of maxfun and min_obj_value throughout
        rvec_list = np.zeros((number_of_samples, self.m()))
        f_list = np.zeros((number_of_samples,))
        num_samples_run = 0
        incremented_nx = False
        exit_info = None

        for i in range(number_of_samples):
            if self.nf >= self.maxfun:
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                break  # quit

            self.nf += 1
            if not incremented_nx:
                self.nx += 1
                incremented_nx = True
            rvec_list[i, :], f_list[i] = eval_least_squares_objective(self.objfun, remove_scaling(x, self.scaling_changes), args=self.args, eval_num=self.nf, pt_num=self.nx,
                                            full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                            check_for_overflow=params("general.check_objfun_for_overflow"))
            num_samples_run += 1

        # Check if the average value was below our threshold
        if num_samples_run > 0 and \
                        sumsq(np.mean(rvec_list[:num_samples_run, :], axis=0)) <= self.model.min_objective_value():
            exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")

        return rvec_list, f_list, num_samples_run, exit_info

    def choose_point_to_replace(self, d, skip_kopt=True):
        delsq = self.delta ** 2
        scaden = None
        knew = None  # may knew never be set here?
        exit_info = None

        for k in range(self.model.npt()):
            if skip_kopt and k == self.model.kopt:
                continue  # skip this k

            # Build Lagrange polynomial
            try:
                c, g = self.model.lagrange_gradient(k)
            except LA.LinAlgError:
                exit_info = ExitInformation(EXIT_LINALG_ERROR, "Singular matrix when choosing point to replace")
                break  # end & quit

            den = c + np.dot(g, d)

            distsq = sumsq(self.model.xpt(k) - self.model.xopt())
            temp = max(1.0, (distsq / delsq) ** 2)
            if scaden is None or temp * abs(den) > scaden:
                scaden = temp * abs(den)
                knew = k

        return knew, exit_info

    def done_with_current_rho(self, xnew, gnew, crvmin, hq, current_iter):
        # (xnew, gnew, crvmin) come from trust region step
        # hq is Hessian of model for the full objective

        # Wait at least 3 iterations between reductions of rho
        if current_iter <= self.last_successful_iter + 2:
            return False

        errbig = max(self.diffs)
        frhosq = 0.125 * self.rho ** 2
        if crvmin > 0.0 and errbig > frhosq * crvmin:
            return False

        bdtol = errbig / self.rho
        for j in range(self.n()):
            bdtest = bdtol
            if xnew[j] == self.model.sl[j]:
                bdtest = gnew[j]
            if xnew[j] == self.model.su[j]:
                bdtest = -gnew[j]
            if bdtest < bdtol:
                curv = hq.get_element(j, j)  # curv = Hessian(j, j)
                bdtest += 0.5 * curv * self.rho
                if bdtest < bdtol:
                    return False
        # Otherwise...
        return True

    def reduce_rho(self, current_iter, params):
        alpha1 = params("tr_radius.alpha1")
        alpha2 = params("tr_radius.alpha2")
        ratio = self.rho / self.rhoend
        if ratio <= 16.0:
            new_rho = self.rhoend
        elif ratio <= 250.0:
            new_rho = sqrt(ratio) * self.rhoend  # geometric average of rho and rhoend
        else:
            new_rho = alpha1 * self.rho

        self.delta = max(alpha2 * self.rho, new_rho)  # self.rho = old rho
        self.rho = new_rho
        self.last_successful_iter = current_iter  # reset successful iteration check
        return

    def calculate_ratio(self, current_iter, rvec_list, d, gopt, hq, ratio_factor):
        exit_info = None
        f = sumsq(np.mean(rvec_list, axis=0))  # estimate actual objective value
        pred_reduction = - model_value(gopt, hq, d)
        actual_reduction = self.model.fopt() - f
        self.diffs = [abs(actual_reduction - pred_reduction), self.diffs[0], self.diffs[1]]
        if min(sqrt(sumsq(d)), self.delta) > self.rho:  # if ||d|| >= rho, successful!
            self.last_successful_iter = current_iter
        if pred_reduction < 0.0:
            exit_info = ExitInformation(EXIT_TR_INCREASE_ERROR, "Trust region step gave model increase")

        ratio = ratio_factor * actual_reduction / pred_reduction
        return ratio, exit_info

    def terminate_from_slow_iterations(self, current_iter, params):
        if len(self.last_iters_step_taken) <= params("slow.history_for_slow"):
            # Not enough info, simply append
            self.last_iters_step_taken.append(current_iter)
            self.last_fopts_step_taken.append(self.model.fopt())
            this_iter_slow = False
        else:
            # Enough info - shift values
            self.last_iters_step_taken = self.last_iters_step_taken[1:] + [current_iter]
            self.last_fopts_step_taken = self.last_fopts_step_taken[1:] + [self.model.fopt()]
            this_iter_slow = (log(self.last_fopts_step_taken[0]) - log(self.model.fopt())) / \
                             float(params("slow.history_for_slow")) < params("slow.thresh_for_slow")
        # Update counter of number of slow iterations
        if this_iter_slow:
            self.num_slow_iters += 1
            logging.info("Slow iteration (%g consecutive so far, max allowed %g)"
                         % (self.num_slow_iters, params("slow.max_slow_iters")))
        else:
            self.num_slow_iters = 0
            logging.debug("Non-slow iteration")
        return this_iter_slow, self.num_slow_iters >= params("slow.max_slow_iters")

    def soft_restart(self, number_of_samples, nruns_so_far, params, x_in_abs_coords_to_save=None, rvec_to_save=None,
                     nsamples_to_save=None):
        # A successful run is one where we reduced fopt
        if self.model.fopt() < self.last_run_fopt:
            self.last_successful_run = nruns_so_far
        self.last_run_fopt = self.model.fopt()

        ok_to_do_restart = (nruns_so_far - self.last_successful_run < params("restarts.max_unsuccessful_restarts")) and \
                           (self.nf < self.maxfun)

        if not ok_to_do_restart:
            # last outputs are (exit_flag, exit_str, return_to_new_tr_iteration)
            exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
            if nruns_so_far - self.last_successful_run >= params("restarts.max_unsuccessful_restarts"):
                exit_info = ExitInformation(EXIT_SUCCESS, "Reached maximum number of unsuccessful restarts")
            return exit_info

        # Save current best point and the one the user wanted to save too
        if x_in_abs_coords_to_save is not None:
            assert rvec_to_save is not None, "Soft restart: specified x_to_save but not rvec_to_save"
            assert nsamples_to_save is not None, "Soft restart: specified x_to_save but not nsamples_to_save"
            self.model.save_point(x_in_abs_coords_to_save, rvec_to_save, nsamples_to_save, x_in_abs_coords=True)
        self.model.save_point(self.model.xopt(abs_coordinates=True), self.model.ropt(),
                              self.model.nsamples[self.model.kopt], x_in_abs_coords=True)

        logging.info("Soft restart [currently, f = %g after %g function evals]" % (self.model.fopt(), self.nf))
        # Resetting method: reset delta and rho, then move the closest 'num_steps' points to xk to improve geometry
        # Note: closest points because we are suddenly increasing delta & rho, so we want to encourage spreading out points
        self.delta = self.rhobeg
        self.rho = self.rhobeg
        self.diffs = [0.0, 0.0, 0.0]

        all_sq_dist = self.model.distances_to_xopt()[:self.model.npt()]
        closest_points = np.argsort(all_sq_dist)
        if params("restarts.soft.move_xk"):
            closest_points = closest_points[:params("restarts.soft.num_geom_steps")]
            upper_limit = self.model.num_pts
        else:
            closest_points = closest_points[1:params("restarts.soft.num_geom_steps")+1]
            upper_limit = self.model.num_pts - 1

        for i in range(min(params("restarts.soft.num_geom_steps"), upper_limit)):
            # Determine which point to update (knew)
            knew = closest_points[i]

            # Using adelt=delta in fix_geometry (adelt determines the ball to max lagrange poly in altmov)
            # [Only reason actual 'delta' is needed in fix_geometry is for calling nsamples()]
            exit_info = self.geometry_step(knew, self.delta, number_of_samples, params)
            if exit_info is not None:
                return exit_info

        # Now increase npt, if required
        if params("restarts.increase_npt") and self.model.npt() < params("restarts.max_npt"):
            num_pts_to_add = min(params("restarts.increase_npt_amt"), params("restarts.max_npt") - self.model.npt())
            # First n points will be random orthogonal directions; the rest will be purely random directions
            # sl <= xopt + dirn <= su   -or equivalently-   sl-xopt <= dirn <= su-xopt
            xopt = self.model.xopt()
            dirns = random_directions_within_bounds(num_pts_to_add, self.delta, self.model.sl - xopt,
                                                    self.model.su - xopt)
            for i in range(num_pts_to_add):
                xnew = self.model.xopt() + dirns[i, :]  # always base move around best value so far
                x = self.model.as_absolute_coordinates(xnew)
                rvec_list, f_list, num_samples_run, exit_info = self.evaluate_objective(x, number_of_samples, params)

                # Handle exit conditions (f < min obj value or maxfun reached)
                if exit_info is not None:
                    if num_samples_run > 0:
                        self.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run,
                                              x_in_abs_coords=True)
                    return exit_info  # return & quit

                # Otherwise, add new results
                self.model.add_new_point(xnew, rvec_list[0, :])  # expect step, not absolute x
                for i in range(1, num_samples_run):
                    self.model.add_new_sample(self.model.npt() - 1, rvec_extra=rvec_list[i, :])

            logging.info("Soft restart: added %g new directions, npt is now %g" % (num_pts_to_add, self.model.npt()))

        # Otherwise, we are doing a restart
        self.last_successful_iter = 0
        return None  # exit_info = None

    def move_furthest_points(self, number_of_samples, num_pts_to_move, params):
        # get points furthest from (new) xopt and move to a good geometry point
        all_sq_dist = self.model.distances_to_xopt()[:self.model.npt()]
        furthest_points = np.argsort(all_sq_dist)[::-1]  # indices from furthest to closest (last is kopt)

        for i in range(min(num_pts_to_move, len(furthest_points) - 1)):
            # Determine which point to update (knew)
            knew = furthest_points[i]

            # Using adelt=delta in fix_geometry (adelt determines the ball to max lagrange poly in altmov)
            # [Only reason actual 'delta' is needed in fix_geometry is for calling nsamples()]
            adelt = self.delta
            exit_info = self.geometry_step(knew, adelt, number_of_samples, params)

            if exit_info is not None:
                return exit_info

        return None

    def all_values_within_noise_level(self, params):
        all_fvals_within_noise = True
        # We scale the expected noise by sqrt(# samples used to generate averages)
        if params("noise.additive_noise_level") is not None:
            add_noise = params("noise.scale_factor_for_quit") * params("noise.additive_noise_level")
            for k in range(self.model.npt()):
                all_fvals_within_noise = all_fvals_within_noise and \
                                (self.model.fval[k] <= self.model.fopt() + add_noise / sqrt(self.model.nsamples[k]))
        else:  # noise_level_multiplicative
            ratio = 1.0 + params("noise.scale_factor_for_quit") * params("noise.multiplicative_noise_level")
            for k in range(self.model.npt()):
                this_ratio = self.model.fval[k] / self.model.fopt()  # fval_opt strictly positive (would have quit o/w)
                all_fvals_within_noise = all_fvals_within_noise and (
                    this_ratio <= ratio / sqrt(self.model.nsamples[k]))
        return all_fvals_within_noise

    def move_furthest_points_momentum(self, d, number_of_samples, num_pts_to_move, params):
        all_sq_dist = self.model.distances_to_xopt()[:self.model.npt()]
        furthest_points = np.argsort(all_sq_dist)[::-1]  # indices from furthest to closest (last is kopt)

        xopt = self.model.xopt()
        dirns = random_directions_within_bounds(num_pts_to_move, self.delta, self.model.sl - xopt, self.model.su - xopt)

        for i in range(min(num_pts_to_move, len(furthest_points) - 1)):
            # Determine which point to update (knew)
            knew = furthest_points[i]
            if np.dot(dirns[i, :], d) < 0.0:
                dirns[i, :] = -dirns[i, :]  # flip so pointing in direction
                flipped = True
            else:
                flipped = False
            # However, this may send us outside our bounds if we are near the boundary; don't flip if the flip
            # reduces the size of the step by too much
            if flipped:
                fixed_dirn = np.maximum(np.minimum(dirns[i, :], self.model.su - self.model.xopt()),
                                         self.model.sl - self.model.xopt())
                if np.linalg.norm(fixed_dirn) < 1e-3 * self.delta:
                    dirns[i, :] = -dirns[i, :]
            xnew = np.maximum(np.minimum(self.model.xopt() + dirns[i, :], self.model.su), self.model.sl)
            x = self.model.as_absolute_coordinates(xnew)
            rvec_list, f_list, num_samples_run, exit_info = self.evaluate_objective(x, number_of_samples, params)

            # Handle exit conditions (f < min obj value or maxfun reached)
            if exit_info is not None:
                if num_samples_run > 0:
                    self.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run,
                                          x_in_abs_coords=True)
                return exit_info  # return & quit

            # Otherwise, add new results
            self.model.change_point(knew, xnew, rvec_list[0, :])  # expect step, not absolute x
            for i in range(1, num_samples_run):
                self.model.add_new_sample(knew, rvec_extra=rvec_list[i, :])
        return None

