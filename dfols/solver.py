"""
Solver
====

The main solver


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
import os
import scipy.linalg as LA
import scipy.stats as STAT
import warnings

from .controller import *
from .diagnostic_info import *
from .params import *
from .util import *

__all__ = ['solve']

module_logger = logging.getLogger(__name__) 


# A container for the results of the optimization routine
class OptimResults(object):
    def __init__(self, xmin, rmin, objmin, jacmin, nf, nx, nruns, exit_flag, exit_msg, xmin_eval_num, jacmin_eval_nums):
        self.x = xmin
        self.resid = rmin
        self.obj = objmin
        self.jacobian = jacmin
        self.nf = nf
        self.nx = nx
        self.nruns = nruns
        self.flag = exit_flag
        self.msg = exit_msg
        self.diagnostic_info = None
        self.xmin_eval_num = xmin_eval_num
        self.jacmin_eval_nums = jacmin_eval_nums
        # Set standard names for exit flags
        self.EXIT_SLOW_WARNING = EXIT_SLOW_WARNING
        self.EXIT_MAXFUN_WARNING = EXIT_MAXFUN_WARNING
        self.EXIT_SUCCESS = EXIT_SUCCESS
        self.EXIT_INPUT_ERROR = EXIT_INPUT_ERROR
        self.EXIT_TR_INCREASE_ERROR = EXIT_TR_INCREASE_ERROR
        self.EXIT_LINALG_ERROR = EXIT_LINALG_ERROR
        self.EXIT_FALSE_SUCCESS_WARNING = EXIT_FALSE_SUCCESS_WARNING

    def __str__(self):
        # Result of calling print(soln)
        output = "****** DFO-LS Results ******\n"
        if self.flag != self.EXIT_INPUT_ERROR:
            output += "Solution xmin = %s\n" % str(self.x)
            if len(self.resid) < 100:
                output += "Residual vector = %s\n" % str(self.resid)
            else:
                output += "Not showing residual vector because it is too long; check self.resid\n"
            output += "Objective value f(xmin) = %.10g\n" % self.obj
            output += "Needed %g objective evaluations (at %g points)\n" % (self.nf, self.nx)
            if self.nruns > 1:
                output += "Did a total of %g runs\n" % self.nruns
            if self.jacobian is not None and np.size(self.jacobian) < 200:
                output += "Approximate Jacobian = %s\n" % str(self.jacobian)
            elif self.jacobian is None:
                output += "No Jacobian returned\n"
            else:
                output += "Not showing approximate Jacobian because it is too long; check self.jacobian\n"
            if self.diagnostic_info is not None:
                output += "Diagnostic information available; check self.diagnostic_info\n"
            output += "Solution xmin was evaluation point %g\n" % self.xmin_eval_num
            if len(self.jacmin_eval_nums) < 100:
                output += "Approximate Jacobian formed using evaluation points %s\n" % str(self.jacmin_eval_nums)
        output += "Exit flag = %g\n" % self.flag
        output += "%s\n" % self.msg
        output += "****************************\n"
        return output


def solve_main(objfun, x0, argsf, xl, xu, projections, npt, rhobeg, rhoend, maxfun, nruns_so_far, nf_so_far, nx_so_far, nsamples, params,
               diagnostic_info, scaling_changes, h=None, lh=None, argsh=(), prox_uh=None, argsprox=None, r0_avg_old=None, r0_nsamples_old=None, default_growing_method_set_by_user=None,
               do_logging=True, print_progress=False):
    # Evaluate at x0 (keep nf, nx correct and check for f < 1e-12)
    # The hard bit is determining what m = len(r0) should be, and allocating memory appropriately
    if r0_avg_old is None:
        number_of_samples = max(nsamples(rhobeg, rhobeg, 0, nruns_so_far), 1)
        # Evaluate the first time...
        nf = nf_so_far + 1
        nx = nx_so_far + 1
        r0, obj0 = eval_least_squares_with_regularisation(objfun, remove_scaling(x0, scaling_changes), h, 
                                              argsf=argsf, argsh=argsh, verbose=do_logging, eval_num=nf, pt_num=nx,
                                              full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                              check_for_overflow=params("general.check_objfun_for_overflow"))
        m = len(r0)

        # Now we have m, we can evaluate the rest of the times
        rvec_list = np.zeros((number_of_samples, m))
        obj_list = np.zeros((number_of_samples,))
        rvec_list[0, :] = r0
        obj_list[0] = obj0
        num_samples_run = 1
        exit_info = None

        for i in range(1, number_of_samples):  # skip first eval - already did this
            if nf >= maxfun:
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                nruns_so_far += 1
                break  # stop evaluating at x0

            nf += 1
            # Don't increment nx for x0 - we did this earlier
            rvec_list[i, :], obj_list[i] = eval_least_squares_with_regularisation(objfun, remove_scaling(x0, scaling_changes), h, 
                                                argsf=argsf, argsh=argsh, verbose=do_logging, eval_num=nf, pt_num=nx,
                                                full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                check_for_overflow=params("general.check_objfun_for_overflow"))
            num_samples_run += 1

        r0_avg = np.mean(rvec_list[:num_samples_run, :], axis=0)
        # NOTE: modify objvalue here
        if h is None:
            if sumsq(r0_avg) <= params("model.abs_tol"):
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
        else:
            if sumsq(r0_avg) + h(remove_scaling(x0, scaling_changes), *argsh)<= params("model.abs_tol"):
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")

        if exit_info is not None:
            return x0, r0_avg, sumsq(r0_avg), None, num_samples_run, nf, nx, nruns_so_far+1, exit_info, diagnostic_info

    else:  # have old r0 information (e.g. from previous restart), use this instead

        # m = len(r0_avg_old)
        r0_avg = r0_avg_old
        num_samples_run = r0_nsamples_old
        nf = nf_so_far
        nx = nx_so_far
    
    # On the first run, set default growing method (unless the user has already done this)
    if default_growing_method_set_by_user is not None and (not default_growing_method_set_by_user):
        # If m>=n, the default growing method (use_full_rank_interp) is best
        # However, this can fail for m<n, so need to use an alternative method (perturb_trust_region_step)
        if m < len(x0):
            if do_logging:
                module_logger.debug("Inverse problem (m<n), switching default growing method")
            params('growing.full_rank.use_full_rank_interp', new_value=False)
            params('growing.perturb_trust_region_step', new_value=True)
            if not params.params_changed['growing.delta_scale_new_dirns']:
                params('growing.delta_scale_new_dirns', new_value=0.1)

    # Initialise controller
    control = Controller(objfun, argsf, x0, r0_avg, num_samples_run, xl, xu, projections, npt, rhobeg, rhoend, nf, nx, maxfun,
                         params, scaling_changes, do_logging, h=h, lh=lh, argsh=argsh,  prox_uh=prox_uh, argsprox=argsprox)

    # Initialise interpolation set
    number_of_samples = max(nsamples(control.delta, control.rho, 0, nruns_so_far), 1)
    num_directions = min(params("growing.ndirs_initial") + params("restarts.hard.increase_ndirs_initial_amt") * nruns_so_far,
                         npt - 1)  # cap at npt
    if params("init.random_initial_directions"):
        if do_logging:
            module_logger.info("Initialising (random directions)")
        exit_info = control.initialise_random_directions(number_of_samples, num_directions, params)
    else:
        if do_logging:
            module_logger.info("Initialising (coordinate directions)")
        exit_info = control.initialise_coordinate_directions(number_of_samples, num_directions, params)
    if exit_info is not None:
        x, rvec, obj, jacmin, nsamples, x_eval_num, jac_eval_nums = control.model.get_final_results()
        return x, rvec, obj, None, nsamples, control.nf, control.nx, nruns_so_far + 1, exit_info, diagnostic_info, x_eval_num, jac_eval_nums

    finished_growing = (control.model.npt() >= control.model.num_pts)  # have we finished growing the initial set yet?

    # Save list of last N successful steps: whether they failed to be an improvement over fsave
    succ_steps_not_improvement = [False]*params("restarts.soft.max_fake_successful_steps")

    # Attempting to auto-detect restart? Need to keep a history of delta and ||chg J|| for non-safety iterations
    restart_auto_detect_full = False  # have we filled up the whole vectors yet? Don't restart from this if not
    if params("restarts.use_restarts") and params("restarts.auto_detect"):
        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
        restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))

    #------------------------------------------
    # Begin main loop
    # ------------------------------------------
    current_iter = -1
    if do_logging:
        module_logger.info("Beginning main loop")
    if print_progress:
        print("{:^5}{:^7}{:^10}{:^10}{:^10}{:^10}{:^7}".format("Run", "Iter", "Obj", "Grad", "Delta", "rho", "Evals"))
    while True:
        current_iter += 1

        if do_logging:
            module_logger.debug("*** Iter %g (delta = %g, rho = %g) ***" % (current_iter, control.delta, control.rho))

        if (not finished_growing) and control.model.npt() >= control.model.num_pts:
            if do_logging:
                module_logger.info("Finished growing init set")
            finished_growing = True

            if params("growing.reset_delta"):
                control.delta = rhobeg

            if params("growing.reset_rho"):
                control.rho = rhobeg

        if not finished_growing:
            if do_logging:
                module_logger.debug("Main loop: still growing (have %g of %g pts)" % (control.model.npt(), control.model.num_pts))

        # Noise level exit check
        if finished_growing and params("noise.quit_on_noise_level") and control.all_values_within_noise_level(params):
            if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                     x_in_abs_coords_to_save=None, rvec_to_save=None, nsamples_to_save=None)
                if exit_info is not None:
                    nruns_so_far += 1
                    break  # quit
                current_iter = -1
                nruns_so_far += 1
                rhoend = params("restarts.rhoend_scale") * rhoend
                restart_auto_detect_full = False
                restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                continue  # next iteration
            else:
                exit_info = ExitInformation(EXIT_SUCCESS, "All points within noise level")
                nruns_so_far += 1
                break  # quit

        # Interpolate mini-models
        interp_ok, interp_error, norm_J_error, linalg_resid, ls_interp_cond_num = \
            control.model.interpolate_mini_models_svd(verbose=params("logging.save_diagnostic_info"),
                make_full_rank=params("growing.full_rank.use_full_rank_interp") and not finished_growing,
                min_sing_val=params("growing.full_rank.min_sing_val"),
                sing_val_frac=params("growing.full_rank.svd_scale_factor"),
                max_jac_cond=params("growing.full_rank.svd_max_jac_cond"),
                get_chg_J=params("restarts.use_restarts") and params("restarts.auto_detect"),
                throw_error_on_nans=params("interpolation.throw_error_on_nans"))
        if not interp_ok:
            if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                 x_in_abs_coords_to_save=None, rvec_to_save=None, nsamples_to_save=None)
                if exit_info is not None:
                    nruns_so_far += 1
                    break  # quit
                current_iter = -1
                nruns_so_far += 1
                rhoend = params("restarts.rhoend_scale") * rhoend
                restart_auto_detect_full = False
                restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                continue  # next iteration
            else:
                exit_info = ExitInformation(EXIT_LINALG_ERROR, "Singular matrix in mini-model interpolation (main loop)")
                nruns_so_far += 1
                break  # quit

        tau = 1.0 # ratio used in the safety phase
        if h is None:
            # Trust region step
            d, gopt, H, gnew, crvmin = control.trust_region_step(params)
        else:
            # Calculate criticality measure
            criticality_measure = control.evaluate_criticality_measure(params)
            # Trust region step
            d, gopt, H, gnew, crvmin = control.trust_region_step(params, criticality_measure)
            try:
                tau = min(criticality_measure/(LA.norm(gopt)+lh), 1.0)
            except ValueError:
                # In some instances, gopt can have nan/inf values -- this ultimately calls a safety step and is generally fine
                # but we need to set a value for tau nonetheless
                tau = 1.0
        
        if do_logging:
            module_logger.debug("Trust region step is d = " + str(d))
        
        xnew = control.model.xopt() + d
        dnorm = min(LA.norm(d), control.delta)

        if print_progress:
            print("{:^5}{:^7}{:^10.2e}{:^10.2e}{:^10.2e}{:^10.2e}{:^7}".format(nruns_so_far+1, current_iter+1, control.model.objopt(), np.linalg.norm(gopt), control.delta, control.rho, control.nf))

        if params("logging.save_diagnostic_info"):
            diagnostic_info.save_info_from_control(control, nruns_so_far, current_iter,
                                                   save_poisedness=params("logging.save_poisedness"))
            # norm_J_error is square of Frobenius norm of chgJ
            diagnostic_info.update_interpolation_information(interp_error, ls_interp_cond_num, linalg_resid,
                                                             sqrt(norm_J_error), LA.norm(gopt), LA.norm(d))

        if dnorm < tau * params("general.safety_step_thresh") * control.rho and not finished_growing and params("growing.safety.do_safety_step"):
            if do_logging:
                module_logger.debug("Safety step during growing phase")

            if params("logging.save_diagnostic_info"):
                diagnostic_info.update_ratio(np.nan)
                diagnostic_info.update_iter_type(ITER_SAFETY)
                diagnostic_info.update_slow_iter(-1)

            # (start safety step while growing)
            if params("growing.safety.reduce_delta"):
                distsq = (10.0 * control.rho) ** 2
                sq_distances = control.model.distances_to_xopt()
                if np.max(sq_distances) > distsq:  # fix geometry and quit
                    distsq = np.max(sq_distances)
                    control.delta = max(min(0.1 * control.delta, 0.5 * sqrt(distsq)), 1.5 * control.rho)
                # Continue to below (i.e. add new direction(s))

            elif params("growing.safety.full_geom_step"):
                distsq = (10.0 * control.rho) ** 2
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                update_delta = True  # we do reduce delta for safety steps
                did_fix_geom, exit_info = control.check_and_fix_geometry(distsq, update_delta, number_of_samples, params)
                if dnorm > control.rho:
                    control.last_successful_iter = current_iter

                if exit_info is not None:
                    if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        nruns_so_far += 1
                        break  # quit

                # If we did fix geometry, continue to the below (add a new direction too).
                # To skip this, uncomment the below 2 lines
                # if did_fix_geom:
                    # continue  # next iteration

            # Default safety step behaviour (while growing) - add new direction(s)
            number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
            exit_info = control.add_new_direction_while_growing(number_of_samples, params, min_num_steps=1)
            if exit_info is not None:
                if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                        "restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    nruns_so_far += 1
                    break  # quit

            continue  # next iteration
            # (end safety step while growing)
        elif dnorm < params("general.safety_step_thresh") * control.rho and finished_growing:
            # (start safety step)
            if do_logging:
                module_logger.debug("Safety step (main phase)")

            if params("logging.save_diagnostic_info"):
                diagnostic_info.update_ratio(np.nan)
                diagnostic_info.update_iter_type(ITER_SAFETY)
                diagnostic_info.update_slow_iter(-1)

            if not control.done_with_current_rho(xnew, gnew, crvmin, H, current_iter):
                distsq = (10.0 * control.rho) ** 2
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                update_delta = True  # we do reduce delta for safety steps
                did_fix_geom, exit_info = control.check_and_fix_geometry(distsq, update_delta, number_of_samples, params)
                if dnorm > control.rho:
                    control.last_successful_iter = current_iter

                if exit_info is not None:
                    if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                            "restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        nruns_so_far += 1
                        break  # quit

                if did_fix_geom:
                    continue  # next iteration

            # If we are done with the current rho, or didn't fix geometry above, reduce rho
            if control.rho > rhoend:
                # Reduce rho
                control.reduce_rho(current_iter, params)
                if do_logging:
                    module_logger.info("New rho = %g after %i function evaluations" % (control.rho, control.nf))
                    if control.n() < params("logging.n_to_print_whole_x_vector"):
                        module_logger.debug("Best so far: f = %.15g at x = " % (control.model.objopt())
                                      + str(control.model.xopt(abs_coordinates=True)))
                    else:
                        module_logger.debug("Best so far: f = %.15g at x = [...]" % (control.model.objopt()))
                continue  # next iteration
            else:
                # Quit on rho=rhoend
                if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                         x_in_abs_coords_to_save=None, rvec_to_save=None, nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    # Cannot reduce rho, so check xnew and quit
                    x = control.model.as_absolute_coordinates(xnew)
                    ##print("x from xnew", x)
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    rvec_list, obj_list, num_samples_run, exit_info = control.evaluate_objective(x, number_of_samples,
                                                                                               params)
                    
                    if num_samples_run > 0:
                            control.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0),
                                                     num_samples_run, control.nx, x_in_abs_coords=True)
                    
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit

                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    nruns_so_far += 1
                    break  # quit
            # (end safety step)
        else:
            # (start trust region step)
            if do_logging:
                module_logger.debug("Standard trust region step")

            # If growing, optionally perturb the trust region step in a new direction
            if not finished_growing and params("growing.perturb_trust_region_step"):
                step_length = params('growing.delta_scale_new_dirns') * control.delta
                dirn = control.get_new_direction_for_growing(step_length)
                if do_logging:
                    module_logger.debug("Perturbing trust region with step = %s" % str(dirn))
                d += dirn
                xnew += dirn
                if do_logging:
                    module_logger.debug("New trust region step = %s" % str(d))

            # If finished growing, add chgJ and delta to restart auto-detect set
            if finished_growing and params("restarts.use_restarts") and params("restarts.auto_detect"):
                chg_J_frobenius = sqrt(norm_J_error)
                if restart_auto_detect_full:
                    # Drop first values, add new values at end
                    restart_auto_detect_delta = np.append(np.delete(restart_auto_detect_delta, [0]), control.delta)
                    restart_auto_detect_chgJ = np.append(np.delete(restart_auto_detect_chgJ, [0]), chg_J_frobenius)
                else:
                    idx = np.argmax(restart_auto_detect_delta < 0.0)  # index of first negative value
                    restart_auto_detect_delta[idx] = control.delta
                    restart_auto_detect_chgJ[idx] = chg_J_frobenius
                    restart_auto_detect_full = (
                    idx >= len(restart_auto_detect_delta) - 1)  # have we now got everything?

            if sumsq(d) <= params("general.rounding_error_constant") * sumsq(control.model.xopt()):
                base_shift = control.model.xopt()
                xnew = xnew - base_shift  # before xopt is updated
                control.model.shift_base(base_shift)

            knew, exit_info = control.choose_point_to_replace(d, skip_kopt=True)
            if exit_info is not None:
                if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                        "restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    nruns_so_far += 1
                    break  # quit

            # Evaluate new point
            x = control.model.as_absolute_coordinates(xnew)
            ##print("x from xnew again", x)
            number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
            rvec_list, obj_list, num_samples_run, exit_info = control.evaluate_objective(x, number_of_samples, params)
            if np.any(np.isnan(rvec_list)):
                # Just exit without saving the current point
                # We should be able to do a hard restart though, because it's unlikely 
                # that we will get the same trust-region step after expanding the radius/re-initialising
                module_logger.warning("NaN encountered in evaluation of trust-region step")
                if params("interpolation.throw_error_on_nans"):
                    raise np.linalg.LinAlgError("NaN encountered in objective evaluations")
                
                exit_info = ExitInformation(EXIT_EVAL_ERROR, "NaN received from objective function evaluation")
                nruns_so_far += 1
                break  # quit
            if exit_info is not None:
                if num_samples_run > 0:
                    control.model.save_point(x, np.mean(rvec_list[:num_samples_run, :], axis=0), num_samples_run, control.nx,
                                             x_in_abs_coords=True)
                nruns_so_far += 1
                break  # quit

            # Estimate f in order to compute 'actual reduction'
            ratio, exit_info = control.calculate_ratio(control.model.xopt(abs_coordinates=True), current_iter, rvec_list[:num_samples_run, :], d, gopt, H)
            if exit_info is not None:
                if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                        "restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    nruns_so_far += 1
                    break  # quit

            # Update delta
            if do_logging:
                module_logger.debug("Ratio = %g" % ratio)
            if params("logging.save_diagnostic_info"):
                diagnostic_info.update_ratio(ratio)
                diagnostic_info.update_slow_iter(-1)  # n/a, unless otherwise update
            if ratio < params("tr_radius.eta1"):  # ratio < 0.1
                if finished_growing:
                    control.delta = min(params("tr_radius.gamma_dec") * control.delta, dnorm) / tau
                else:
                    control.delta = min(params("growing.gamma_dec") * control.delta, dnorm) / tau  # different gamma_dec
                if params("logging.save_diagnostic_info"):
                    diagnostic_info.update_iter_type(ITER_ACCEPTABLE_NO_GEOM if ratio > 0.0
                                                     else ITER_UNSUCCESSFUL_NO_GEOM)  # we flag geom update below
            elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
                if finished_growing:
                    control.delta = max(params("tr_radius.gamma_dec") * control.delta, dnorm)
                else:
                    control.delta = max(params("growing.gamma_dec") * control.delta, dnorm)  # different gamma_dec
                if params("logging.save_diagnostic_info"):
                    diagnostic_info.update_iter_type(ITER_SUCCESSFUL)
            else:  # (ratio > eta2 = 0.7)
                control.delta = min(max(params("tr_radius.gamma_inc") * control.delta,
                                        params("tr_radius.gamma_inc_overline") * dnorm), 1.0e10)
                if params("logging.save_diagnostic_info"):
                    diagnostic_info.update_iter_type(ITER_VERY_SUCCESSFUL)
            if control.delta <= 1.5 * control.rho:  # cap trust region radius at rho
                control.delta = control.rho

            # Steps for successful steps
            if ratio > 0.0:
                # Re-select knew, allowing knew=kopt this time
                knew, exit_info = control.choose_point_to_replace(d, skip_kopt=False)
                if exit_info is not None:
                    if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                            "restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        nruns_so_far += 1
                        break  # quit

            # Update point
            if not finished_growing and params("growing.num_new_dirns_each_iter") == 0:
                # If still growing (but not adding geom steps; i.e. full rank growing), just add new point
                knew = control.model.npt()

            if do_logging:
                module_logger.debug("Updating with knew = %i" % knew)
            control.model.change_point(knew, xnew, rvec_list[0, :], control.nx)  # expect step, not absolute x
            for i in range(1, num_samples_run):
                control.model.add_new_sample(knew, rvec_extra=rvec_list[i, :])

            # Termination check: slow iterations [needs to be after updated with new point, as use model.fopt()
            if ratio > 0.0:
                this_iter_slow, should_terminate = control.terminate_from_slow_iterations(current_iter, params)
                if params("logging.save_diagnostic_info"):
                    diagnostic_info.update_slow_iter(1 if this_iter_slow else 0)
                if finished_growing and should_terminate:
                    if do_logging:
                        module_logger.info("Slow iteration  - terminating/restarting")
                    if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        exit_info = ExitInformation(EXIT_SLOW_WARNING, "Maximum slow iterations reached")
                        nruns_so_far += 1
                        break  # quit

                # Update list of successful steps
                this_step_was_not_improvement = control.model.objsave is not None and control.model.objopt() > control.model.objsave
                succ_steps_not_improvement.pop()  # remove last item
                succ_steps_not_improvement.insert(0, this_step_was_not_improvement)  # add at beginning
                # Terminate (not restart) if all are True
                if all(succ_steps_not_improvement):
                    exit_info = ExitInformation(EXIT_FALSE_SUCCESS_WARNING, "Maximum false successful steps reached")
                    nruns_so_far += 1
                    break  # quit

            # While growing, (optionally) add new directions
            if not finished_growing and params("growing.num_new_dirns_each_iter") > 0:
                if do_logging:
                    module_logger.debug("Still growing: adding %g new directions" % params("growing.num_new_dirns_each_iter"))
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                exit_info = control.add_new_direction_while_growing(number_of_samples, params)
                if exit_info is not None:
                    if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                            "restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        nruns_so_far += 1
                        break  # quit

            if not finished_growing and not params("growing.do_geom_steps"):
                continue  # next iteration

            # For regression, move more points than just the trust region step
            num_regression_steps = min(params("regression.num_extra_steps") + \
                                       nruns_so_far * params("regression.increase_num_extra_steps_with_restart"),
                                       control.model.npt() - 1)  # cap at number of points
            if finished_growing and ratio > 0.0 and num_regression_steps > 0:
                if do_logging:
                    module_logger.info("Regression: moving %g points" % num_regression_steps)
                number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                if params("regression.momentum_extra_steps"):  # move points as random extra steps
                    exit_info = control.move_furthest_points_momentum(d, number_of_samples, num_regression_steps, params)
                else:  # move points as geometry steps
                    exit_info = control.move_furthest_points(number_of_samples, num_regression_steps, params)

                if exit_info is not None:
                    if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                            "restarts.use_soft_restarts"):
                        number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                        exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                         x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                         nsamples_to_save=None)
                        if exit_info is not None:
                            nruns_so_far += 1
                            break  # quit
                        current_iter = -1
                        nruns_so_far += 1
                        rhoend = params("restarts.rhoend_scale") * rhoend
                        restart_auto_detect_full = False
                        restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                        continue  # next iteration
                    else:
                        nruns_so_far += 1
                        break  # quit

            if ratio >= params("tr_radius.eta1"):  # ratio >= 0.1
                continue  # next iteration

            # Auto-detection of restarts - check if we should do a restart
            if finished_growing and params("restarts.use_restarts") and params("restarts.auto_detect") \
                    and restart_auto_detect_full and ratio < 0.0:  # don't restart after a successful step
                do_restart = False
                iters_delta_flat = np.where(np.abs(restart_auto_detect_delta[1:]-restart_auto_detect_delta[:-1])<1e-15)[0]
                iters_delta_down = np.where(restart_auto_detect_delta[1:] - restart_auto_detect_delta[:-1] < -1e-15)[0]
                iters_delta_up = np.where(restart_auto_detect_delta[1:] - restart_auto_detect_delta[:-1] > 1e-15)[0]
                if len(iters_delta_up) == 0 and len(iters_delta_down) > 2*len(iters_delta_flat):
                    # no very successful iterations, and twice as many unsuccessful than moderately successful iterations

                    # If delta criteria met, check chgJ criteria
                    # Fit line to k vs. log(||chgJ||_F), but floor ||chgJ||_F away from zero
                    slope, intercept, r_value, p_value, std_err = STAT.linregress(np.arange(len(restart_auto_detect_chgJ)),
                                                                                np.log(np.maximum(restart_auto_detect_chgJ, 1e-15)))

                    if do_logging:
                        module_logger.debug("Iter %g: (slope, intercept, r_value) = (%g, %g, %g)" % (current_iter, slope, intercept, r_value))
                    if slope > params("restarts.auto_detect.min_chgJ_slope") \
                            and r_value > params("restarts.auto_detect.min_correl"):
                        # increasing trend, with at least some positive correlation
                        do_restart = True
                    else:
                        do_restart = False

                # Data available (full NumPy vectors of fixed length): restart_auto_detect_delta, restart_auto_detect_chgJ
                if do_restart and params("restarts.use_soft_restarts"):
                    if do_logging:
                        module_logger.info("Auto detection: need to do a restart")
                        module_logger.debug("delta history = %s" % str(restart_auto_detect_delta))
                        module_logger.debug("chgJ history = %s" % str(restart_auto_detect_chgJ))
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                elif do_restart:
                    if do_logging:
                        module_logger.info("Auto detection: need to do a restart")
                    exit_info = ExitInformation(EXIT_AUTO_DETECT_RESTART_WARNING, "Auto-detected restart")
                    nruns_so_far += 1
                    break  # quit
                    # If not doing restart, just continue as below (geom steps, etc.)

            # Otherwise (ratio < eta1 = 0.1), check & fix geometry
            if do_logging:
                module_logger.debug("Checking and possibly improving geometry (unsuccessful step)")
            distsq = max((2.0 * control.delta) ** 2, (10.0 * control.rho) ** 2)
            update_delta = False
            number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
            did_fix_geom, exit_info = control.check_and_fix_geometry(distsq, update_delta, number_of_samples, params)
            if dnorm > control.rho:
                control.last_successful_iter = current_iter

            if exit_info is not None:
                if exit_info.able_to_do_restart() and params("restarts.use_restarts") and params(
                        "restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, rvec_to_save=None,
                                                     nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    nruns_so_far += 1
                    break  # quit

            if did_fix_geom:
                if params("logging.save_diagnostic_info"):
                    diagnostic_info.update_iter_type(ITER_ACCEPTABLE_GEOM if ratio > 0.0 else ITER_UNSUCCESSFUL_GEOM)
                continue  # next iteration

            # If we didn't fix geometry but we still got an objective reduction (i.e. 0 < ratio < eta1 = 0.1), continue
            if ratio > 0.0:
                continue  # next iteration

            # Otherwise, ratio <= 0 (i.e. delta was reduced) and we didn't fix geometry - check if we need to reduce rho
            if max(control.delta, dnorm) > control.rho:
                continue  # next iteration
            elif control.rho > rhoend:
                # Reduce rho
                control.reduce_rho(current_iter, params)
                if do_logging:
                    module_logger.info("New rho = %g after %i function evaluations" % (control.rho, control.nf))
                    if control.n() < params("logging.n_to_print_whole_x_vector"):
                        module_logger.debug("Best so far: f = %.15g at x = " % (control.model.objopt())
                                      + str(control.model.xopt(abs_coordinates=True)))
                    else:
                        module_logger.debug("Best so far: f = %.15g at x = [...]" % (control.model.objopt()))
                continue  # next iteration
            else:
                # Quit on rho=rhoend
                if params("restarts.use_restarts") and params("restarts.use_soft_restarts"):
                    number_of_samples = max(nsamples(control.delta, control.rho, current_iter, nruns_so_far), 1)
                    exit_info = control.soft_restart(number_of_samples, nruns_so_far, params,
                                                     x_in_abs_coords_to_save=None, rvec_to_save=None, nsamples_to_save=None)
                    if exit_info is not None:
                        nruns_so_far += 1
                        break  # quit
                    current_iter = -1
                    nruns_so_far += 1
                    rhoend = params("restarts.rhoend_scale") * rhoend
                    restart_auto_detect_full = False
                    restart_auto_detect_delta = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    restart_auto_detect_chgJ = -1.0 * np.ones((params("restarts.auto_detect.history"),))
                    continue  # next iteration
                else:
                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    nruns_so_far += 1
                    break  # quit
                    # (end trust region step)
    # (end main loop)

    # Quit & return the important information
    x, rvec, obj, jacmin, nsamples, x_eval_num, jac_eval_nums = control.model.get_final_results()
    if do_logging:
        module_logger.debug("At return from DFO-LS, number of function evals = %i" % nf)
        module_logger.debug("Smallest objective value = %.15g at x = " % obj + str(x))
    return x, rvec, obj, jacmin, nsamples, control.nf, control.nx, nruns_so_far, exit_info, diagnostic_info, x_eval_num, jac_eval_nums


def solve(objfun, x0, h=None, lh=None, prox_uh=None, argsf=(), argsh=(), argsprox=(), bounds=None, projections=[], npt=None, rhobeg=None, rhoend=1e-8, maxfun=None, nsamples=None, user_params=None,
          objfun_has_noise=False, scaling_within_bounds=False, do_logging=True, print_progress=False):
    x0 = x0.astype(float)
    n = len(x0)

    # Set missing inputs (if not specified) to some sensible defaults
    if bounds is None:
        xl = None
        xu = None
    else:
        assert len(bounds) == 2, "bounds must be a 2-tuple of (lower, upper), where both are arrays of size(x0)"
        xl = bounds[0].astype(float) if bounds[0] is not None else None
        xu = bounds[1].astype(float) if bounds[1] is not None else None

    if (xl is None or xu is None) and scaling_within_bounds:
        scaling_within_bounds = False
        warnings.warn("Ignoring scaling_within_bounds=True for unconstrained problem/1-sided bounds", RuntimeWarning)

    if projections and scaling_within_bounds:
        scaling_within_bounds = False
        warnings.warn("Ignoring scaling_within_bounds=True for problem with arbitrary constraints", RuntimeWarning)

    if xl is None:
        xl = -1e20 * np.ones((n,))  # unconstrained
    if xu is None:
        xu = 1e20 * np.ones((n,))  # unconstrained
    if npt is None:
        npt = n + 1
    if rhobeg is None:
        rhobeg = 0.1 if scaling_within_bounds else 0.1 * max(np.max(np.abs(x0)), 1.0)
    if maxfun is None:
        maxfun = min(100 * (n + 1), 1000)  # 100 gradients, capped at 1000
    if nsamples is None:
        nsamples = lambda delta, rho, iter, nruns: 1  # no averaging

    # If using arbitrary constraints, create projection from bounds
    if projections:
        xlb = xl.copy()
        xub = xu.copy()
        bproj = lambda w: pbox(w,xlb,xub)
        projections = list(projections)
        projections.append(bproj)

        # since using arbitrary constraints, don't constrain otherwise
        xl = -1e20 * np.ones((n,))
        xu = 1e20 * np.ones((n,))

    # Set parameters
    params = ParameterList(int(n), int(npt), int(maxfun), objfun_has_noise=objfun_has_noise)  # make sure int, not np.int
    if user_params is not None:
        for (key, val) in user_params.items():
            params(key, new_value=val)
    
    # Default growing method depends on if m>=n or m<n - need to set this once we know m
    # But should only do this if the user hasn't forced a choice on us
    default_growing_method_set_by_user = user_params is not None and \
        ('growing.full_rank.use_full_rank_interp' in user_params or 'growing.perturb_trust_region_step' in user_params)

    scaling_changes = None
    if scaling_within_bounds:
        shift = xl.copy()
        scale = xu - xl
        scaling_changes = (shift, scale)

    x0 = apply_scaling(x0, scaling_changes)
    xl = apply_scaling(xl, scaling_changes)
    xu = apply_scaling(xu, scaling_changes)

    exit_info = None
    # Input & parameter checks
    if exit_info is None and h is not None:
        if prox_uh is None:
            exit_info = ExitInformation(EXIT_INPUT_ERROR, "Must provide prox_uh input if h is not None")
        elif lh is None:
            exit_info = ExitInformation(EXIT_INPUT_ERROR, "Must provide lh input if h is not None")
        elif lh <= 0.0:
            exit_info = ExitInformation(EXIT_INPUT_ERROR, "lh must be strictly positive")

    if exit_info is None and npt < n + 1:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "npt must be >= n+1 for linear models with inexact interpolation")

    if exit_info is None and rhobeg <= 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be strictly positive")

    if exit_info is None and rhoend <= 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhoend must be strictly positive")

    if exit_info is None and rhobeg <= rhoend:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be > rhoend")

    if exit_info is None and maxfun <= 0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "maxfun must be strictly positive")

    if exit_info is None and np.shape(x0) != (n,):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "x0 must be a vector")

    if exit_info is None and np.shape(x0) != np.shape(xl):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "lower bounds must have same shape as x0")

    if exit_info is None and np.shape(x0) != np.shape(xu):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "upper bounds must have same shape as x0")

    if exit_info is None and np.min(xu - xl) < 2.0 * rhobeg:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "gap between lower and upper must be at least 2*rhobeg")

    if maxfun <= npt:
        warnings.warn("maxfun <= npt: Are you sure your budget is large enough?", RuntimeWarning)

    # Check invalid parameter values

    all_ok, bad_keys = params.check_all_params(npt)
    if exit_info is None and not all_ok:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "Bad parameters: %s" % str(bad_keys))

    if exit_info is None and params("growing.safety.full_geom_step"):
        if params("growing.safety.reduce_delta"):
            exit_info = ExitInformation(EXIT_INPUT_ERROR,
                                        "Safety step while growing: either reduce delta -or- full geom step")

    if exit_info is None and params("growing.full_rank.use_full_rank_interp"):
        if params("growing.perturb_trust_region_step"):
            exit_info = ExitInformation(EXIT_INPUT_ERROR,
                                        "Growing: either make J full rank -or- perturb trust region step")

    # Ensure no doubling-up on noise estimates
    if exit_info is None and params("noise.quit_on_noise_level"):
        if params("noise.multiplicative_noise_level") is None:
            if params("noise.additive_noise_level") is None:
                params("noise.additive_noise_level", new_value=0.0)  # do not quit on noise level
        else:
            if params("noise.additive_noise_level") is not None:
                exit_info = ExitInformation(EXIT_INPUT_ERROR,
                                            "Must have exactly one of additive or multiplicative noise estimate")

    if exit_info is None and params("init.run_in_parallel") and not params("init.random_initial_directions"):
        exit_info = ExitInformation(EXIT_INPUT_ERROR,
                                    "Parallel initialisation not yet developed for coordinate initial directions")

    if exit_info is None and params("growing.reset_rho"):
        if not params("growing.reset_delta"):
            exit_info = ExitInformation(EXIT_INPUT_ERROR, "Growing: if resetting rho, must also reset delta")

    # If we had an input error, quit gracefully
    if exit_info is not None:
        exit_flag = exit_info.flag
        exit_msg = exit_info.message(with_stem=True)
        results = OptimResults(None, None, None, None, 0, 0, 0, exit_flag, exit_msg)
        return results

    # Enforce arbitrary constraint bounds on x0
    if projections:
        xp = dykstra(projections,x0,max_iter=params("dykstra.max_iters"),tol=params("dykstra.d_tol"))
        if not np.allclose(xp,x0):
            warnings.warn("x0 not feasible w.r.t given constraints, adjusting", RuntimeWarning)
            x0 = xp.copy()

    # Enforce lower & upper bounds on x0
    idx = (x0 < xl)
    if np.any(idx):
        warnings.warn("x0 below lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx]

    idx = (x0 > xu)
    if np.any(idx):
        warnings.warn("x0 above upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx]

    # Call main solver (first time)
    diagnostic_info = DiagnosticInfo()
    nruns = 0
    nf = 0
    nx = 0
    xmin, rmin, objmin, jacmin, nsamples_min, nf, nx, nruns, exit_info, diagnostic_info, xmin_eval_num, jacmin_eval_nums = \
        solve_main(objfun, x0, argsf, xl, xu, projections, npt, rhobeg, rhoend, maxfun, nruns, nf, nx, nsamples, params,
                    diagnostic_info, scaling_changes, h, lh, argsh, prox_uh, argsprox, default_growing_method_set_by_user=default_growing_method_set_by_user,
                   do_logging=do_logging, print_progress=print_progress)

    # Hard restarts loop
    last_successful_run = nruns
    while params("restarts.use_restarts") and not params("restarts.use_soft_restarts") and nf < maxfun and \
            exit_info.able_to_do_restart() and nruns - last_successful_run < params("restarts.max_unsuccessful_restarts"):
        rhoend = params("restarts.rhoend_scale") * rhoend
        
        if params("restarts.increase_npt"):
            npt += params("restarts.increase_npt_amt")
            npt = min(npt, params("restarts.max_npt"))

        if do_logging:
            module_logger.info("Restarting from finish point (f = %g) after %g function evals; using rhobeg = %g and rhoend = %g"
                     % (objmin, nf, rhobeg, rhoend))
        if params("restarts.hard.use_old_rk"):
            xmin2, rmin2, objmin2, jacmin2, nsamples2, nf, nx, nruns, exit_info, diagnostic_info, xmin_eval_num2, jacmin_eval_nums2 = \
                solve_main(objfun, xmin, argsf, xl, xu, projections, npt, rhobeg, rhoend, maxfun, nruns, nf, nx, nsamples, params,
                            diagnostic_info, scaling_changes, h, lh, argsh, prox_uh, argsprox, r0_avg_old=rmin, r0_nsamples_old=nsamples_min,
                           do_logging=do_logging, print_progress=print_progress)
        else:
            xmin2, rmin2, objmin2, jacmin2, nsamples2, nf, nx, nruns, exit_info, diagnostic_info, xmin_eval_num2, jacmin_eval_nums2 = \
                solve_main(objfun, xmin, argsf, xl, xu, projections, npt, rhobeg, rhoend, maxfun, nruns, nf, nx, nsamples, params,
                           diagnostic_info, scaling_changes, h, lh, argsh, prox_uh, argsprox, do_logging=do_logging, print_progress=print_progress)

        if objmin2 < objmin or np.isnan(objmin):
            if do_logging:
                module_logger.info("Successful run with new f = %s compared to old f = %s" % (objmin2, objmin))
            last_successful_run = nruns
            (xmin, rmin, objmin, nsamples_min, xmin_eval_num) = (xmin2, rmin2, objmin2, nsamples2, xmin_eval_num2)
            if jacmin2 is not None:  # may be None if finished during setup phase, in which case just use old Jacobian
                jacmin = jacmin2
                jacmin_eval_nums = jacmin_eval_nums2
        else:
            if do_logging:
                module_logger.info("Unsuccessful run with new f = %s compared to old f = %s" % (objmin2, objmin))

    if nruns - last_successful_run >= params("restarts.max_unsuccessful_restarts"):
        exit_info = ExitInformation(EXIT_SUCCESS, "Reached maximum number of unsuccessful restarts")

    # Process final return values & package up
    exit_flag = exit_info.flag
    exit_msg = exit_info.message(with_stem=True)
    # Un-scale Jacobian
    if scaling_changes is not None and jacmin is not None:
        for i in range(n):
            jacmin[:, i] = jacmin[:, i] / scaling_changes[1][i]
    results = OptimResults(remove_scaling(xmin, scaling_changes), rmin, objmin, jacmin, nf, nx, nruns, exit_flag, exit_msg, xmin_eval_num, jacmin_eval_nums)
    if params("logging.save_diagnostic_info"):
        df = diagnostic_info.to_dataframe(with_xk=params("logging.save_xk"), with_rk=params("logging.save_rk"))
        results.diagnostic_info = df

    if do_logging:
        module_logger.info("Did a total of %g run(s)" % nruns)

    return results

