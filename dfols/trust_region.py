"""
Trust Region Subproblem Solver
====

Specifically, the call
    d, gnew, crvmin = trsbox(xopt, g, H, sl, su, delta)
produces a new vector d which (approximately) solves the trust region subproblem:
    min_{d}  g'*d + 0.5*d'*H*d
    s.t.    ||d|| <= delta
            sl <= xopt + d <= su
The other outputs: gnew is the gradient of the model at d, and crvmin has
information about the curvature of the model at the solution.

For handling arbitrary constraints, the call is
    d, gnew, crvmin = ctrsbox(xopt, g, H, projections, delta)
which produces a new vector d approximately solving the constrained trust region subproblem:
    min_{d}  g'*d + 0.5*d'*H*d
    s.t.    ||d|| <= delta
            xopt + d is feasible w.r.t. the constraint set C
The other outputs: gnew is the gradient of the model at d, and crvmin has
information about the curvature of the model at the solution.

We also provide a function for maximising the absolute value of a linear function
inside a similar trust region - this is useful for geometry steps.
The call
    x = trsbox_geometry(xbase, c, g, lower, upper, delta)
solves
    min_x  abs(c + g' * (x - xbase))
    s.t.  lower <= x <= upper
          ||x-xbase|| <= Delta
With this value, the variable d=x-xbase solves the problem
    min_d  abs(c + g' * d)
    s.t.   lower <= xbase + d <= upper
          ||d|| <= delta
Again, we have a version of this for handling arbitrary constraints
The call
    x = ctrsbox_geometry(xbase, c, g, projections, Delta)
Solves
    min_d  abs(c + g' * d)
    s.t.   xbase + d is feasible w.r.t. the constraint set C
          ||d|| <= delta

Notes
----
The solver trsbox is an implementation of the routine TRSBOX from BOBYQA (Powell, 2009).
Some modifications to the termination conditions are from the equivalent routine
from DFBOLS (Zhang et al, 2010).


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

from math import sqrt, ceil
import numpy as np
try:
    import trustregion
    USE_FORTRAN = True
except ImportError:
    # Fall back to Python implementation
    USE_FORTRAN = False

from .util import dykstra, pball, pbox, sumsq, model_value, remove_scaling

__all__ = ['ctrsbox_sfista', 'ctrsbox_pgd', 'ctrsbox_geometry', 'trsbox', 'trsbox_geometry']

ZERO_THRESH = 1e-14

def ctrsbox_sfista(xopt, g, H, projections, delta, h, L_h, prox_uh, argsh=(), argsprox=(), func_tol=1e-3, max_iters=500, d_max_iters=100, d_tol=1e-10, use_fortran=USE_FORTRAN, scaling_changes=None, sfista_iters_scale=1.0):
    n = xopt.size
    assert xopt.shape == (n,), "xopt has wrong shape (should be vector)"
    assert g.shape == (n,), "g and xopt have incompatible sizes"
    assert len(H.shape) == 2, "H must be a matrix"
    assert H.shape == (n,n), "H and xopt have incompatible sizes"
    assert np.allclose(H, H.T), "H must be symmetric"
    assert delta > 0.0, "delta must be strictly positive"

    # Initialization
    d = np.zeros(n) # start with zero vector
    y = np.zeros(n)
    t = 1
    k_H = np.linalg.norm(H, 2)
    crvmin = -1.0
    
    # Number of iterations & smoothing parameter, from Theorem 10.57 in 
    #   [A. Beck. First-order methods in optimization, SIAM, 2017]
    # We do not use the values of k and mu given in the theorem statement, but rather the intermediate
    # results on p313 (K1 for number of iterations, and the immediate next line for mu)
    # Note: in the book's notation, Gamma=delta^2, alpha=1, beta=L_h^2/2, Lf=k_H [alpha and beta from Thm 10.51]
    try:
        MAX_LOOP_ITERS = ceil(sfista_iters_scale * delta * (L_h+sqrt(L_h*L_h+2*k_H*func_tol)) / func_tol)
        MAX_LOOP_ITERS = min(MAX_LOOP_ITERS, max_iters)
    except ValueError:
        MAX_LOOP_ITERS = max_iters
    u =  2 * delta / (MAX_LOOP_ITERS * L_h) # smoothing parameter
    # u = 2 * func_tol / (L_h ** 2 + L_h * sqrt(L_h ** 2 + 2 * k_H * func_tol))  # the above choice works better in practice

    def gradient_Fu(xopt, g, H, u, prox_uh, d):
    # Calculate gradient_Fu,
    # where Fu(d) := g(d) + h_u(d) and h_u(d) is a 1/u-smooth approximation of h.
    # We assume that h is globally Lipschitz continous with constant L_h,
    # then we can let h_u(d) be the Moreau Envelope M_h_u(d) of h.
        return g + H @ d + (xopt + d - prox_uh(remove_scaling(xopt + d, scaling_changes), u, *argsprox)) / u

    # Lipschitz constant of gradient_Fu
    l = k_H + 1 / u 

    # trust region is a ball of radius delta around xopt
    trproj = lambda w: pball(w, xopt, delta)

    # combine trust region constraints with user-entered constraints
    P = list(projections)  # make a copy of the projections list
    P.append(trproj)
    def proj(d0):
        p = dykstra(P, xopt+d0, max_iter=d_max_iters, tol=d_tol)
        # we want the step only, so we subtract xopt
        # from the new point: proj(xk+d) - xk
        return p - xopt

    # general step
    model_value_best = model_value(g, H, d, xopt, h, argsh, scaling_changes)
    d_best = d.copy()
    for k in range(MAX_LOOP_ITERS):
        prev_d = d.copy()
        prev_t = t
        # gradient_Fu at y
        g_Fu = gradient_Fu(xopt, g, H, u, prox_uh, d, *argsprox)

        # main update step
        d = proj(y - g_Fu / l)
        new_model_value = model_value(g, H, d, xopt, h, argsh, scaling_changes)
        if new_model_value < model_value_best:
            d_best = d.copy()
            model_value_best = new_model_value

        # update true gradient
        # gnew is the gradient of the smoothed function
        gnew = gradient_Fu(xopt, g, H, u, prox_uh, d, *argsprox)

        # update CRVMIN
        crv = d.dot(H).dot(d)/sumsq(d) if sumsq(d) >= ZERO_THRESH else crvmin
        crvmin = min(crvmin, crv) if crvmin != -1.0 else crv
        
        # momentum update
        t = (1 + sqrt(1 + 4*t*t)) / 2
        y = d + (prev_t - 1) * (d - prev_d) / t
    return d, gnew, crvmin

def ctrsbox_pgd(xopt, g, H, projections, delta, d_max_iters=100, d_tol=1e-10, use_fortran=USE_FORTRAN):
    n = xopt.size
    assert xopt.shape == (n,), "xopt has wrong shape (should be vector)"
    assert g.shape == (n,), "g and xopt have incompatible sizes"
    assert len(H.shape) == 2, "H must be a matrix"
    assert H.shape == (n,n), "H and xopt have incompatible sizes"
    assert np.allclose(H, H.T), "H must be symmetric"
    assert delta > 0.0, "delta must be strictly positive"

    d = np.zeros((n,))
    gnew = g.copy()
    gy = g.copy()
    crvmin = -1.0
    y = d.copy()
    eta = 1.2 # L backtrack scaling factor
    t = 1

    # Initial guess of L is norm(Hessian)
    L = np.linalg.norm(H, 2)

    # trust region is a ball of radius delta around xopt
    trproj = lambda w: pball(w, xopt, delta)

    # combine trust region constraints with user-entered constraints
    P = list(projections)  # make a copy of the projections list
    P.append(trproj)
    def proj(d0):
        p = dykstra(P, xopt+d0, max_iter=d_max_iters, tol=d_tol)
        # we want the step only, so we subtract xopt
        # from the new point: proj(xk+d) - xk
        return p - xopt

    MAX_LOOP_ITERS = 100 * n ** 2

    # projected GD loop 
    for ii in range(MAX_LOOP_ITERS):
        w = y - (1/L)*gy
        prev_d = d.copy()
        d = proj(w)

        # size of step taken
        s = d - prev_d
        stplen = np.linalg.norm(s)

        # update true gradient
        gnew += H.dot(s)

        # update CRVMIN
        crv = s.dot(H).dot(s)/sumsq(s) if sumsq(s) >= ZERO_THRESH else crvmin
        crvmin = min(crvmin, crv) if crvmin != -1.0 else crv

        # exit condition
        if stplen <= ZERO_THRESH:
            break

        # momentum update
        prev_t = t
        t = (1 + np.sqrt(1 + 4 * t ** 2))/2
        prev_y = y.copy()
        y = d + s*(prev_t - 1)/t

        # update gradient w.r.t y
        gy += H.dot(y - prev_y)

    return d, gnew, crvmin

def trsbox(xopt, g, H, sl, su, delta, use_fortran=USE_FORTRAN):
    if use_fortran:
        return trustregion.solve(g, H, delta,
                                 sl=np.minimum(sl - xopt, -ZERO_THRESH),
                                 su=np.maximum(su - xopt, ZERO_THRESH),
                                 verbose_output=True)

    n = xopt.size
    assert xopt.shape == (n,), "xopt has wrong shape (should be vector)"
    assert g.shape == (n,), "g and xopt have incompatible sizes"
    assert len(H.shape) == 2, "H must be a matrix"
    assert H.shape == (n,n), "H and xopt have incompatible sizes"
    assert np.allclose(H, H.T), "H must be symmetric"
    assert sl.shape == (n,), "sl and xopt have incompatible sizes"
    assert su.shape == (n,), "su and xopt have incompatible sizes"
    assert np.all(sl <= xopt), "xopt violates lower bound sl"
    assert np.all(xopt <= su), "xopt violates upper bound su"
    assert delta > 0.0, "delta must be strictly positive"
    # Assume g and H have full quadratic model for objective
    # i.e. skip straight to label 8 in DFBOLS version

    # The sign of G(I) gives the sign of the change to the I-th variable
    # that will reduce Q from its value at XOPT. Thus XBDI(I) shows whether
    # or not to fix the I-th variable at one of its bounds initially, with
    # NACT being set to the number of fixed variables. D and GNEW are also
    # set for the first iteration. DELSQ is the upper bound on the sum of
    # squares of the free variables. QRED is the reduction in Q so far.

    iterc = 0
    nact = 0  # number of fixed variables

    xbdi = np.zeros((n,), dtype=int)  # fix x_i at bounds? [values -1, 0, 1]
    xbdi[(xopt <= sl) & (g >= 0.0)] = -1
    xbdi[(xopt >= su) & (g <= 0.0)] = 1

    d = np.zeros((n,))
    s = np.zeros((n,))
    gnew = g.copy()
    qred = 0.0
    delsq = delta ** 2
    crvmin = -1.0
    beta = 0.0  # label 20

    need_alt_trust_step = False  # will either quit main CG loop to finish, or do alternative step
    MAX_LOOP_ITERS = 100 * n ** 2  # avoid infinite loops
    # while True:  # main CG loop [label 30]
    for ii in range(MAX_LOOP_ITERS):
        s[xbdi != 0] = 0.0
        if beta == 0.0:
            s[xbdi == 0] = -gnew[xbdi == 0]
        else:
            s[xbdi == 0] = beta * s[xbdi == 0] - gnew[xbdi == 0]
        stepsq = sumsq(s)

        if stepsq == 0.0:
            need_alt_trust_step = False
            break  # break and quit

        if beta == 0.0:
            gredsq = stepsq
            itermax = iterc + n - nact

        if iterc == 0:
            gredsq0 = gredsq

        # Exit conditions
        if gredsq <= min(1.0e-6 * gredsq0, 1.0e-18) or gredsq * delsq <= min(1.0e-6 * qred ** 2, 1.0e-18):  # DFBOLS
            need_alt_trust_step = False
            break  # break and quit

        # Multiply the search direction by the second derivative matrix of Q and
        # calculate some scalars for the choice of steplength. Then set BLEN to
        # the length of the the step to the trust region boundary and STPLEN to
        # the steplength, ignoring the simple bounds.

        hs = H.dot(s)

        # label 50
        ds = np.dot(s[xbdi == 0], d[xbdi == 0])
        shs = np.dot(s[xbdi == 0], hs[xbdi == 0])
        resid = delsq - sumsq(d[xbdi == 0])
        if resid <= 0.0:
            need_alt_trust_step = True
            break  # break and calculate alt step instead

        temp = sqrt(stepsq * resid + ds ** 2)
        blen = (resid / (temp + ds) if ds >= 0.0 else (temp - ds) / stepsq)
        stplen = (blen if shs <= 0.0 else min(blen, gredsq / shs))

        # Exit condition
        if stplen <= 1.0e-30:  # DFBOLS
            need_alt_trust_step = False
            break  # break and quit

        # Reduce STPLEN if necessary in order to preserve the simple bounds,
        # letting IACT be the index of the new constrained variable.
        iact = None
        for i in range(n):
            if s[i] != 0.0:
                temp = (su[i] - xopt[i] - d[i] if s[i] > 0.0 else sl[i] - xopt[i] - d[i]) / s[i]
                if temp < stplen:
                    stplen = temp
                    iact = i

        # Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q.
        sdec = 0.0
        if stplen > 0.0:
            iterc += 1
            temp = shs / stepsq
            if iact is None and temp > 0.0:
                crvmin = min(crvmin, temp) if crvmin != -1.0 else temp
            ggsav = gredsq
            gnew += stplen * hs
            d += stplen * s
            gredsq = sumsq(gnew[xbdi == 0])
            sdec = max(stplen * (ggsav - 0.5 * stplen * shs), 0.0)
            qred += sdec

        # Restart the conjugate gradient method if it has hit a new bound.
        if iact is not None:
            nact += 1
            xbdi[iact] = (1 if s[iact] >= 0.0 else -1)
            delsq = delsq - d[iact] ** 2
            if delsq <= 0.0:
                need_alt_trust_step = True
                break  # break and calculate alt step instead
            beta = 0.0  # label 20
            continue  # restart loop (new CG iteration)

        # If STPLEN is less than BLEN, then either apply another conjugate
        # gradient iteration or RETURN.
        if stplen >= blen:
            need_alt_trust_step = True
            break  # break and calculate alt step instead

        # Exit condition
        if iterc == itermax or sdec <= 1.0e-6 * qred:  # DFBOLS
            need_alt_trust_step = False
            break  # break and quit

        beta = gredsq / ggsav
        continue  # new CG iteration
    # end of CG loop

    # either done or need to take and alternative step
    if need_alt_trust_step:
        crvmin = 0.0
        d, gnew = alt_trust_step(n, xopt, H, sl, su, d, xbdi, nact, gnew, qred)
        return d, gnew, crvmin
    else:
        return d_within_bounds(d, xopt, sl, su, xbdi), gnew, crvmin


# Alternative Trust Region Step (label 100 of TRSBOX in BOBYQA, where crvmin=0)
def alt_trust_step(n, xopt, H, sl, su, d, xbdi, nact, gnew, qred):
    MAX_LOOP_ITERS = 100 * n ** 2  # avoid infinite loops
    # while True:  # label 100 here
    for ii in range(MAX_LOOP_ITERS):
        if nact >= n - 1:
            return d_within_bounds(d, xopt, sl, su, xbdi), gnew

        # Prepare for the alternative iteration by calculating some scalars
        # and by multiplying the reduced D by the second derivative matrix of
        # Q, where S holds the reduced D in the call of GGMULT.
        s = np.zeros((n,))
        s[xbdi == 0] = d[xbdi == 0]
        dredsq = sumsq(d[xbdi == 0])
        dredg = np.dot(d[xbdi == 0], gnew[xbdi == 0])
        gredsq = sumsq(gnew[xbdi == 0])

        # Label 210 (crvmin = 0, itcsav = iterc)
        hs = H.dot(s)

        hred = hs.copy()
        # quit 210 by goto 120

        # Let the search direction S be a linear combination of the reduced D
        # and the reduced G that is orthogonal to the reduced D.
        restart_alt_loop = False  # once the below loop finishes, quit unless need to go again
        # while True:  # label 120
        for jj in range(MAX_LOOP_ITERS):
            temp = gredsq * dredsq - dredg ** 2
            if temp <= 1.0e-4 * qred ** 2:
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results
            temp = sqrt(temp)
            s = np.zeros((n,))
            s[xbdi == 0] = (dredg * d[xbdi == 0] - dredsq * gnew[xbdi == 0]) / temp
            sredg = -temp

            # By considering the simple bounds on the variables, calculate an upper
            # bound on the tangent of half the angle of the alternative iteration,
            # namely ANGBD, except that, if already a free variable has reached a
            # bound, there is a branch back to label 100 after fixing that variable.
            free_variable_reached_bound = False
            angbd = 1.0
            iact = None
            for i in range(n):
                if xbdi[i] == 0:
                    tempa = xopt[i] + d[i] - sl[i]
                    tempb = su[i] - xopt[i] - d[i]
                    if tempa <= 0.0:
                        nact += 1
                        xbdi[i] = -1
                        free_variable_reached_bound = True
                        break  # skip the rest of this for loop
                    elif tempb <= 0.0:
                        nact += 1
                        xbdi[i] = 1
                        free_variable_reached_bound = True
                        break  # skip the rest of this for loop
                    ssq = d[i] ** 2 + s[i] ** 2
                    temp = ssq - (xopt[i] - sl[i]) ** 2
                    if temp > 0.0:
                        temp = sqrt(temp) - s[i]
                        if angbd * temp > tempa:
                            angbd = tempa / temp
                            iact = i
                            xsav = -1
                    temp = ssq - (su[i] - xopt[i]) ** 2
                    if temp > 0.0:
                        temp = sqrt(temp) + s[i]
                        if angbd * temp > tempb:
                            angbd = tempb / temp
                            iact = i
                            xsav = 1
            # End for loop
            if free_variable_reached_bound:  # deal with break conditions above
                restart_alt_loop = True
                break  # quit inner label 120 loop and restart alt iteration loop (label 100)

            # Label 210 (crvmin = 0, itcsav < iterc since iterc+=1 earlier)
            hs = H.dot(s)

            # Label 150
            # Calculate HHD and some curvatures for the alternative iteration.
            shs = np.sum(s[xbdi == 0] * hs[xbdi == 0])
            dhs = np.sum(d[xbdi == 0] * hs[xbdi == 0])
            dhd = np.sum(d[xbdi == 0] * hred[xbdi == 0])

            # Seek the greatest reduction in Q for a range of equally spaced values
            # of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle of
            # the alternative iteration.
            redmax = 0.0
            isav = -1
            redsav = 0.0
            temp = 0.0  # force scope outside i loop below since needed later
            iu = int(17 * angbd + 3.1)
            for i in range(iu):  # i = 0, ..., iu-1
                angt = angbd * float(i + 1) / float(iu)
                sth = 2.0 * angt / (1.0 + angt ** 2)
                temp = shs + angt * (angt * dhd - 2.0 * dhs)
                rednew = sth * (angt * dredg - sredg - 0.5 * sth * temp)
                if rednew > redmax:
                    redmax = rednew
                    isav = i
                    rdprev = redsav
                elif i == isav + 1:
                    rdnext = rednew
                redsav = rednew

            # Return if the reduction is zero. Otherwise, set the sine and cosine
            # of the angle of the alternative iteration, and calculate SDEC.
            if isav == -1:
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results

            if isav < iu - 1:
                temp = (rdnext - rdprev) / (2.0 * redmax - rdprev - rdnext)
                angt = angbd * (float(isav + 1) + 0.5 * temp) / float(iu)

            cth = (1.0 - angt ** 2) / (1.0 + angt ** 2)
            sth = 2.0 * angt / (1.0 + angt ** 2)
            temp = shs + angt * (angt * dhd - 2.0 * dhs)
            sdec = sth * (angt * dredg - sredg - 0.5 * sth * temp)

            if sdec <= 0.0:
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results

            # Update GNEW, D and HRED. If the angle of the alternative iteration
            # is restricted by a bound on a free variable, that variable is fixed
            # at the bound.
            gnew += (cth - 1.0) * hred + sth * hs
            d[xbdi == 0] = cth * d[xbdi == 0] + sth * s[xbdi == 0]
            dredg = np.dot(d[xbdi == 0], gnew[xbdi == 0])
            gredsq = sumsq(gnew[xbdi == 0])
            hred = cth * hred + sth * hs

            qred += sdec
            if iact is not None and isav == iu - 1:
                nact += 1
                xbdi[iact] = xsav
                restart_alt_loop = True
                break  # quit inner label 120 loop and restart alt iteration loop (label 100)

            if (sdec <= 0.01 * qred):
                restart_alt_loop = False
                break  # quit inner label 120 loop and return results
            continue  # back to inner label 120 loop

        # End inner label 120 loop

        if restart_alt_loop:
            continue
        else:
            break  # end outer loop and quit

    # End while True (label 100)
    return d_within_bounds(d, xopt, sl, su, xbdi), gnew


def d_within_bounds(d, xopt, sl, su, xbdi):
    # Used in TRSBOX, force d to be within bounds
    # In Fortran code, is at label 190
    xnew = np.maximum(np.minimum(xopt + d, su), sl)
    xnew[xbdi == -1] = sl[xbdi == -1]
    xnew[xbdi == 1] = su[xbdi == 1]
    d = xnew - xopt
    return d


def ball_step(x0, g, Delta):
    # Given initial point x0, take largest step in direction g allowed by ||x|| <= Delta
    # That is, solve
    #   ||x0 + alpha*g||^2 = Delta^2, alpha >= 0
    # Using this method, solution exists whenever ||x0|| <= Delta^2 [take alpha=0 if g=0]
    gdotx0 = np.dot(g, x0)
    gsqnorm = np.dot(g, g)
    x0sqnorm = np.dot(x0, x0)
    if sqrt(gsqnorm) < ZERO_THRESH:  # Error catching: if g=0, make no step
        return 0.0
    else:
        # Sqrt had negative input on prob 46 in OG DFOLS with noise
        #  print("Inside of the sqrt:", gdotx0**2 + gsqnorm*(Delta**2 - x0sqnorm))
        # Got Inside of the sqrt: -3.608971127647144e-42
        # Added max(0,...) here
        return (sqrt(np.maximum(0,gdotx0**2 + gsqnorm*(Delta**2 - x0sqnorm))) - gdotx0) / gsqnorm

def ctrsbox_linear(xbase, g, projections, Delta, d_max_iters=100, d_tol=1e-10, use_fortran=USE_FORTRAN):
    # Solve the convex program:
    #   min_d   g' * d
    #   s.t.    xbase + d is feasible w.r.t. constraint set C
    #           ||d||^2 <= Delta^2

    n = g.size
    d = np.zeros((n,))
    y = d.copy()
    t = 1
    dirn = -g
    cons_dirns = []

    # If g[i] = 0, never step along this direction
    constant_directions = np.where(np.abs(dirn) < ZERO_THRESH)[0]
    dirn[constant_directions] = 0.0

    # trust region is a ball of radius delta centered around xbase
    trproj = lambda w: pball(w, xbase, Delta)

    # combine trust region constraints with user-entered constraints
    P = list(projections)  # make a copy of the projections list
    P.append(trproj)
    def proj(d0):
        p = dykstra(P, xbase + d0, max_iter=d_max_iters, tol=d_tol)
        # we want the step only, so we subtract
        # xbase from the new point: proj(xk + d) - xk
        return p - xbase

    MAX_LOOP_ITERS = 100 * n ** 2

    # projected GD loop 
    for ii in range(MAX_LOOP_ITERS):
        w = y + dirn
        prev_d = d.copy()
        d = proj(w)

        s = d - prev_d
        stplen = np.linalg.norm(s)

        # exit condition
        if stplen <= ZERO_THRESH:
            break

        # 'momentum' update
        prev_t = t
        t = (1 + np.sqrt(1 + 4 * t ** 2))/2
        prev_y = y.copy()
        y = d + s*(prev_t - 1)/t

    return d

def trsbox_linear(g, a_in, b_in, Delta, use_fortran=USE_FORTRAN):
    # Solve the convex program:
    #   min_x   g' * x
    #   s.t.   a <= x <= b
    #           ||x||^2 <= Delta^2
    # using an active-set type approach
    a = np.minimum(a_in, -ZERO_THRESH)
    b = np.maximum(b_in, ZERO_THRESH)
    if use_fortran:
        return trustregion.solve(g, None, Delta,
                                 sl=a,
                                 su=b,
                                 verbose_output=False)

    n = g.size
    x = np.zeros((n,))
    dirn = -g
    cons_dirns = []

    # If g[i] = 0, never step along this direction
    constant_directions = np.where(np.abs(dirn) < ZERO_THRESH)[0]
    dirn[constant_directions] = 0.0
    cons_dirns += list(constant_directions)

    for i in range(n):
        if np.linalg.norm(dirn) < ZERO_THRESH:
            return x
        alpha_unc = ball_step(x, dirn, Delta)
        xnew = x + alpha_unc * dirn
        # Check if hit box bounds
        on_box_bdry = False
        hit_upper = None
        idx_hit = None
        for j in range(n):
            if j in cons_dirns:
                continue  # only looking at unconstrained directions
            if xnew[j] <= a[j]:
                on_box_bdry = True
                hit_upper = False
                idx_hit = j
                break
            elif xnew[j] >= b[j]:
                on_box_bdry = True
                hit_upper = True
                idx_hit = j
                break

        if not on_box_bdry:
            return xnew  # unconstrained solution
        else:
            # Go as far as possible until hit box, then remove that direction from 'dirn'
            cons_dirns.append(idx_hit)  # new constrained direction
            alpha_con = ((b[idx_hit] if hit_upper else a[idx_hit]) - x[idx_hit]) / dirn[idx_hit]
            x = x + alpha_con * dirn
            x[idx_hit] = b[idx_hit] if hit_upper else a[idx_hit]  # force boundary exactly
            dirn[idx_hit] = 0.0  # no more searching this direction
    return x

def ctrsbox_geometry(xbase, c, g, projections, Delta, d_max_iters=100, d_tol=1e-10, use_fortran=USE_FORTRAN):
    # Given a Lagrange polynomial defined by: L(x) = c + g' * (x - xbase)
    # Maximise |L(x)| in a box + trust region - that is, solve:
    #   max_x  abs(c + g' * (x - xbase))
    #    s.t.  x is feasible w.r.t constraint set C
    #          ||x-xbase|| <= Delta
    # Setting s = x-xbase (or x = xbase + s), this is equivalent to:
    #   max_s  abs(c + g' * s)
    #   s.t.   xbase + s is is feasible w.r.t constraint set C
    #          ||s|| <= Delta
    smin = ctrsbox_linear(xbase, g, projections, Delta, d_max_iters=100, d_tol=1e-10, use_fortran=use_fortran)  # minimise g' * s
    smax = ctrsbox_linear(xbase, -g, projections, Delta, d_max_iters=100, d_tol=1e-10, use_fortran=use_fortran)  # maximise g' * s
    if abs(c + np.dot(g, smin)) >= abs(c + np.dot(g, smax)):  # choose the one with largest absolute value
        return smin
    else:
        return smax

def trsbox_geometry(xbase, c, g, lower, upper, Delta, use_fortran=USE_FORTRAN):
    # Given a Lagrange polynomial defined by: L(x) = c + g' * (x - xbase)
    # Maximise |L(x)| in a box + trust region - that is, solve:
    #   max_x  abs(c + g' * (x - xbase))
    #    s.t.  lower <= x <= upper
    #          ||x-xbase|| <= Delta
    # Setting s = x-xbase (or x = xbase + s), this is equivalent to:
    #   max_s  abs(c + g' * s)
    #   s.t.   lower - xbase <= s <= upper - xbase
    #          ||s|| <= Delta
    assert np.all(lower <= xbase + ZERO_THRESH), "xbase violates lower bound"
    assert np.all(xbase - ZERO_THRESH <= upper), "xbase violates upper bound"
    smin = trsbox_linear(g, lower - xbase, upper - xbase, Delta, use_fortran=use_fortran)  # minimise g' * s
    smax = trsbox_linear(-g, lower - xbase, upper - xbase, Delta, use_fortran=use_fortran)  # maximise g' * s
    if abs(c + np.dot(g, smin)) >= abs(c + np.dot(g, smax)):  # choose the one with largest absolute value
        return xbase + smin
    else:
        return xbase + smax
