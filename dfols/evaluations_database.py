"""
Class to create/store database of existing evaluations, and routines to select
existing evaluations to build an initial linear model
"""
import logging
import numpy as np

from .util import apply_scaling, dykstra
from .trust_region import ctrsbox_geometry, trsbox_geometry

__all__ = ['EvaluationDatabase']

module_logger = logging.getLogger(__name__)


# Class to store set of evaluations (x, rx)
class EvaluationDatabase(object):
    def __init__(self, eval_list=None, starting_eval=None):
        # eval_list is a list of tuples (x, rx)
        self._evals = []
        if eval_list is not None:
            for e in eval_list:
                self._evals.append(e)

        # Which evaluation index should be the starting point of the optimization?
        self.starting_eval = None
        if starting_eval is not None and 0 <= starting_eval <= len(self._evals):
            self.starting_eval = starting_eval

        # Scaling changes
        self.scaling_changes = None

    def __len__(self):
        return len(self._evals)

    def append(self, x, rx, make_starting_eval=False):
        self._evals.append((x, rx))
        if make_starting_eval:
            self.starting_eval = len(self) - 1

    def set_starting_eval(self, index):
        if 0 <= index < len(self):
            self.starting_eval = index
        else:
            raise IndexError("Invalid index %g given current set of %g evaluations" % (index, len(self)))

    def get_starting_eval_idx(self):
        if len(self) == 0:
            raise RuntimeError("No evaluations available, no suitable starting evaluation ")
        elif self.starting_eval is None:
            module_logger.warning("Starting evaluation index not set, using most recently appended evaluation")
            self.starting_eval = len(self) - 1

        return self.starting_eval

    def get_eval(self, index, scaled_x=False):
        # Return (x, rx) for given index
        if 0 <= index < len(self):
            x = self._evals[index][0]
            rx = self._evals[index][1]
            if scaled_x and self.scaling_changes is not None:
                return apply_scaling(x, self.scaling_changes), rx
            else:
                return x, rx
        else:
            raise IndexError("Invalid index %g given current set of %g evaluations" % (index, len(self)))

    def get_x(self, index, scaled_x=False):
        return self.get_eval(index, scaled_x=scaled_x)[0]

    def get_rx(self, index):
        return self.get_eval(index)[1]

    def apply_scaling(self, scaling_changes):
        # Save scaling information for on-the-fly use
        # (don't change self._evals internally as prevents the database being re-used)
        self.scaling_changes = scaling_changes
        return

    def select_starting_evals(self, delta, xl=None, xu=None, projections=[], tol=1e-8,
                              dykstra_max_iters=100, dykstra_tol=1e-10):
        # Given a database 'evals' with prescribed starting index, and initial trust-region radius delta > 0
        # determine a subset of the database to use

        # The bounds xl <= x <= xu and projection list are used to determine where to evaluate any new points
        # (ensuring they are feasible)

        if delta <= 0.0:
            raise RuntimeError("delta must be strictly positive")
        if len(self) == 0:
            raise RuntimeError("Need at least one evaluation to select starting evaluations")

        base_idx = self.get_starting_eval_idx()
        xbase = self.get_x(self.get_starting_eval_idx(), scaled_x=True)
        n = len(xbase)
        module_logger.debug("Selecting starting evaluations from existing database")
        module_logger.debug("Have %g evaluations to choose from" % len(self))
        module_logger.debug("Using base index %g" % base_idx)

        # For linear interpolation, we will use the matrix
        # M = [[1, 0], [0, L]] where L has rows (xi-xbase)/delta
        # So, just build a large matrix Lfull with everything
        n_perturbations = len(self) - 1
        Lfull = np.zeros((n_perturbations, n))
        row_idx = 0
        for i in range(n_perturbations + 1):
            if i == base_idx:
                continue
            Lfull[row_idx, :] = (self.get_x(i, scaled_x=True) - xbase) / delta  # Lfull[i,:] = (xi-xbase) / delta
            row_idx += 1

        xdist = np.linalg.norm(Lfull, axis=1)  # xdist[i] = ||Lfull[i,:]|| = ||xi-xbase|| / delta
        # module_logger.debug("xdist =", xdist)

        # We ideally want xdist ~ 1, so reweight these distances based on that (large xdist_reweighted --> xdist ~ 1 --> good)
        xdist_reweighted = 1.0 / np.maximum(xdist, 1.0 / xdist)
        # module_logger.debug("xdist_reweighted =", xdist_reweighted)

        if n_perturbations == 0:
            module_logger.debug("Only one evaluation available, just selecting that")
            return base_idx, [], delta * np.eye(n)

        # Now, find as many good perturbations as we can
        # Good = not too far from xbase (relative to delta) and sufficiently linearly independent
        #        from other selected perturbations (i.e. Lfull[perturbation_idx,:] well-conditioned
        #        and len(perturbation_idx) <= n
        perturbation_idx = []  # what point indices to use as perturbations

        for iter in range(min(n_perturbations, n)):
            # Add one more good perturbation, if available
            # Note: can only add at most the number of available perturbations, or n perturbations, whichever is smaller
            if iter == 0:
                # First perturbation: every direction is equally good, so pick the point closest to the
                # trust-region boundary
                idx = int(np.argmax(xdist_reweighted))
                module_logger.debug("Adding index %g with ||xi-xbase|| / delta = %g" % (idx if idx < base_idx else idx+1, xdist[idx]))
                perturbation_idx.append(idx)
            else:
                Q, R = np.linalg.qr(Lfull[perturbation_idx, :].T, mode='reduced')
                # module_logger.debug("Current perturbation_idx =", perturbation_idx)
                L_rem = Lfull @ (np.eye(n) - Q @ Q.T)  # part of (xi-xbase)/delta orthogonal to current perturbations
                # rem_size = fraction of original length ||xi-xbase||/delta that is orthogonal to current perturbations
                # all entries are in [0,1], and is zero for already selected perturbations
                rem_size = np.linalg.norm(L_rem, axis=1) / xdist
                rem_size[perturbation_idx] = 0  # ensure this holds exactly
                # module_logger.debug("rem_size =", rem_size)
                # module_logger.debug("rem_size * xdist_reweighted =", rem_size * xdist_reweighted)

                # We want a point with large rem_size and xdist ~ 1 (i.e. xdist_reweighted large)
                idx = int(np.argmax(rem_size * xdist_reweighted))
                if rem_size[idx] * xdist_reweighted[idx] > tol:
                    # This ensures new perturbation is sufficiently linearly independent of existing perturbations
                    # (and also ensures idx hasn't already been chosen)
                    module_logger.debug("Adding index %g" % (idx if idx < base_idx else idx+1))
                    perturbation_idx.append(idx)
                else:
                    module_logger.debug("No more linearly independent directions, quitting")
                    break

        # Find new linearly independent directions
        if len(perturbation_idx) < n:
            module_logger.debug("Selecting %g new linearly independent directions" % (n - len(perturbation_idx)))
            Q, _ = np.linalg.qr(Lfull[perturbation_idx, :].T, mode='complete')
            new_perturbations = delta * Q[:, len(perturbation_idx):].T

            # Make perturbations feasible w.r.t. xl <= x <= xu and projections
            # Note: if len(projections) > 0, then the projection list *already* includes bounds
            # Don't need to make pre-existing evaluations feasible, since we already have r(x) for these

            # Start construction of interpolation matrix for later
            L = np.zeros((n, n), dtype=float)
            L[:len(perturbation_idx), :] = Lfull[perturbation_idx, :]
            L[len(perturbation_idx):, :] = new_perturbations / delta

            # Since we already have a full set of linearly independent directions,
            # we do this by moving each infeasible perturbation to a geometry-improving location
            for i in range(new_perturbations.shape[0]):
                xnew = xbase + new_perturbations[i, :]
                # Check feasibility
                if len(projections) == 0:
                    # Bounds only
                    feasible = np.all(xnew >= xl) and np.all(xnew <= xu)
                else:
                    # Projections
                    xnew_C = dykstra(projections, xnew, max_iter=dykstra_max_iters, tol=dykstra_tol)
                    feasible = np.linalg.norm(xnew - xnew_C) < dykstra_tol

                if feasible:
                    # Skip feasible points, nothing to do
                    continue

                # If infeasible, build Lagrange polynomial and move to geometry-improving location in B(xbase,delta)
                # which will automatically be feasible
                module_logger.debug("Moving default %g-th new perturbation to ensure feasibility" % i)
                c = 0.0  # Lagrange polynomial centered at xbase
                ei = np.zeros((n,), dtype=float)
                ei[len(perturbation_idx) + i] = 1.0
                g = np.linalg.solve(L, ei) / delta  # divide by delta because L is scaled by 1/delta
                if len(projections) == 0:
                    new_perturbations[i, :] = trsbox_geometry(xbase, c, g, xl, xu, delta)
                else:
                    new_perturbations[i, :] = ctrsbox_geometry(xbase, c, g, projections, delta)

                # Update L after replacement
                L[len(perturbation_idx) + i, :] = new_perturbations[i,:] / delta
        else:
            module_logger.debug("Full set of directions found, no need for new evaluations")
            new_perturbations = None

        # perturbation_idx in [0, ..., n_perturbations-1], reset to be actual indices
        for i in range(len(perturbation_idx)):
            if perturbation_idx[i] >= base_idx:
                perturbation_idx[i] += 1
        return base_idx, perturbation_idx, new_perturbations
