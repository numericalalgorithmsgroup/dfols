Version History
===============
This section lists the different versions of DFO-LS and the updates between them.

Version 1.0 (6 Feb 2018)
------------------------
* Initial release of DFO-LS

Version 1.0.1 (20 Feb 2018)
---------------------------
* Minor bug fix to trust region subproblem solver (the output :code:`crvmin` is calculated correctly) - this has minimal impact on the performance of DFO-LS.

Version 1.0.2 (20 Jun 2018)
---------------------------
* Extra optional input :code:`args` which passes through arguments for :code:`objfun`.
* Bug fixes: default parameters for reduced initialization cost regime, returning correct value if exiting from within a safety step, retrieving dependencies during installation.

Version 1.1 (16 Jan 2019)
-------------------------
* Use different default reduced initialization cost method for inverse problems to ensure whole space is searched correctly.
* Bug fixes: default trust region radius when scaling feasible region, exit correctly when no Jacobian returned, handling overflow at initial value

Version 1.1.1 (5 Apr 2019)
--------------------------
* Link code to Zenodo, to create DOI - no changes to the DFO-LS algorithm.

Version 1.2
-----------
* Use deterministic initialisation by default (so it is no longer necessary to set a random seed for reproducibility of DFO-LS results).
* Full model Hessian stored rather than just upper triangular part - this improves the runtime of Hessian-based operations.
* Faster trust-region and geometry subproblem solutions in Fortran using the `trustregion <https://github.com/lindonroberts/trust-region>`_ package.
* Faster interpolation solution for multiple right-hand sides.
* Don't adjust starting point if it is close to the bounds (as long as it is feasible).
* Option to stop default logging behavior and/or enable per-iteration printing.
* Bugfix: correctly handle 1-sided bounds as inputs, avoid divide-by-zero warnings when auto-detecting restarts.
