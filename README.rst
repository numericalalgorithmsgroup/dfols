===================================================
DFO-LS: Derivative-Free Optimizer for Least-Squares
===================================================

.. image::  https://github.com/numericalalgorithmsgroup/dfols/actions/workflows/python_testing.yml/badge.svg
   :target: https://github.com/numericalalgorithmsgroup/dfols/actions
   :alt: Build Status

.. image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: GNU GPL v3 License

.. image:: https://img.shields.io/pypi/v/DFO-LS.svg
   :target: https://pypi.python.org/pypi/DFO-LS
   :alt: Latest PyPI version

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2630426.svg
   :target: https://doi.org/10.5281/zenodo.2630426
   :alt: DOI:10.5281/zenodo.2630426
   
.. image:: https://static.pepy.tech/personalized-badge/dfo-ls?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads
   :target: https://pepy.tech/project/dfo-ls
   :alt: Total downloads

DFO-LS is a flexible package for solving nonlinear least-squares minimization, without requiring derivatives of the objective. It is particularly useful when evaluations of the objective function are expensive and/or noisy. DFO-LS is more flexible version of `DFO-GN <https://github.com/numericalalgorithmsgroup/dfogn>`_.

This is an implementation of the algorithm from our paper: C. Cartis, J. Fiala, B. Marteau and L. Roberts, `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41 [`preprint <https://arxiv.org/abs/1804.00154>`_]. For reproducibility of all figures in this paper, please feel free to contact the authors. 

If you are interested in solving general optimization problems (without a least-squares structure), you may wish to try `Py-BOBYQA <https://github.com/numericalalgorithmsgroup/pybobyqa>`_, which has many of the same features as DFO-LS.

Documentation
-------------
See manual.pdf or `here <https://numericalalgorithmsgroup.github.io/dfols/>`_.

Citation
--------
If you use DFO-LS in a paper, please cite:

Cartis, C., Fiala, J., Marteau, B. and Roberts, L., `Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers <https://doi.org/10.1145/3338517>`_, *ACM Transactions on Mathematical Software*, 45:3 (2019), pp. 32:1-32:41.

If you use DFO-LS for problems with constraints, including bound constraints, please also cite:

Hough, M. and Roberts, L., `Model-Based Derivative-Free Methods for Convex-Constrained Optimization <https://doi.org/10.1137/21M1460971>`_, *SIAM Journal on Optimization*, 21:4 (2022), pp. 2552-2579.

Requirements
------------
DFO-LS requires the following software to be installed:

* Python 3.9 or higher (http://www.python.org/)

Additionally, the following python packages should be installed (these will be installed automatically if using *pip*, see `Installation using pip`_):

* NumPy (http://www.numpy.org/)
* SciPy version 1.11 or higher (http://www.scipy.org/)
* Pandas (http://pandas.pydata.org/)

**Optional package:** DFO-LS versions 1.2 and higher also support the `trustregion <https://github.com/lindonroberts/trust-region>`_ package for fast trust-region subproblem solutions. To install this, make sure you have a Fortran compiler (e.g. `gfortran <https://gcc.gnu.org/wiki/GFortran>`_) and NumPy installed, then run :code:`pip install trustregion`. You do not have to have trustregion installed for DFO-LS to work, and it is not installed by default.

Installation using conda
------------------------
DFO-LS can be directly installed in Anaconda environments using `conda-forge <https://anaconda.org/conda-forge/dfo-ls>`_:

.. code-block:: bash

    $ conda install -c conda-forge dfo-ls

Installation using pip
----------------------
For easy installation, use `pip <http://www.pip-installer.org/>`_ as root:

.. code-block:: bash

    $ [sudo] pip install DFO-LS

or alternatively *easy_install*:

.. code-block:: bash

    $ [sudo] easy_install DFO-LS

If you do not have root privileges or you want to install DFO-LS for your private use, you can use:

.. code-block:: bash

    $ pip install --user DFO-LS

which will install DFO-LS in your home directory.

Note that if an older install of DFO-LS is present on your system you can use:

.. code-block:: bash

    $ [sudo] pip install --upgrade DFO-LS

to upgrade DFO-LS to the latest version.

Manual installation
-------------------
Alternatively, you can download the source code from `Github <https://github.com/numericalalgorithmsgroup/dfols>`_ and unpack as follows:

 .. code-block:: bash

    $ git clone https://github.com/numericalalgorithmsgroup/dfols
    $ cd dfols

DFO-LS is written in pure Python and requires no compilation. It can be installed using:

 .. code-block:: bash

    $ [sudo] pip install .

If you do not have root privileges or you want to install DFO-LS for your private use, you can use:

 .. code-block:: bash

    $ pip install --user .

instead.

To upgrade DFO-LS to the latest version, navigate to the top-level directory (i.e. the one containing :code:`pyproject.toml`) and rerun the installation using :code:`pip`, as above:

 .. code-block:: bash

    $ git pull
    $ [sudo] pip install .  # with admin privileges

Testing
-------
If you installed DFO-LS manually, you can test your installation using the pytest package:

 .. code-block:: bash

    $ pip install pytest
    $ python -m pytest --pyargs dfols

Alternatively, the HTML documentation provides some simple examples of how to run DFO-LS.

Examples
--------
Examples of how to run DFO-LS are given in the `documentation <https://numericalalgorithmsgroup.github.io/dfols/>`_, and the `examples <https://github.com/numericalalgorithmsgroup/dfols/tree/master/examples>`_ directory in Github.

Uninstallation
--------------
If DFO-LS was installed using *pip* you can uninstall as follows:

 .. code-block:: bash

    $ [sudo] pip uninstall DFO-LS

If DFO-LS was installed manually you have to remove the installed files by hand (located in your python site-packages directory).

Bugs
----
Please report any bugs using `GitHub's issue tracker <https://github.com/numericalalgorithmsgroup/dfols/issues>`_.

License
-------
This algorithm is released under the GNU GPL license. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing.
