Installing DFO-LS
=================

Requirements
------------
DFO-LS requires the following software to be installed:

* Python 2.7 or Python 3 (http://www.python.org/)

Additionally, the following python packages should be installed (these will be installed automatically if using *pip*, see `Installation using pip`_):

* NumPy 1.11 or higher (http://www.numpy.org/)
* SciPy 0.18 or higher (http://www.scipy.org/)
* Pandas 0.17 or higher (http://pandas.pydata.org/)

**Optional package:** DFO-LS versions 1.2 and higher also support the `trustregion <https://github.com/lindonroberts/trust-region>`_ package for fast trust-region subproblem solutions. To install this, make sure you have a Fortran compiler (e.g. `gfortran <https://gcc.gnu.org/wiki/GFortran>`_) and NumPy installed, then run :code:`pip install trustregion`. You do not have to have trustregion installed for DFO-LS to work, and it is not installed by default.

Installation using conda
------------------------
DFO-LS can be directly installed in Anaconda environments using `conda-forge <https://anaconda.org/conda-forge/dfo-ls>`_:

.. code-block:: bash

    $ conda install -c conda-forge dfo-ls

Installation using pip
----------------------
For easy installation, use *pip* (http://www.pip-installer.org/) as root::

 .. code-block:: bash

    $ [sudo] pip install DFO-LS

or alternatively *easy_install*::

 .. code-block:: bash

    $ [sudo] easy_install DFO-LS

If you do not have root privileges or you want to install DFO-LS for your private use, you can use::

 .. code-block:: bash

    $ pip install --user DFO-LS

which will install DFO-LS in your home directory.

Note that if an older install of DFO-LS is present on your system you can use::

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

To upgrade DFO-LS to the latest version, navigate to the top-level directory (i.e. the one containing :code:`setup.py`) and rerun the installation using :code:`pip`, as above:

 .. code-block:: bash

    $ git pull
    $ [sudo] pip install .  # with admin privileges

Testing
-------
If you installed DFO-LS manually, you can test your installation by running:

 .. code-block:: bash

    $ python setup.py test

Alternatively, the HTML documentation provides some simple examples of how to run DFO-LS.

Uninstallation
--------------
If DFO-LS was installed using *pip* you can uninstall as follows:

 .. code-block:: bash

    $ [sudo] pip uninstall DFO-LS

If DFO-LS was installed manually you have to remove the installed files by hand (located in your python site-packages directory).

