.. _install:

************
Installation
************

Install via pip
---------------

To install the most recent release of ``shone``, you may::

    python -m pip install shone

Install from source
-------------------

Clone the repository, change directories into it, and build from source::

    git clone https://github.com/bmorris3/shone.git
    cd shone
    python -m pip install -e .

.. note::

    Known issue for M2 Macs: as of January 2024, pip will install a version of jaxlib
    that may not work, raising the following error::

        RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support.

    The easiest workaround is to uninstall this version of jax with::

        pip uninstall jax jaxlib

    and then install jax via conda::

        conda install -c conda-forge jaxlib
        conda install -c conda-forge jax

