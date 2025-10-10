batter
==============================

.. [//]: # (Badges)

.. image:: https://github.com/yuxuanzhuang/batter/workflows/CI/badge.svg
   :target: https://github.com/yuxuanzhuang/batter/actions?query=workflow%3ACI

.. image:: https://codecov.io/gh/yuxuanzhuang/batter/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/yuxuanzhuang/batter/branch/main

This is a refactored version of the `BAT.py` package rewritten in object-oriented style.

The major changes include support for calculating the absolute binding free energy of membrane proteins, currently in AMBER+sdg+express mode.

**Note:** Some parts of the code may still be under development, but the overall structure is in place.

Installation
-------------------------------
To install, first clone the repository and then run the following command:

.. code-block:: bash

    git clone git@github.com:yuxuanzhuang/batter.git
    cd batter
    git submodule update --init --recursive

    conda env create -f extern/openfe/environment.yml -n batter
    conda env update --name batter --file environment.yml

    conda activate batter
    
    pip install -e ./extern/alchemlyb
    pip install -e ./extern/openfe
    pip install -e ./extern/rocklinc
    pip install -e .

This will install the package in editable mode, so you can make changes to the code and see the changes reflected in the package.

Usage
-------------------------------
It is advisable to use the package as a Python script or inside a Jupyter notebook.

See the `scripts` folder for scripts that might be helpful in running the simulations:

- `cancel_jobs.sh`: Script to cancel all FE-related jobs on a cluster.

Copyright
-------------------------------
**Copyright (c) 2024, Yuxuan Zhuang**

Acknowledgements
-------------------------------
This project is based on the 
`Computational Molecular Science Python Cookiecutter <https://github.com/molssi/cookiecutter-cms>`_ version 1.10.
