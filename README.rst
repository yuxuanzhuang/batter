batter
==============================

.. [//]: # (Badges)
.. image:: https://github.com/yuxuanzhuang/batter/workflows/CI/badge.svg
   :target: https://github.com/yuxuanzhuang/batter/actions?query=workflow%3ACI

.. image:: https://codecov.io/gh/yuxuanzhuang/batter/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/yuxuanzhuang/batter/branch/main

This is a refactored version of the `BAT.py` package rewritten in object-oriented style.

The major changes include support for calculating the absolute binding free energy of membrane proteins, currently in AMBER+sdg+express mode.

Note: Some parts of the code may still be under development, but the overall structure is in place.

Installation
-------------------------------
To install, first clone the repository and then run the following command:

.. code-block:: bash

    git clone git@github.com:yuxuanzhuang/batter.git
    cd batter

    conda env create -f environment.yml
    conda activate batter
    pip install -e .

This will install the package in editable mode, so you can make changes to the code and see the changes reflected in the package.

Usage
-------------------------------
It is advisable to use the package as a python script or inside a jupyter notebook.

See the `examples` folder for examples of how to use the package.

- `run_batter_abfe.ipynb` is an example of how to run ABFE simulations.
- `run_batter_rbfe.ipynb` is an example of how to run RBFE simulations.
- `abfe_rest.ipynb` and `rbfe_rest.ipynb` are examples of running with flat-bottom restraints
    - `generate_rmsf_restratins.ipynb` is an example of how to generate RMSF restraints.
    - you will also need a patched version of AMBER24 to run these simulations.
- `rbfe_pipeline.py` and `abfe_pipeline.py` are examples of how to run the simulations in an
automatic pipeline.
    - `submit_pipeline.sbatch` is an example of how to submit the pipeline to a cluster.

See the `scripts` folder for scripts that might be helpful in running the simulations.

- `cancel_jobs.sh` is a script that can be used to cancel all FE-related jobs on a cluster.
- `mbar_rest.ipynb` is a notebook to check the validity of the MBAR results with REST.
- `mbar_sdr.ipynb` is a notebook to check the validity of the MBAR results with SDR.

**NOT RECOMMENDED**

Alternatively, below are examples of how to run the code as a command-line tool.
It still ultilizes the old scripts and it is not well tested.
Input files can be found in the `examples` folder.

**Run Equilibration**

.. code-block:: bash

    batter batpy -i membrane_input.in -s equil -w FEP -p all-poses

**Run Free Energy Calculations (FE)**

.. code-block:: bash

    batter batpy -i input.in -s fe -w FEP -p all-poses

**Run Analysis**

.. code-block:: bash

    batter batpy -i input.in -s analysis -w FEP -p all-poses

Copyright
-------------------------------
Copyright (c) 2024, Yuxuan Zhuang

Acknowledgements
-------------------------------
Project based on the 
`Computational Molecular Science Python Cookiecutter <https://github.com/molssi/cookiecutter-cms>`_ version 1.10.