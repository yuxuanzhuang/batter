batter
==============================

.. image:: https://github.com/yuxuanzhuang/batter/workflows/CI/badge.svg
   :target: https://github.com/yuxuanzhuang/batter/actions?query=workflow%3ACI

.. image:: https://codecov.io/gh/yuxuanzhuang/batter/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/yuxuanzhuang/batter/branch/main

``batter`` is a modern, object-oriented toolkit for free-energy workflows.
It adds first-class support for **absolute binding free energy (ABFE)** of membrane proteins and **absolute solvation free energy (ASFE)**,
with an AMBER + sdg + express pipeline to the original ``BAT.py`` package.

.. note::
   The API is stabilizing. Some modules are still under active development, but the overall structure is in place.

Installation
-------------------------------

Clone the repository, initialize submodules, and create the environment:

.. code-block:: bash

   git clone git@github.com:yuxuanzhuang/batter.git
   # If SSH clone fails, configure your GitHub SSH keys:
   # https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

   cd batter
   git submodule update --init --recursive

   conda create -n batter -y python=3.12
   conda env update -f environment.yml -n batter
   conda activate batter

   # Editable installs for bundled deps
   pip install -e ./extern/alchemlyb
   pip install -e ./extern/rocklinc

   # Install batter (editable)
   pip install -e .

This installs in editable mode so your code changes are immediately reflected.

Quickstart
-------------------------------

Run an example configuration:

.. code-block:: bash

   cd examples
   batter run mabfe.yaml

Use ``--help`` to see all commands:

.. code-block:: bash

   batter -h
   batter run -h

Examples
==============================

YAMLs in ``examples/`` illustrate common setups:

**Absolute Binding Free Energy (ABFE)**
   1. ``mabfe.yaml`` — membrane protein (e.g., B2AR) in a lipid bilayer
   2. ``mabfe_nonmembrane.yaml`` — soluble protein (e.g., BRD4) in water
   3. ``extra_restraints/mabfe.yaml`` — add additional positional restraints to selected atoms
   4. ``conformational_restraints/mabfe.yaml`` — add additional conformational restraints (distance between atoms)

**Absolute Solvation Free Energy (ASFE)**
   1. ``masfe.yaml`` — small molecule (e.g., epinephrine) in water

Notes
-------------------------------
- Backends include local execution and SLURM-based submission (see CLI options).
- Example YAMLs are intended as starting points; adjust force fields, restraints, and sampling knobs to your system.

Copyright
-------------------------------
**Copyright (c) 2024, Yuxuan Zhuang**

Acknowledgements
-------------------------------
Built with the
`Computational Molecular Science Python Cookiecutter <https://github.com/molssi/cookiecutter-cms>`_ (v1.10).