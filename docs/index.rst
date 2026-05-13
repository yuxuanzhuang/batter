BATTER Documentation
====================

BATTER is a modular orchestration layer for **absolute binding free energy (ABFE)**,
**relative binding free energy (RBFE)**, and **absolute solvation free energy (ASFE)**
workflows, especially for membrane proteins.
This site provides getting-started guides, tutorials, API references, developer
resources, and a cookbook of smaller workflow recipes.

Quick Links
-----------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc
      :margin: 0 3 0 3

      New to BATTER? Begin with environment setup, the core workflow, and example inputs.

   .. grid-item-card:: Tutorial
      :link: tutorial/index
      :link-type: doc
      :margin: 0 3 0 3

      Run complete ABFE and RBFE calculations step by step and learn the command-line entry points.

   .. grid-item-card:: Configuration Reference
      :link: cookbook/configuration
      :link-type: doc
      :margin: 0 3 0 3

      Explore all options in the run and simulation YAML schemas, including SLURM settings,
      restraints, and REMD parameters.

   .. grid-item-card:: Cookbook
      :link: cookbook/index
      :link-type: doc
      :margin: 0 3 0 3

      Browse focused recipes such as the RBFE guide and lambda-schedule optimization tricks.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc
      :margin: 0 3 0 3

      Browse the Python API for configuration helpers, artifact stores, and results repositories.

   .. grid-item-card:: Developer Guide
      :link: developer_guide
      :link-type: doc
      :margin: 0 3 0 3

      Dive into the internal architecture, typed pipeline payloads, and the metadata model.

   .. grid-item-card:: CLI Reference
      :link: cli
      :link-type: doc
      :margin: 0 3 0 3

      Discover available ``batter`` commands, options, and SLURM utilities.

Primary Guides
--------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   getting_started
   tutorial/index
   cookbook/index
   cli
   api
   developer_guide
