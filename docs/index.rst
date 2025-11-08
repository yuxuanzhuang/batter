=============================
Welcome to BATTER's Docs
=============================

BATTER is a modular orchestration layer for absolute binding free-energy (ABFE)
workflows—especially for membrane proteins—and
for solvation (ASFE). This documentation includes
user guides, configuration references,
tutorials, and developer resources.


Quick Links
===========

.. grid:: 1 1 2 2

   .. grid-item-card:: Getting Started
      :margin: 0 3 0 3

      New to BATTER? Start here for environment setup, the core workflow, and links
      to example inputs.

      .. button-link:: getting_started.html
         :color: primary
         :outline:
         :expand:

         Go to Getting Started

   .. grid-item-card:: Tutorial
      :margin: 0 3 0 3

      Follow the step-by-step tutorial to run a complete ABFE calculation and become
      familiar with the command-line entry points.

      .. button-link:: tutorial.html
         :color: primary
         :outline:
         :expand:

         Explore the Tutorial

   .. grid-item-card:: Configuration Reference
      :margin: 0 3 0 3

      Learn every option available in the run and simulation YAML schemas, including
      SLURM controls, restraints, and REMD knobs.

      .. button-link:: configuration.html
         :color: primary
         :outline:
         :expand:

         Read the Configuration Reference

   .. grid-item-card:: API Reference
      :margin: 0 3 0 3

      Browse the Python API for programmatic access to configuration helpers, artifact
      stores, and results repositories.

      .. button-link:: api.html
         :color: primary
         :outline:
         :expand:

         View the API Reference

   .. grid-item-card:: Developer Guide
      :margin: 0 3 0 3

      Dive into the internal architecture, typed pipeline payloads, and metadata model
      when extending or contributing to BATTER.

      .. button-link:: developer_guide.html
         :color: primary
         :outline:
         :expand:

         Open the Developer Guide

   .. grid-item-card:: Command-Line Usage
      :margin: 0 3 0 3

      Learn the available ``batter`` commands, options, and SLURM utilities.

      .. button-link:: cli.html
         :color: primary
         :outline:
         :expand:

         View the CLI Reference


Primary Guides
==============

.. toctree::
   :maxdepth: 2
   :titlesonly:

   getting_started
   tutorial
   configuration
   cli
   api
   developer_guide
