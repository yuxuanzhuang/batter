.. _md_tutorial:

MD Tutorial
===========

Molecular Dynamics (MD) Workflow with ``batter``
------------------------------------------------

This tutorial shows how to run ligand-bound or apo MD through BATTER's
orchestrator. The MD protocol reuses BATTER's system staging, ligand
parameterisation, equilibration input generation, job monitoring, and
equilibration-analysis machinery, but it stops before FE-window setup. No
lambda windows, MBAR analysis, or FE result records are produced for
``protocol: md``.

Use ``examples/md_example.yaml`` for a ligand-bound run and
``examples/md_apo_example.yaml`` for an apo-only run.

Quick Walkthrough
-----------------

The MD pipeline is:

#. **System staging** - create ``<run.output_folder>/executions/<run_id>/`` and
   stage receptor, membrane/box inputs, ligands, and shared metadata.
#. **Ligand parameterisation** - prepare real ligand parameters with the chosen
   GAFF/OpenFF settings. Apo dummy ligands are recognized and skipped by the
   ligand-parameterisation path.
#. **Equilibration preparation** - write the AMBER topology, coordinates, mdin
   files, and local/SLURM run scripts for equilibration.
#. **Equilibration** - run the configured MD length from ``fe_sim.eq_steps``.
#. **Equilibration analysis** - write analysis outputs and representative
   structures under each staged ligand simulation directory.

Preparing a Ligand-Bound MD Run
-------------------------------

Start from ``examples/md_example.yaml``. The minimal shape is:

.. code-block:: yaml

   version: 1
   protocol: md
   backend: local

   create:
     system_name: my_system
     protein_input: reference/protein.pdb
     system_input: reference/system.pdb
     system_coordinate: reference/system.rst7
     protein_align: name CA and resid 50 to 200
     ligand_input: reference/ligands_dict.json
     lipid_mol: ["POPC"]
     water_model: OPC

   run:
     output_folder: my_system_md
     run_id: auto
     on_failure: prune
     max_workers: 8

   fe_sim:
     eq_steps: 50000000
     cut: 9.0
     hmr: yes
     dt: 0.004
     temperature: 298.15

``create.ligand_input`` points to a JSON dictionary whose keys are ligand names
and whose values are ligand structure paths:

.. code-block:: json

   {
     "LIG1": "ligands/lig1.sdf",
     "LIG2": "ligands/lig2.sdf"
   }

Paths inside this JSON are resolved relative to the JSON file location. You can
also write the same mapping directly in YAML with ``create.ligand_paths``.

Running Apo-Only MD
-------------------

For apo-only MD, BATTER still needs one ligand-like entry so it can reuse the
existing ligand-centered staging path. Use a null apo ligand entry; BATTER
replaces it with its bundled dummy ligand PDB and recognizes it as apo-only
during later setup.

The inline YAML form used by ``examples/md_apo_example.yaml`` is:

.. code-block:: yaml

   create:
     ligand_paths:
       None: None

The equivalent ``create.ligand_input`` JSON form is:

.. code-block:: json

   {
     "None": null
   }

Custom apo labels are also valid and are useful for repeated apo simulations:

.. code-block:: json

   {
     "apo_rep1": null,
     "apo_rep2": null
   }

BATTER sanitizes these names for output directories and maps each null value to
the bundled apo dummy ligand. Apo-only MD can omit ``create.anchor_atoms``; BATTER
uses a protein-only anchor heuristic so dummy ligand coordinates do not define
the binding-site geometry.

Anchor Atoms
------------

``create.anchor_atoms`` is optional for MD. If omitted:

* ligand-bound MD uses the first real ligand pose to select receptor anchors;
* apo-only MD uses the protein-only heuristic.

BATTER uses three distinct, unambiguous selections directly as P1/P2/P3. One
selection is a P1 hint. Two selections are treated as incomplete: BATTER keeps
the first as P1 and chooses P2/P3 together. Invalid or ambiguous input warns and
falls back to automatic selection. In ligand-bound runs, a detected salt bridge
can define P1/L1; L2/L3 then prefer ring atoms connected to at least two heavy
atoms, followed by atoms with more than two heavy-atom connections and other
nonterminal heavy atoms. Auto-selected receptor anchors come from stable
non-loop Cα atoms and are screened against near-linear frames. Compact systems
may relax the preferred spacing while retaining the non-collinearity guard.
Provide three selections only when you need fully manual P1/P2/P3 geometry:

.. code-block:: yaml

   create:
     anchor_atoms:
       - "name CA and resid 113"
       - "name CA and resid 82"
       - "name CA and resid 316"

The resolved selections are written to
``executions/<run_id>/all-ligands/manifest.json`` under ``anchors`` and
``anchor_atom_selections``. Prepared-system masks are written per ligand to
``equil/anchors.json``.

Running the Workflow
--------------------

First run a dry run to validate paths and generate inputs up to the first MD
submission point:

.. code-block:: bash

   batter run examples/md_example.yaml --dry-run

For apo-only MD:

.. code-block:: bash

   batter run examples/md_apo_example.yaml --dry-run

Then launch the run:

.. code-block:: bash

   batter run examples/md_example.yaml

For SLURM-managed execution, keep ``backend: local`` in the YAML and submit the
manager through SLURM:

.. code-block:: bash

   batter run examples/md_example.yaml --slurm-submit

Useful knobs:

``fe_sim.eq_steps``
   Total equilibration MD steps.
``fe_sim.hmr`` / ``fe_sim.dt``
   Hydrogen mass repartitioning and timestep settings.
``fe_sim.enable_mcwat``
   Enable or disable MC water moves during equilibration; the default is enabled.
``fe_sim.mcwat_fe``
   Enable MC water moves in FE production templates; the default is enabled.
   Standard grouped batch runs disable these moves at launch unless REMD is used.
``run.clean_failures`` / ``--clean-failures``
   Clear failed sentinels and retry counters before rerunning an execution.
``run.run_id``
   Use a stable run id for reproducible paths, or ``auto`` to create/reuse a
   run id automatically.

Inspecting Outputs
------------------

MD-only runs keep their primary artifacts under:

.. code-block:: text

   <run.output_folder>/executions/<run_id>/
   |-- all-ligands/
   |   `-- manifest.json
   |-- artifacts/
   `-- simulations/
       `-- <ligand_or_apo_name>/
           |-- inputs/
           `-- equil/

Equilibration analysis files are written under each ligand's simulation tree.
Because ``protocol: md`` has no FE phase, BATTER logs that FE production was
skipped and does not export FE records under ``results/``.
