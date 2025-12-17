=========================
Ligand Parameterisation
=========================

BATTER ships a lightweight parameterisation toolkit that converts staged ligand
inputs into a content-addressed store of AMBER or OpenFF artefacts.  The main
entry point is :func:`batter.param.ligand.batch_ligand_process`, which produces
GAFF/GAFF2 ``mol2``/``frcmod``/``lib`` bundles or OpenFF ``prmtop`` files that
can be reused across simulations.

Typical usage
-------------

.. code-block:: python

   from batter.param.ligand import batch_ligand_process

   hashes, metadata = batch_ligand_process(
       ligand_paths={
           "ligA": "ligands/adp.sdf",
           "ligB": "ligands/amp.mol2",
       },
       output_path="cache/ligands",
       ligand_ff="gaff2",
       charge_method="am1bcc",
   )

   print("Prepared hashes:", hashes)
   print("Canonical SMILES:", metadata["ligands/adp.sdf"][1])

API Reference
-------------

.. automodule:: batter.param.ligand
   :members:
   :undoc-members:
   :show-inheritance:

Caching and validation
----------------------

Ligand artifacts are content-addressed: input coordinates plus force-field/charge
settings are hashed, so rerunning ``batch_ligand_process`` with the same inputs reuses
cached ``mol2``/``frcmod``/``lib`` bundles instead of recomputing. Charge assignment
errors and missing protonation states surface as exceptions; callers should surface
those errors up the pipeline rather than silently skipping ligands. Validation also
canonicalises SMILES so cache keys stay stable across input formats.

Output layout
-------------

By default, outputs land under the provided ``output_path`` in per-ligand folders
that include the hash. Metadata (canonical SMILES, charge method, parameter files)
is returned to the caller so builders can record it in ``SystemMeta`` and reuse the
same parameter set across runs.
