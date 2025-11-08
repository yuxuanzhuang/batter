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
