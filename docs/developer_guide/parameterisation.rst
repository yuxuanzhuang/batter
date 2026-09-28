=========================
Ligand Parameterisation
=========================

BATTER ships a lightweight parameterisation toolkit that converts staged ligand
inputs into a content-addressed store of AMBER or OpenFF artefacts.  The main
entry point is :func:`batter.param.ligand.batch_ligand_process`, which produces
GAFF/GAFF2 ``mol2``/``frcmod``/``lib`` bundles or OpenFF ``prmtop`` files that
can be reused across simulations.

Standalone CLI
--------------

Use ``param-ligand`` to parameterise one SDF without preparing a complete BATTER
execution::

   batter param-ligand ligand.sdf

The command defaults to:

* output store: ``./ligand_param``;
* force field: ``openff-2.3.0``;
* charge method: ``openff-gnn-am1bcc-1.0.0.pt``;
* retaining explicit input hydrogens; and
* reusing a complete matching cache entry.

The exact output directory is printed on success. For example, a cache key of
``a340da44aec6`` produces::

   ligand_param/
   |-- .locks/
   `-- a340da44aec6/
       |-- lig.sdf
       |-- lig.mol2
       |-- lig.frcmod
       |-- lig.lib
       |-- lig.prmtop
       |-- lig.inpcrd
       |-- lig.pdb
       |-- lig.json
       `-- metadata.json

Common overrides include::

   batter param-ligand ligand.sdf \
       --output shared_ligand_params \
       --charge-method am1bcc \
       --no-retain-h \
       --overwrite

``--no-retain-h`` removes the input hydrogens and regenerates them before
parameterisation. ``--overwrite`` rebuilds only the matching content-addressed
entry; it does not clear the whole parameter store. The command reports
parameterisation failures rather than pruning them.

The command runs synchronously on the local host; it does not submit a Slurm
job. The full parameterisation environment is required. In particular,
OpenFF output also uses an Amber bootstrap, so ``antechamber``, ``parmchk2``,
and ``tleap`` must be available alongside OpenFF Toolkit, Interchange, and the
selected charge backend.

``lig.json`` describes the prepared ligand (including its residue name, net
charge, force field, and canonicalised SDF), while ``metadata.json`` records
cache provenance such as the input path, content hash, atom-order fingerprint,
and parameterisation settings.

Reusing the store in a workflow
-------------------------------

Point ``create.param_outdir`` at the store root to let a normal BATTER run reuse
the standalone result::

   create:
     ligand_input: ligands.json
     param_outdir: /path/to/ligand_param
     ligand_ff: openff-2.3.0
     param_charge: openff-gnn-am1bcc-1.0.0.pt
     retain_lig_prot: true

The force field, charge method, and hydrogen-retention setting must match the
standalone invocation because all three contribute to the content hash. The
standalone command defaults to ``./ligand_param`` (singular); when
``create.param_outdir`` is omitted, a normal workflow uses
``<output_folder>/ligand_params`` (plural).

Input and reuse rules
---------------------

Supply an SDF containing a chemically valid ligand with 3D coordinates. A
protonated structure with explicit hydrogens is recommended when using the
default ``--retain-h`` behavior. If an SDF contains multiple records, only the
first molecule is parameterised.

The cache key contains canonical chemistry, indexed atom topology, force field,
charge method, and hydrogen-retention mode. Coordinates are deliberately absent
from the key, so coordinate-only conformers with the same atom ordering can
reuse parameters. Isomorphic molecules with different atom ordering receive
different entries because BATTER applies their generated parameter files
positionally.

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

Ligand artifacts are content-addressed. Cache keys include canonical chemistry,
indexed atom topology, force-field and charge settings, and the explicit-hydrogen
retention mode. Coordinates are excluded, so conformers with compatible atom
ordering can reuse parameters. Graph-isomorphic files with different atom ordering
receive separate entries because downstream AMBER files are applied positionally.
Charge assignment errors and missing protonation states surface as exceptions;
callers should surface those errors up the pipeline rather than silently skipping
ligands.

Writers targeting the same content hash are serialized with a file lock under
``<parameter-store>/.locks/``. A process that waits for another run rechecks the
cache while holding the lock, reuses a complete entry, and rebuilds an incomplete
entry. This prevents concurrent executions from publishing partial artifacts.

Output layout
-------------

By default, outputs land under the provided ``output_path`` in per-ligand folders
that include the hash. Metadata (canonical SMILES, charge method, parameter files)
is returned to the caller so builders can record it in ``SystemMeta`` and reuse the
same parameter set across runs.
