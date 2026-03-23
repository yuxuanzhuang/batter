RBFE Guide
==========

This page documents the current RBFE workflow in BATTER and the network-mapping
options in the ``rbfe`` section of ``run.yaml``.

Minimal RBFE configuration
--------------------------

.. code-block:: yaml

   protocol: rbfe

   run:
     output_folder: /path/to/output
     run_id: auto

   create:
     system_name: my_system
     protein_input: /path/to/protein.pdb
     ligand_paths:
       LIG1: /path/to/lig1.sdf
       LIG2: /path/to/lig2.sdf
       LIG3: /path/to/lig3.sdf

   rbfe:
     mapping: default
     atom_mapper: kartograf

If you omit ``rbfe.mapping`` (and do not provide files), BATTER uses
``default``.

Default mapping algorithm
-------------------------

The default mapping is a star network:

* Ligands are taken in input order.
* The first ligand is used as reference.
* Pairs are built as ``(lig1, lig2)``, ``(lig1, lig3)``, ...

This corresponds to ``RBFENetwork.default_mapping``.

Mapping options and precedence
------------------------------

RBFE mapping is controlled by ``rbfe`` in ``run.yaml``:

* ``rbfe.mapping_file``
* ``rbfe.mapping`` (default ``default``)

If both are provided, BATTER uses ``mapping_file``.

Supported ``rbfe.mapping`` values
---------------------------------

* ``default`` (also ``star`` / ``first`` aliases)
* ``konnektor``

When using ``konnektor``, you can optionally set ``rbfe.konnektor_layout``.

.. code-block:: yaml

   rbfe:
     mapping: konnektor
     atom_mapper: kartograf
     konnektor_layout: star
     both_directions: false

Atom mapper backends
--------------------

RBFE atom mapping backend is controlled by ``rbfe.atom_mapper``:

* ``kartograf`` (default) – current BATTER Kartograf-based mapping behavior:

  .. code-block:: python

     # network planning mapper (rbfe.py)
     KartografAtomMapper(
         atom_max_distance=0.95,
         map_hydrogens_on_hydrogens_only=True,
         atom_map_hydrogens=False,
         map_exact_ring_matches_only=True,
         allow_partial_fused_rings=True,
         allow_bond_breaks=False,
         additional_mapping_filter_functions=[filter_element_changes],
     )

  During RBFE transformation setup (``_internal/ops/simprep.py``), BATTER uses
  the same Kartograf settings except ``atom_map_hydrogens=True`` and then removes
  hydrogen pairs from the final core mapping.

* ``lomap`` – uses:

  .. code-block:: python

     LomapAtomMapper(
         time=20,
         threed=True,
         max3d=1.5,
         element_change=False,
         shift=True,
     )

Example:

.. code-block:: yaml

   rbfe:
     mapping: konnektor
     atom_mapper: lomap

Bidirectional RBFE edges
------------------------

Set ``rbfe.both_directions: true`` to run both directions for each mapped edge.
For example, a mapped pair ``LIG1~LIG2`` will generate both:

* ``LIG1~LIG2``
* ``LIG2~LIG1``

``mapping_file`` formats
------------------------

``rbfe.mapping_file`` supports:

* JSON/YAML list of pairs, e.g. ``[["LIG1","LIG2"], ["LIG2","LIG3"]]``
* JSON/YAML dict with ``pairs`` or ``edges`` keys
* JSON/YAML adjacency dict, e.g. ``{"LIG1": ["LIG2","LIG3"]}``
* text file with one pair per line (``A~B``, ``A,B``, or ``A B``)

Konnektor layouts: how to list all available options
-----------------------------------------------------

BATTER resolves Konnektor layouts dynamically from
``konnektor.network_planners`` by collecting class names ending with
``NetworkGenerator``.

To list available layout names in your environment:

.. code-block:: bash

   python - <<'PY'
   from konnektor import network_planners as np
   names = []
   for name in dir(np):
       if name.endswith("NetworkGenerator"):
           names.append(name)
           names.append(name[:-len("NetworkGenerator")].lower())
   print(sorted(set(names)))
   PY

Note: Konnektor ``explicit`` layouts require explicit edges; in BATTER use
``rbfe.mapping_file`` for that case.

Where RBFE mapping is stored
----------------------------

BATTER writes the resolved network to:

* ``executions/<run_id>/artifacts/config/rbfe_network.json``

Transformation systems are created under:

* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/``

For each transformation pair, BATTER stores atom-mapping artifacts in:

* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/fe/x/x-1/mapping.json``
* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/fe/x/x-1/mapping.pkl``
* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/fe/x/x-1/mapping.png``

These generic ``mapping.*`` filenames are used for both ``kartograf`` and
``lomap`` atom-mapper backends.

Pipeline notes
--------------

In RBFE, BATTER runs ligand-level prep/equilibration first, then builds
transformation pairs from the resolved network and runs component ``x`` FE on
those pairs.
