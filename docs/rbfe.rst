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
     konnektor_layout: star

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

Pipeline notes
--------------

In RBFE, BATTER runs ligand-level prep/equilibration first, then builds
transformation pairs from the resolved network and runs component ``x`` FE on
those pairs.
