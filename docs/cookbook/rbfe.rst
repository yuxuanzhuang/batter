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
     # Optional: omit anchors, provide one P1 anchor, or provide explicit P1/P2/P3.
     # anchor_atoms:
     #   - "name CA and resid 113"
     #   - "name CA and resid 82"
     #   - "name CA and resid 316"

   rbfe:
     mapping: default
     atom_mapper: kartograf

   fe_sim:
     lambdas: [0.0, 0.5, 1.0]
     x_n_steps: 300000

If you omit ``rbfe.mapping`` (and do not provide files), BATTER uses
``default``.

Anchor selection
----------------

``create.anchor_atoms`` is optional. When it is omitted, BATTER chooses the three
receptor anchors heuristically during system preparation:

* For ABFE/RBFE and MD runs with at least one real ligand, BATTER uses the first
  real ligand pose as the binding-site reference. It prefers stable non-loop
  receptor Cα atoms near the ligand, keeps P1-P2 and P2-P3 separated by the
  usual BATTER distance guideline, and scores P1 using nearby ligand interaction
  atoms. With zero or one receptor override, a detected salt bridge can define
  P1/L1; L2/L3 then prefer ring atoms connected to at least two heavy atoms,
  followed by other highly connected nonterminal heavy atoms.
* For apo-only MD runs, BATTER uses a protein-only heuristic that chooses a
  stable, non-degenerate receptor-anchor triplet without relying on dummy
  ligand coordinates.

If you provide one selection, BATTER treats that atom as P1 and chooses P2/P3
automatically. Provide three selections only when you need fully manual
P1/P2/P3 geometry. Resolved
global anchors are stored in ``executions/<run_id>/all-ligands/manifest.json``
under ``anchors`` and ``anchor_atom_selections``. Prepared-system anchor masks
used later by equilibration and FE setup are written to each ligand's
``equil/anchors.json``.

Standard RBFE and rbfe_septop
-----------------------------

Use ``protocol: rbfe`` when ligand pairs have a chemically meaningful common
core. BATTER prepares reusable atom-mapping artifacts and the transformation
setup uses those mappings for the standard RBFE softcore construction.

Use ``protocol: rbfe_septop`` for scaffold hops, protonation-state comparisons,
or pairs where a common-core mapping would be misleading. The same RBFE network
planner is used, but SEPTOP FE setup treats each full ligand as softcore and
uses opposite lambda-dependent Boresch restraints for the two bound ligands.
Mapping artifacts are still generated for network scoring and review, and
``rbfe.network_scorer: auto`` switches to receptor-frame pocket-shape scoring by
default.

Default mapping algorithm
-------------------------

The default mapping starts as a star network:

* Ligands are taken in input order.
* The first ligand is used as the initial reference.
* Pairs are built as ``(lig1, lig2)``, ``(lig1, lig3)``, ...

This corresponds to ``RBFENetwork.default_mapping``. Unless you set
``rbfe.direction_policy: preserve`` or provide ``rbfe.mapping_file``, BATTER then
orients generated edges with ``rbfe.direction_policy``; the default
``larger_volume`` policy uses the larger grid-volume ligand as the reference.

Mapping options and precedence
------------------------------

RBFE mapping is controlled by ``rbfe`` in ``run.yaml``:

* ``rbfe.mapping_file``
* ``rbfe.mapping`` (default ``default``)
* ``rbfe.atom_mapping_file`` for optional per-pair atom mapping overrides
* ``rbfe.network_scorer`` (default ``auto``)
* ``rbfe.direction_policy`` (default ``larger_volume``)
* ``rbfe.minimal_mapping_atom`` (default ``3``)
* ``rbfe.add_atom_mapping_edges`` (default ``false``)

If both are provided, BATTER uses ``mapping_file``.

Supported ``rbfe.mapping`` values
---------------------------------

* ``default`` (also ``star`` / ``first`` aliases)
* ``konnektor``

When using ``konnektor``, you can optionally set ``rbfe.konnektor_layout``.
``rbfe.network_scorer`` controls how Konnektor scores candidate edges:
``auto`` resolves to LoMap scoring for standard RBFE and pocket-shape scoring
for ``rbfe_septop``; explicit values include ``lomap``, ``shape_difference`` and
``pocket_shape``.

.. code-block:: yaml

   rbfe:
     mapping: konnektor
     atom_mapper: kartograf
     konnektor_layout: star
     network_scorer: auto
     direction_policy: larger_volume
     both_directions: false

.. _rbfe_atom_mapper_options:

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
  the same Kartograf settings except ``atom_map_hydrogens=True`` unless
  ``rbfe.kartograf`` overrides it.

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
     lomap:
       time: 7
       max3d: 2.0
       shift: false
     kartograf:
       atom_max_distance: 1.1
       allow_bond_breaks: true
       filter_element_changes: false

Only the selected backend's options are used for a run. The available Kartograf
keys are ``atom_max_distance``, ``map_exact_ring_matches_only``,
``allow_partial_fused_rings``, ``allow_bond_breaks``,
``filter_element_changes``, and ``filter_mismatched_attached_h_count``.
BATTER does not expose Kartograf's hydrogen-mapping toggles in YAML because the
AMBER setup path relies on the previous fixed behavior: network planning uses
``atom_map_hydrogens=False``, RBFE transformation setup uses
``atom_map_hydrogens=True``, and both use
``map_hydrogens_on_hydrogens_only=True``.
The available LoMap keys are ``time``, ``threed``, ``max3d``,
``element_change``, and ``shift``.

Atom mapping override files
---------------------------

Set ``rbfe.atom_mapping_file`` to provide atom mappings for selected ligand
pairs while leaving all uncovered pairs on the configured ``rbfe.atom_mapper``.
BATTER uses the override while Konnektor scores candidate network edges and
writes the same prepared ``mapping.json`` for later transformation setup.
When you choose to use manual atom mappings, it is recommended to keep all
ligand-pair atom mappings you intend to rely on in ``rbfe.atom_mapping_file`` so
network scoring and simulation setup use the same curated mapping source.
If an override pair is valid but neither direction appears in the planned
network, BATTER leaves it unused by default and logs a warning. Set
``rbfe.add_atom_mapping_edges: true`` to append those valid override pairs as
extra planned edges.

The simplest JSON/YAML format maps pair labels to atom index maps:

.. code-block:: json

   {
     "LIG1~LIG2": {"0": 0, "1": 4, "2": 5}
   }

Pair maps are interpreted as ``componentB_to_componentA``:
target/alternate atom index to reference atom index, using 0-based atom indices.
For ``LIG1~LIG2`` above, keys are atoms in ``LIG2`` and values are atoms in
``LIG1``. If BATTER later needs ``LIG2~LIG1``, the map is inverted
automatically.

Structured entries are also accepted:

.. code-block:: yaml

   pairs:
     - pair: LIG1~LIG2
       componentB_to_componentA:
         0: 0
         1: 4
     - ref: LIG3
       alt: LIG4
       componentA_to_componentB:
         2: 7
         3: 8

``componentA_to_componentB`` / ``reference_to_target`` maps are inverted during
loading so the stored prepared artifact remains in BATTER's
``componentB_to_componentA`` orientation.

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
* ``executions/<run_id>/artifacts/config/rbfe_network.html``
* ``executions/<run_id>/artifacts/config/rbfe_network.png`` when Konnektor can
  render a static plot
* ``executions/<run_id>/artifacts/config/rbfe_mappings/<LIG1~LIG2>/mapping.json``
* ``executions/<run_id>/artifacts/config/rbfe_mappings/<LIG1~LIG2>/mapping.pkl``
* ``executions/<run_id>/artifacts/config/rbfe_mappings/<LIG1~LIG2>/mapping.png``

Open ``rbfe_network.html`` before production windows are submitted to inspect the
planned ligand graph and atom-mapping images in the same interactive style as the
Cinnabar dashboard. Use ``--only-rbfe-network`` or
``run.only_rbfe_network: true`` if you want BATTER to stop immediately after
writing these network artifacts.

The HTML view is the primary network-review artifact:

* Pan or zoom the graph with the mouse, trackpad, or the ``+`` / ``-`` /
  ``Fit`` / ``Reset`` controls.
* Click a ligand node to open a draggable note with the ligand label and 2D
  structure when RDKit rendering succeeds.
* Click an edge label or edge path to inspect transformation index, direction,
  mapper, mapped-atom count, atom-mapping image, and scores.
* Reverse-direction pairs are collapsed into one display edge, but each
  directed transformation remains listed in the edge note.
* Edges are colored by network redundancy by default. The ``Edge color``
  selector can switch to available Kartograf mapping metrics such as RMSD
  score, mapped-atom ratio, volume ratio, shape mismatch, and shape overlap.
  Missing optional metrics are simply absent from the selector.

By default, BATTER keeps duplicate ligands in RBFE network planning. Set
``rbfe.skip_duplicate_ligands: true`` to omit later ligands with identical
molecular identity before the network is planned. Full atom or full heavy-atom
mappings are retained as normal edges and recorded as coverage metadata in the
per-edge mapping status and network JSON. Skipped identical ligands are recorded
in ``rbfe_network.json`` as skip metadata.

Transformation systems are created under:

* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/``

For each transformation pair, BATTER copies the prepared atom-mapping artifacts into:

* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/fe/x/x-1/mapping.json``
* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/fe/x/x-1/mapping.pkl``
* ``executions/<run_id>/simulations/transformations/<LIG1~LIG2>/fe/x/x-1/mapping.png``

These generic ``mapping.*`` filenames are used for both ``kartograf`` and
``lomap`` atom-mapper backends.

Pipeline notes
--------------

In RBFE, BATTER runs ``prepare_rbfe`` after ligand parametrization and before
``prepare_equil``. This stage resolves the network and atom mappings once under
``artifacts/config``. Later transformation setup copies those prepared mapping
artifacts instead of generating new atom maps.
