Small Ligands and Ions
======================

This page describes the internal implementation for ligands that cannot define a
full Boresch ligand frame, such as monoatomic ions and two-heavy-atom molecules.

The feature spans three layers:

* parameterisation detects charged monoatomic ions and writes AMBER artifacts;
* build/restraint preparation permits partial ligand anchors only when the ligand
  is physically too small for a full frame;
* FE restraint preparation can add ion guard lower walls so bulk ions do not
  approach the bound or solvent ligand reference atoms during alchemical windows;
* ABFE z-window restraint preparation adds a bulk-ligand flat-bottom distance
  restraint between the first bound ligand atom and the first bulk ligand atom;
* analysis applies an analytical reduced external-restraint correction when
  ``disang.rest`` contains three or five ligand translation/rotation terms.

Parameterisation path
---------------------

``batter.param.ligand.LigandProcessing`` detects monoatomic ions immediately after
loading the RDKit molecule:

.. code-block:: text

   _is_charged_monoatomic_ion(mol)
   _formal_charge(mol)
   LigandProcessing._prepare_monoatomic_ion_parameters()

The detection rule is intentionally strict:

* ``mol.GetNumAtoms() == 1``
* ``mol.GetNumBonds() == 0``
* the atom formal charge is non-zero

When the rule matches, ``_calculate_partial_charge`` sets ``ligand_charge`` from
formal charge and returns without calling ``Molecule.assign_partial_charges``.
This avoids toolkit failures for elements that NAGL or antechamber charge models
do not support.

``_prepare_monoatomic_ion_parameters`` writes ``mol2`` and ``frcmod`` directly,
then runs ``tleap`` to emit ``lib``, ``prmtop``, ``inpcrd``, and ``pdb``. The
current built-in parameter table is ``_OPC_MONOATOMIC_ION_PARAMS``. Add new ions
there only when the mass, radius, and epsilon are appropriate for the intended
water model.

Anchor preparation path
-----------------------

Full Boresch restraints require three receptor anchors and three ligand anchors:
``P1/P2/P3`` and ``L1/L2/L3``. BATTER now allows fewer ligand anchors only under a
small-ligand guard.

The core helpers live in ``batter._internal.ops.restraints``:

.. code-block:: text

   _ligand_anchor_count(L1, L2, L3)
   _validate_ligand_anchor_set(...)
   _equil_anchor_restraint_expressions(...)
   _reduced_or_boresch_tr_expressions(...)

``_validate_ligand_anchor_set`` accepts partial ligand anchors only when:

* at least one ligand anchor exists;
* fewer than three ligand anchors exist;
* the ligand has fewer than three heavy atoms.

Otherwise it raises, because missing ``L2`` or ``L3`` on a larger ligand would
mean the Boresch frame was not constructed correctly.

For equilibration, ``_equil_anchor_restraint_expressions`` returns receptor
anchor restraints plus the ligand terms that can be defined. For a one-anchor
ligand it emits three ligand terms; for a two-anchor ligand it emits five; for a
normal ligand it emits six.

``batter._internal.ops.build_complex`` mirrors the same guard when reading the
prepared ``anchors*.txt/json`` files via ``_partial_ligand_anchors_are_expected``.
This keeps ABFE preparation from failing when a genuinely small ligand can only
define one or two ligand anchors.

For normal ligands with an automatically chosen L1, charged protein-ligand
contacts can promote a salt-bridge atom to L1. L2/L3 are then ranked to avoid
terminal atoms: valid ring atoms with at least two heavy neighbours are preferred
first, then atoms with more than two heavy neighbours, then other nonterminal
heavy atoms. The Boresch guard applies the same ranking after filtering out
near-planar receptor-ligand frames.

Ion guard path
--------------

``fe_sim.ion_guard`` defaults to ``"yes"`` and is resolved onto
``SimulationConfig.ion_guard``. The FE restraint writers for ABFE ``z`` and RBFE
``x`` components call ``_append_ion_guard_restraints`` after the normal
``disang.rest`` terms are written.

The helper reads ``full.pdb`` because the guarded ions are part of the full FE
topology. For ABFE it uses the first heavy atom in the first ligand residue as
the binding-site reference. For RBFE and RBFE-SEPTOP it first checks
``x-1/scmask.json`` and uses the explicit common-core site index when available,
then falls back to the first ligand residue. The ligand reference residue is
excluded from the ion list so a monoatomic ion ligand is not restrained against
itself.

Each configured ``cation``/``anion`` atom receives lower-wall distance restraints
tagged ``#Ion_Guard``:

.. code-block:: text

   r1 = 0.0
   r2 = 10.0
   r3 = 999.0
   r4 = 999.0
   rk2 = 10.0
   rk3 = 0.0

The tag is intentionally distinct from ``#Lig_TR`` so reduced/Boresch analytical
correction detection ignores these guard terms.

Bulk ligand restraint path
--------------------------

For ABFE z-windows with a bound ligand and a translated bulk-solvent ligand copy,
BATTER no longer appends the bulk ligand atom to the positional ``ATOM 1 2``
restraint in ``mdin-template``. Instead, ``disang.rest`` receives a separate
``#Bulk_Lig`` flat-bottom restraint between AMBER atom 2 in the binding-site
ligand and the first heavy atom in the bulk ligand copy:

.. code-block:: text

   &rst
     iat=-1,-1,
     r1=-999.0, r2=-3.0, r3=3.0, r4=999.0,
     rk2=10.0, rk3=10.0,
     igr1=2,0,
     igr2=<bulk first atom>,0,
   &end

The bulk dummy coordinate used during MC-water setup is placed at the first
bulk-ligand atom rather than at the bulk-ligand center of mass.

Analysis path
-------------

``batter.analysis.analysis.analyze_lig_task`` chooses the analytical correction
from tagged AMBER restraint lines in ``disang.rest``:

.. code-block:: text

   #Lig_TR        ABFE / single-site component
   #Lig_TR_REF    SEPTOP reference ligand
   #Lig_TR_ALT    SEPTOP alternate ligand

The helper ``_disang_restraint_tag_count`` counts exact trailing tags. The
selection rule is:

.. list-table::
   :header-rows: 1

   * - Tagged ligand TR terms
     - Analysis class
     - Result label
   * - ``>= 6``
     - ``BoreschAnalysis``
     - ``Boresch``, ``Boresch_REF``, or ``Boresch_ALT``
   * - ``3`` or ``5``
     - ``ReducedExternalRestraintAnalysis``
     - ``Reduced_TR``, ``Reduced_TR_REF``, or ``Reduced_TR_ALT``
   * - Other non-zero count
     - skipped with warning
     - none

``ReducedExternalRestraintAnalysis`` parses the ``r2`` reference values from the
tagged terms and integrates the restrained partition function numerically. For
one-anchor ligands, the integrated terms are distance, angle, and dihedral, with
standard-state denominator ``1660.0``. For two-anchor ligands, the second angle
and second dihedral are included and the denominator is multiplied by ``4π``.

The sign is still controlled by ``COMPONENT_DIRECTION_DICT``. ABFE release terms
use ``Reduced_TR: -1``; SEPTOP uses ``Reduced_TR_REF: -1`` and
``Reduced_TR_ALT: +1`` to match the opposing endpoint restraints.

Testing checklist
-----------------

When changing this code, run at least:

.. code-block:: bash

   python -m py_compile \
     batter/_internal/ops/box.py \
     batter/_internal/ops/build_complex.py \
     batter/_internal/ops/restraints.py \
     batter/analysis/analysis.py \
     batter/param/ligand.py

   pytest \
     tests/test_analysis_logging.py \
     tests/test_restraints_colvar_mirror.py \
     tests/test_box_helpers.py \
     tests/test_param_ligand.py \
     tests/test_orchestrate_run.py \
     tests/test_orchestrate_ligands.py \
     tests/test_sim_files_non_loop.py \
     -q

For real-workflow smoke tests, inspect:

* ``ligand_params/<hash>/lig.json`` for ``ligand_charge`` and ion artifacts;
* ``equil/anchors.txt`` and ``equil/anchors.json`` for partial ligand anchors;
* ``fe/<component>/<component>-1/disang.rest`` for three, five, or six
  ``#Lig_TR`` terms;
* ``fe/z/*/disang.rest`` or ``fe/x/*/disang.rest`` for ``#Ion_Guard`` terms when
  bulk ions are present and ``fe_sim.ion_guard`` is enabled;
* ``fe/Results/Results.dat`` for ``Reduced_TR`` or ``Boresch`` labels.
