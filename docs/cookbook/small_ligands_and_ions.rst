Small Ligands and Ions
======================

BATTER supports ABFE and SEPTOP-style RBFE preparation for ligands that are too
small to define a full three-atom ligand Boresch frame. This includes
monoatomic ions such as ``[Na+]`` and two-heavy-atom ligands.

This support has two separate parts:

* monoatomic-ion parameterisation writes AMBER-compatible ligand artifacts
  without calling OpenFF/NAGL charge assignment;
* reduced external-restraint corrections replace the full six-term Boresch
  correction when the ligand has only one or two heavy atoms.

Monoatomic ion parameterisation
-------------------------------

Charged monoatomic ligands are detected from the input molecule: exactly one atom,
no bonds, and a non-zero formal charge. For these ligands BATTER uses the formal
charge directly and writes AMBER ion artifacts into the ligand-parameter cache.

The generated cache entry contains the same file set as a normal ligand:

.. code-block:: text

   lig.mol2
   lig.frcmod
   lig.lib
   lig.prmtop
   lig.inpcrd
   lig.pdb
   lig.sdf
   lig.json

The ion path is independent of the configured charge model. For example, this is
valid even though OpenFF NAGL cannot assign charges to sodium:

.. code-block:: yaml

   create:
     ligand_input: ligand_dict_na.json
     ligand_ff: openff-2.3.0
     param_charge: openff-gnn-am1bcc-1.0.0.pt
     water_model: OPC

Supported built-in ion parameter rows are OPC-compatible ``Li+``, ``Na+``,
``K+``, ``Cl-``, ``Mg2+``, and ``Ca2+``. Other monoatomic ions fail with a clear
error until their mass and nonbonded parameters are added to BATTER.

Reduced external-restraint correction
-------------------------------------

The standard Boresch correction needs three receptor anchors and three ligand
anchors. A one-atom ion or two-heavy-atom molecule cannot provide the missing
ligand anchors, so BATTER writes only the restraint terms that are physically
defined and analyzes them with ``Reduced_TR`` instead of ``Boresch``.

For a one-atom ligand, BATTER uses three ligand terms:

.. code-block:: text

   P1 - L1                 distance
   P2 - P1 - L1            angle
   P3 - P2 - P1 - L1       dihedral

These restrain the ligand as a point in the receptor frame: radius, polar angle,
and azimuth.

For a two-heavy-atom ligand, BATTER uses five ligand terms:

.. code-block:: text

   P1 - L1                 distance
   P2 - P1 - L1            angle
   P3 - P2 - P1 - L1       dihedral
   P1 - L1 - L2            angle
   P2 - P1 - L1 - L2       dihedral

The missing sixth Boresch torsion is rotation around the ligand bond axis; a
two-atom ligand has no third ligand atom to define that rotation.

In ``Results.dat`` the analytical line is labeled according to the number of
restraint terms found in ``disang.rest``:

.. code-block:: text

   Reduced_TR    ...
   z             ...
   Total         ...

For SEPTOP RBFE, the corresponding labels are ``Reduced_TR_REF`` and
``Reduced_TR_ALT``.

Practical guidance
------------------

Use more than endpoint lambdas for ion decoupling. A schedule such as
``[0.0, 1.0]`` is usually inadequate for charged ion transformations and can
produce effectively zero MBAR overlap even when the simulation completes.

``fe_sim.ion_guard`` is enabled by default for FE window generation. It writes
``#Ion_Guard`` flat-bottom lower-wall restraints in ``fe/z/*/disang.rest`` for
ABFE and ``fe/x/*/disang.rest`` for RBFE/RBFE-SEPTOP, keeping configured bulk
ions at least 10 Å from the binding-site ligand reference atom. This guard is
not applied to top-level equilibration. Disable it with
``fe_sim.ion_guard: no`` only when those ion-ligand close approaches are expected
for the system being modeled.

.. code-block:: yaml

   fe_sim:
     lambdas:
       - 0.0
       - 0.10565
       - 0.14832
       - 0.17842
       - 0.20427
       - 0.22603
       - 0.24433
       - 0.2625
       - 0.27917
       - 0.29575
       - 0.31188
       - 0.32772
       - 0.34359
       - 0.35892
       - 0.37497
       - 0.39086
       - 0.40584
       - 0.42001
       - 0.43453
       - 0.45056
       - 0.46585
       - 0.47913
       - 0.49085
       - 0.5015
       - 0.51177
       - 0.52289
       - 0.53652
       - 0.55271
       - 0.56723
       - 0.58061
       - 0.59453
       - 0.60923
       - 0.62468
       - 0.63991
       - 0.65631
       - 0.67222
       - 0.68487
       - 0.69656
       - 0.70897
       - 0.72458
       - 0.74158
       - 0.75893
       - 0.779
       - 0.79905
       - 0.82704
       - 0.85882
       - 0.9069
       - 1.0
     z_n_steps: 500000
     analysis_start_step: 100000

The reduced restraint correction is not an electrostatic finite-size correction.
For charged ligands, decide separately whether your protocol needs a Rocklin-style
or related charged-system correction and configure the FE analysis accordingly.

Troubleshooting
---------------

``Molecule contains forbidden element 11``
    The monoatomic-ion bypass was not used. Confirm your ``batter`` command imports
    the active BATTER checkout and that the ion input is a charged one-atom molecule.

``Boresch restraints require ligand anchors L1/L2/L3``
    BATTER allows reduced anchors only for ligands with fewer than three heavy atoms.
    A larger ligand must provide a complete three-anchor ligand frame.

``Reduced_TR`` is missing from ``Results.dat``
    Check ``fe/<component>/<component>-1/disang.rest``. BATTER chooses the correction
    from the number of ``#Lig_TR`` terms: six or more uses ``Boresch``; three or five
    uses ``Reduced_TR``.
