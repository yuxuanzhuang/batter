AMBER GPU Compilation Notes
===========================

These notes document the local AMBER GPU patches used for BATTER runs.  Apply
the AMBER26 ``netfrc`` patch first because it affects both NVIDIA CUDA and AMD
HIP GPU GTI/TI runs.  The AMD section below is limited to GTI/HIP runtime
stability patches.

AMBER26 ``netfrc`` Patch
------------------------

This is an AMBER26 GPU GTI/TI issue and is not AMD-only.  It can affect both
NVIDIA CUDA and AMD HIP builds.

``netfrc`` is the PME net-force correction switch in the ``&ewald`` namelist:

* ``netfrc = 1`` removes the average nonbonded/PME force offset;
* ``netfrc = 0`` leaves the forces exactly as computed.

AMBER24 defaulted to ``netfrc = 1`` for MD runs whenever ``imin = 0``.  AMBER26
changed the default so that restrained runs with ``ntr = 1`` use
``netfrc = 0``:

.. code-block:: fortran

   if (imin .eq. 0 .and. ntr .eq. 0) then
     netfrc = 1
   else
     netfrc = 0
   end if

For GPU GTI softcore/TI, this can destabilize the force finalization path.  In
the failing BATTER case, the run eventually reported an illegal memory access
while copying the 42-term energy buffer, but the trigger was the AMBER26
``ntr = 1`` default selecting ``netfrc = 0``.

The local AMBER26 patch is in:

``pmemd26_src/src/pmemd/src/mdin_ewald_dat.F90``

It keeps the AMBER26 default for non-GTI and non-TI restrained runs, but restores
the AMBER24-style MD default for CUDA/HIP GTI TI runs:

.. code-block:: fortran

   if (netfrc .lt. 0) then
   #if defined(CUDA) && defined(GTI)
     ! GPU GTI force finalization is sensitive to disabling the PME net-force
     ! correction.  Keep the normal MD default for TI even when ntr is set.
     if (imin .eq. 0 .and. (ntr .eq. 0 .or. icfe .ne. 0)) then
   #else
     if (imin .eq. 0 .and. ntr .eq. 0) then
   #endif
       netfrc = 1
     else
       netfrc = 0
     end if
   end if

The warning for ``netfrc == 1`` with restraints is also narrowed so GPU GTI TI
does not warn for this intentional default, while minimization and non-TI
restrained runs still warn.

If using an unpatched AMBER26 build, add this namelist to affected BATTER input
files as a workaround:

.. code-block:: fortran

   &ewald
     netfrc = 1,
   /

AMD GPU GTI/HIP Runtime Patches
-------------------------------

These patches are for ROCm/HIP stability in GTI/TI runs.  The observed symptoms
were HSA memory aperture violations, illegal memory accesses, or follow-on
``hipGetDevice`` failures during TI/softcore simulations.

``gti_cuda.cu``
   In ``ik_BuildTINBList``, keep the launch shape for the GTI neighbor-list
   phases at:

   .. code-block:: c++

      threadsPerBlock = 128;
      blocksToUse = gpu->blocks;

   This matches the older working pmemd24 HIP behavior.  On Frontier/ROCm, the
   larger architecture-dependent launch shapes could fail later in kernels such
   as ``kCalculateTIKineticEnergy_kernel`` even though the original fault came
   from GTI neighbor-list construction.

``gti_general_kernels.cu``
   In ``vec_sync``, keep the ``combinedMode`` cases split explicitly.  The
   ROCm-sensitive case is ``combinedMode == 2``: copy ``a0`` into temporaries
   before writing ``a1``.

   .. code-block:: c++

      T vx = pVector[a0];
      T vy = pVector[a0 + cSim.stride];
      T vz = pVector[a0 + cSim.stride2];
      pVector[a1] = vx;
      pVector[a1 + cSim.stride] = vy;
      pVector[a1 + cSim.stride2] = vz;

   This preserves the intended ``V0 -> V1`` sync while avoiding the old shared
   branch that faulted in ``kgSyncVector_kernel`` under ROCm.

These AMD/HIP runtime patches should not be applied blindly to NVIDIA-specific
launch tuning.  The ``netfrc`` patch above is the cross-vendor AMBER26 fix; the
GTI/HIP launch and ``vec_sync`` changes are ROCm stability fixes.
