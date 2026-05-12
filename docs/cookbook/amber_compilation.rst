AMBER GPU Compilation Notes
===========================

These notes document the local AMBER GPU patches used for BATTER runs on
Frontier-style AMD/ROCm systems and for AMBER26 GPU GTI/TI stability.  They are
intended as a short checklist when rebuilding AMBER or comparing a new AMBER
release against the working local tree.

AMD HIP Build Patch
-------------------

The AMD build patch has two goals:

* build ``pmemd.hip`` for the AMD GPU architectures needed on the target
  systems, rather than hard-coding a single architecture;
* avoid CUDA-only source paths that fail or miscompile under HIP/ROCm.

The build scripts set the HIP targets through one environment variable:

.. code-block:: bash

   AMBER_AMDGPU_TARGETS="${AMBER_AMDGPU_TARGETS:-gfx803;gfx900;gfx906;gfx908;gfx90a}"

and pass it into CMake as both names used by different parts of the build:

.. code-block:: bash

   -D HIP=ON
   -D AMDGPU_TARGETS="${AMBER_AMDGPU_TARGETS}"
   -D GPU_TARGETS="${AMBER_AMDGPU_TARGETS}"
   -D HIP_RDC=ON
   -D VKFFT=ON
   -D GTI=TRUE

The relevant files are:

* ``pmemd26_src/compile_with_hip.sh``
* ``ambertools26_src/compile_with_hip.sh``
* ``build_amber_frontier.sh``
* ``pmemd26_src/src/pmemd/src/cuda/CMakeLists.txt``
* ``pmemd26_src/src/pmemd/src/xray/cuda/CMakeLists.txt``

The CMake patch sets ``HIP_ARCHITECTURES`` from ``GPU_TARGETS`` for both serial
and MPI CUDA/HIP objects.  This makes the compiled binaries contain code objects
for the requested AMD GPU families.  On Frontier, ``gfx90a`` is the key target;
the wider default list keeps the same source tree usable on older AMD systems.

HIP compatibility patches are also needed in CUDA-named source files:

* ``CudaWrapper.h`` and ``pba3DHost.cu`` include ``hip/hip_runtime.h`` plus
  ``hip_definitions.h`` when ``AMBER_PLATFORM_AMD`` is defined, instead of
  including ``cuda_runtime.h`` unconditionally.
* ``hip_definitions.h`` maps missing CUDA symbols such as ``cudaError``,
  ``cudaStreamCreate``, and ``cudaStreamDestroy`` onto HIP equivalents.
* ``kCalculateSA.cu`` chooses the correct Thrust execution policy:
  ``thrust::hip::par`` for AMD and ``thrust::cuda::par`` for NVIDIA.

GTI HIP Runtime Patch
---------------------

Some GPU GTI kernels are launch-shape sensitive under ROCm.  The observed
symptoms were HSA memory aperture violations, illegal memory accesses, or later
``hipGetDevice`` failures during TI/softcore simulations.

Two local GTI patches were kept:

``gti_cuda.cu``
   In ``ik_BuildTINBList``, force the two problematic neighbor-list phases to
   use:

   .. code-block:: c++

      threadsPerBlock = 128;
      blocksToUse = gpu->blocks;

   This matches the older working pmemd24 HIP behavior and avoids the failing
   launch shapes seen with larger architecture-dependent thread counts.

``gti_general_kernels.cu``
   In ``vec_sync``, split ``combinedMode`` handling into explicit cases.  The
   important ROCm-sensitive case is ``combinedMode == 2``: copy ``a0`` into
   temporaries before writing ``a1``.  This avoids a fault in
   ``kgSyncVector_kernel`` while preserving the intended ``V0 -> V1`` sync.

These are AMD/HIP stability patches.  The launch-shape patch should not be
applied blindly to unrelated kernels, and NVIDIA builds should be tested before
changing their tuned launch shapes.  The ``vec_sync`` split is logically
equivalent to the old code and is expected to be safe, but it was added because
of ROCm behavior.

AMBER26 ``netfrc`` Patch
------------------------

This is a separate AMBER26 GTI/TI issue and is not AMD-only.  It can affect both
NVIDIA CUDA and AMD HIP GPU GTI runs.

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
while copying the 42-term energy buffer, but the real trigger was the AMBER26
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

Validation
----------

The local patched build was validated by rebuilding with:

.. code-block:: console

   cmake --build build/amber --target install -j16

and rerunning a 100-step HIP GTI probe with ``ntr = 1``, ``icfe = 1``,
``ifsc = 1``, the real restraint GROUP block, and ``mcwat = 1``.  The output
reported ``netfrc = 1``, reached ``NSTEP = 100``, and did not produce NaNs or an
illegal memory access.
