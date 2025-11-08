================
Execution Model
================

BATTER's orchestration layer is composed of three lightweight building blocks:
pipelines, runtime artifact stores, and system descriptors. Understanding how
they interact helps when customising workflows or writing new handlers.

Pipelines
=========

``batter.pipeline.pipeline.Pipeline`` stores an acyclic list of
:class:`~batter.pipeline.step.Step` objects plus useful introspection helpers:

* :meth:`Pipeline.describe` returns a serialisable summary with step names,
  dependencies, and payload types.
* :meth:`Pipeline.adjacency` exposes the DAG in adjacency-list form, which is
  handy when rendering graphs or verifying dependency structure in tests.
* :meth:`Pipeline.dependencies` lets you query the prerequisites for any step,
  mirroring the :attr:`Step.requires` metadata.

Because :class:`Step` instances are immutable dataclasses, call
:meth:`Step.replace` to clone a step with a modified payload.

Runtime Artifacts
=================

``batter.runtime.portable.ArtifactStore`` manages portable outputs (restarts,
analysis tables, etc.) and keeps a manifest in JSON form for reproducibility.
Use :meth:`ArtifactStore.list_artifacts` to inspect what has been stored, filter
by name prefix (for example ``prefix="fe/"``), or limit results to files versus
directories. The manifest is shared across rebased views so you can relocate a
run's artifacts to another filesystem while preserving metadata.

System Descriptors
==================

``batter.systems.core.SimSystem`` encapsulates the layout of a simulation on
disk. The helper methods introduced here simplify everyday tasks:

* :meth:`SimSystem.path` joins ``root`` with arbitrary path fragments, reducing
  boilerplate when locating artifacts.
* :meth:`SimSystem.with_meta` merges updates into the attached
  :class:`SystemMeta` object, keeping provenance and ligand-specific metadata in
  sync without mutating the original descriptor.

When building or transforming systems, prefer :meth:`SimSystem.with_artifacts`
for structural changes (new topology/coordinate paths) and
:meth:`SimSystem.with_meta` for contextual tweaks (ligand IDs, run labels, etc.).
