==============================
Implementing Internal Builders
==============================

Per-step work inside BATTER (``prepare_equil``, ``prepare_fe``, etc.) is handled by
classes under ``batter/_internal``. To add or extend one:

1. **Subclass :class:`batter._internal.builders.base.BaseBuilder`** and implement the
   abstract hooks (``_build_complex``, ``_create_box``, ``_restraints``,
   ``_sim_files``, and ``_run_files``). Each hook receives a populated
   :class:`~batter._internal.builders.interfaces.BuildContext`.
2. **Reuse shared ops** (``batter/_internal/ops``) for Amber template rendering,
   restraint generation, and simulation-file patching instead of duplicating logic.
3. **Register the builder** if it maps to a specific component or stage using the
   decorators in :mod:`batter._internal.builders.fe_registry`.
4. **Wire handlers** – update the relevant exec handler (for example
   ``prepare_equil``) so it instantiates your builder when appropriate.
5. **Test in isolation** by synthesising a :class:`BuildContext` fixture and verifying
   the expected files appear under ``ctx.working_dir``.

Builders operate strictly inside the per-ligand working directory, which keeps them
easy to reason about. Following the template above ensures new stages integrate with
the existing handler and registry infrastructure.
