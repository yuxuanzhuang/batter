Typed Pipeline Payloads and System Metadata
===========================================

This guide explains the structured data models that flow through the orchestrator,
pipeline steps, and execution handlers. The new payload and metadata classes live in
``batter/pipeline/payloads.py`` and ``batter/systems/core.py`` respectively.

Step Payloads
-------------

Each pipeline :class:`Step` now carries a :class:`~batter.pipeline.payloads.StepPayload`
instead of an untyped dictionary. The payload is a Pydantic model that exposes
attributes for the simulation configuration and shared system parameters while still
allowing arbitrary extras for backwards compatibility.

.. code-block:: python

   from batter.pipeline.payloads import StepPayload

   payload = StepPayload(
       sim=sim_config,                  # SimulationConfig instance
       sys_params=system_params,        # SystemParams instance
       extra_flag=True,                 # Additional arbitrary fields are allowed
   )

   payload.sim.temperature             # typed attribute access
   payload.sys_params.param_outdir     # system-level paths as Path objects
   payload["extra_flag"]               # dict-like access still works

   # Clone with modifications (returns a new StepPayload)
   payload = payload.copy_with(job_mgr=manager)

Execution handlers should begin by validating the incoming payload to support both
typed and legacy data:

.. code-block:: python

   from batter.pipeline.payloads import StepPayload

   def prepare_equil_handler(step, system, params):
       payload = StepPayload.model_validate(params)
       sim = payload.sim                  # SimulationConfig (required)
       sys_params = payload.sys_params    # SystemParams (optional)
       ...

System Parameters
-----------------

The :class:`~batter.pipeline.payloads.SystemParams` model wraps shared system-level
inputs (paths, force-field choices, extra restraints). It behaves like a mapping but
surface frequently accessed fields as attributes and ensures paths are converted to
``pathlib.Path`` instances.

.. code-block:: python

   from batter.pipeline.payloads import SystemParams
   from pathlib import Path

   sys_params = SystemParams(
       param_outdir=Path("work/params"),
       system_name="SYS",
       ligand_paths={"LIG1": Path("lig.sdf")},
       custom_value=5,
   )

   sys_params.param_outdir            # -> Path
   sys_params["custom_value"]         # dict-style access
   sys_params.get("missing", default) # safe lookup with default

SimSystem Metadata (SystemMeta)
--------------------------------

``SimSystem.meta`` is now a :class:`~batter.systems.core.SystemMeta` instance. The
class keeps track of common keys (ligand identifier, residue name, parameter
directories) and permits arbitrary extra values. Builders and orchestrator stages use
``SystemMeta.merge`` to propagate metadata to child systems without mutating the
source object.

.. code-block:: python

   from batter.systems.core import SimSystem, SystemMeta
   from pathlib import Path

   sys = SimSystem(name="SYS", root=Path("work"), meta={"ligand": "LIG1"})
   sys.meta.ligand        # -> "LIG1"

   updated = sys.with_artifacts(meta=sys.meta.merge(residue_name="LIG"))
   updated.meta.residue_name  # -> "LIG"

Handlers should prefer ``system.meta.get("key", default)`` instead of assuming a plain
dictionary.

Migration Notes
---------------

* ``step.params`` remains as an alias to ``step.payload`` for compatibility.
* When creating new steps, pass :class:`StepPayload` and :class:`SystemParams`
  instances instead of raw dictionaries.
* Always call ``StepPayload.model_validate`` inside handlers so older manifests that
  still store dictionaries can be replayed safely.
* Use ``SystemMeta.merge(...)`` whenever constructing child systems so metadata
  such as ligand identifiers and parameter directories move forward intact.
