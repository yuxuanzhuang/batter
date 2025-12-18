=========================
Authoring Custom Pipelines
=========================

The orchestrator treats pipelines as declarative DAGs of
:class:`~batter.pipeline.step.Step` objects. Adding a new protocol therefore boils
down to composing steps and teaching :mod:`batter.orchestrate` how to select them.

1. **Write a factory** (see :mod:`batter.pipeline.factory`). Build a list of
   :class:`Step <batter.pipeline.step.Step>` instances using the helper ``_step`` so
   each node receives a :class:`~batter.pipeline.payloads.StepPayload`.
2. **Expose selection logic** via
   :func:`batter.orchestrate.pipeline_utils.select_pipeline`. Add a branch that
   calls your factory when ``protocol == "<name>"``.
3. **(Optional) Register an :class:`~batter.orchestrate.protocols.FEProtocol`** if
   you need protocol-specific validation or metadata. Look at
   :mod:`batter.orchestrate.protocol_impl` for reference.
4. **Update :class:`batter.config.run.RunConfig`** so the new protocol string is
   accepted and ``fe_sim`` is coerced into the appropriate Pydantic model.
5. **Test and document** – add a lightweight unit test that asserts the ordered step
   names and capture any new YAML options in :doc:`../configuration`.

Once these pieces are in place, :func:`batter.orchestrate.run.run_from_yaml` will
automatically schedule the new pipeline when the user selects your protocol.

Minimal factory example
=======================

.. code-block:: python

   # batter/pipeline/factory.py
   from batter.pipeline.pipeline import Pipeline
   from batter.pipeline.step import Step
   from batter.pipeline.payloads import StepPayload
   from batter.pipeline.factory import _step

   def make_custom_pipeline(sim, sys_params):
       payload = StepPayload(sim=sim, sys_params=sys_params)
       steps: list[Step] = [
           _step("prep_inputs", payload),
           _step("run_custom_md", payload, requires=["prep_inputs"]),
           _step("analyze_custom", payload, requires=["run_custom_md"]),
       ]
       return Pipeline(steps)

   # batter/orchestrate/pipeline_utils.py
   from batter.pipeline.factory import make_custom_pipeline

   def select_pipeline(protocol, sim_cfg, only_fe_prep=False, sys_params=None):
       ...
       if protocol == "custom":
           return make_custom_pipeline(sim_cfg, sys_params)
       ...

Keep step names terse and stable—tests typically assert the ordered names. When
adding a branch, ensure ``RunConfig`` accepts the protocol string and that the
factory handles ``only_fe_prep`` or other flags consistently with existing pipelines.
