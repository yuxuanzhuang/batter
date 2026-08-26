# Building BATTER's Documentation

BATTER's documentation is built with [Sphinx](https://www.sphinx-doc.org/). From
this directory, create the dedicated environment and build the HTML output:

```bash
conda env create -f requirements.yaml
conda activate docs_batter
make html
```

The generated site starts at `_build/html/index.html`. The source tree contains
the getting-started guide, workflow tutorials, cookbook, CLI/API references, and
developer guide.

Read the Docs uses the repository's existing `.readthedocs.yaml`, which points at
`docs/conf.py` and installs `docs/requirements.yaml`.
