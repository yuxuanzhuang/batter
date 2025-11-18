import shutil
import pytest

if shutil.which("antechamber") is None:
    pytest.skip("AmberTools `antechamber` not available; skipping heavy integration tests", allow_module_level=True)
