import os
import pytest

HEAVY_ENV = "BATTER_TEST_RUN_CLI"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "heavy: marks tests that are slow or resource-intensive (env: BATTER_TEST_RUN_CLI)"
    )


def pytest_collection_modifyitems(config, items):
    heavy_enabled = os.environ.get(HEAVY_ENV) == "1"
    for item in items:
        if "heavy" in item.keywords and not heavy_enabled:
            item.add_marker(
                pytest.mark.skip(reason="Set BATTER_TEST_RUN_CLI=1 to enable heavy CLI tests.")
            )
