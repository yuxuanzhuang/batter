"""
Unit and regression test for the batter package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import batter


def test_batter_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "batter" in sys.modules
