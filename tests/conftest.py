"""Fail fast if the tests are not exercising this checkout.

pyproject.toml's `pythonpath = ["."]` puts the repo root first on sys.path, so
`mars` and `marslib` should resolve here. The one case that still slips through
silently is a missing local marslib*.so (fresh clone, after `make clean`): the
local mars.py imports fine and its `import marslib` falls back to a
site-packages copy. Catch both modules explicitly.
"""

import os

import pytest

import mars
import marslib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    for mod in (mars, marslib):
        path = os.path.abspath(mod.__file__)
        if not path.startswith(ROOT + os.sep):
            raise pytest.UsageError(
                f"{mod.__name__} imported from {path}, not from this checkout "
                f"({ROOT}). Run `make` to build the local marslib, or uninstall "
                f"the site-packages copy (`pip uninstall mars`)."
            )
