"""Import sweep over the light modules.

The heavy pipeline entrypoints (training/train.py, inference/*) pull the full
Wan stack and are covered by py_compile in the migration verification instead.
"""
import importlib

import pytest

MODULES = [
    "native_animation",
    "native_animation.data.build_metadata",
    "native_animation.data.sampling",
    "native_animation.data.extract_keyframes",
    "native_animation.modeling.native_flowmatch",
    "native_animation.evaluation.evaluate",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
