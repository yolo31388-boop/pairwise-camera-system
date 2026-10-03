"""Test-suite plumbing.

The generated test classes call unittest-style assertions
(``self.assertIsNotNone``) but do not inherit from ``unittest.TestCase``.
Mix the assertion helpers in at collection time so the tests run as
written, without modifying the test files.
"""

import unittest

_ASSERT_ATTRS = [name for name in dir(unittest.TestCase) if name.startswith("assert")]
_SUPPORT_ATTRS = ("failureException", "fail")


def pytest_pycollect_makeitem(collector, name, obj):
    if (
        isinstance(obj, type)
        and name.startswith("Test")
        and not issubclass(obj, unittest.TestCase)
    ):
        for attr in _ASSERT_ATTRS + list(_SUPPORT_ATTRS):
            if not hasattr(obj, attr):
                setattr(obj, attr, getattr(unittest.TestCase, attr))
    return None
