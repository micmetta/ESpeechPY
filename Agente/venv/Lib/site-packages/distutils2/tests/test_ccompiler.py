"""Tests for distutils.compiler.ccompiler."""

from distutils2.compiler import ccompiler
from distutils2.tests import unittest, support


class CCompilerTestCase(unittest.TestCase):
    pass  # XXX need some tests on CCompiler


def test_suite():
    return unittest.makeSuite(CCompilerTestCase)

if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
