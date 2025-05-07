"""Tests for distutils2.extension."""
import os

from distutils2.compiler.extension import Extension
from distutils2.tests import unittest

class ExtensionTestCase(unittest.TestCase):

    pass

def test_suite():
    return unittest.makeSuite(ExtensionTestCase)

if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
