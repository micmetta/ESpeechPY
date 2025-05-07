import os
import sys

from distutils2.tests import unittest


here = os.path.dirname(__file__) or os.curdir

def test_suite():
    suite = unittest.TestSuite()
    for fn in os.listdir(here):
        if fn.startswith("test") and fn.endswith(".py"):
            modname = "distutils2._backport.tests." + fn[:-3]
            __import__(modname)
            module = sys.modules[modname]
            suite.addTest(module.test_suite())
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
