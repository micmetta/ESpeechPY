"""Tests for the distutils2.pypi.wrapper module."""


from distutils2.tests import unittest
from distutils2.pypi.wrapper import switch_index_if_fails


class Index1(object):
    def test(self):
        raise Exception("boo")


class Index2(object):
    def test(self):
        return 'OK'


class Indexes(object):
    _indexes = {'one': Index1(), 'two': Index2()}


class TestPyPIWrapper(unittest.TestCase):

    def test_wrapper(self):
        index = Indexes._indexes['one']
        func = switch_index_if_fails(index.test, Indexes)
        self.assertEqual(func(), 'OK')


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPyPIWrapper))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
