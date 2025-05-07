"""Distutils2 test suite runner."""

# Ripped from importlib tests, thanks Brett!

import os
from test.support import reap_children, reap_threads, run_unittest

from distutils2.tests import unittest


@reap_threads
def test_main():
    try:
        start_dir = os.path.dirname(__file__)
        top_dir = os.path.dirname(os.path.dirname(start_dir))
        test_loader = unittest.TestLoader()
        # XXX find out how to use unittest.main, to get command-line options
        # (failfast, catch, etc.)
        run_unittest(test_loader.discover(start_dir, top_level_dir=top_dir))
    finally:
        reap_children()


if __name__ == '__main__':
    test_main()
