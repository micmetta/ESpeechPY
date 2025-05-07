"""Run the project's test suite."""

import os
import sys
import logging
import unittest

from distutils2 import logger
from distutils2.command.cmd import Command
from distutils2.database import get_distribution
from distutils2.errors import PackagingOptionError
from distutils2.util import resolve_name


class test(Command):

    description = "run the project's test suite"

    user_options = [
        ('suite=', 's',
         "test suite to run (for example: 'some_module.test_suite')"),
        ('runner=', None,
         "test runner to be called."),
        ('tests-require=', None,
         "list of distributions required to run the test suite."),
    ]

    def initialize_options(self):
        self.suite = None
        self.runner = None
        self.tests_require = []

    def finalize_options(self):
        self.build_lib = self.get_finalized_command("build").build_lib
        for requirement in self.tests_require:
            if get_distribution(requirement) is None:
                logger.warning("test dependency %s is not installed, "
                               "tests may fail", requirement)
        if (not self.suite and not self.runner and
            self.get_ut_with_discovery() is None):
            raise PackagingOptionError(
                "no test discovery available, please give a 'suite' or "
                "'runner' option or install unittest2")

    def get_ut_with_discovery(self):
        if hasattr(unittest.TestLoader, "discover"):
            return unittest
        else:
            try:
                import unittest2
                return unittest2
            except ImportError:
                return None

    def run(self):
        prev_syspath = sys.path[:]
        try:
            # build release
            build = self.reinitialize_command('build')
            self.run_command('build')
            sys.path.insert(0, build.build_lib)

            # XXX maybe we could pass the verbose argument of pysetup here
            logger = logging.getLogger('distutils2')
            verbose = logger.getEffectiveLevel() >= logging.DEBUG
            verbosity = verbose + 1

            # run the tests
            if self.runner:
                resolve_name(self.runner)()
            elif self.suite:
                runner = unittest.TextTestRunner(verbosity=verbosity)
                runner.run(resolve_name(self.suite)())
            elif self.get_ut_with_discovery():
                ut = self.get_ut_with_discovery()
                test_suite = ut.TestLoader().discover(os.curdir)
                runner = ut.TextTestRunner(verbosity=verbosity)
                runner.run(test_suite)
        finally:
            sys.path[:] = prev_syspath
