"""Tests for distutils2.run."""

import os
import sys
import textwrap
from io import StringIO

from distutils2 import install
from distutils2.tests import unittest, support
from distutils2.run import main

from distutils2.tests.support import assert_python_ok, assert_python_failure

# setup script that uses __file__
setup_using___file__ = """\
__file__

from distutils2.run import setup
setup()
"""

setup_prints_cwd = """\
import os
print os.getcwd()

from distutils2.run import setup
setup()
"""


class RunTestCase(support.TempdirManager,
                  support.LoggingCatcher,
                  unittest.TestCase):

    maxDiff = None

    # TODO restore the tests removed six months ago and port them to pysetup

    def test_install(self):
        # making sure install returns 0 or 1 exit codes
        project = os.path.join(os.path.dirname(__file__), 'package.tgz')
        install_path = self.mkdtemp()
        old_get_path = install.get_path
        install.get_path = lambda path: install_path
        old_mod = os.stat(install_path).st_mode
        os.chmod(install_path, 0)
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            self.assertFalse(install.install(project))
            self.assertEqual(main(['install', 'blabla']), 1)
        finally:
            sys.stderr = old_stderr
            os.chmod(install_path, old_mod)
            install.get_path = old_get_path

    def get_pythonpath(self):
        pythonpath = os.environ.get('PYTHONPATH')
        d2parent = os.path.dirname(os.path.dirname(__file__))  # XXX buggy
        if pythonpath is not None:
            pythonpath = os.pathsep.join((pythonpath, d2parent))
        else:
            pythonpath = d2parent
        return pythonpath

    def call_pysetup(self, *args):
        _, out, err = assert_python_ok('-m', 'distutils2.run', *args,
                                       PYTHONPATH=self.get_pythonpath())
        return out, err

    def call_pysetup_fail(self, *args):
        _, out, err = assert_python_failure('-m', 'distutils2.run', *args,
                                            PYTHONPATH=self.get_pythonpath())
        return out, err

    def test_show_help(self):
        # smoke test, just makes sure some help is displayed
        out, err = self.call_pysetup('--help')
        self.assertGreater(out, b'')
        self.assertEqual(err, b'')

    def test_list_commands(self):
        out, err = self.call_pysetup('run', '--list-commands')
        # check that something is displayed
        self.assertGreater(out, b'')
        self.assertEqual(err, b'')

        # make sure the manual grouping of commands is respected
        check_position = out.find(b'  check: ')
        build_position = out.find(b'  build: ')
        self.assertTrue(check_position, out)  # "out" printed as debugging aid
        self.assertTrue(build_position, out)
        self.assertLess(check_position, build_position, out)

    # TODO test that custom commands don't break --list-commands

    def test_unknown_command_option(self):
        out, err = self.call_pysetup_fail('run', 'build', '--unknown')
        self.assertGreater(out, b'')
        # sadly this message comes straight from the getopt module and can't be
        # modified to use repr instead of str for the unknown option; to be
        # changed when the command line parsers are replaced by something clean
        self.assertEqual(err.splitlines(),
                         [b'error: option --unknown not recognized'])

    def test_invalid_command(self):
        out, err = self.call_pysetup_fail('run', 'com#mand')
        self.assertGreater(out, b'')
        self.assertEqual(err.splitlines(),
                         [b"error: invalid command name 'com#mand'"])

    def test_unknown_command(self):
        out, err = self.call_pysetup_fail('run', 'invalid_command')
        self.assertGreater(out, b'')
        self.assertEqual(err.splitlines(),
                         [b"error: command 'invalid_command' not recognized"])

    def test_unknown_action(self):
        out, err = self.call_pysetup_fail('invalid_action')
        self.assertGreater(out, b'')
        self.assertEqual(err.splitlines(),
                         [b"error: action 'invalid_action' not recognized"])

    def test_setupcfg_parsing(self):
        # #14733: pysetup used to parse setup.cfg too late
        project_dir = self.mkdtemp()
        os.chdir(project_dir)
        custompy = textwrap.dedent(
            """\
            from distutils2.command.cmd import Command

            class custom(Command):

                user_options = []

                def initialize_options(self):
                    pass

                def finalize_options(self):
                    pass

                def run(self):
                    print('custom: ok')
            """)
        setupcfg = textwrap.dedent(
            """\
            [global]
            commands = custom.custom
            """)
        self.write_file('custom.py', custompy)
        self.write_file('setup.cfg', setupcfg)

        out, err = self.call_pysetup('run', 'custom')
        self.assertEqual(err.splitlines(), [b'running custom'])
        self.assertEqual(out.splitlines(), [b'custom: ok'])


def test_suite():
    return unittest.makeSuite(RunTestCase)

if __name__ == "__main__":
    unittest.main(defaultTest="test_suite")
