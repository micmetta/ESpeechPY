"""Call pytest from a distutils setup.py script.
"""

import sys
import os
import os.path
from distutils.core import Command
from distutils.spawn import spawn

__version__ = "0.1"


def _inject_distutils_command():
    """Inject this module into the distutils.command package.

    This is needed because distutils.dist.Distribution searches
    commands by trying to import the respective module from this
    package.
    """
    import distutils.command
    mod = sys.modules[__name__]
    cmds = ['build_test', 'test']
    for c in cmds:
        sys.modules['distutils.command.%s' % c] = mod
        setattr(distutils.command, c, mod)
    i = distutils.command.__all__.index('clean')
    distutils.command.__all__[i:i] = cmds

_inject_distutils_command()


class _tmpchdir:
    """Temporarily change the working directory.
    """
    def __init__(self, wdir):
        self.savedir = os.getcwd()
        self.wdir = wdir
    def __enter__(self):
        os.chdir(self.wdir)
        return os.getcwd()
    def __exit__(self, type, value, tb):
        os.chdir(self.savedir)


class build_test(Command):
    """Dummy.  This command is called at the beginning of test after
    build.  It does nothing, but it can be overridden by custom code
    in setup.py to build the test environment.
    """
    description = "set up test environment"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        pass


class test(Command):

    description = "run the tests"
    user_options = [
        ('build-lib=', None, "build directory for modules"),
        ('build-scripts=', None, "build directory for scripts"),
        ('skip-build', None,
         "skip rebuilding everything (for testing/debugging)"),
        ('test-args=', None, "extra arguments to pass to pytest"),
    ]
    boolean_options = ['skip-build']

    def initialize_options(self):
        self.build_lib = None
        self.build_scripts = None
        self.skip_build = 0
        self.test_args = None

    def finalize_options(self):
        self.set_undefined_options('build', 
                                   ('build_lib', 'build_lib'), 
                                   ('build_scripts', 'build_scripts'))

    def run(self):
        if not self.skip_build:
            self.run_command('build')
            self.run_command('build_test')

        # Add build_lib to the module search path to make sure the
        # built packages can be imported by the tests.  Manipulate
        # both, sys.path to affect the current running Python, and
        # os.environ['PYTHONPATH'] to affect subprocesses spawned by
        # the tests.
        build_lib = os.path.abspath(self.build_lib)
        sys.path.insert(0,build_lib)
        try:
            # if PYTHONPATH is already set, prepend build_lib.
            os.environ['PYTHONPATH'] = "%s:%s" % (build_lib,
                                                  os.environ['PYTHONPATH'])
        except KeyError:
            # no, PYTHONPATH was not set.
            os.environ['PYTHONPATH'] = build_lib

        # Set build_scripts in the environment so that tests are able
        # to find and execute them.
        build_scripts = os.path.abspath(self.build_scripts)
        os.environ['BUILD_SCRIPTS_DIR'] = build_scripts

        # Do not create byte code during test.
        sys.dont_write_bytecode = True
        os.environ['PYTHONDONTWRITEBYTECODE'] = "1"

        # Must change the directory, otherwise modules in the cwd
        # would override the one from build_lib.  Alas, there seem to
        # be no way to tell Python not to put the cwd in front of
        # $PYTHONPATH in sys.path.
        testcmd = [sys.executable, "-m", "pytest"]
        if self.test_args:
            testcmd.extend(self.test_args.split())
        if self.dry_run:
            testcmd.append("--collect-only")
        with _tmpchdir("tests"):
            spawn(testcmd)
