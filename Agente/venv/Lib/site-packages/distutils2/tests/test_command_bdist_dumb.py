"""Tests for distutils.command.bdist_dumb."""

import os
import imp
import sys
import zipfile
import distutils2.util

from distutils2.dist import Distribution
from distutils2.command.bdist_dumb import bdist_dumb
from distutils2.tests import unittest, support
from distutils2.tests.support import requires_zlib


class BuildDumbTestCase(support.TempdirManager,
                        support.LoggingCatcher,
                        unittest.TestCase):

    def setUp(self):
        super(BuildDumbTestCase, self).setUp()
        self.old_location = os.getcwd()

    def tearDown(self):
        os.chdir(self.old_location)
        distutils2.util._path_created.clear()
        super(BuildDumbTestCase, self).tearDown()

    @requires_zlib
    def test_simple_built(self):

        # let's create a simple package
        tmp_dir = self.mkdtemp()
        pkg_dir = os.path.join(tmp_dir, 'foo')
        os.mkdir(pkg_dir)
        self.write_file((pkg_dir, 'foo.py'), '#')
        self.write_file((pkg_dir, 'MANIFEST.in'), 'include foo.py')
        self.write_file((pkg_dir, 'README'), '')

        dist = Distribution({'name': 'foo', 'version': '0.1',
                             'py_modules': ['foo'],
                             'home_page': 'xxx', 'author': 'xxx',
                             'author_email': 'xxx'})
        os.chdir(pkg_dir)
        cmd = bdist_dumb(dist)

        # so the output is the same no matter
        # what is the platform
        cmd.format = 'zip'

        cmd.ensure_finalized()
        cmd.run()

        # see what we have
        dist_created = os.listdir(os.path.join(pkg_dir, 'dist'))
        base = "%s.%s.zip" % (dist.get_fullname(), cmd.plat_name)
        if os.name == 'os2':
            base = base.replace(':', '-')

        self.assertEqual(dist_created, [base])

        # now let's check what we have in the zip file
        fp = zipfile.ZipFile(os.path.join('dist', base))
        try:
            contents = fp.namelist()
        finally:
            fp.close

        if sys.version_info[:2] == (3, 1):
            pyc = 'foo.pyc'
        else:
            pyc = 'foo.%s.pyc' % imp.get_tag()
        contents = sorted(os.path.basename(fn) for fn in contents)
        wanted = ['foo.py', pyc,
                  'METADATA', 'INSTALLER', 'REQUESTED', 'RECORD']
        self.assertEqual(contents, sorted(wanted))

    def test_finalize_options(self):
        pkg_dir, dist = self.create_dist()
        os.chdir(pkg_dir)
        cmd = bdist_dumb(dist)
        self.assertEqual(cmd.bdist_dir, None)
        cmd.finalize_options()

        # bdist_dir is initialized to bdist_base/dumb if not set
        base = cmd.get_finalized_command('bdist').bdist_base
        self.assertEqual(cmd.bdist_dir, os.path.join(base, 'dumb'))

        # the format is set to a default value depending on the os.name
        default = cmd.default_format[os.name]
        self.assertEqual(cmd.format, default)


def test_suite():
    return unittest.makeSuite(BuildDumbTestCase)

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
