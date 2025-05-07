"""Support for packaging, distribution and installation of Python projects.

Third-party tools can use parts of distutils2 as building blocks
without causing the other modules to be imported:

    import distutils2.version
    import distutils2.metadata
    import distutils2.pypi.simple
    import distutils2.tests.pypi_server
"""

from logging import getLogger

__all__ = ['__version__', 'logger']

__version__ = "1.0a5.dev0"
logger = getLogger('distutils2')
