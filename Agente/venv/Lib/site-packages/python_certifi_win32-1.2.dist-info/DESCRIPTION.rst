====================
python-certifi-win32
====================

This package patches certifi at runtime to also include certificates from the windows certificate store.

This will allow packages such as requests (and tools based on it, like pip) to verify tls/ssl 
connections to servers who's ca is trusted by your windows install.

This module will automatically keep the merged certifi+windows cacerts file up to date, even when the
certifi module is updated.

Simply install with::

  pip install python-certifi-win32

and requests should trust your hosts if your host system does.


Acknowledgements
----------------
This module is inspired by a patch for python-certifi to implement this functionality: https://github.com/certifi/python-certifi/pull/54
The method of patching at runtime is built from the autowrapt module: https://pypi.python.org/pypi/autowrapt

