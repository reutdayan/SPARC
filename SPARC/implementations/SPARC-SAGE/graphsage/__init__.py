from __future__ import print_function
from __future__ import division

import os
import sys

# Ensure ``from graphsage...`` works when the package is imported from any cwd.
_PACKAGE_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
