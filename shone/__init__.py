try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

from .chemistry import *  # noqa
from .opacity import *  # noqa
from .transmission import *  # noqa
from .spectrum import *  # noqa
