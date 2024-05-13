try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

from .opacity import *  # noqa
from .chemistry import *  # noqa
from .transmission import *  # noqa
