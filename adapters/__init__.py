"""Dataset adapter implementations.

Importing this package registers the built-in adapters with `AdapterRegistry`.
"""

from .wildlife import WildlifeAdapter  # noqa: F401
from .newsgroups import NewsgroupsAdapter  # noqa: F401
from .emotions import EmotionsAdapter  # noqa: F401
from .reuters import ReutersAdapter  # noqa: F401
from .webkb import WebKBAdapter  # noqa: F401
