"""Run spaceshuttle.py with non-interactive matplotlib backend."""

import matplotlib

matplotlib.use("Agg")
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
# Monkey-patch plt.show to be a no-op
import matplotlib.pyplot as _plt

_plt.show = lambda *a, **kw: None
exec(open(os.path.join(os.path.dirname(__file__), "spaceshuttle.py")).read())
