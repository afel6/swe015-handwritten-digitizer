"""Test-session setup that must run before any test module imports transformers."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
