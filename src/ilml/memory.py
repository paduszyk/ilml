from __future__ import annotations

from joblib.memory import Memory

_memory = Memory("./.ilml_cache", verbose=0)

cache = _memory.cache
