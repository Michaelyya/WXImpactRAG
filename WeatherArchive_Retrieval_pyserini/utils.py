# -*- coding: utf-8 -*-
"""
Parity shim mirroring WeatherArchive_Retrieval/utils.py.

The Pyserini implementation keeps all helpers in ``pyserini_utils``; this module
re-exports the two names the original ``utils`` exposed (``BASE_ADDRESS`` and
``evaluate_retriever_performance``) so existing imports / call sites keep working.
"""

from WeatherArchive_Retrieval_pyserini.pyserini_utils import (  # noqa: F401
    BASE_ADDRESS,
    evaluate_retriever_performance,
)
