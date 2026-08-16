"""Public SDK façade (K15).

Provides the documented import path ``from shortchain.sdk import ShortChain``
without making ``shortchain.runtime`` the user-facing name.
"""

from shortchain.runtime.sdk import ShortChain

__all__ = ["ShortChain"]