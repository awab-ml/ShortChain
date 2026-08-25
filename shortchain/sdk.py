"""Public SDK façade.

Provides the documented import path ``from shortchain.sdk import ShortChain``
without making ``shortchain.telemetry`` the user-facing name.
"""

from shortchain.telemetry.sdk import ShortChain

__all__ = ["ShortChain"]