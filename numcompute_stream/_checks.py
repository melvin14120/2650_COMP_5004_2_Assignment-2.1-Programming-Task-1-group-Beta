"""
_checks.py
Small validation helpers shared by the numcompute_stream modules.

Provides
--------
- _check_int(value, name, minimum)   an integer >= minimum, or a clear error

Design notes
------------
One copy of the check, so every module reports a bad hyperparameter with the
same wording. The integer test itself is the one from ``numcompute.utils``
(A1): ``bool`` does not count as an integer.
"""

from __future__ import annotations

from numcompute.utils import _is_integer


def _check_int(value, name: str, minimum: int) -> int:
    """Return ``value`` as ``int`` after checking it is an integer >= ``minimum``.

    Raises
    ------
    TypeError
        If ``value`` is not an integer (``bool`` and floats are rejected).
    ValueError
        If ``value`` is smaller than ``minimum``.
    """
    if not _is_integer(value):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)