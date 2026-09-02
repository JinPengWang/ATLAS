"""Utilities for expanding CEC suite group names to individual problem names."""

from __future__ import annotations

from typing import List

CEC_SUITE_GROUPS = {
    "cec2005": [f"cec2005_f{i}" for i in range(1, 26)],   # F1-F25
    "cec2013": [f"cec2013_f{i}" for i in range(1, 29)],   # F1-F28
    "cec2014": [f"cec2014_f{i}" for i in range(1, 31)],   # F1-F30
    "cec2017": [f"cec2017_f{i}" for i in range(1, 31) if i != 2],  # F1,F3-F30 (F2 excluded by CEC2017 organizers)
    "cec2019": [f"cec2019_f{i}" for i in range(1, 11)],   # F1-F10
    "cec2020": [f"cec2020_f{i}" for i in range(1, 11)],   # F1-F10
    "cec2022": [f"cec2022_f{i}" for i in range(1, 13)],   # F1-F12
}

# Supported dimensions for each CEC suite (from opfunu)
CEC_SUITE_DIMS = {
    "cec2005": [10, 30, 50],
    "cec2013": [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "cec2014": [10, 20, 30, 50, 100],
    "cec2017": [2, 10, 20, 30, 50, 100],
    "cec2019": [9, 10, 16, 18],  # F1=9, F2=16, F3=18, F4-F10=10
    "cec2020": [2, 5, 10, 15, 20, 30, 50, 100],
    "cec2022": [2, 10, 20],
}


def expand_problem_names(names: List[str]) -> List[str]:
    """Expand suite group names to individual problem names.

    Names that match a CEC suite key (e.g. ``"cec2017"``) are expanded to
    all functions in that suite.  Other names are kept as-is.

    Args:
        names: List of problem or suite names.

    Returns:
        Expanded list of individual problem names.

    Examples::

        >>> expand_problem_names(["cec2017"])
        ["cec2017_f1", "cec2017_f3", ..., "cec2017_f30"]

        >>> expand_problem_names(["sphere", "cec2022"])
        ["sphere", "cec2022_f1", ..., "cec2022_f12"]
    """
    expanded: List[str] = []
    for name in names:
        key = name.lower().strip()
        if key in CEC_SUITE_GROUPS:
            expanded.extend(CEC_SUITE_GROUPS[key])
        else:
            expanded.append(name)
    return expanded


def get_supported_dims(suite_name: str) -> List[int]:
    """Get supported dimensions for a CEC suite.

    Args:
        suite_name: Suite name (e.g. ``"cec2017"``).

    Returns:
        List of supported dimensions, or empty list if not a CEC suite.

    Examples::

        >>> get_supported_dims("cec2017")
        [2, 10, 20, 30, 50, 100]

        >>> get_supported_dims("sphere")
        []
    """
    return CEC_SUITE_DIMS.get(suite_name.lower().strip(), [])


def get_all_supported_dims(problem_names: List[str]) -> List[int]:
    """Get all supported dimensions across multiple problems.

    For CEC suite names, returns the intersection of supported dimensions.
    For individual problems, extracts the suite name and looks it up.
    For non-CEC problems, includes common dimensions [10, 30, 50, 100].

    Args:
        problem_names: List of problem or suite names.

    Returns:
        Sorted list of dimensions supported by all problems.

    Examples::

        >>> get_all_supported_dims(["cec2017"])
        [2, 10, 20, 30, 50, 100]

        >>> get_all_supported_dims(["cec2017", "cec2022"])
        [2, 10, 20]
    """
    all_dims = None
    for name in problem_names:
        key = name.lower().strip()
        # Extract suite name from individual function name (e.g. cec2017_f1 -> cec2017)
        if "_f" in key:
            suite = key.split("_f")[0]
        else:
            suite = key
        dims = CEC_SUITE_DIMS.get(suite)
        if dims is not None:
            if all_dims is None:
                all_dims = set(dims)
            else:
                all_dims &= set(dims)
    if all_dims is None:
        return [10, 30, 50, 100]
    return sorted(all_dims)
