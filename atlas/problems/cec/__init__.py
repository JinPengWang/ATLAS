"""CEC benchmark suite integration via ``opfunu``.

Requires the optional ``opfunu`` dependency (``pip install opfunu``).
"""

from atlas.problems.cec.cec2005 import register_cec2005
from atlas.problems.cec.cec2013 import register_cec2013
from atlas.problems.cec.cec2014 import register_cec2014
from atlas.problems.cec.cec2017 import register_cec2017
from atlas.problems.cec.cec2019 import register_cec2019
from atlas.problems.cec.cec2020 import register_cec2020
from atlas.problems.cec.cec2022 import register_cec2022


def register_all_cec() -> None:
    """Register all CEC benchmark functions."""
    register_cec2005()
    register_cec2013()
    register_cec2014()
    register_cec2017()
    register_cec2019()
    register_cec2020()
    register_cec2022()
