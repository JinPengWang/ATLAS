"""Optional setup script for ATLAS.

Install in development mode::

    pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent

setup(
    name="atlas",
    version="0.2.0",
    description=(
        "All-in-one Toolkit for Learning and Applying metaheuristicS (ATLAS) – "
        "a modular metaheuristic algorithm research platform."
    ),
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="ATLAS Contributors",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "experiments", "results", "docs"]),
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.9",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "pandas>=1.5",
        "PyYAML>=6.0",
        "tqdm>=4.64",
        "tabulate>=0.9.0",
        "plotly>=5.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "ruff", "isort"],
        "cec": ["opfunu>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "atlas-benchmark=experiments.run_benchmark:main",
        ],
    },
)
