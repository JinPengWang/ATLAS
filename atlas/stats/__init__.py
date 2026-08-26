"""Statistical analysis and academic table export suite for ATLAS."""

from atlas.stats.critical_difference import (
    calculate_critical_difference,
    nemenyi_post_hoc,
    plot_critical_difference,
)
from atlas.stats.hypothesis_testing import (
    FriedmanTestResult,
    PairwiseTestResult,
    calculate_average_ranks,
    friedman_test,
    wilcoxon_signed_rank_test,
)
from atlas.stats.table_exporter import (
    export_csv_summary,
    export_latex_table,
    export_markdown_table,
)

__all__ = [
    "wilcoxon_signed_rank_test",
    "friedman_test",
    "calculate_average_ranks",
    "PairwiseTestResult",
    "FriedmanTestResult",
    "calculate_critical_difference",
    "nemenyi_post_hoc",
    "plot_critical_difference",
    "export_latex_table",
    "export_markdown_table",
    "export_csv_summary",
]
