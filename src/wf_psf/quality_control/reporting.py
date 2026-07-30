"""Quality control reporting.

Provides reporting utilities for summarising quality metrics,
rejection statistics, and filtering outcomes produced by the quality
control pipeline.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""


class QualityControlReportGenerator:
    """Interface for generating quality control reports.

    Implementations may summarize metrics, rejection statistics, and
    filtering outcomes produced by the quality control pipeline.
    """

    def generate():
        """Generate a report from the quality control results."""
        pass


