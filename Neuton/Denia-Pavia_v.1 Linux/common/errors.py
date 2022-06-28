class MetricNotDescribed(Exception):

    def __init__(self, metric):
        super().__init__(f"Scoring for the metric {metric} is not described.")

class StageNotFound(Exception):

    def __init__(self, stage) -> None:
        super().__init__(
            f"The stage {stage} was not found in the list of valid stages."
        )
