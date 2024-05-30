import enum

from pages.cmethods import render_methods
from pages.datasets import render_datasets
from pages.metrics import render_metrics


class Pages(enum.Enum):
    datasets = "Input Datasets"
    methods = "Clustering Methods"
    results = "Results & Metrics"


render_functions = {
    Pages.datasets.value: render_datasets,
    Pages.methods.value: render_methods,
    Pages.results.value: render_metrics,
}
