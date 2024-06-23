import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


from pages.datasets import dataset_scatter_plot, get_datasets
from utils import Method, Metric, compute_metric, get_clustering_parameters, make_grid


@st.cache_data
def plot_metric(data: dict[str, tuple[np.ndarray, np.ndarray]], metric: Metric, _params: dict) -> plt.Figure:
    score = compute_metric(data, metric, _params)
    fig, ax = plt.subplots()
    ax.bar(list(score.keys()), list(score.values()))
    ax.set_title(metric.value.title())
    return fig


def render_metrics() -> None:
    datasets = get_datasets()
    params = get_clustering_parameters()
    for dataset_name, dataset in datasets.items():
        st.header(f'Clustering Methods Performance on "{dataset_name.title()}" Dataset')
        grid = make_grid(1, len(Metric) + 1)
        grid[0][0].pyplot(dataset_scatter_plot(dataset, 'Ground Of Truth'))
        for metric_idx, metric in enumerate(Metric):
            grid[0][metric_idx + 1].pyplot(plot_metric(dataset, metric, params[dataset_name]))