import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from pages.datasets import dataset_scatter_plot, get_datasets
from utils import Method, compute_dbscan, compute_hierarchical, compute_kmeans, make_grid 


def clusteing_results_plot(X: np.ndarray, labels: np.ndarray, plot_title: str, centroids: np.ndarray | None = None) -> plt.Figure:
    fig, ax = plt.subplots()
    out = labels == -1
    plt.scatter(X[~out][:, 0], X[~out][:, 1], c=labels[~out], alpha=0.45, cmap='viridis')
    if X[out].size > 0:
        plt.scatter(X[out][:, 0], X[out][:, 1], c='gray', alpha=0.7, label='outliers')
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='X', label='centroids')
    ax.set_title(f'{plot_title.title()}')
    if X[out].size > 0 or centroids is not None:
        plt.legend()
    fig.tight_layout()
    return fig


@st.cache_data
def kmeans_plot(data: tuple[np.ndarray, np.ndarray], plot_title: str) -> plt.Figure:
    X = data[0]
    labels, centroids = compute_kmeans(data)
    return clusteing_results_plot(X, labels, plot_title, centroids)


@st.cache_data
def hierarchical_plot(data: tuple[np.ndarray, np.ndarray], plot_title: str) -> plt.Figure:
    X, labels = data[0], compute_hierarchical(data)
    return clusteing_results_plot(X, labels, plot_title)


@st.cache_data
def dbscan_plot(data: tuple[np.ndarray, np.ndarray], plot_title: str) -> plt.Figure:
    X, labels = data[0], compute_dbscan(data)
    return clusteing_results_plot(X, labels, plot_title)


_plot_rendering_methods = {
    Method.kmeans: kmeans_plot,
    Method.hierarchical: hierarchical_plot,
    Method.dbscan: dbscan_plot,
}


def render_methods() -> None:
    datasets = get_datasets()
    for dataset_name in datasets:
        st.header(f'Dataset "{dataset_name.title()}"')
        grid = make_grid(1, len(Method) + 1)
        grid[0][0].pyplot(dataset_scatter_plot(datasets[dataset_name], title='Ground of Truth'))
        for idx, method in enumerate(Method):
            grid[0][idx + 1].pyplot(_plot_rendering_methods[method](datasets[dataset_name], method.name))
