import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import enum

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from pages.datasets import get_datasets
from utils.grid import make_grid 


class ClusteringMethods(enum.Enum):
    kmeans = 'K-Means'
    hierarchical = 'Hierarchical Clustering'
    dbscan = 'DBSCAN'


def clusteing_results_plot(X: np.ndarray, labels: np.ndarray, dataset_name: str, centroids: np.ndarray | None = None) -> plt.Figure:
    fig, ax = plt.subplots()
    out = labels == -1
    plt.scatter(X[~out][:, 0], X[~out][:, 1], c=labels[~out], alpha=0.5, s=100, cmap='viridis')
    if X[out].size > 0:
        plt.scatter(X[out][:, 0], X[out][:, 1], c='gray', alpha=0.5, s=100, label='outliers')
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='X', s=100)
    ax.set_title(f'{dataset_name.title()} Partition')
    if X[out].size > 0:
        plt.legend()
    fig.tight_layout()
    return fig


@st.cache_data
def compute_kmeans(X: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    centriods = kmeans.cluster_centers_

    return labels, centriods


@st.cache_data
def compute_hierarchical(X: np.ndarray, n_clusters: int) -> np.ndarray:
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = hc.fit_predict(X)
    return labels


@st.cache_data
def compute_dbscan(X: np.ndarray) -> np.ndarray:
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    return dbscan.fit_predict(X)


@st.cache_data
def kmeans_plot(data: tuple[np.ndarray, np.ndarray], dataset_name: str) -> plt.Figure:
    X, initial_labels = data
    n_clusters = len(np.unique(initial_labels))
    labels, centroids = compute_kmeans(X, n_clusters)
    return clusteing_results_plot(X, labels, dataset_name, centroids)


@st.cache_data
def hierarchical_plot(data: tuple[np.ndarray, np.ndarray], dataset_name: str) -> plt.Figure:
    X, initial_labels = data
    n_clusters = len(np.unique(initial_labels))
    labels = compute_hierarchical(X, n_clusters)
    return clusteing_results_plot(X, labels, dataset_name)


@st.cache_data
def dbscan_plot(data: tuple[np.ndarray, np.ndarray], dataset_name: str) -> plt.Figure:
    X, _ = data
    labels = compute_dbscan(X)
    return clusteing_results_plot(X, labels, dataset_name)


_plot_rendering_methods = {
    ClusteringMethods.kmeans: kmeans_plot,
    ClusteringMethods.hierarchical: hierarchical_plot,
    ClusteringMethods.dbscan: dbscan_plot,
}


def render_plot(method: ClusteringMethods, datasets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    st.header(f'{method.value.title()} Clustering Method', divider='gray')
    grid = make_grid(1, 4)
    for idx, (dataset_name, data) in enumerate(datasets.items()):
        grid[0][idx].pyplot(_plot_rendering_methods[method](data, dataset_name))


def render_methods() -> None:
    datasets = get_datasets()
    for method in ClusteringMethods:
        render_plot(method, datasets)
