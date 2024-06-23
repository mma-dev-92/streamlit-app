import enum
import streamlit as st
import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler


class Method(enum.Enum):
    kmeans = 'K-Means'
    hierarchical = 'Hierarchical Clustering'
    dbscan = 'DBSCAN'


class Page(enum.Enum):
    datasets = "Input Datasets"
    methods = "Clustering Methods Application"
    results = "Performance Evaluation"


class Metric(enum.Enum):
    ari = "Adjusted Random Score"
    nmi = "Normalized Mutual Information"
    vms = "V-Measure Score"


_seed = 12345
_n_features = 2


@st.cache_data
def uniformly_distributed_data(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    random_dataset = StandardScaler().fit_transform(np.random.uniform((0, 0), (1, 1), (n_samples, _n_features)))
    labels = np.ones(len(random_dataset)) * np.nan
    labels[(random_dataset[:, 0] >= 0) & (random_dataset[:, 1] >= 0)] = 1
    labels[(random_dataset[:, 0] <= 0) & (random_dataset[:, 1] >= 0)] = 2
    labels[(random_dataset[:, 0] >= 0) & (random_dataset[:, 1] <= 0)] = 3
    labels[(random_dataset[:, 0] <= 0) & (random_dataset[:, 1] <= 0)] = 4

    return random_dataset, labels


@st.cache_data
def different_deisities(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    centers = [[0, 0], [1, 1], [5, 0], [0, 5], [2.5, 2.5]]
    cluster_std = [0.5, 1.0, 0.5, 1.0, 0.3] 

    data = make_blobs(n_samples, centers=centers, cluster_std=cluster_std, random_state=seed)
    return StandardScaler().fit_transform(data[0]), data[1]


@st.cache_data
def nested_clusters(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Outer clusters
    outer_centers = [[-5, -5], [5, 5]]
    outer_std = [1.5, 1.5]
    X_outer, y_outer = make_blobs(n_samples=n_samples // 2, centers=outer_centers, cluster_std=outer_std, random_state=seed)
    y_outer = y_outer

    # Inner clusters within each outer cluster
    inner_centers_1 = [[-6, -6], [-4, -4]]
    inner_std_1 = [0.5, 0.5]
    X_inner_1, y_inner_1 = make_blobs(n_samples=n_samples // 4, centers=inner_centers_1, cluster_std=inner_std_1, random_state=seed)
    y_inner_1 = y_inner_1 + 2

    inner_centers_2 = [[4, 4], [6, 6]]
    inner_std_2 = [0.5, 0.5]
    X_inner_2, y_inner_2 = make_blobs(n_samples=n_samples // 4, centers=inner_centers_2, cluster_std=inner_std_2, random_state=seed)
    y_inner_2 = y_inner_2 + 4

    X = np.vstack([X_outer, X_inner_1, X_inner_2])
    y = np.hstack([y_outer, y_inner_1, y_inner_2])

    return StandardScaler().fit_transform(X), y


@st.cache_data
def get_datasets(n_samples: int = 2000, seed: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Creates normalized datasets of shape (n_samples, n_features)
    """
    seed = seed or _seed

    result = dict(
        moons=make_moons(n_samples=n_samples, noise=0.07, random_state=seed),
        rings=make_circles(n_samples=n_samples, noise=0.08, random_state=seed, factor=0.4),
        blobs=make_blobs(n_samples=n_samples, centers=8, cluster_std=0.8, n_features=_n_features, random_state=seed),
    )

    result = {k: (StandardScaler().fit_transform(v[0]), v[1]) for k, v in result.items()}
    result['uniformly distributed'] = uniformly_distributed_data(n_samples, seed)
    result['different densities'] = different_deisities(n_samples, seed)
    result['nested clusters'] = nested_clusters(n_samples, seed)

    return result


@st.cache_data
def get_clustering_parameters() -> dict[str, dict]:
    """Set of parameters for each dataset and for each clustering method."""
    return {
        'moons': {
            Method.kmeans.name: {
                'k': 4
            },
            Method.hierarchical.name: {
                'method': 'ward',
                'k': 2,
            },
            Method.dbscan.name: {
                'min_pts': 5,
                'eps': 0.1,
            },
        },
        'rings': {
            Method.kmeans.name: {'k': 6},
            Method.hierarchical.name: {
                'method': 'ward',
                'k': 2,
            },
            Method.dbscan.name: {
                'min_pts': 7,
                'eps': 0.18,
            },
        },
        'blobs': {
            Method.kmeans.name: {'k': 4},
            Method.hierarchical.name: {
                'method': 'ward',
                'k': 8,
            },
            Method.dbscan.name: {
                'min_pts': 4,
                'eps': 0.1,
            },
        },
        'uniformly distributed': {
            Method.kmeans.name: {'k': 4},
            Method.hierarchical.name: {
                'method': 'ward',
                'k': 4,
            },
            Method.dbscan.name: {
                'min_pts': 6,
                'eps': 0.13,
            },
        },
        'different densities': {
            Method.kmeans.name: {'k': 4},
            Method.hierarchical.name: {
                'method': 'ward',
                'k': 5,
            },
            Method.dbscan.name: {
                'min_pts': 4,
                'eps': 0.105,
            },
        },
        'nested clusters': {
            Method.kmeans.name: {'k': 2},
            Method.hierarchical.name: {
                'method': 'ward',
                'k': 4,
            },
            Method.dbscan.name: {
                'min_pts': 4,
                'eps': 0.09,
            },
        },
    }


def make_grid(cols: int, rows: int) -> list:
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows, gap="medium")
    return grid


@st.cache_data
def compute_kmeans(data: tuple[np.ndarray, np.ndarray], params: dict) -> tuple[np.ndarray, np.ndarray]:
    X, _ = data
    kmeans = KMeans(n_clusters=params['k'], random_state=0)
    labels = kmeans.fit_predict(X)
    centriods = kmeans.cluster_centers_
    return labels, centriods


@st.cache_data
def compute_hierarchical(data: tuple[np.ndarray, np.ndarray], params: dict) -> np.ndarray:
    X, _ = data
    hc = AgglomerativeClustering(n_clusters=params['k'], metric='euclidean', linkage=params['method'])
    return hc.fit_predict(X)


@st.cache_data
def compute_dbscan(data: tuple[np.ndarray, np.ndarray], params: dict) -> np.ndarray:
    X, _ = data
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_pts'])
    return dbscan.fit_predict(X)


_metric_funs = {
    Metric.ari: adjusted_rand_score,
    Metric.nmi: normalized_mutual_info_score,
    Metric.vms: v_measure_score,
}


@st.cache_data
def compute_metric(data: tuple[np.ndarray, np.ndarray], metric: Metric, params: dict) -> dict[str, float]:
    _, y = data
    return {
        Method.kmeans.value: _metric_funs[metric](labels_true=y, labels_pred=compute_kmeans(data, params[Method.kmeans.name])[0]),
        Method.hierarchical.value: _metric_funs[metric](labels_true=y, labels_pred=compute_hierarchical(data, params[Method.hierarchical.name])),
        Method.dbscan.value: _metric_funs[metric](labels_true=y, labels_pred=compute_dbscan(data, params[Method.dbscan.name]))
    }
