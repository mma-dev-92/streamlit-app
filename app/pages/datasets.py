import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from utils.grid import make_grid


_seed = 1234567890
_n_features = 2


@st.cache_data
def create_datasets(n_samples: int = 1000, seed: int | None = None) -> dict[str, np.ndarray]:
    """
    Creates normalized datasets of shape (n_samples, n_features)
    """
    seed = seed or _seed
    result = dict(
        moons=datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)[0],
        rings=datasets.make_circles(n_samples=n_samples, noise=0.04, random_state=seed)[0],
        blobs=datasets.make_blobs(n_samples=n_samples, n_features=_n_features, random_state=seed)[0],
        uniform=np.random.uniform((0, 0), (1, 1), (n_samples, _n_features)),
    )

    return {k: StandardScaler().fit_transform(v) for k, v in result.items()}


def dataset_scatter_plot(dataset: dict[str, np.ndarray], dataset_name: str) -> plt.Figure:
    fig, ax = plt.subplots()
    data = dataset[dataset_name]
    ax.scatter(x=data[:, 0], y=data[:, 1])
    ax.set_title(f'{dataset_name} dataset'.title())

    return fig


def render_datasets() -> None:
    data_to_plot = create_datasets()
    grid = make_grid(2, 4)

    grid[0][1].pyplot(dataset_scatter_plot(data_to_plot, "rings"))
    grid[0][2].pyplot(dataset_scatter_plot(data_to_plot, "moons"))
    grid[1][1].pyplot(dataset_scatter_plot(data_to_plot, "uniform"))
    grid[1][2].pyplot(dataset_scatter_plot(data_to_plot, "blobs"))
