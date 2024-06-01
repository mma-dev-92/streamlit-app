import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from utils.grid import make_grid


_seed = 1234567890
_n_features = 2


@st.cache_data
def get_datasets(n_samples: int = 1000, seed: int | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Creates normalized datasets of shape (n_samples, n_features)
    """
    seed = seed or _seed
    result = dict(
        moons=datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed),
        rings=datasets.make_circles(n_samples=n_samples, noise=0.04, random_state=seed, factor=0.4),
        blobs=datasets.make_blobs(n_samples=n_samples, n_features=_n_features, random_state=seed),
        uniform=(np.random.uniform((0, 0), (1, 1), (n_samples, _n_features)), np.zeros(n_samples)),
    )

    return {k: (StandardScaler().fit_transform(v[0]), v[1]) for k, v in result.items()}


@st.cache_data
def dataset_scatter_plot(dataset: dict[str, np.ndarray], dataset_name: str) -> plt.Figure:
    fig, ax = plt.subplots()
    data, classes = dataset[dataset_name]
    ax.scatter(x=data[:, 0], y=data[:, 1], c=classes, alpha=0.5, cmap='viridis')
    ax.set_title(f'{dataset_name} dataset'.title())

    return fig


def render_datasets() -> None:
    data_to_plot = get_datasets()
    st.header('Example Datasets', divider='gray')
    grid = make_grid(1, 4)

    for idx, dataset_name in enumerate(data_to_plot):
        grid[0][idx].pyplot(dataset_scatter_plot(data_to_plot, dataset_name))
    
    st.header('The Aim of This Exercise', divider='gray')
    st.write('Bla bla bla - I need to say a few words probably...')
