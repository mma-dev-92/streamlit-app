import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from utils import get_datasets, make_grid


@st.cache_data
def dataset_scatter_plot(data: tuple[np.ndarray, np.ndarray], title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    X, classes = data
    ax.scatter(x=X[:, 0], y=X[:, 1], c=classes, alpha=0.45, cmap='viridis')
    ax.set_title(f'{title.title()}')

    return fig


_n_plots_per_row = 3

def render_datasets() -> None:
    data_to_plot = get_datasets()
    st.header('Example Datasets', divider='gray')
    grid = make_grid(len(data_to_plot) // _n_plots_per_row, _n_plots_per_row)

    for idx, dataset_name in enumerate(data_to_plot):
        grid[idx // _n_plots_per_row][(idx % 3)].pyplot(dataset_scatter_plot(data_to_plot[dataset_name], dataset_name))
