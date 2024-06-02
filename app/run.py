import streamlit as st
from streamlit_navigation_bar import st_navbar

from pages.datasets import render_datasets
from pages.methods import render_methods
from pages.metrics import render_metrics
from utils import Page


render_functions = {
    Page.datasets.value: render_datasets,
    Page.methods.value: render_methods,
    Page.results.value: render_metrics,
}


def set_page_config() -> None:
    st.set_page_config(
        # initial_sidebar_state="collapsed", 
        page_title="Review of Selected Clustering Methods",
        layout="wide",
        initial_sidebar_state='collapsed',
    )


def display_navbar() -> None:
    page = st_navbar(pages=[pg.value for pg in Page])
    render_functions.get(page)()


if __name__ == '__main__':
    set_page_config()
    display_navbar()

