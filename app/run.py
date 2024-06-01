import streamlit as st
from streamlit_navigation_bar import st_navbar

from config import Pages, render_functions


def set_page_config() -> None:
    st.set_page_config(
        # initial_sidebar_state="collapsed", 
        page_title="Review of Selected Clustering Methods",
        layout="wide",
    )


def display_navbar() -> None:
    page = st_navbar(pages=[pg.value for pg in Pages])
    render_functions.get(page)()


if __name__ == '__main__':
    set_page_config()
    display_navbar()

