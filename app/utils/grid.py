import streamlit as st

def make_grid(cols: int, rows: int) -> list:
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows, gap="medium")
    return grid