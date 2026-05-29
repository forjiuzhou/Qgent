"""Streamlit-based interactive dashboard for Qgent.

Launch with: streamlit run -m qgent.viz.dashboard
Or: python -m qgent.viz.dashboard
"""
from __future__ import annotations

import sys


def run_dashboard():
    """Entry point for the Streamlit dashboard."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is required: pip install streamlit")
        sys.exit(1)

    st.set_page_config(page_title="Qgent Dashboard", layout="wide")
    st.title("Qgent - Quantitative Research Dashboard")

    st.sidebar.header("Configuration")

    # Placeholder sections for future implementation
    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Factors", "Backtest", "Portfolio"])

    with tab1:
        st.subheader("Data Management")
        st.info("Load and explore OHLCV data for crypto and US stocks.")
        st.text("Use DataLoader to load saved data, or CryptoFetcher / USStockFetcher to download.")

    with tab2:
        st.subheader("Factor Research")
        st.info("Compute, evaluate, and compare factors.")
        st.text("Available factors can be explored via FactorRegistry.list_factors()")

    with tab3:
        st.subheader("Backtesting")
        st.info("Run strategies and analyze performance.")

    with tab4:
        st.subheader("Portfolio Optimization")
        st.info("Compare allocation methods across multiple assets.")


if __name__ == "__main__":
    run_dashboard()
