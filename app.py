# app.py
# Streamlit UI for EPICS Road Risk System

import streamlit as st
import os
from generate_city_risk import process_city
import streamlit.components.v1 as components

st.set_page_config(page_title="EPICS Road Risk", layout="wide")

st.title("🚗 EPICS Road Risk — Live City Risk Generator")
st.markdown("Enter a city name and click **Generate** to run the full pipeline.")

# Inputs
city = st.text_input("City name", "Bhopal, Madhya Pradesh, India")
outdir = st.text_input("Output directory", "data")

if st.button("Generate"):
    st.info("Processing city… This may take a few minutes.")

    try:
        result = process_city(city, out_dir=outdir, verbose=True)
        st.success("Completed successfully!")
        st.session_state["result"] = result

    except Exception as e:
        st.error(f"Error: {e}")

# SHOW RESULTS
st.markdown("---")

if "result" in st.session_state:

    result = st.session_state["result"]

    seg_map_file = result["seg_map_html"]
    heat_map_file = result["heat_map_html"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Risk Map")
        if os.path.exists(seg_map_file):
            with open(seg_map_file, "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=600)

    with col2:
        st.subheader("Heat Map")
        if os.path.exists(heat_map_file):
            with open(heat_map_file, "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=600)

st.markdown("All files saved inside the project folder under `data/<city_slug>`.")
