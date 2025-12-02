# app.py
# Streamlit UI to run generate_city_risk.process_city and show results
# Requirements: streamlit, streamlit_folium

import streamlit as st
import os
from generate_city_risk import process_city

st.set_page_config(page_title="Road Risk — EPICS", layout="wide")

st.title("Road Risk — EPICS (Generate & View)")
st.markdown("Select a city and click **Generate**. This runs the pipeline (OSMnx → segments → risk → maps).")

city = st.text_input("City (place name for OSM)", "Bhopal, Madhya Pradesh, India")
outdir = st.text_input("Output folder", "data")

col1, col2 = st.columns([1,3])

with col1:
    if st.button("Generate risk map"):
        st.info("Processing — this can take a few minutes for a large city.")
        try:
            result = process_city(city, out_dir=outdir, verbose=True)
            st.success("Processing finished.")
            st.write(result)
            seg_map = result.get("seg_map")
            heat_map = result.get("heat_map")
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

with col2:
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        if res.get("seg_map") and os.path.exists(res["seg_map"]):
            st.subheader("Risk map (segments)")
            from streamlit_folium import st_folium
            st_folium(res["seg_map"], width=900, height=600)
        if res.get("heat_map") and os.path.exists(res["heat_map"]):
            st.subheader("Heatmap")
            from streamlit_folium import st_folium
            st_folium(res["heat_map"], width=900, height=600)
    else:
        st.info("No results yet. Click Generate.")

st.markdown("---")
st.markdown("Files will be saved under `data/<city_slug>/`.")
