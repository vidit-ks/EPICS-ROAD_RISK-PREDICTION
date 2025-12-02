# app.py
# Streamlit UI to run generate_city_risk.process_city and show results
# Requirements: streamlit, streamlit_folium, folium

import streamlit as st
import os
import folium
from streamlit_folium import st_folium
from generate_city_risk import process_city

st.set_page_config(page_title="Road Risk — EPICS", layout="wide")
st.title("Road Risk — EPICS (Generate & View)")
st.markdown("Select a city and click **Generate**. This runs the pipeline (OSMnx → segments → risk → maps).")

city = st.text_input("City (place name for OSM)", "Bhopal, Madhya Pradesh, India")
outdir = st.text_input("Output folder", "data")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Generate risk map"):
        st.info("Processing — this can take a few minutes for a large city.")
        try:
            result = process_city(city, out_dir=outdir, verbose=True)
            st.success("Processing finished.")
            st.write(result)
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

with col2:
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]

        # ---------------- Segments map ----------------
        seg_map = res.get("seg_map_obj")  # folium.Map object
        if seg_map:
            st.subheader("Risk map (segments)")
            st_folium(seg_map, width=900, height=600)
        else:
            # fallback if HTML file exists
            seg_map_file = res.get("seg_map")
            if seg_map_file and os.path.exists(seg_map_file):
                st.subheader("Risk map (segments)")
                html = open(seg_map_file, 'r', encoding='utf-8').read()
                st_folium(folium.Html(html, script=True), width=900, height=600)

        # ---------------- Heatmap ----------------
        heat_map = res.get("heat_map_obj")  # folium.Map object
        if heat_map:
            st.subheader("Heatmap")
            st_folium(heat_map, width=900, height=600)
        else:
            heat_map_file = res.get("heat_map")
            if heat_map_file and os.path.exists(heat_map_file):
                st.subheader("Heatmap")
                html = open(heat_map_file, 'r', encoding='utf-8').read()
                st_folium(folium.Html(html, script=True), width=900, height=600)

    else:
        st.info("No results yet. Click Generate.")

st.markdown("---")
st.markdown("Files will be saved under `data/<city_slug>/`.")
