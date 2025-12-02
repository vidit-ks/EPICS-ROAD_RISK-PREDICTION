# app.py
# Streamlit UI to run route-based risk mapping
import streamlit as st
import os
from generate_city_risk import process_route, process_city
import streamlit.components.v1 as components

st.set_page_config(page_title="EPICS Road Risk — Route", layout="wide")
st.title("EPICS Road Risk — Route-based Risk Maps")
st.markdown("Enter Start and Destination places (plain text). The app will geocode and compute risk for that route only.")

col_inputs, col_blank = st.columns([3,1])
with col_inputs:
    start_place = st.text_input("Start (address/place)", "Bhopal Junction Railway Station, Bhopal")
    end_place = st.text_input("End (address/place)", "DB City Mall, Bhopal")
    city_name = st.text_input("City for network lookup", "Bhopal, Madhya Pradesh, India")
    outdir = st.text_input("Output folder", "data")

if st.button("Generate Route Risk"):
    st.info("Computing route and risk — this may take 30-90 seconds depending on network and city size.")
    try:
        result = process_route(start_place, end_place, city_name, out_dir=outdir, verbose=True)
        st.success("Route processed.")
        st.session_state["route_result"] = result
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.markdown("---")

if "route_result" in st.session_state:
    res = st.session_state["route_result"]
    st.subheader("Route info")
    st.write(f"Start: {res['start_place']}")
    st.write(f"End: {res['end_place']}")
    st.write("Files saved to:", res["base"])
    st.write("Preview of first few route segments (dict):")
    st.json(res.get("gdf_route_preview", []))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Route Segment Risk Map")
        seg_map_file = res.get("seg_map")
        if seg_map_file and os.path.exists(seg_map_file):
            html = open(seg_map_file, 'r', encoding='utf-8').read()
            components.html(html, height=650)
        else:
            st.warning("Segment map not available.")

    with col2:
        st.subheader("Route Heatmap")
        heat_map_file = res.get("heat_map")
        if heat_map_file and os.path.exists(heat_map_file):
            html = open(heat_map_file, 'r', encoding='utf-8').read()
            components.html(html, height=650)
        else:
            st.warning("Heatmap not available.")

st.markdown("---")
st.markdown("Tip: For better results, use clear place names or well-known landmarks (e.g., 'Bhopal Junction Railway Station').")
