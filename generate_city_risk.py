# generate_city_risk.py
# Full working version — downloads city network, computes risk, exports maps.

import os
import json
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import LineString

# ----------------------------------------------------------------------
# 1. Helper — clean city name into folder path
# ----------------------------------------------------------------------
def slugify(text):
    return text.lower().replace(" ", "_").replace(",", "")

# ----------------------------------------------------------------------
# 2. Function: Download OSM network + convert into segments GeoDataFrame
# ----------------------------------------------------------------------
def download_city_network(place_name):
    print(f"[INFO] Downloading road network for: {place_name}")

    try:
        G = ox.graph_from_place(place_name, network_type="drive")
    except Exception as e:
        raise RuntimeError(f"OSM download failed: {e}")

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Ensure geometry exists
    edges = edges[edges.geometry.notnull()].reset_index()

    print(f"[INFO] Edges: {len(edges)}")

    # Create road_id (from OSM 'osmid')
    edges["road_id"] = edges["osmid"].astype(str)

    # Compute length
    edges["length_m"] = edges.geometry.length * 111000  # rough meters

    return edges

# ----------------------------------------------------------------------
# 3. Dummy risk model — You can replace later with ML model
# ----------------------------------------------------------------------
def compute_risk(segments: gpd.GeoDataFrame):
    np.random.seed(42)
    segments["risk"] = np.random.rand(len(segments))
    return segments

# ----------------------------------------------------------------------
# 4. Export maps as HTML
# ----------------------------------------------------------------------
def save_segment_map(gdf, out_html):
    center = [gdf.geometry.unary_union.centroid.y,
              gdf.geometry.unary_union.centroid.x]

    m = folium.Map(location=center, zoom_start=12)

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        folium.PolyLine(
            locations=[(y, x) for x, y in coords],
            color="#FF0000" if row["risk"] > 0.7 else "#00AAFF",
            weight=4,
            opacity=0.7
        ).add_to(m)

    m.save(out_html)
    return m

def save_heatmap(gdf, out_html):
    from folium.plugins import HeatMap

    center = [gdf.geometry.unary_union.centroid.y,
              gdf.geometry.unary_union.centroid.x]

    m = folium.Map(location=center, zoom_start=12)

    heat_data = []
    for _, row in gdf.iterrows():
        x, y = row.geometry.centroid.x, row.geometry.centroid.y
        heat_data.append([y, x, float(row["risk"])])

    HeatMap(heat_data, radius=15, blur=12).add_to(m)
    m.save(out_html)
    return m

# ----------------------------------------------------------------------
# 5. Main function: process_city
# ----------------------------------------------------------------------
def process_city(place_name, out_dir="data", verbose=True):

    city_slug = slugify(place_name)
    city_path = os.path.join(out_dir, city_slug)

    os.makedirs(city_path, exist_ok=True)

    # 1. Download road network
    segments = download_city_network(place_name)

    # 2. Compute risk
    segments = compute_risk(segments)

    # 3. Save GeoJSON + CSV
    seg_geo = os.path.join(city_path, "segments.geojson")
    seg_csv = os.path.join(city_path, "segments.csv")

    segments.to_file(seg_geo, driver="GeoJSON")
    segments.drop(columns=["geometry"]).to_csv(seg_csv, index=False)

    # 4. Save maps
    seg_html = os.path.join(city_path, "segments_map.html")
    heat_html = os.path.join(city_path, "heat_map.html")

    save_segment_map(segments, seg_html)
    save_heatmap(segments, heat_html)

    if verbose:
        print(f"[DONE] Files saved under {city_path}")

    # return file paths to Streamlit
    return {
        "segments_file": seg_geo,
        "segments_csv": seg_csv,
        "seg_map_html": seg_html,
        "heat_map_html": heat_html
    }

# ----------------------------------------------------------------------
# END FILE
# ----------------------------------------------------------------------
