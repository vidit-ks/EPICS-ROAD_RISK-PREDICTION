# generate_city_risk.py
# Exposes process_city(city_name, out_dir) for UI/backend
# Updated for Streamlit usage and proper Folium map handling

import os
import sys
import math
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
import folium
from folium.plugins import HeatMap

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
SEGMENT_LENGTH_M = 100
MAP_OUTPUT = True
W_INTER = 0.5
W_CURV = 0.3
W_INVLEN = 0.2
CLIP_PCT = 0.99
EPS = 1e-6

# --------- Helper Functions ----------
def haversine_distance(a, b):
    R = 6371000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    aa = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(aa))

def linestring_length_m(ls):
    coords = list(ls.coords)
    total = 0.0
    for i in range(len(coords)-1):
        a = (coords[i][1], coords[i][0])
        b = (coords[i+1][1], coords[i+1][0])
        total += haversine_distance(a, b)
    return total

def split_linestring_fixed_length(ls, segment_length_m=SEGMENT_LENGTH_M):
    total_len = linestring_length_m(ls)
    if total_len <= segment_length_m:
        return [ls]
    n_segments = int(math.ceil(total_len / segment_length_m))
    coords = list(ls.coords)
    cumdist = [0.0]
    for i in range(len(coords)-1):
        a = (coords[i][1], coords[i][0])
        b = (coords[i+1][1], coords[i+1][0])
        cumdist.append(cumdist[-1] + haversine_distance(a, b))
    total = cumdist[-1]
    if total == 0: return [ls]
    out_segments = []
    for i in range(n_segments):
        start_d = i * segment_length_m
        end_d = min((i+1) * segment_length_m, total)
        def interp_at_distance(d):
            if d <= 0: return coords[0]
            if d >= total: return coords[-1]
            for j in range(len(cumdist)-1):
                if cumdist[j] <= d <= cumdist[j+1]:
                    denom = (cumdist[j+1]-cumdist[j]) if (cumdist[j+1]-cumdist[j])!=0 else 1e-9
                    frac = (d - cumdist[j]) / denom
                    x1,y1 = coords[j]; x2,y2 = coords[j+1]
                    return (x1 + frac*(x2-x1), y1 + frac*(y2-y1))
            return coords[-1]
        p1 = interp_at_distance(start_d); p2 = interp_at_distance(end_d)
        out_segments.append(LineString([p1,p2]))
    return out_segments

def curvature_for_linestring(ls):
    coords = list(ls.coords)
    if len(coords) < 3: return 0.0
    angles = []
    for i in range(1, len(coords)-1):
        x0,y0 = coords[i-1]; x1,y1 = coords[i]; x2,y2 = coords[i+1]
        v1 = (x1-x0, y1-y0); v2 = (x2-x1, y2-y1)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(v1[0], v1[1]); mag2 = math.hypot(v2[0], v2[1])
        if mag1*mag2 == 0: continue
        cosang = max(-1.0, min(1.0, dot/(mag1*mag2)))
        angles.append(abs(math.acos(cosang)))
    ang_sum = sum(angles)
    length_m = linestring_length_m(ls)
    return ang_sum/length_m if length_m>0 else ang_sum

def robust_normalize(series, clip_pct=CLIP_PCT):
    s = series.astype(float).fillna(0.0)
    lo = s.min()
    hi = np.percentile(s, clip_pct*100)
    if hi <= lo: hi = s.max() if s.max()>lo else lo+1.0
    norm = (s - lo) / (hi - lo)
    return norm.clip(0.0,1.0), lo, hi

# --------- Main Pipeline ----------
def process_city(city_name, out_dir="data", verbose=True):
    city_slug = city_name.lower().replace(",", "").replace(" ", "_")
    base = os.path.join(out_dir, city_slug)
    os.makedirs(base, exist_ok=True)

    if verbose: print("Downloading road network for:", city_name)
    G = ox.graph_from_place(city_name, network_type="drive")
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    if 'geometry' not in gdf_edges.columns:
        nodes = dict(G.nodes(data=True))
        def mkedge(r): u,v = int(r['u']), int(r['v']); return LineString([(nodes[u]['x'], nodes[u]['y']), (nodes[v]['x'], nodes[v]['y'])])
        gdf_edges['geometry'] = gdf_edges.apply(mkedge, axis=1)
    gdf_edges = gdf_edges[~gdf_edges.geometry.is_empty].copy()

    # Split into segments
    rows = []
    for _, row in gdf_edges.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiLineString": geom = max(list(geom), key=lambda s: linestring_length_m(s))
        segs = split_linestring_fixed_length(geom, SEGMENT_LENGTH_M)
        for seg in segs:
            centroid = seg.centroid
            rows.append({
                "u": row.get("u"), "v": row.get("v"), "osmid": row.get("osmid"),
                "highway": row.get("highway"),
                "length_m": linestring_length_m(seg),
                "curvature": curvature_for_linestring(seg),
                "geometry": seg,
                "centroid_x": centroid.x, "centroid_y": centroid.y
            })
    gdf_segments = gpd.GeoDataFrame(rows, geometry='geometry', crs="EPSG:4326")

    # Compute intersection counts (basic)
    gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
    gdf_nodes_proj = gdf_nodes.set_geometry('geometry').to_crs(epsg=3857)
    gdf_segments_proj = gdf_segments.to_crs(epsg=3857)
    gdf_segments_proj['centroid'] = gdf_segments_proj.geometry.centroid
    gdf_segments_proj['buffer25m'] = gdf_segments_proj['centroid'].buffer(25)
    nodes_for_join = gdf_nodes_proj.set_geometry('geometry')
    segments_for_join = gdf_segments_proj.set_geometry('buffer25m')
    nodes_for_join = nodes_for_join.to_crs(segments_for_join.crs)
    join = gpd.sjoin(nodes_for_join, segments_for_join, how='right', predicate='intersects')
    counts = join.groupby('index_right').size().reindex(gdf_segments_proj.index, fill_value=0)
    gdf_segments['intersection_count'] = counts.values.astype(int)

    # Normalize highway & compute risk
    gdf_segments["inv_length"] = 1.0 / (gdf_segments["length_m"] + EPS)
    gdf_segments["inter_norm"], _, _ = robust_normalize(gdf_segments["intersection_count"])
    gdf_segments["curv_norm"], _, _ = robust_normalize(gdf_segments["curvature"])
    gdf_segments["invlen_norm"], _, _ = robust_normalize(gdf_segments["inv_length"])
    gdf_segments["risk_score_raw"] = (W_INTER * gdf_segments["inter_norm"] +
                                      W_CURV * gdf_segments["curv_norm"] +
                                      W_INVLEN * gdf_segments["invlen_norm"])
    low = gdf_segments["risk_score_raw"].min()
    high = gdf_segments["risk_score_raw"].quantile(0.995)
    denom = (high-low) if (high-low) > 0 else (gdf_segments["risk_score_raw"].max()-low+EPS)
    gdf_segments["risk_score"] = ((gdf_segments["risk_score_raw"]-low)/denom).clip(0.0,1.0)
    gdf_segments["risk_band"] = pd.cut(gdf_segments["risk_score"], bins=[-0.01,0.25,0.5,0.75,1.0],
                                       labels=["Low","Medium","High","Very High"])

    # Save outputs
    raw_geo = os.path.join(base, f"{city_slug}_segments.geojson")
    raw_csv = os.path.join(base, f"{city_slug}_segments.csv")
    risk_geo = os.path.join(base, f"{city_slug}_segments_risk.geojson")
    risk_csv = os.path.join(base, f"{city_slug}_segments_risk.csv")
    gdf_segments.to_file(raw_geo, driver="GeoJSON")
    gdf_segments.drop(columns='geometry').to_csv(raw_csv, index=False)
    gdf_segments.to_file(risk_geo, driver="GeoJSON")
    gdf_segments.drop(columns='geometry').to_csv(risk_csv, index=False)

    # Create Folium maps (return objects for Streamlit)
    seg_map_obj = heat_map_obj = None
    if MAP_OUTPUT and not gdf_segments.empty:
        center = [gdf_segments.geometry.centroid.y.mean(), gdf_segments.geometry.centroid.x.mean()]
        # Segments colored by risk
        m_seg = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
        for _, r in gdf_segments.iterrows():
            geom = r.geometry
            if geom.is_empty: continue
            if isinstance(geom, MultiLineString): geom = max(list(geom), key=lambda s: s.length)
            if not isinstance(geom, LineString): continue
            coords = [(lat, lon) for lon, lat in geom.coords]
            color = '#2ecc71' if r['risk_score'] < 0.33 else ('#f1c40f' if r['risk_score'] < 0.66 else '#e74c3c')
            folium.PolyLine(coords, color=color, weight=3, opacity=0.7).add_to(m_seg)
        seg_map_obj = m_seg

        # Heatmap
        heat_points = [[r.geometry.centroid.y, r.geometry.centroid.x, float(r['risk_score'])] 
                       for _, r in gdf_segments.iterrows() if not r.geometry.is_empty]
        m_heat = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
        if heat_points: HeatMap(heat_points, radius=12, blur=10, max_zoom=13).add_to(m_heat)
        heat_map_obj = m_heat

    return {
        "city": city_name,
        "base": base,
        "raw_geo": raw_geo, "raw_csv": raw_csv,
        "risk_geo": risk_geo, "risk_csv": risk_csv,
        "seg_map": seg_map_obj, "heat_map": heat_map_obj
    }

# CLI
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python generate_city_risk.py \"City, Country\"")
        sys.exit(1)
    city = sys.argv[1]
    out = process_city(city)
    print("Done. Outputs:", json.dumps({k:str(v) for k,v in out.items()}, indent=2))
