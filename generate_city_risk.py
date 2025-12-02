# generate_city_risk.py
# Usage: import process_city/process_route from this file
# Requires: osmnx, geopandas, shapely, numpy, pandas, folium, scikit-learn

import os
import sys
import time
import math
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point, MultiLineString
from folium.plugins import HeatMap
import folium

warnings.filterwarnings("ignore")

# ---------------- OSMnx settings (polite) ----------------
ox.settings.default_user_agent = "EPICS-RoadRisk/1.0 (vidit@example.com)"
ox.settings.timeout = 180
ox.settings.overpass_rate_limit = True

# ---------------- Config ----------------
SEGMENT_LENGTH_M = 100
MAP_OUTPUT = True
W_INTER = 0.5
W_CURV = 0.3
W_INVLEN = 0.2
CLIP_PCT = 0.99
EPS = 1e-6

# ---------------- Helpers ----------------
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
        a = (coords[i][1], coords[i][0]); b = (coords[i+1][1], coords[i+1][0])
        cumdist.append(cumdist[-1] + haversine_distance(a, b))
    total = cumdist[-1]
    if total == 0:
        return [ls]
    out_segments = []
    for i in range(n_segments):
        start_d = i * segment_length_m
        end_d = min((i+1) * segment_length_m, total)
        def interp_at_distance(d):
            if d <= 0: return coords[0]
            if d >= total: return coords[-1]
            for j in range(len(cumdist)-1):
                if cumdist[j] <= d <= cumdist[j+1]:
                    denom = (cumdist[j+1] - cumdist[j]) or 1e-9
                    frac = (d - cumdist[j]) / denom
                    x1, y1 = coords[j]; x2, y2 = coords[j+1]
                    return (x1 + frac*(x2-x1), y1 + frac*(y2-y1))
            return coords[-1]
        p1 = interp_at_distance(start_d)
        p2 = interp_at_distance(end_d)
        out_segments.append(LineString([p1, p2]))
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
    if hi <= lo:
        hi = s.max() if s.max()>lo else lo+1.0
    norm = (s - lo) / (hi - lo)
    return norm.clip(0.0,1.0), lo, hi

# ---------------- Safe geocode/load_graph ----------------
def safe_geocode(place, attempts=5, delay=2):
    for i in range(attempts):
        try:
            # returns (lat, lon)
            pt = ox.geocode(place)
            return pt
        except Exception as e:
            if i < attempts-1:
                time.sleep(delay)
                continue
            raise

def load_graph_for_place(place, attempts=4):
    # wrapper to load graph with retries
    for i in range(attempts):
        try:
            G = ox.graph_from_place(place, network_type="drive", which_result=1)
            return G
        except Exception as e:
            if i < attempts-1:
                time.sleep(3)
                continue
            # final attempt: try without which_result
            try:
                return ox.graph_from_place(place, network_type="drive")
            except Exception:
                raise RuntimeError(f"Failed to download graph for {place}: {e}")

# ----------------- process_route -----------------
def process_route(start_place, end_place, city_name, out_dir="data", verbose=True):
    """
    start_place, end_place: free-text addresses (geocoded via OSM)
    city_name: used to download graph (can be same city, e.g. "Bhopal, India")
    """

    city_slug = city_name.lower().replace(",", "").replace(" ", "_")
    route_slug = f"{city_slug}_route"
    base = os.path.join(out_dir, route_slug)
    os.makedirs(base, exist_ok=True)

    if verbose: print("Loading graph for:", city_name)
    G = load_graph_for_place(city_name)

    # geocode start and end (text)
    if verbose: print("Geocoding start and end...")
    try:
        start_pt = safe_geocode(start_place)
        end_pt = safe_geocode(end_place)
    except Exception as e:
        raise RuntimeError(f"Geocoding failed: {e}")

    # find nearest nodes
    if verbose: print("Finding nearest graph nodes to points...")
    u = ox.nearest_nodes(G, start_pt[1], start_pt[0])
    v = ox.nearest_nodes(G, end_pt[1], end_pt[0])

    # compute shortest path by length
    if verbose: print(f"Computing shortest path from node {u} to {v} ...")
    try:
        route_nodes = ox.shortest_path(G, u, v, weight="length")
    except Exception as e:
        raise RuntimeError(f"Route computation failed: {e}")

    # build LineString of route (follow node coordinates)
    coords = []
    for n in route_nodes:
        data = G.nodes[n]
        coords.append((data['x'], data['y']))  # lon, lat (OSMnx uses x=lon,y=lat)

    # convert to lat,lon pairs for shapely LineString expects (x,y)=lon,lat
    route_ls = LineString(coords)  # coords are (lon, lat)

    # split route into fixed-length segments
    segs = split_linestring_fixed_length(route_ls, segment_length_m=SEGMENT_LENGTH_M)

    # build GeoDataFrame for route segments
    rows = []
    for seg in segs:
        length_m = linestring_length_m(seg)
        curv = curvature_for_linestring(seg)
        centroid = seg.centroid
        rows.append({
            "length_m": length_m,
            "curvature": curv,
            "geometry": seg,
            "centroid_x": centroid.x,
            "centroid_y": centroid.y
        })
    gdf_route = gpd.GeoDataFrame(rows, geometry='geometry', crs="EPSG:4326")

    # compute intersection counts using graph nodes (project to metric)
    gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
    # ensure node geometry exists
    gdf_nodes = gdf_nodes.set_geometry('geometry')

    # project both
    gdf_route_proj = gdf_route.to_crs(epsg=3857)
    gdf_nodes_proj = gdf_nodes.to_crs(epsg=3857)

    gdf_route_proj['centroid'] = gdf_route_proj.geometry.centroid
    gdf_route_proj['buffer25m'] = gdf_route_proj['centroid'].buffer(25)

    # set active geometry and perform safe join
    nodes_for_join = gdf_nodes_proj.set_geometry('geometry').copy()
    segments_for_join = gdf_route_proj.set_geometry('buffer25m').copy()
    # ensure CRSs match
    nodes_for_join = nodes_for_join.to_crs(segments_for_join.crs)

    try:
        join = gpd.sjoin(nodes_for_join[['geometry']], segments_for_join[['buffer25m']], how='right', predicate='intersects')
        if 'index_right' in join.columns and not join.empty:
            counts = join.groupby('index_right').size().reindex(gdf_route_proj.index, fill_value=0)
        else:
            counts = pd.Series(0, index=gdf_route_proj.index)
    except Exception as e:
        if verbose: print("Spatial join failed for route (falling back to zero counts):", e)
        counts = pd.Series(0, index=gdf_route_proj.index)

    gdf_route['intersection_count'] = counts.values.astype(int)

    # compute risk metrics (same approach as city)
    gdf_route["inv_length"] = 1.0 / (gdf_route["length_m"].astype(float) + EPS)
    gdf_route["inter_norm"], _, _ = robust_normalize(gdf_route["intersection_count"])
    gdf_route["curv_norm"], _, _ = robust_normalize(gdf_route["curvature"])
    gdf_route["invlen_norm"], _, _ = robust_normalize(gdf_route["inv_length"])
    gdf_route["risk_score_raw"] = (W_INTER * gdf_route["inter_norm"] + W_CURV * gdf_route["curv_norm"] + W_INVLEN * gdf_route["invlen_norm"])

    low = gdf_route["risk_score_raw"].min()
    high = gdf_route["risk_score_raw"].quantile(0.995)
    denom = (high - low) if (high - low) > 0 else (gdf_route["risk_score_raw"].max() - low + EPS)
    gdf_route["risk_score"] = ((gdf_route["risk_score_raw"] - low) / denom).clip(0,1)
    gdf_route["risk_band"] = pd.cut(gdf_route["risk_score"], bins=[-0.01,0.25,0.5,0.75,1.0], labels=["Low","Medium","High","Very High"])

    # Save route-level files
    geo_out = os.path.join(base, "route_segments.geojson")
    csv_out = os.path.join(base, "route_segments.csv")
    gdf_route.to_file(geo_out, driver="GeoJSON")
    gdf_route.drop(columns='geometry').to_csv(csv_out, index=False)

    # Create route folium map + heatmap (only route segments)
    seg_map_obj = None
    heat_map_obj = None
    if MAP_OUTPUT and not gdf_route.empty:
        try:
            center = [ (start_pt[0] + end_pt[0]) / 2.0, (start_pt[1] + end_pt[1]) / 2.0 ]
        except:
            try:
                center = [ gdf_route.geometry.centroid.y.mean(), gdf_route.geometry.centroid.x.mean() ]
            except:
                center = [23.2599, 77.4126]

        m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
        for _, r in gdf_route.iterrows():
            geom = r.geometry
            if geom.is_empty: continue
            if isinstance(geom, MultiLineString):
                geom = max(list(geom), key=lambda s: s.length)
            if not isinstance(geom, LineString): continue
            coords = [(lat, lon) for lon, lat in geom.coords]
            score = r["risk_score"]
            color = '#2ecc71' if score < 0.33 else ('#f1c40f' if score < 0.66 else '#e74c3c')
            folium.PolyLine(coords, color=color, weight=5, opacity=0.9).add_to(m)
        seg_map_file = os.path.join(base, "route_segments_risk_map.html")
        m.save(seg_map_file)
        seg_map_obj = seg_map_file  # return path (app will read HTML)

        # heatmap
        heat_points = []
        for _, r in gdf_route.iterrows():
            try:
                heat_points.append([ r.geometry.centroid.y, r.geometry.centroid.x, float(r["risk_score"]) ])
            except:
                continue
        m2 = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
        if heat_points:
            HeatMap(heat_points, radius=12, blur=10, max_zoom=13).add_to(m2)
        heat_map_file = os.path.join(base, "route_heatmap.html")
        m2.save(heat_map_file)
        heat_map_obj = heat_map_file
    else:
        seg_map_obj = None
        heat_map_obj = None

    return {
        "city": city_name,
        "start_place": start_place,
        "end_place": end_place,
        "base": base,
        "route_geo": geo_out,
        "route_csv": csv_out,
        "seg_map": seg_map_obj,
        "heat_map": heat_map_obj,
        "gdf_route_preview": gdf_route.head().to_dict(orient="records")
    }


# ----------------- optionally keep process_city (full city) -----------------
def process_city(city_name, out_dir="data", verbose=True):
    """
    Original pipeline for whole city (kept for backward compatibility)
    """
    city_slug = city_name.lower().replace(",", "").replace(" ", "_").replace(",", "")
    base = os.path.join(out_dir, city_slug)
    os.makedirs(base, exist_ok=True)

    G = load_graph_for_place(city_name)
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    gdf_edges = gdf_edges[gdf_edges.geometry.notna() & ~gdf_edges.geometry.is_empty].copy()

    rows = []
    for _, row in gdf_edges.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiLineString":
            geom = max(list(geom), key=lambda s: linestring_length_m(s))
        segs = split_linestring_fixed_length(geom, SEGMENT_LENGTH_M)
        for seg in segs:
            centroid = seg.centroid
            rows.append({
                "osmid": row.get("osmid"),
                "highway": row.get("highway"),
                "length_m": linestring_length_m(seg),
                "curvature": curvature_for_linestring(seg),
                "geometry": seg,
                "centroid_x": centroid.x, "centroid_y": centroid.y
            })
    gdf_segments = gpd.GeoDataFrame(rows, geometry='geometry', crs="EPSG:4326")

    # intersection counts (reuse approach)
    gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False).reset_index()
    seg_proj = gdf_segments.to_crs(3857)
    nodes_proj = gdf_nodes.to_crs(3857)
    seg_proj['centroid'] = seg_proj.geometry.centroid
    seg_proj['buffer25m'] = seg_proj['centroid'].buffer(25)
    try:
        join = gpd.sjoin(nodes_proj.set_geometry('geometry'), seg_proj.set_geometry('buffer25m'), how='right', predicate='intersects')
        counts = join.groupby('index_right').size().reindex(seg_proj.index, fill_value=0) if ('index_right' in join.columns and not join.empty) else pd.Series(0, index=seg_proj.index)
    except Exception:
        counts = pd.Series(0, index=seg_proj.index)
    gdf_segments['intersection_count'] = counts.values.astype(int)

    # compute risk as before
    gdf = gdf_segments.copy()
    gdf["inv_length"] = 1.0 / (gdf["length_m"] + EPS)
    gdf["inter_norm"], _, _ = robust_normalize(gdf["intersection_count"])
    gdf["curv_norm"], _, _ = robust_normalize(gdf["curvature"])
    gdf["invlen_norm"], _, _ = robust_normalize(gdf["inv_length"])
    gdf["risk_score_raw"] = (W_INTER * gdf["inter_norm"] + W_CURV * gdf["curv_norm"] + W_INVLEN * gdf["invlen_norm"])
    low = gdf["risk_score_raw"].min()
    high = gdf["risk_score_raw"].quantile(0.995)
    denom = (high - low) if (high - low) > 0 else (gdf["risk_score_raw"].max() - low + EPS)
    gdf["risk_score"] = ((gdf["risk_score_raw"] - low) / denom).clip(0,1)
    gdf["risk_band"] = pd.cut(gdf["risk_score"], bins=[-0.01,0.25,0.5,0.75,1.0], labels=["Low","Medium","High","Very High"])

    risk_geo = os.path.join(base, f"{city_slug}_segments_risk.geojson")
    risk_csv = os.path.join(base, f"{city_slug}_segments_risk.csv")
    gdf.to_file(risk_geo, driver="GeoJSON")
    gdf.drop(columns='geometry').to_csv(risk_csv, index=False)

    # create maps
    if MAP_OUTPUT and not gdf.empty:
        try:
            center = [gdf.geometry.unary_union.centroid.y, gdf.geometry.unary_union.centroid.x]
        except:
            center = [23.2599, 77.4126]
        m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
        for _, r in gdf.iterrows():
            try:
                geom = r.geometry
                if geom.is_empty: continue
                if isinstance(geom, MultiLineString): geom = max(list(geom), key=lambda s: s.length)
                if not isinstance(geom, LineString): continue
                coords = [(lat, lon) for lon, lat in geom.coords]
                color = '#2ecc71' if r['risk_score'] < 0.33 else ('#f1c40f' if r['risk_score'] < 0.66 else '#e74c3c')
                folium.PolyLine(coords, color=color, weight=3, opacity=0.7).add_to(m)
            except Exception:
                continue
        seg_map = os.path.join(base, f"{city_slug}_segments_risk_map.html")
        m.save(seg_map)

        heat_points = []
        for _, r in gdf.iterrows():
            try:
                lat = r.geometry.centroid.y; lon = r.geometry.centroid.x
                heat_points.append([lat, lon, float(r['risk_score'])])
            except Exception:
                continue
        m2 = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
        if heat_points:
            HeatMap(heat_points, radius=12, blur=10, max_zoom=13).add_to(m2)
        heat_map = os.path.join(base, f"{city_slug}_segments_risk_heatmap.html")
        m2.save(heat_map)
    else:
        seg_map = heat_map = None

    return {
        "city": city_name,
        "base": base,
        "raw_geo": os.path.join(base, f"{city_slug}_segments.geojson"),
        "raw_csv": os.path.join(base, f"{city_slug}_segments.csv"),
        "risk_geo": risk_geo,
        "risk_csv": risk_csv,
        "seg_map": seg_map,
        "heat_map": heat_map
    }

# ---------------- CLI for quick testing ----------------
if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "route":
        # usage: python generate_city_risk.py route "start address" "end address" "city"
        start = sys.argv[2]
        end = sys.argv[3]
        city = sys.argv[4] if len(sys.argv) > 4 else "Bhopal, Madhya Pradesh, India"
        out = process_route(start, end, city)
        print("Route outputs:", json.dumps(out, indent=2))
    elif len(sys.argv) >= 2:
        city = sys.argv[1]
        out = process_city(city)
        print("City outputs:", json.dumps(out, indent=2))
    else:
        print("Usage:")
        print('  python generate_city_risk.py "City, Country"')
        print('  python generate_city_risk.py route "Start address" "End address" "City name"')
