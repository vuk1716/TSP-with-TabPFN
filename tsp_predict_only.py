# tsp_predict_only.py
# Predict-only (NO SOLVER): greedy route guided by model predictions.
# - Loads saved TabPFN models (model_x.joblib, model_y.joblib, features.json)
# - Creates a new random TSP instance
# - At each step, predicts (x,y), then moves to the closest UNVISITED point to that prediction
# - Writes route CSV (with predicted vs actual at each step) and a PNG plot
# - Prints compute device (CPU or GPU name via PyTorch) before running

import json
import re
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------- device reporting -----------------
def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return "GPU (CUDA)", name
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "GPU (Apple MPS)", "Apple M-series (MPS)"
    except Exception:
        pass
    return "CPU", None

def print_device_info():
    driver, name = _detect_device()
    if driver.startswith("GPU"):
        print(f"Compute device: {driver} — {name}")
    else:
        print("Compute device: CPU")

# ----------------- basic geometry + instance -----------------
Point = Tuple[float, float]

def euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def generate_random_points(n: int, seed: int = 123, span: float = 1000.0) -> List[Point]:
    random.seed(seed)
    return [(random.uniform(0, span), random.uniform(0, span)) for _ in range(n)]

def precompute_knn(points: List[Point], k: int) -> Dict[int, List[Tuple[int, float]]]:
    """For each node i, store list of (j, distance(i,j)) sorted asc, excluding i."""
    n = len(points)
    knn: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        x0, y0 = points[i]
        arr = []
        for j, (x, y) in enumerate(points):
            if j == i:
                continue
            arr.append((j, euclid((x0, y0), (x, y))))
        arr.sort(key=lambda t: t[1])
        knn[i] = arr[:k]
    return knn

# ----------------- models I/O -----------------
def load_models(models_dir: str):
    p = Path(models_dir)
    if not (p / "model_x.joblib").exists() or not (p / "model_y.joblib").exists() or not (p / "features.json").exists():
        raise FileNotFoundError(
            f"Missing models in '{models_dir}'. Expected model_x.joblib, model_y.joblib, features.json"
        )
    # NOTE: loading will require the 'tabpfn' package installed (same as training env)
    model_x = joblib.load(p / "model_x.joblib")
    model_y = joblib.load(p / "model_y.joblib")
    meta = json.loads((p / "features.json").read_text())
    feature_cols: List[str] = meta["feature_cols"]

    # infer k from saved feature columns (expects knn1_x, knn2_x, ...)
    k = max([int(re.findall(r"knn(\d+)_x", c)[0]) for c in feature_cols if re.findall(r"knn(\d+)_x", c)] or [0])
    if k == 0:
        raise ValueError("Could not infer k from saved features (no knn*_x columns).")
    return model_x, model_y, feature_cols, k

# ----------------- feature construction -----------------
def make_feature_row(step: int,
                     prev_xy: Point,
                     step_dist: float,
                     cum_dist: float,
                     cur_node: int,
                     points: List[Point],
                     knn_map: Dict[int, List[Tuple[int, float]]],
                     feature_cols: List[str],
                     k: int) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the EXACT training feature columns order.
    Assumes training used columns like:
      step, step_distance, cumulative_distance, prev_x, prev_y, knn_k,
      knn1_x, knn1_y, knn1_distance, ..., knn{k}_x, knn{k}_y, knn{k}_distance
    """
    row: Dict[str, Any] = {}
    row["step"] = int(step)
    row["step_distance"] = float(step_dist)
    row["cumulative_distance"] = float(cum_dist)
    row["prev_x"] = float(prev_xy[0]) if step > 0 else np.nan
    row["prev_y"] = float(prev_xy[1]) if step > 0 else np.nan
    row["knn_k"] = int(k)

    # KNN of the CURRENT node (based on actual coordinates)
    knns = knn_map[cur_node]
    for r in range(1, k + 1):
        if r <= len(knns):
            j, d = knns[r - 1]
            nx, ny = points[j]
            row[f"knn{r}_x"] = float(nx)
            row[f"knn{r}_y"] = float(ny)
            row[f"knn{r}_distance"] = float(d)
        else:
            row[f"knn{r}_x"] = np.nan
            row[f"knn{r}_y"] = np.nan
            row[f"knn{r}_distance"] = np.nan

    # Keep ONLY the columns the model expects, preserving order
    data = {c: (row[c] if c in row else np.nan) for c in feature_cols}
    return pd.DataFrame([data], columns=feature_cols)

# ----------------- greedy route guided by predictions -----------------
def build_route_greedy(points: List[Point],
                       model_x,
                       model_y,
                       feature_cols: List[str],
                       k: int,
                       start: int = 0,
                       close_tour: bool = True):
    """
    Greedy construction:
      - At each step, create features for CURRENT node
      - Predict (x,y) with the two regressors
      - Move to the unvisited point closest to that predicted (x,y)
    """
    n = len(points)
    knn_map = precompute_knn(points, k=k)

    route = [start]
    visited = set(route)

    # For CSV output
    steps_info: List[Dict[str, Any]] = []
    cum_dist = 0.0
    prev_xy: Point = points[start]
    step_dist = 0.0

    # Step 0 (record a prediction too)
    print("Current step: ", 0)
    feat0 = make_feature_row(0, prev_xy, 0.0, 0.0, start, points, knn_map, feature_cols, k)
    pred0_x = float(model_x.predict(feat0)[0])
    pred0_y = float(model_y.predict(feat0)[0])
    steps_info.append({
        "step": 0, "node_index": start,
        "x": points[start][0], "y": points[start][1],
        "pred_x": pred0_x, "pred_y": pred0_y,
        "step_distance": 0.0, "cumulative_distance": 0.0,
        "prev_x": "", "prev_y": ""
    })

    cur = start
    for step in range(1, n):
        # Build features from the current node context
        print("Current step: ", step)
        feat = make_feature_row(step, prev_xy, step_dist, cum_dist, cur, points, knn_map, feature_cols, k)
        # Predict next (x, y)
        px = float(model_x.predict(feat)[0])
        py = float(model_y.predict(feat)[0])

        # Choose closest UNVISITED actual point to the predicted (px,py)
        best_j, best_d = None, float("inf")
        for j in range(n):
            if j in visited:
                continue
            d = euclid((px, py), points[j])
            if d < best_d:
                best_j, best_d = j, d

        next_node = best_j
        step_dist = euclid(points[cur], points[next_node])
        cum_dist += step_dist

        steps_info.append({
            "step": step,
            "node_index": next_node,
            "x": points[next_node][0], "y": points[next_node][1],
            "pred_x": px, "pred_y": py,
            "step_distance": step_dist, "cumulative_distance": cum_dist,
            "prev_x": points[cur][0], "prev_y": points[cur][1]
        })

        route.append(next_node)
        visited.add(next_node)
        # Prepare for next iteration: previous == this step's 'cur' BEFORE moving
        prev_xy = points[cur]
        cur = next_node

    if close_tour:
        step = n
        step_dist = euclid(points[cur], points[start])
        cum_dist += step_dist
        route.append(start)
        steps_info.append({
            "step": step, "node_index": start,
            "x": points[start][0], "y": points[start][1],
            "pred_x": "", "pred_y": "",
            "step_distance": step_dist, "cumulative_distance": cum_dist,
            "prev_x": points[cur][0], "prev_y": points[cur][1]
        })

    return route, steps_info

# ----------------- outputs -----------------
def write_route_csv(outfile: str, steps_info: List[Dict[str, Any]]):
    fieldnames = ["step","node_index","x","y","pred_x","pred_y",
                  "step_distance","cumulative_distance","prev_x","prev_y"]
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in steps_info:
            row = dict(r)
            # round floats for readability
            for k in ("x","y","pred_x","pred_y","step_distance","cumulative_distance","prev_x","prev_y"):
                if isinstance(row.get(k), (int,float)) and row[k] != "":
                    row[k] = float(f"{row[k]:.6f}")
            w.writerow(row)

def plot_route_png(imgfile: str, points: List[Point], route: List[int], title: str = "Greedy (model-guided) route"):
    xs = [points[i][0] for i in route]
    ys = [points[i][1] for i in route]
    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, marker="o")          # default colors
    # highlight start
    sx, sy = points[route[0]]
    plt.scatter([sx], [sy], s=100)
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(imgfile, dpi=150); plt.close()

# ----------------- main -----------------
if __name__ == "__main__":
    # Show device first
    print_device_info()

    # Config (edit if you like)
    MODELS_DIR = "models_tabpfn"            # folder with model_x.joblib, model_y.joblib, features.json
    N = 100                                   # number of points in new instance
    SEED = 123                                # RNG seed
    SPAN = 1000.0                             # coordinate span
    START_NODE = 0                            # starting index
    CLOSE_TOUR = True                         # return to start at the end?
    OUT_CSV = "greedy_pred_vs_actual.csv"     # output CSV
    OUT_PNG = "greedy_model_route.png"        # output image

    # Load saved models + schema
    model_x, model_y, feature_cols, k = load_models(MODELS_DIR)

    # Create a new instance (NO SOLVER!)
    points = generate_random_points(N, seed=SEED, span=SPAN)
    print("Build route")

    # Build route greedily using predictions
    route, steps_info = build_route_greedy(
        points, model_x, model_y, feature_cols, k,
        start=START_NODE, close_tour=CLOSE_TOUR
    )

    # Save outputs
    write_route_csv(OUT_CSV, steps_info)
    plot_route_png(OUT_PNG, points, route, title="Greedy model-guided TSP (no solver)")

    total_len = steps_info[-1]["cumulative_distance"]
    print(f"Route length ≈ {total_len:.3f}")
    print(f"CSV written: {OUT_CSV}")
    print(f"Image written: {OUT_PNG}")
