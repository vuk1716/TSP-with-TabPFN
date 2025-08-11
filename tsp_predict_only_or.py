# tsp_predict_only.py
# Predict-only (greedy, no solver) + OR-Tools comparison in ONE side-by-side figure.
# Outputs saved into outputs/<YYYYMMDD_HHMMSS>/ with timestamped filenames.

import json
import re
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# OR-Tools (for the solver baseline)
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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

def route_length(points: List[Point], route: List[int]) -> float:
    total = 0.0
    for i in range(1, len(route)):
        total += euclid(points[route[i-1]], points[route[i]])
    return total

def generate_random_points(n: int, seed: int = 123, span: float = 1000.0) -> List[Point]:
    random.seed(seed)
    return [(random.uniform(0, span), random.uniform(0, span)) for _ in range(n)]

def precompute_knn(points: List[Point], k: int) -> Dict[int, List[Tuple[int, float]]]:
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
    # Unpickling may require the same libs used in training (e.g., tabpfn installed)
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
    row: Dict[str, Any] = {}
    row["step"] = int(step)
    row["step_distance"] = float(step_dist)
    row["cumulative_distance"] = float(cum_dist)
    row["prev_x"] = float(prev_xy[0]) if step > 0 else np.nan
    row["prev_y"] = float(prev_xy[1]) if step > 0 else np.nan
    row["knn_k"] = int(k)

    # KNN of the CURRENT node
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
    n = len(points)
    knn_map = precompute_knn(points, k=k)

    route = [start]
    visited = set(route)

    steps_info: List[Dict[str, Any]] = []
    cum_dist = 0.0
    prev_xy: Point = points[start]
    step_dist = 0.0

    # Step 0 (record a prediction too)
    feat0 = make_feature_row(0, prev_xy, 0.0, 0.0, start, points, knn_map, feature_cols, k)
    print("Current step: ", 0)
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
        print("Current step: ", step)
        feat = make_feature_row(step, prev_xy, step_dist, cum_dist, cur, points, knn_map, feature_cols, k)
        px = float(model_x.predict(feat)[0])
        py = float(model_y.predict(feat)[0])

        # closest UNVISITED point to predicted (px,py)
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

# ----------------- OR-Tools baseline on the same instance -----------------
def _build_distance_matrix(points: List[Point], scale: float = 1000.0):
    n = len(points)
    m = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(n):
            if i != j:
                xj, yj = points[j]
                m[i][j] = int(round(euclid((xi, yi), (xj, yj)) * scale))
    return m, scale

def solve_with_ortools(points: List[Point], depot: int = 0, time_limit_s: int = 10) -> List[int]:
    n = len(points)
    dist_mat, _ = _build_distance_matrix(points)
    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def cb(fi, ti):
        i = manager.IndexToNode(fi)
        j = manager.IndexToNode(ti)
        return dist_mat[i][j]

    transit_idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    sol = routing.SolveWithParameters(params)
    if not sol:
        raise RuntimeError("OR-Tools could not find a solution.")

    idx = routing.Start(0)
    route = [manager.IndexToNode(idx)]
    while not routing.IsEnd(idx):
        idx = sol.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
    return route  # closed (ends at depot)

# ----------------- outputs -----------------
def write_route_csv(outfile: Path, steps_info: List[Dict[str, Any]]):
    fieldnames = ["step","node_index","x","y","pred_x","pred_y",
                  "step_distance","cumulative_distance","prev_x","prev_y"]
    with open(str(outfile), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in steps_info:
            row = dict(r)
            for k in ("x","y","pred_x","pred_y","step_distance","cumulative_distance","prev_x","prev_y"):
                if isinstance(row.get(k), (int,float)) and row[k] != "":
                    row[k] = float(f"{row[k]:.6f}")
            w.writerow(row)

def write_simple_route_csv(outfile: Path, points: List[Point], route: List[int]):
    fieldnames = ["step","node_index","x","y","step_distance","cumulative_distance"]
    cum = 0.0
    rows = []
    for s, node in enumerate(route):
        x, y = points[node]
        if s == 0:
            step_d = 0.0
        else:
            prev = route[s-1]
            step_d = euclid(points[prev], (x, y))
            cum += step_d
        rows.append({
            "step": s, "node_index": node, "x": x, "y": y,
            "step_distance": step_d, "cumulative_distance": cum
        })
    with open(str(outfile), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r = r.copy()
            for k in ("x","y","step_distance","cumulative_distance"):
                r[k] = float(f"{r[k]:.6f}")
            w.writerow(r)

def plot_routes_side_by_side_png(imgfile: Path,
                                 points: List[Point],
                                 greedy_route: List[int], greedy_len: float,
                                 solver_route: List[int], solver_len: float,
                                 start_idx: int):
    gap_abs = greedy_len - solver_len
    gap_pct = (gap_abs / solver_len) * 100.0 if solver_len > 0 else float("nan")

    xs_g = [points[i][0] for i in greedy_route]
    ys_g = [points[i][1] for i in greedy_route]
    xs_s = [points[i][0] for i in solver_route]
    ys_s = [points[i][1] for i in solver_route]

    plt.figure(figsize=(14, 6))

    # Left: Greedy
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(xs_g, ys_g, marker="o")
    sx, sy = points[start_idx]
    ax1.scatter([sx], [sy], s=120)
    ax1.set_title(f"Greedy (len={greedy_len:.2f})")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y")
    ax1.axis("equal")

    # Right: OR-Tools
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(xs_s, ys_s, marker="o", linestyle="--")
    ax2.scatter([sx], [sy], s=120)
    ax2.set_title(f"OR-Tools (len={solver_len:.2f})")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax2.axis("equal")

    plt.suptitle(f"Greedy vs OR-Tools • Gap={gap_abs:.2f} ({gap_pct:.2f}%)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(str(imgfile), dpi=150)
    plt.close()

# ----------------- main -----------------
if __name__ == "__main__":
    # Show device first
    print_device_info()

    # Config (edit if you like)
    MODELS_DIR = "models_tabpfn"   # folder with model_x.joblib, model_y.joblib, features.json
    N = 1000                       # number of points in new instance
    SEED = 123                     # RNG seed
    SPAN = 1000.0                  # coordinate span
    START_NODE = 0                 # starting index
    CLOSE_TOUR = True              # return to start at the end?

    # --------- timestamped output folder & filenames ---------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("outputs")
    run_dir = base_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    OUT_GREEDY_CSV = run_dir / f"greedy_pred_vs_actual_{ts}.csv"
    OUT_SOLVER_CSV = run_dir / f"solver_route_{ts}.csv"
    OUT_BOTH_PNG   = run_dir / f"routes_side_by_side_{ts}.png"

    print(f"Output directory: {run_dir.resolve()}")

    # Load saved models + schema
    model_x, model_y, feature_cols, k = load_models(MODELS_DIR)

    # Create a new instance
    points = generate_random_points(N, seed=SEED, span=SPAN)

    # Build greedy (model-guided) route
    greedy_route, greedy_steps_info = build_route_greedy(
        points, model_x, model_y, feature_cols, k,
        start=START_NODE, close_tour=CLOSE_TOUR
    )
    greedy_len = route_length(points, greedy_route)

    # Solve the same instance with OR-Tools (baseline)
    solver_route = solve_with_ortools(points, depot=START_NODE, time_limit_s=10)
    solver_len = route_length(points, solver_route)

    # Save outputs (timestamped in subfolder)
    write_route_csv(OUT_GREEDY_CSV, greedy_steps_info)
    write_simple_route_csv(OUT_SOLVER_CSV, points, solver_route)
    plot_routes_side_by_side_png(OUT_BOTH_PNG, points,
                                 greedy_route, greedy_len,
                                 solver_route, solver_len,
                                 start_idx=START_NODE)

    # Report comparison in console too
    gap_abs = greedy_len - solver_len
    gap_pct = (gap_abs / solver_len) * 100.0 if solver_len > 0 else float("nan")
    print(f"Greedy route length  : {greedy_len:.6f}")
    print(f"Solver route length  : {solver_len:.6f}")
    print(f"Gap (greedy - solver): {gap_abs:.6f}  ({gap_pct:.2f}%)")

    print(f"\nCSV written: {OUT_GREEDY_CSV.name} (greedy), {OUT_SOLVER_CSV.name} (solver)")
    print(f"PNG written: {OUT_BOTH_PNG.name}")
