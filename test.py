import re
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -----------------------
# Geometry / TSP helpers
# -----------------------
Point = Tuple[float, float]

def euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def generate_random_points(n: int, seed: int = 0, span: float = 1000.0) -> List[Point]:
    random.seed(seed)
    return [(random.uniform(0, span), random.uniform(0, span)) for _ in range(n)]

def build_distance_matrix(points: List[Point], scale: float = 1000.0):
    n = len(points)
    m = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(n):
            if i != j:
                xj, yj = points[j]
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj) * scale))
    return m, scale

def solve_tsp(points: List[Point]):
    n = len(points)
    if n < 2:
        return [0, 0], 0.0

    dist_mat, scale = build_distance_matrix(points)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # depot=0
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
    params.time_limit.FromSeconds(10)

    sol = routing.SolveWithParameters(params)
    if not sol:
        raise RuntimeError("No solution found by OR-Tools.")

    idx = routing.Start(0)
    route = [manager.IndexToNode(idx)]
    total_cost = 0
    while not routing.IsEnd(idx):
        prev = idx
        idx = sol.Value(routing.NextVar(idx))
        total_cost += routing.GetArcCostForVehicle(prev, idx, 0)
        route.append(manager.IndexToNode(idx))

    return route, total_cost / 1000.0  # inverse of scale above

def k_nearest(points: List[Point], node_idx: int, k: int):
    x0, y0 = points[node_idx]
    dists = []
    for j, (x, y) in enumerate(points):
        if j == node_idx:
            continue
        dists.append((j, x, y, euclid((x0, y0), (x, y))))
    dists.sort(key=lambda t: t[3])
    return dists[:k]

# -------------------------------------------
# Writer for the "minimal header" like yours
# -------------------------------------------
def write_minimal_csv(outfile: str, points: List[Point], route: List[int], k: int):
    """
    Header (matches your screenshot style):
      step, x, y, step_distance, cumulative_distance, prev_x, prev_y, knn_k,
      knn1_x, knn1_y, knn1_distance, ..., knn{k}_x, knn{k}_y, knn{k}_distance
    """
    base = ["step", "x", "y", "step_distance", "cumulative_distance", "prev_x", "prev_y", "knn_k"]
    knn_fields = []
    for r in range(1, k+1):
        knn_fields += [f"knn{r}_x", f"knn{r}_y", f"knn{r}_distance"]
    fieldnames = base + knn_fields

    rows = []
    cum = 0.0
    for step, node in enumerate(route):
        x, y = points[node]
        if step == 0:
            step_dist = 0.0
            prev_x = ""
            prev_y = ""
        else:
            px, py = points[route[step-1]]
            step_dist = euclid((px, py), (x, y))
            cum += step_dist
            prev_x, prev_y = px, py

        knn = k_nearest(points, node, k)
        row = {
            "step": step,
            "x": x,
            "y": y,
            "step_distance": round(step_dist, 6),
            "cumulative_distance": round(cum, 6),
            "prev_x": prev_x,
            "prev_y": prev_y,
            "knn_k": k,
        }
        for r in range(1, k+1):
            if r <= len(knn):
                _, nx, ny, d = knn[r-1]
                row[f"knn{r}_x"] = nx
                row[f"knn{r}_y"] = ny
                row[f"knn{r}_distance"] = round(d, 6)
            else:
                row[f"knn{r}_x"] = ""
                row[f"knn{r}_y"] = ""
                row[f"knn{r}_distance"] = ""
        rows.append(row)

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# -----------------------
# ML helpers
# -----------------------
def infer_k_from_columns(columns: List[str]) -> int:
    k = 0
    pat = re.compile(r"^knn(\d+)_x$")
    for c in columns:
        m = pat.match(c)
        if m:
            k = max(k, int(m.group(1)))
    return k

def train_two_models(train_csv: str):
    df = pd.read_csv(train_csv)
    # Features = everything except x, y
    feature_cols = [c for c in df.columns if c not in ("x", "y")]
    X = df[feature_cols]
    yx = df["x"].astype(float)
    yy = df["y"].astype(float)

    # quick split just to report performance on holdout (optional)
    X_tr, X_te, yx_tr, yx_te = train_test_split(X, yx, test_size=0.25, random_state=42)
    _,  _,  yy_tr, yy_te = train_test_split(X, yy, test_size=0.25, random_state=42)

    print("Training TabPFNRegressor for x …")
    model_x = TabPFNRegressor()
    model_x.fit(X_tr, yx_tr)
    pred_x = model_x.predict(X_te)
    print(f"[x] MSE={mean_squared_error(yx_te, pred_x):.6f}  R2={r2_score(yx_te, pred_x):.4f}")

    print("Training TabPFNRegressor for y …")
    model_y = TabPFNRegressor()
    model_y.fit(X_tr, yy_tr)  # same split for fair-ish comparison
    pred_y = model_y.predict(X_te)
    print(f"[y] MSE={mean_squared_error(yy_te, pred_y):.6f}  R2={r2_score(yy_te, pred_y):.4f}")

    return model_x, model_y, feature_cols

# -----------------------
# Main pipeline function
# -----------------------
def make_and_compare(
    train_csv: str,
    out_new_tsp_csv: str = "new_tsp_instance.csv",
    out_compare_csv: str = "pred_vs_actual.csv",
    n_new: int = 100,
    seed: int = 123,
    span: float = 1000.0,
):
    # 1) Train models on the provided CSV
    model_x, model_y, feature_cols = train_two_models(train_csv)

    # 2) Infer k from training header, then create a brand-new TSP instance
    train_df = pd.read_csv(train_csv, nrows=1)
    k = infer_k_from_columns(list(train_df.columns))
    if k == 0:
        raise ValueError("Could not infer k from training CSV header (expect knn1_x, …).")

    points_new = generate_random_points(n_new, seed=seed, span=span)
    route_new, _ = solve_tsp(points_new)
    write_minimal_csv(out_new_tsp_csv, points_new, route_new, k=k)
    print(f"New TSP CSV written: {out_new_tsp_csv}")

    # 3) Predict x,y for the new TSP CSV using ONLY the feature columns
    df_new = pd.read_csv(out_new_tsp_csv)

    # Ensure the new CSV has all the feature columns used in training
    missing = [c for c in feature_cols if c not in df_new.columns]
    if missing:
        raise ValueError(f"New TSP CSV is missing expected feature columns: {missing}")

    X_new = df_new[feature_cols]
    pred_x = model_x.predict(X_new)
    pred_y = model_y.predict(X_new)

    # 4) Compare with ground truth x,y from the new TSP
    actual_x = df_new["x"].to_numpy(dtype=float)
    actual_y = df_new["y"].to_numpy(dtype=float)
    err_x = pred_x - actual_x
    err_y = pred_y - actual_y
    err_l2 = np.sqrt(err_x**2 + err_y**2)

    comp = pd.DataFrame({
        "step": df_new["step"],
        "actual_x": actual_x,
        "pred_x": pred_x,
        "err_x": err_x,
        "actual_y": actual_y,
        "pred_y": pred_y,
        "err_y": err_y,
        "err_l2": err_l2,
    })
    comp.to_csv(out_compare_csv, index=False)
    print(f"Comparison CSV written: {out_compare_csv}")

    # quick summary
    print("\n--- Summary on new instance ---")
    print(f"RMSE_x: {math.sqrt(mean_squared_error(actual_x, pred_x)):.6f}")
    print(f"RMSE_y: {math.sqrt(mean_squared_error(actual_y, pred_y)):.6f}")
    print(f"Mean L2 error: {err_l2.mean():.6f}")

# -------------
# Entry point
# -------------
if __name__ == "__main__":
    # EDIT THESE THREE PATHS / PARAMS AS YOU LIKE:
    TRAIN_CSV = "tsp_route.csv"         # <- your original CSV (with header like screenshot)
    NEW_TSP_CSV = "new_tsp_instance.csv"         # output: freshly generated TSP CSV (same header shape)
    COMPARE_CSV = "pred_vs_actual.csv"           # output: predictions vs actuals for the new instance

    make_and_compare(
        train_csv=TRAIN_CSV,
        out_new_tsp_csv=NEW_TSP_CSV,
        out_compare_csv=COMPARE_CSV,
        n_new=100,   # size of new TSP instance
        seed=123,    # RNG seed
        span=1000.0  # coordinate span for random generation
    )
