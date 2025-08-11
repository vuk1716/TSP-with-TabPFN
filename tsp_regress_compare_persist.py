import json, re, csv, math, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import joblib

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ----------------- TSP helpers -----------------
Point = Tuple[float, float]
def euclid(a: Point, b: Point) -> float: return math.hypot(a[0]-b[0], a[1]-b[1])

def generate_random_points(n: int, seed: int = 0, span: float = 1000.0) -> List[Point]:
    random.seed(seed); return [(random.uniform(0, span), random.uniform(0, span)) for _ in range(n)]

def build_distance_matrix(points: List[Point], scale: float = 1000.0):
    n = len(points); m = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(n):
            if i != j:
                xj, yj = points[j]
                m[i][j] = int(round(math.hypot(xi - xj, yi - yj) * scale))
    return m, scale

def solve_tsp(points: List[Point]):
    n = len(points)
    if n < 2: return [0, 0], 0.0
    dist_mat, scale = build_distance_matrix(points)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def cb(fi, ti):
        i = manager.IndexToNode(fi); j = manager.IndexToNode(ti)
        return dist_mat[i][j]
    transit_idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(10)
    sol = routing.SolveWithParameters(params)
    if not sol: raise RuntimeError("No solution found by OR-Tools.")
    idx = routing.Start(0); route = [manager.IndexToNode(idx)]; total = 0
    while not routing.IsEnd(idx):
        prev = idx; idx = sol.Value(routing.NextVar(idx))
        total += routing.GetArcCostForVehicle(prev, idx, 0)
        route.append(manager.IndexToNode(idx))
    return route, total / 1000.0

def k_nearest(points: List[Point], node_idx: int, k: int):
    x0, y0 = points[node_idx]
    arr = [(j, *points[j], euclid((x0,y0), points[j])) for j in range(len(points)) if j != node_idx]
    arr.sort(key=lambda t: t[3]); return arr[:k]

def write_minimal_csv(outfile: str, points: List[Point], route: List[int], k: int):
    base = ["step", "x", "y", "step_distance", "cumulative_distance", "prev_x", "prev_y", "knn_k"]
    knn_fields = []
    for r in range(1, k + 1):
        knn_fields += [f"knn{r}_x", f"knn{r}_y", f"knn{r}_distance"]
    fieldnames = base + knn_fields

    rows = []
    cum = 0.0
    for step, node in enumerate(route):
        x, y = points[node]
        if step == 0:
            step_dist = 0.0
            prevx, prevy = "", ""
        else:
            px, py = points[route[step - 1]]
            step_dist = euclid((px, py), (x, y))
            cum += step_dist
            prevx, prevy = px, py

        knn = k_nearest(points, node, k)

        row = {
            "step": step,
            "x": x,
            "y": y,
            "step_distance": round(step_dist, 6),
            "cumulative_distance": round(cum, 6),
            "prev_x": prevx,
            "prev_y": prevy,
            "knn_k": k,
        }

        for r in range(1, k + 1):
            if r <= len(knn):
                _, nx, ny, d = knn[r - 1]
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


# --------------- ML: save/load ---------------
def infer_k_from_columns(columns: List[str]) -> int:
    k = 0; pat = re.compile(r"^knn(\d+)_x$")
    for c in columns:
        m = pat.match(c)
        if m: k = max(k, int(m.group(1)))
    return k

def save_models(models_dir: str, model_x, model_y, feature_cols: List[str]):
    p = Path(models_dir); p.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_x, p / "model_x.joblib")
    joblib.dump(model_y, p / "model_y.joblib")
    (p / "features.json").write_text(json.dumps({"feature_cols": feature_cols}, indent=2))
    print(f"Saved models to {p.resolve()}")

def load_models(models_dir: str):
    p = Path(models_dir)
    model_x = joblib.load(p / "model_x.joblib")
    model_y = joblib.load(p / "model_y.joblib")
    meta = json.loads((p / "features.json").read_text())
    feature_cols = meta["feature_cols"]
    print(f"Loaded models from {p.resolve()}")
    return model_x, model_y, feature_cols

def train_two_models(train_csv: str):
    df = pd.read_csv(train_csv)
    feature_cols = [c for c in df.columns if c not in ("x", "y")]
    X = df[feature_cols]
    yx = df["x"].astype(float); yy = df["y"].astype(float)
    X_tr, X_te, yx_tr, yx_te = train_test_split(X, yx, test_size=0.25, random_state=42)
    _, _, yy_tr, yy_te = train_test_split(X, yy, test_size=0.25, random_state=42)

    print("Training TabPFNRegressor for x …")
    model_x = TabPFNRegressor(); model_x.fit(X_tr, yx_tr)
    pred_x = model_x.predict(X_te)
    print(f"[x] MSE={mean_squared_error(yx_te, pred_x):.6f}  R2={r2_score(yx_te, pred_x):.4f}")

    print("Training TabPFNRegressor for y …")
    model_y = TabPFNRegressor(); model_y.fit(X_tr, yy_tr)
    pred_y = model_y.predict(X_te)
    print(f"[y] MSE={mean_squared_error(yy_te, pred_y):.6f}  R2={r2_score(yy_te, pred_y):.4f}")
    return model_x, model_y, feature_cols

# --------------- Pipeline (train/load + predict) ---------------
def run_pipeline(
    train_csv: str = None,
    models_dir: str = "models_tabpfn",
    out_new_tsp_csv: str = "new_tsp_instance.csv",
    out_compare_csv: str = "pred_vs_actual.csv",
    n_new: int = 100,
    seed: int = 123,
    span: float = 1000.0,
    force_train: bool = False,
):
    models_exist = Path(models_dir, "model_x.joblib").exists() and Path(models_dir, "model_y.joblib").exists()
    if force_train or (not models_exist):
        if not train_csv:
            raise ValueError("Training CSV is required for first-time training.")
        print("Training models…")
        model_x, model_y, feature_cols = train_two_models(train_csv)
        save_models(models_dir, model_x, model_y, feature_cols)
    else:
        print("Models found; loading …")
        model_x, model_y, feature_cols = load_models(models_dir)

    # Make a brand-new TSP instance and write CSV with SAME header pattern (infers k from training CSV)
    if train_csv:
        k = infer_k_from_columns(list(pd.read_csv(train_csv, nrows=1).columns))
    else:
        # If training CSV not provided (predict-only run), assume k from saved feature columns
        k = max([int(re.findall(r"knn(\d+)_x", c)[0]) for c in feature_cols if re.findall(r"knn(\d+)_x", c)] or [0])
        if k == 0: raise ValueError("Could not infer k; run once with a training CSV.")
    points_new = generate_random_points(n_new, seed=seed, span=span)
    route_new, _ = solve_tsp(points_new)
    write_minimal_csv(out_new_tsp_csv, points_new, route_new, k=k)
    print(f"New TSP CSV written: {out_new_tsp_csv}")

    # Predict and compare (ensure same feature set/order)
    df_new = pd.read_csv(out_new_tsp_csv)
    # Features used in training:
    X_new = df_new[feature_cols]
    pred_x = model_x.predict(X_new); pred_y = model_y.predict(X_new)
    actual_x = df_new["x"].to_numpy(float); actual_y = df_new["y"].to_numpy(float)
    err_x = pred_x - actual_x; err_y = pred_y - actual_y
    err_l2 = np.sqrt(err_x**2 + err_y**2)
    comp = pd.DataFrame({
        "step": df_new["step"],
        "actual_x": actual_x, "pred_x": pred_x, "err_x": err_x,
        "actual_y": actual_y, "pred_y": pred_y, "err_y": err_y,
        "err_l2": err_l2,
    })
    comp.to_csv(out_compare_csv, index=False)
    print(f"Comparison CSV written: {out_compare_csv}")
    print(f"RMSE_x={math.sqrt(mean_squared_error(actual_x, pred_x)):.6f}  "
          f"RMSE_y={math.sqrt(mean_squared_error(actual_y, pred_y)):.6f}  "
          f"Mean L2={err_l2.mean():.6f}")

if __name__ == "__main__":
    # First run (trains and saves):
    # run_pipeline(train_csv="your_training_file.csv", models_dir="models_tabpfn")

    # Subsequent runs (no retrain, just generate new instance + predict):
    # run_pipeline(models_dir="models_tabpfn", train_csv=None)

    # Default example (edit paths):
    run_pipeline(
        train_csv="tsp_route.csv",   # set to None after first run
        models_dir="models_tabpfn",
        out_new_tsp_csv="new_tsp_instance.csv",
        out_compare_csv="pred_vs_actual.csv",
        n_new=100, seed=123, span=1000.0,
        force_train=False,  # set True to retrain and overwrite saved models
    )
