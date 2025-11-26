import numpy as np
import os
import json
from fct.admm_functions import compute_rho_for_acc_admm
 


def choose_adaptive_intervals(kappa: float, n_points: int = 20):
    """
    Automatically generate adaptive grids for v1 and v2 centered around
    the theoretical (v1, v2) values for a given κ and algorithm.
    Intervals expand logarithmically with κ and never collapse to zero.
    """
    L = kappa

    # --- Compute theoretical centers with TM parameters ---
    gamma = 1 - 1 / np.sqrt(kappa)
    v1_center = (1 + gamma) / L
    v2_center = (gamma ** 2) / (2 - gamma)

    # --- Log-scaling of interval spread with kappa ---
    spread_scale = np.log10(kappa + 1) / np.log10(10000 + 1)  # ∈ [0, 1]
    rel_spread_v1 = 0.05 + 0.2 * spread_scale   # ~5–30%

    # Minimum width to avoid zero grid near center 0
    min_width_v1 = 1e-3 / L

    # Compute span values
    v1_span = max(v1_center * rel_spread_v1, min_width_v1 / 2)
    v2_span = 0.2

    v2_min = max(0.0, v2_center - v2_span)
    v2_max = min(1.0, v2_center + v2_span)

    v1_min = max(0.0, (v1_center - v1_span))
    v1_max = v1_center + 0.2*v1_span

    # --- Build grids ---
    v1_grid = np.linspace(v1_min, v1_max, n_points)
    v1_target = v1_center
    v1_grid = np.unique(np.append(v1_grid, v1_target))
    v2_grid = np.linspace(v2_min, v2_max, n_points)

    ### Note:
    # For these two values of kappa, we need higher resolution to obtain good results
    if kappa == 100:
        v2_grid = np.linspace(0.695, 0.715, 40)
        v2_target = v2_center
        v2_grid = np.unique(np.append(v2_grid, v2_target))

    if kappa == 1000:
        v2_grid = np.linspace(0.76, 0.8, 40)
        v2_grid = np.linspace(v2_min, v2_max, n_points)
        v2_target = v2_center
        v2_grid = np.unique(np.append(v2_grid, v2_target))

    return v1_grid, v2_grid


def run_grid_search(kappa, *, threshold, n_ZF, algo, alpha, n_points, json_filename):
    """
    If kappa is a float → evaluate one κ.
    If kappa is a list/array → evaluate all κ values in a loop.
    Save all results into `json_filename`.
    """

    # CASE 1: kappa is a list/array so we do a loop here
    if not isinstance(kappa, (int, float)):   # list, array, iterable
        all_results = {}

        for k in kappa:
            r = run_grid_search(float(k), threshold=threshold, n_ZF=n_ZF, algo=algo, alpha=alpha,
                n_points=n_points, json_filename=json_filename)
            all_results[str(k)] = r
        return all_results

    # CASE 2: kappa is a single float, then we just do the original logic
    kappa = float(kappa)

    def _resolve(th, k):
        if callable(th):
            return float(th(k))
        if isinstance(th, dict):
            return float(th.get(k, np.inf))
        return float(th)

    L = kappa
    v1_values, v2_values = choose_adaptive_intervals(kappa, n_points)
    cutoff = _resolve(threshold, kappa)

    best_v1 = []
    best_v2 = []
    best_rate = float("inf")

    for v1 in v1_values:
        for v2 in v2_values:
            try:
                rate = compute_rho_for_acc_admm(1, L, n_ZF, algo=algo, v1=v1, v2=v2, rho_max=1.3,
                    eps=1e-6, alpha=alpha)
            except Exception:
                continue

            if rate <= cutoff:
                print(f"κ={kappa:.3g} | v1={v1:.5f} v2={v2:.5f} | rate={rate:.5f} <= cutoff={cutoff:.5f}")
                if rate < best_rate:
                    print(f"New BEST: rate={rate:.6f}, v1={v1:.6f}, v2={v2:.6f}")
                    best_rate = rate
                    best_v1 = [float(v1)]
                    best_v2 = [float(v2)]
                elif rate == best_rate:
                    print(f"Equal BEST: v1={v1:.6f}, v2={v2:.6f}")
                    best_v1.append(float(v1))
                    best_v2.append(float(v2))

    def _unique(lst):
        out, seen = [], set()
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    result = {
        "kappa": kappa,
        "best_rate": best_rate,
        "v1": _unique(best_v1),
        "v2": _unique(best_v2),
    }

    ### save jason ###
    # load existing content if present
    data = {}
    if os.path.exists(json_filename):
        try:
            with open(json_filename, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    data[str(kappa)] = result

    with open(json_filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Stored κ={kappa} results in '{json_filename}'")

    return result


def extract_data(data):
  
    grid_kappa = []
    grid_rate = []
    grid_v1 = []
    grid_v2 = []

    G_kappa = []
    G_rate = []

    for key, entry in data.items():

        if key == "G_results":
            for k, v in entry.items():
                kappa_val = float(k)  
                G_kappa.append(kappa_val)
                G_rate.append(v["G_rate"])
  
            continue

        try:
            _ = float(key)   
        except ValueError:
            continue

        grid_kappa.append(entry["kappa"])
        grid_rate.append(entry["rate"][0])
        grid_v1.append(entry["v1"][0])
        grid_v2.append(entry["v2"][0])

    def sort_by_kappa(k, *others):
        idx = sorted(range(len(k)), key=lambda i: k[i])
        return ([k[i] for i in idx],) + tuple([ [arr[i] for i in idx] for arr in others ])

    grid_sorted = sort_by_kappa(grid_kappa, grid_rate, grid_v1, grid_v2)
    G_sorted    = sort_by_kappa(G_kappa,    G_rate)

    return grid_sorted + G_sorted

