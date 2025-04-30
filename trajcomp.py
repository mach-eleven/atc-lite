import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.signal import resample
import argparse

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None
try:
    from shapely.geometry import LineString
except ImportError:
    LineString = None

def load_trajectory(pyfile):
    spec = importlib.util.spec_from_file_location("traj_module", pyfile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.trajectories

def resample_traj(traj, n_points):
    arr = np.array(traj)
    if len(arr) == n_points:
        return arr
    return resample(arr, n_points)

def trajectory_length(traj):
    diffs = np.diff(traj, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

def curvature(traj):
    diffs = np.diff(traj, axis=0)
    angles = np.arctan2(diffs[:,1], diffs[:,0])
    curv = np.abs(np.diff(angles))
    return np.mean(curv), np.max(curv)

def speed_profile(traj):
    diffs = np.diff(traj, axis=0)
    return np.linalg.norm(diffs, axis=1)

def estimate_fuel_use_realistic(traj, v_min=100, v_max=300, cruise_consumption=1.0):
    traj = np.array(traj)
    total_fuel = 0.0
    for i in range(1, len(traj)):
        dx, dy = traj[i,0] - traj[i-1,0], traj[i,1] - traj[i-1,1]
        dist = np.sqrt(dx**2 + dy**2)
        # Use a reasonable speed for fuel model (could use dist or a fixed cruise speed)
        speed = max(dist, v_min)
        speed_factor = (speed - v_min) / (v_max - v_min)
        base_consumption = cruise_consumption * (0.8 + 0.4 * speed_factor**2)
        # Fuel burned is proportional to distance, not time steps
        fuel_burned = base_consumption * dist
        total_fuel += fuel_burned
    return total_fuel

def compute_metrics(traj1, traj2, nofrechet=False):
    t1 = np.array(traj1[0][0])
    t2 = np.array(traj2[0][0])
    n_points = min(len(t1), len(t2))
    t1_rs = resample_traj(t1, n_points)
    t2_rs = resample_traj(t2, n_points)
    dists = np.linalg.norm(t1_rs - t2_rs, axis=1)
    avg_dist = np.mean(dists)
    max_dist = np.max(dists)
    len_diff = abs(len(t1) - len(t2))
    # DTW distance
    if fastdtw:
        dtw_dist, _ = fastdtw(t1, t2)
        # Speed profile DTW
        sp1 = speed_profile(t1)
        sp2 = speed_profile(t2)
        sp_dtw, _ = fastdtw(sp1, sp2)
    else:
        dtw_dist = None
        sp_dtw = None
    # Fréchet distance
    if LineString:
        ls1 = LineString(t1)
        ls2 = LineString(t2)
        import shapely
        print(f"Shapely version: {shapely.__version__}")
        print(f"LineString has frechet_distance: {hasattr(ls1, 'frechet_distance')}")
        if hasattr(ls1, "frechet_distance") and not nofrechet:
            try:
                frechet = ls1.frechet_distance(ls2)
            except Exception as e:
                print(f"Error computing Fréchet distance: {e}")
                frechet = None
        else:
            if not nofrechet:
                print("Fréchet distance not available in this Shapely version or object.")
            frechet = None
        hausdorff = ls1.hausdorff_distance(ls2)
        try:
            from shapely.geometry import Polygon
            poly = Polygon(np.vstack([t1, t2[::-1]]))
            area_between = poly.area
        except Exception:
            area_between = None
    else:
        frechet = None
        hausdorff = None
        area_between = None
    # Angle difference (shape similarity)
    def angle_diff(a, b):
        da = np.diff(a, axis=0)
        db = np.diff(b, axis=0)
        ang_a = np.arctan2(da[:,1], da[:,0])
        ang_b = np.arctan2(db[:,1], db[:,0])
        return np.mean(np.abs(np.unwrap(ang_a) - np.unwrap(ang_b)))
    angle_similarity = angle_diff(t1_rs, t2_rs)
    # Curvature
    mean_curv1, max_curv1 = curvature(t1)
    mean_curv2, max_curv2 = curvature(t2)
    # Trajectory length
    length1 = trajectory_length(t1)
    length2 = trajectory_length(t2)
    # Speed profile similarity (L2 norm)
    sp1 = speed_profile(t1_rs)
    sp2 = speed_profile(t2_rs)
    minlen = min(len(sp1), len(sp2))
    speed_l2 = np.linalg.norm(sp1[:minlen] - sp2[:minlen])
    # Fuel/energy efficiency comparison (realistic)
    fuel1 = estimate_fuel_use_realistic(t1)
    fuel2 = estimate_fuel_use_realistic(t2)
    fuel1_per_unit = fuel1 / length1 if length1 > 0 else float('nan')
    fuel2_per_unit = fuel2 / length2 if length2 > 0 else float('nan')
    fuel_efficiency_ratio = fuel1_per_unit / fuel2_per_unit if fuel2_per_unit > 0 else float('nan')
    metrics = {
        'fuel_use_trajectory1': fuel1,
        'fuel_use_trajectory2': fuel2,
        'fuel_per_unit_distance_trajectory1': fuel1_per_unit,
        'fuel_per_unit_distance_trajectory2': fuel2_per_unit,
        'fuel_efficiency_ratio (saved/real)': fuel_efficiency_ratio,
    }
    metrics.update({
        'avg_pointwise_distance': avg_dist,
        'max_pointwise_distance': max_dist,
        'trajectory_length_difference': len_diff,
        'trajectory1_length': len(t1),
        'trajectory2_length': len(t2),
        'dtw_distance': dtw_dist,
        'frechet_distance': frechet,
        'hausdorff_distance': hausdorff,
        'area_between_curves': area_between,
        'mean_angle_difference': angle_similarity,
        'trajectory1_total_length': length1,
        'trajectory2_total_length': length2,
        'trajectory1_mean_curvature': mean_curv1,
        'trajectory2_mean_curvature': mean_curv2,
        'trajectory1_max_curvature': max_curv1,
        'trajectory2_max_curvature': max_curv2,
        'speed_profile_l2': speed_l2,
        'speed_profile_dtw': sp_dtw,
    })
    return metrics, t1, t2

def plot_trajectories(t1, t2, outpath="traj_comparison.png"):
    plt.figure(figsize=(10, 8))
    plt.plot(t1[:,0], t1[:,1], label="Saved Trajectory", linewidth=3, alpha=0.8)
    plt.plot(t2[:,0], t2[:,1], label="Reference Trajectory", linewidth=3, alpha=0.8)
    plt.scatter(t1[0,0], t1[0,1], c='green', s=80, label='Start (Saved)')
    plt.scatter(t2[0,0], t2[0,1], c='red', s=80, label='Start (Reference)')
    plt.title("Trajectory Comparison")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved trajectory comparison plot to {outpath}")

def print_summary_table(metrics):
    # Calculate percentage match (inverse of avg_pointwise_distance normalized by trajectory length)
    # and fuel efficiency improvement
    tlen = (metrics['trajectory1_total_length'] + metrics['trajectory2_total_length']) / 2
    if tlen > 0:
        percent_match = max(0, 100 * (1 - metrics['avg_pointwise_distance'] / tlen))
    else:
        percent_match = float('nan')
    fuel_eff = metrics['fuel_per_unit_distance_trajectory2']
    fuel_eff_saved = metrics['fuel_per_unit_distance_trajectory1']
    total_fuel_real = metrics['fuel_use_trajectory2']
    total_fuel_saved = metrics['fuel_use_trajectory1']
    if total_fuel_real > 0:
        total_fuel_improvement = 100 * (total_fuel_real - total_fuel_saved) / total_fuel_real
    else:
        total_fuel_improvement = float('nan')
    print("\n=== FINAL TRAJECTORY MATCH & EFFICIENCY SUMMARY ===")
    print("{:<35} {:>15}".format("Metric", "Value"))
    print("-"*52)
    print("{:<35} {:>14.2f} %".format("Trajectory Match (%)", percent_match))
    print("{:<35} {:>14.2f} %".format("Total Fuel Improvement", total_fuel_improvement))
    print("{:<35} {:>14.2f}".format("Avg Pointwise Distance", metrics['avg_pointwise_distance']))
    print("{:<35} {:>14.2f}".format("Hausdorff Distance", metrics['hausdorff_distance'] if metrics['hausdorff_distance'] is not None else float('nan')))
    print("{:<35} {:>14.2f}".format("Area Between Curves", metrics['area_between_curves'] if metrics['area_between_curves'] is not None else float('nan')))
    print("{:<35} {:>14.2f}".format("Saved Traj Fuel/Unit Dist", metrics['fuel_per_unit_distance_trajectory1']))
    print("{:<35} {:>14.2f}".format("Real Traj Fuel/Unit Dist", metrics['fuel_per_unit_distance_trajectory2']))
    print("{:<35} {:>14.2f}".format("Saved Traj Total Fuel", metrics['fuel_use_trajectory1']))
    print("{:<35} {:>14.2f}".format("Real Traj Total Fuel", metrics['fuel_use_trajectory2']))
    print("-"*52)
    print("(Higher match % and higher total fuel improvement % are better)")

def main():
    import glob
    import os
    parser = argparse.ArgumentParser(description="Compare saved and real trajectories with optional cropping.")
    parser.add_argument('--crop', type=int, default=0, help='Crop N points (as a fraction of the shorter trajectory) from the end of both trajectories before comparison')
    parser.add_argument('--nofrechet', action='store_true', help='Skip Fréchet distance calculation')
    args = parser.parse_args()
    saved_trajs = sorted(glob.glob("trajectory_*.py"), reverse=True)
    if not saved_trajs:
        print("No saved trajectory_*.py file found.")
        return
    saved_traj = saved_trajs[0]
    real_traj = "real_traj.py"
    if not os.path.exists(real_traj):
        print("real_traj.py not found.")
        return
    print(f"Comparing {saved_traj} (saved) and {real_traj} (real)...")
    traj1 = load_trajectory(saved_traj)
    traj2 = load_trajectory(real_traj)
    crop = args.crop
    t1_len = len(traj1[0][0])
    t2_len = len(traj2[0][0])
    if crop > 0:
        min_len = min(t1_len, t2_len)
        crop_frac = crop / min_len
        crop1 = int(round(t1_len * crop_frac))
        crop2 = int(round(t2_len * crop_frac))
        if crop1 > 0:
            traj1 = [[traj1[0][0][:-crop1]]]
        if crop2 > 0:
            traj2 = [[traj2[0][0][:-crop2]]]
        print(f"Cropped last {crop1} points from saved trajectory and {crop2} points from real trajectory (fractional crop: {crop_frac:.3f}).")
    metrics, t1, t2 = compute_metrics(traj1, traj2, nofrechet=args.nofrechet)
    print("\n=== Trajectory Comparison Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    plot_trajectories(np.array(t1), np.array(t2), outpath="traj_comparison.png")
    print_summary_table(metrics)

if __name__ == '__main__':
    main()
