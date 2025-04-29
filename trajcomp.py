import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.signal import resample

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

def compute_metrics(traj1, traj2):
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
        if hasattr(ls1, "frechet_distance"):
            frechet = ls1.frechet_distance(ls2)
        else:
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
    return {
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
    }, t1, t2

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

def main():
    import glob
    import os
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
    metrics, t1, t2 = compute_metrics(traj1, traj2)
    print("\n=== Trajectory Comparison Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    plot_trajectories(np.array(t1), np.array(t2), outpath="traj_comparison.png")

if __name__ == '__main__':
    main()
