#!/usr/bin/env python3
"""
Analyze per-step delta positions and rotations from HDF5 episodes using
obs/ee_pose only (shape: T x 7, with columns [x, y, z, qx, qy, qz, qw]).

Outputs aggregate statistics and recommends Planner thresholds:
    - max_pos_delta (meters)
    - max_rot_delta (radians)

Assumptions:
    - Quaternion order is [x, y, z, w] (xyzw).
    - Data are found at: data/<episode>/obs/ee_pose
"""

import argparse
import os
from typing import List, Tuple

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


# ---------------------------- Core computation ----------------------------
def compute_pos_deltas(pos: np.ndarray) -> np.ndarray:
    """Return L2 distance between consecutive positions (T-1,)."""
    d = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    return d


def compute_rot_deltas_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Return geodesic rotation angle (radians) between consecutive quats.

    quat_xyzw: (T, 4) np.ndarray in [x, y, z, w] order
    """
    r0 = Rotation.from_quat(quat_xyzw[:-1])
    r1 = Rotation.from_quat(quat_xyzw[1:])
    r_rel = r0.inv() * r1
    ang = np.linalg.norm(r_rel.as_rotvec(), axis=1)
    return ang


def summarize(name: str, arr: np.ndarray, percentile: float) -> dict:
    stats = {
        "count": int(arr.size),
        "min": float(np.min(arr)) if arr.size else np.nan,
        "mean": float(np.mean(arr)) if arr.size else np.nan,
        "median": float(np.median(arr)) if arr.size else np.nan,
        "std": float(np.std(arr)) if arr.size else np.nan,
        "p90": float(np.percentile(arr, 90)) if arr.size else np.nan,
        "p95": float(np.percentile(arr, 95)) if arr.size else np.nan,
        "p99": float(np.percentile(arr, 99)) if arr.size else np.nan,
        "max": float(np.max(arr)) if arr.size else np.nan,
        "p_target": float(np.percentile(arr, percentile)) if arr.size else np.nan,
    }
    return stats


def print_stats(title: str, stats: dict, rot: bool = False):
    print(f"\n=== {title} ===")
    for k in ["count", "min", "mean", "median", "std", "p90", "p95", "p99", "max", "p_target"]:
        v = stats[k]
        if rot and np.isfinite(v):
            # also show degrees
            if k == "count":
                print(f"{k:>8}: {v}")
            else:
                print(f"{k:>8}: {v:.6f} rad  ({np.degrees(v):.3f} deg)")
        else:
            if k == "count":
                print(f"{k:>8}: {v}")
            else:
                print(f"{k:>8}: {v:.6f}")


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def episode_path_length(pos: np.ndarray) -> float:
    """Total spatial path length for one episode positions (T,3)."""
    if pos.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pos[1:] - pos[:-1], axis=1)))


def yaw_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Return yaw (Z-rotation) in radians from (T,4) xyzw quaternion."""
    r = Rotation.from_quat(quat)
    eul = r.as_euler('ZYX', degrees=False)
    return eul[:, 0]


def draw_fork(ax, x: float, y: float, yaw: float, scale: float = 0.02, color='k'):
    """Draw a small fork-like glyph at (x,y) oriented by yaw on a 2D axis.
    The fork consists of a forward stem and two prongs.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = np.array([c, s])
    left = np.array([-s, c])
    base = np.array([x, y])
    tip = base + fwd * scale
    prong_left = tip + (0.5 * scale) * left
    prong_right = tip - (0.5 * scale) * left
    ax.plot([base[0], tip[0]], [base[1], tip[1]], color=color, linewidth=1)
    ax.plot([tip[0], prong_left[0]], [tip[1], prong_left[1]], color=color, linewidth=1)
    ax.plot([tip[0], prong_right[0]], [tip[1], prong_right[1]], color=color, linewidth=1)


def draw_fork_3d(ax, base: np.ndarray, R: np.ndarray, scale: float = 0.02, color='k'):
    """Draw a small 3D fork glyph at base (3,) using rotation matrix R (3x3).
    Forward is R @ [1,0,0]; prongs are offset along a left vector orthogonal to forward.
    """
    base = np.asarray(base).reshape(3)
    fwd = R @ np.array([1.0, 0.0, 0.0])
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, fwd)
    if np.linalg.norm(left) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        left = np.cross(world_up, fwd)
    left /= (np.linalg.norm(left) + 1e-9)
    tip = base + fwd * scale
    prong_left = tip + left * (0.5 * scale)
    prong_right = tip - left * (0.5 * scale)
    ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], color=color, linewidth=1)
    ax.plot([tip[0], prong_left[0]], [tip[1], prong_left[1]], [tip[2], prong_left[2]], color=color, linewidth=1)
    ax.plot([tip[0], prong_right[0]], [tip[1], prong_right[1]], [tip[2], prong_right[2]], color=color, linewidth=1)


# --------------------------------- CLI ----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze deltas, lengths, trajectories, actions, and success from HDF5.")
    ap.add_argument("--hdf5", required=True, help="Path to dataset.hdf5")
    ap.add_argument("--out_dir", required=True, help="Directory to save plots")
    ap.add_argument("--percentile", type=float, default=95.0, help="Target percentile for recommended thresholds")
    ap.add_argument("--limit", type=int, default=0, help="Max episodes to process (0 = all)")
    ap.add_argument("--plot_episodes", type=int, default=10, help="Number of episodes to visualize for trajectories")
    ap.add_argument("--fork_stride", type=int, default=10, help="Stride for drawing fork glyphs along EE trajectory")
    ap.add_argument("--verbose", action="store_true", help="Print per-episode info")
    args = ap.parse_args()

    if not os.path.exists(args.hdf5):
        raise FileNotFoundError(args.hdf5)
    ensure_out_dir(args.out_dir)

    pos_deltas_all: List[np.ndarray] = []
    rot_deltas_all: List[np.ndarray] = []
    act_pos_deltas_all: List[np.ndarray] = []
    act_rot_deltas_all: List[np.ndarray] = []
    act_suction_toggles: List[int] = []
    ep_lengths: List[int] = []
    ep_path_lengths: List[float] = []
    init_obj_positions: List[np.ndarray] = []
    init_ee_positions: List[np.ndarray] = []
    final_rewards: List[float] = []
    cum_rewards: List[float] = []

    traj_ee_list: List[Tuple[np.ndarray, np.ndarray]] = []  # (pos(T,3), quat(T,4))
    traj_obj_list: List[np.ndarray] = []  # pos(T,3)

    with h5py.File(args.hdf5, "r") as f:
        if "data" not in f:
            raise KeyError("HDF5 missing 'data' group")
        data_grp = f["data"]
        ep_names = sorted(list(data_grp.keys()))
        processed = 0

        for ep in ep_names:
            if args.limit and processed >= args.limit:
                break

            g = data_grp[ep]
            if "obs" not in g:
                if args.verbose:
                    print(f"Skip {ep}: no obs group")
                continue
            obs = g["obs"]
            if "ee_pose" not in obs:
                if args.verbose:
                    print(f"Skip {ep}: missing obs/ee_pose")
                continue

            ee = np.asarray(obs["ee_pose"][...], dtype=np.float64)
            if ee.ndim != 2 or ee.shape[-1] != 7:
                if args.verbose:
                    print(f"Skip {ep}: obs/ee_pose has shape {ee.shape}, expected (T,7)")
                continue

            T = ee.shape[0]
            if T < 2:
                continue

            pos = ee[:, :3]
            quat = ee[:, 3:7]  # assumed xyzw

            # Deltas
            pos_d = compute_pos_deltas(pos)
            rot_d = compute_rot_deltas_quat(quat)

            pos_deltas_all.append(pos_d)
            rot_deltas_all.append(rot_d)
            processed += 1

            ep_lengths.append(T)
            ep_path_lengths.append(episode_path_length(pos))
            init_ee_positions.append(pos[0])

            # Object pose
            if "obj_pose" in obs:
                obj = np.asarray(obs["obj_pose"][...], dtype=np.float64)
                if obj.ndim == 2 and obj.shape[0] >= 1 and obj.shape[-1] >= 3:
                    init_obj_positions.append(obj[0, :3])
                    if len(traj_obj_list) < args.plot_episodes:
                        traj_obj_list.append(obj[:, :3])

            if len(traj_ee_list) < args.plot_episodes:
                traj_ee_list.append((pos, quat))

            # Actions
            if "actions" in g:
                actions = np.asarray(g["actions"][...], dtype=np.float64)  # (T,8): [x,y,z, qw,qx,qy,qz, suction]
                A = min(T, actions.shape[0])
                if A >= 2:
                    a = actions[:A]
                    a_pos = a[:, :3]
                    a_quat_xyzw = np.stack([a[:, 4], a[:, 5], a[:, 6], a[:, 3]], axis=1)
                    a_suction = a[:, 7]
                    act_pos_deltas_all.append(np.linalg.norm(a_pos[1:] - a_pos[:-1], axis=1))
                    act_rot_deltas_all.append(compute_rot_deltas_quat(a_quat_xyzw))
                    toggles = int(np.sum(np.abs(np.diff((a_suction > 0.5).astype(int)))))
                    act_suction_toggles.append(toggles)

            # Rewards
            if "rewards" in g:
                r = np.asarray(g["rewards"][...], dtype=np.float64)
                if r.size >= 1:
                    final_rewards.append(float(r[-1]))
                    cum_rewards.append(float(np.sum(r)))

            if args.verbose:
                print(f"Processed {ep}: T={T}  pos_deltas={pos_d.shape}  rot_deltas={rot_d.shape}")

    if processed == 0:
        raise RuntimeError("No episodes processed. Check keys or dataset.")

    pos_deltas = np.concatenate(pos_deltas_all) if pos_deltas_all else np.array([], dtype=np.float64)
    rot_deltas = np.concatenate(rot_deltas_all) if rot_deltas_all else np.array([], dtype=np.float64)

    pos_stats = summarize("pos_deltas", pos_deltas, args.percentile)
    rot_stats = summarize("rot_deltas", rot_deltas, args.percentile)

    print_stats("Position delta (meters)", pos_stats, rot=False)
    print_stats("Rotation delta (radians)", rot_stats, rot=True)

    # Recommendations
    rec_pos = pos_stats["p_target"]
    rec_rot = rot_stats["p_target"]
    # Round up slightly for safety
    if np.isfinite(rec_pos):
        rec_pos = float(np.round(rec_pos + 1e-6, 4))  # 0.1 mm resolution
    if np.isfinite(rec_rot):
        rec_rot = float(np.round(rec_rot + 1e-6, 5))  # fine rad resolution

    print("\nRecommended thresholds (set Planner(max_pos_delta=..., max_rot_delta=...)):")
    print(f"  max_pos_delta ≈ {rec_pos} m  (percentile {args.percentile})")
    if np.isfinite(rot_stats["p_target"]):
        print(f"  max_rot_delta ≈ {rec_rot} rad ({np.degrees(rec_rot):.3f} deg)  (percentile {args.percentile})")
    else:
        print("  max_rot_delta: not available")

    # 1) Histograms: episode lengths and EE path lengths
    if ep_lengths:
        plt.figure(figsize=(6,4))
        plt.hist(ep_lengths, bins=20, color='#4C78A8', alpha=0.9)
        plt.xlabel('Episode length (T)'); plt.ylabel('Count'); plt.title('Histogram of Episode Lengths')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'hist_episode_lengths.png'), dpi=150); plt.close()
    if ep_path_lengths:
        plt.figure(figsize=(6,4))
        plt.hist(ep_path_lengths, bins=20, color='#72B7B2', alpha=0.9)
        plt.xlabel('EE path length (m)'); plt.ylabel('Count'); plt.title('Histogram of EE Path Lengths')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'hist_ee_path_lengths.png'), dpi=150); plt.close()

    # 2) Initial object positions (XY colored by Z)
    if init_obj_positions:
        pts = np.stack(init_obj_positions, axis=0)
        plt.figure(figsize=(5,5))
        sc = plt.scatter(pts[:,0], pts[:,1], c=pts[:,2], cmap='viridis', s=15)
        plt.colorbar(sc, label='z (m)'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.title('Initial Object Positions (XY)'); plt.axis('equal')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'init_object_positions_xy.png'), dpi=150); plt.close()

    # 3) Initial EE positions (XY colored by Z)
    if init_ee_positions:
        pts = np.stack(init_ee_positions, axis=0)
        plt.figure(figsize=(5,5))
        sc = plt.scatter(pts[:,0], pts[:,1], c=pts[:,2], cmap='plasma', s=15)
        plt.colorbar(sc, label='z (m)'); plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.title('Initial EE Positions (XY)'); plt.axis('equal')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'init_ee_positions_xy.png'), dpi=150); plt.close()

    # 4) Trajectories (objects and EE) in 3D
    if traj_obj_list:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        for idx, pos in enumerate(traj_obj_list):
            ax.plot(pos[:,0], pos[:,1], pos[:,2], alpha=0.9, label=f'ep{idx}')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_title('Object trajectories (3D)')
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'trajs_object.png'), dpi=150); plt.close(fig)

    if traj_ee_list:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab10(np.linspace(0,1,max(1,len(traj_ee_list))))
        for idx, (pos, quat) in enumerate(traj_ee_list):
            col = colors[idx % len(colors)]
            ax.plot(pos[:,0], pos[:,1], pos[:,2], color=col, alpha=0.9, label=f'ep{idx}')
            # Orientation glyphs every stride
            Rmats = Rotation.from_quat(quat).as_matrix()
            for t in range(0, pos.shape[0], max(1, args.fork_stride)):
                draw_fork_3d(ax, pos[t], Rmats[t], scale=0.02, color=col)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_title('EE trajectories (3D) with orientation glyphs')
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'trajs_ee.png'), dpi=150); plt.close(fig)

    # 5) Action delta statistics and histograms
    if act_pos_deltas_all:
        act_pos_d = np.concatenate(act_pos_deltas_all)
        s = summarize('action_pos_deltas', act_pos_d, args.percentile)
        print_stats('Action Position delta (meters)', s, rot=False)
        plt.figure(figsize=(6,4)); plt.hist(act_pos_d, bins=30, color='#E45756', alpha=0.9)
        plt.xlabel('Action pos delta (m)'); plt.ylabel('Count'); plt.title('Histogram: Action Position Deltas')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'hist_action_pos_deltas.png'), dpi=150); plt.close()
    if act_rot_deltas_all:
        act_rot_d = np.concatenate(act_rot_deltas_all)
        s = summarize('action_rot_deltas', act_rot_d, args.percentile)
        print_stats('Action Rotation delta (radians)', s, rot=True)
        plt.figure(figsize=(6,4)); plt.hist(act_rot_d, bins=30, color='#F58518', alpha=0.9)
        plt.xlabel('Action rot delta (rad)'); plt.ylabel('Count'); plt.title('Histogram: Action Rotation Deltas')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'hist_action_rot_deltas.png'), dpi=150); plt.close()
    if act_suction_toggles:
        plt.figure(figsize=(6,4))
        plt.hist(act_suction_toggles, bins=range(0, max(act_suction_toggles)+2), align='left', color='#54A24B', alpha=0.9)
        plt.xlabel('Suction toggles per episode'); plt.ylabel('Count'); plt.title('Histogram: Suction Toggles')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'hist_suction_toggles.png'), dpi=150); plt.close()

    # 6) Success rates from rewards
    if final_rewards:
        fr = np.array(final_rewards)
        cr = np.array(cum_rewards) if cum_rewards else None
        succ_final = float(np.mean(fr > 0.0))
        print(f"\nSuccess rate (final_reward>0): {succ_final*100:.2f}%  over {len(fr)} episodes")
        plt.figure(figsize=(6,4)); plt.hist(fr, bins=30, color='#EECA3B', alpha=0.9)
        plt.xlabel('Final reward'); plt.ylabel('Count'); plt.title('Histogram: Final Rewards')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'hist_final_rewards.png'), dpi=150); plt.close()
        if cr is not None:
            succ_cum = float(np.mean(cr > 0.0))
            print(f"Success rate (cumulative_reward>0): {succ_cum*100:.2f}%  over {len(cr)} episodes")
            plt.figure(figsize=(6,4)); plt.hist(cr, bins=30, color='#B279A2', alpha=0.9)
            plt.xlabel('Cumulative reward'); plt.ylabel('Count'); plt.title('Histogram: Cumulative Rewards')
            plt.grid(linestyle='--', linewidth=0.5, alpha=0.6)
            plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'hist_cumulative_rewards.png'), dpi=150); plt.close()


if __name__ == "__main__":
    main()
