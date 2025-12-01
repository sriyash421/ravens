#!/usr/bin/env python3
import argparse, os, h5py, numpy as np
import cv2

# ---------- Cropping (placeholder) ----------
def crop_frame(img: np.ndarray, cam_idx: int) -> np.ndarray:
    """
    Placeholder crop: takes a single frame (H, W, 3) and a camera index,
    returns the SAME image (no-op). Replace with your crop logic.

    Example (later):
        # top-left crop  Hc x Wc
        H, W = img.shape[:2]
        Hc, Wc = 200, 200
        return img[0:Hc, 0:Wc]
    """
    H,W = 224, 224
    return img
    # if cam_idx == 0:
    #     # return img[-H:, :W, :]  # side camera: bottom-left
    #     img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    # elif cam_idx == 1:
    #     # resive the wrist camera to 224x224
    #     img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    #     return img
    # return img

def crop_stream(frames: np.ndarray, cam_idx: int) -> np.ndarray:
    """
    Apply crop_frame to a sequence (T, H, W, 3) for a given camera index.
    """
    T = frames.shape[0]
    out = []
    for t in range(T):
        out.append(crop_frame(frames[t], cam_idx))
    out = np.asarray(out, dtype=np.uint8)
    # Sanity: ensure all frames have identical H,W after your crop edits.
    if len({f.shape for f in out}) != 1:
        raise ValueError(f"Crop produced variable shapes for camera {cam_idx}. Ensure consistent crop.")
    return out
# -------------------------------------------

def list_image_keys(obs_group):
    return sorted([
        k for k, ds in obs_group.items()
        if isinstance(ds, h5py.Dataset) and k.endswith("_image") and ds.ndim == 4 and ds.shape[-1] == 3
    ])

def write_video(path, frames, fps, gif=False):
    """Write video or GIF from frames (T,H,W,3) uint8.

    Uses OpenCV for MP4 to avoid external dependency. GIF optional (requires imageio).
    """
    T = frames.shape[0]
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    H, W = frames.shape[1:3]
    if gif:
        raise RuntimeError("GIF output not supported (imageio not available). Omit --gif to write MP4.")
    # MP4 via OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        writer.write(frames[t])
    writer.release()

def overlay_annotations(frames, actions=None, rewards=None, expert_masks=None, success=None):
    """Apply textual overlays similar to generate_dagger_data.write_video.

    Parameters
    ----------
    frames : np.ndarray (T,H,W,3) uint8
    actions : np.ndarray (T, A) or list, optional
    rewards : np.ndarray (T,) optional
    expert_masks : np.ndarray (T,) uint8/bool optional
    success : bool or float optional (if float>0.5 => SUCCESS)
    """
    if frames is None:
        return frames
    T = frames.shape[0]
    # Fallbacks
    if actions is None:
        actions = [None] * T
    if rewards is None:
        rewards = [0.0] * T
    if expert_masks is None:
        expert_masks = [False] * T
    # Normalize containers
    actions_list = actions if isinstance(actions, (list, tuple)) else [a for a in actions]
    rewards_list = rewards if isinstance(rewards, (list, tuple)) else list(rewards)
    expert_list = expert_masks if isinstance(expert_masks, (list, tuple)) else list(expert_masks)
    succ_bool = False
    if success is not None:
        succ_bool = bool(success if not isinstance(success, (int, float)) else success > 0.5)
    else:
        # infer from max reward if available
        try:
            succ_bool = max(rewards_list) > 0.99
        except Exception:
            succ_bool = False

    for t in range(T):
        frame = frames[t]
        frame = np.ascontiguousarray(frame).copy().astype(np.uint8)
        # Action
        a = actions_list[t]
        if a is not None:
            # ensure iterable
            try:
                action_str = ", ".join([f"{float(x):.2f}" for x in np.array(a).flatten()])
            except Exception:
                action_str = str(a)
            cv2.putText(frame, action_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        # Success
        success_str = "SUCCESS" if succ_bool else "FAILURE"
        cv2.putText(frame, success_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0) if succ_bool else (0,0,255), 1, cv2.LINE_AA)
        # Reward
        if t < len(rewards_list):
            try:
                r = float(rewards_list[t])
            except Exception:
                r = 0.0
            cv2.putText(frame, f"Reward: {r:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # Expert vs Policy
        if t < len(expert_list):
            expert_flag = bool(expert_list[t])
            cv2.putText(frame, "EXPERT" if expert_flag else "POLICY", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,255), 1, cv2.LINE_AA)
        # Timestep
        cv2.putText(frame, f"Timestep: {t}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        frames[t] = frame
    return frames

def main():
    ap = argparse.ArgumentParser(description="Render stacked camera videos from robomimic-style HDF5 (no resizing).")
    ap.add_argument("--hdf5", required=True, help="Path to dataset.hdf5")
    ap.add_argument("--out_dir", required=True, help="Output directory for videos")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    ap.add_argument("--max_videos", type=int, default=2, help="Maximum number of episodes to render")
    ap.add_argument("--gif", action="store_true", help="Write GIFs instead of MP4")
    ap.add_argument("--overlay", action="store_true", help="Add textual overlays (actions, rewards, expert mask, success)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        data_grp = f["data"]
        ep_names = sorted(list(data_grp.keys()))
        count = 0
        for ep in ep_names:
            if count >= args.max_videos:
                break
            g = data_grp[ep]
            if "obs" not in g:
                continue
            obs = g["obs"]
            img_keys = list_image_keys(obs)
            if not img_keys:
                continue

            # Load streams and apply per-camera crop (currently no-op)
            streams = []
            for cam_idx, k in enumerate(img_keys):
                arr = obs[k][...]  # (T,H,W,3) uint8
                print(f"  Ep {ep} cam {cam_idx} key {k}: shape={arr.shape}")
                streams.append(crop_stream(arr, cam_idx))

            # Check identical (H,W) — we do not resize
            shapes = [s.shape[1:3] for s in streams]
            if any(sh != shapes[0] for sh in shapes):
                raise ValueError(f"Episode {ep} has mismatched image sizes {shapes} after cropping. All must match.")

            # Align T if needed
            Ts = [s.shape[0] for s in streams]
            T = min(Ts)
            streams = [s[:T] for s in streams]

            stacked = np.concatenate(streams, axis=2)  # (T, H, sumW, 3)
            stacked = np.ascontiguousarray(stacked, dtype=np.uint8)

            # Load metadata for overlays if requested
            if args.overlay:
                actions = g.get("actions", None)
                rewards = g.get("rewards", None)
                expert_mask = g.get("expert_mask", None)
                # Convert datasets to numpy arrays (trim to T)
                def _ds_to_np(ds):
                    if ds is None: return None
                    arr = ds[...]
                    return arr[:T]
                actions_np = _ds_to_np(actions)
                rewards_np = _ds_to_np(rewards)
                expert_np = _ds_to_np(expert_mask)
                # Compute success heuristic (consistent w/ generate_dagger_data)
                success_flag = None
                if rewards_np is not None and rewards_np.size > 0:
                    success_flag = float(np.sum(rewards_np)) > 0.99 or np.max(rewards_np) > 0.99
                stacked = overlay_annotations(stacked, actions=actions_np, rewards=rewards_np,
                                              expert_masks=expert_np, success=success_flag)

            ext = ".gif" if args.gif else ".mp4"
            out_path = os.path.join(args.out_dir, f"{ep}_stacked{ext}")
            print(f"[{count+1}] {ep}: keys={img_keys} -> {out_path}  shape={stacked.shape}")
            write_video(out_path, stacked, fps=args.fps, gif=args.gif)
            count += 1

    print(f"Done. Wrote {count} video(s) to {args.out_dir}.")

if __name__ == "__main__":
    main()
