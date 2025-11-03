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
    """
    frames: (T,H,W,3) uint8, contiguous
    For MP4 uses ffmpeg via imageio-ffmpeg; for GIF uses Pillow via imageio v2.
    """
    import imageio.v2 as iio
    T = frames.shape[0]
    frames = np.ascontiguousarray(frames, dtype=np.uint8)

    if gif:
        duration = 1.0 / max(1, fps)
        with iio.get_writer(path, mode="I", duration=duration, loop=0) as w:
            for t in range(T):
                w.append_data(frames[t])
    else:
        try:
            with iio.get_writer(path, fps=fps, codec="libx264", format="FFMPEG",
                                macro_block_size=None, output_params=["-pix_fmt", "yuv420p"]) as w:
                for t in range(T):
                    w.append_data(frames[t])
        except TypeError:
            with iio.get_writer(path, fps=fps, codec="libx264") as w:
                for t in range(T):
                    w.append_data(frames[t])

def main():
    ap = argparse.ArgumentParser(description="Render stacked camera videos from robomimic-style HDF5 (no resizing).")
    ap.add_argument("--hdf5", required=True, help="Path to dataset.hdf5")
    ap.add_argument("--out_dir", required=True, help="Output directory for videos")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    ap.add_argument("--max_videos", type=int, default=2, help="Maximum number of episodes to render")
    ap.add_argument("--gif", action="store_true", help="Write GIFs instead of MP4")
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

            ext = ".gif" if args.gif else ".mp4"
            out_path = os.path.join(args.out_dir, f"{ep}_stacked{ext}")
            print(f"[{count+1}] {ep}: keys={img_keys} -> {out_path}  shape={stacked.shape}")
            write_video(out_path, stacked, fps=args.fps, gif=args.gif)
            count += 1

    print(f"Done. Wrote {count} video(s) to {args.out_dir}.")

if __name__ == "__main__":
    main()
