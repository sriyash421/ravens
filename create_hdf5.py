#!/usr/bin/env python3
import os, json, pickle, pathlib, h5py, numpy as np, argparse
from tqdm import tqdm

def _resize_frames_224(frames):
    """
    frames: (T, H, W, 3) uint8
    returns: (T, 224, 224, 3) uint8
    """
    import cv2  # lazy import so cv2 is only required if resizing
    T = frames.shape[0]
    out = np.empty((T, 224, 224, 3), dtype=np.uint8)
    for t in range(T):
        # cv2.resize expects (width, height)
        out[t] = cv2.resize(frames[t], (224, 224), interpolation=cv2.INTER_AREA)
    return out

def create_robomimic_hdf5(parent_dir, output_hdf5, resize_to_224=False):
    """
    Expects:
      parent_dir/
        color/*.pkl      # (T, N, H, W, 3), uint8; N>=2; 0=side_camera, 1=wrist_camera
        action/*.pkl     # list of dicts per step: move_cmd=(pos(3,), quat(4,)), suction_cmd (scalar)
        info/*.pkl       # list of dicts per step: ee_pose(7,), goal_pose(7,), obj_pose(7,)
        rewards/*.pkl    # np.array (T,)
    """
    parent = pathlib.Path(parent_dir)
    color_dir   = parent / "color"
    action_dir  = parent / "action"
    info_dir    = parent / "info"
    rewards_dir = parent / "reward"

    color_paths = sorted(color_dir.glob("*.pkl"))
    if not color_paths:
        raise FileNotFoundError(f"No episodes found under {color_dir}")

    with h5py.File(output_hdf5, "w") as f:
        g_data = f.create_group("data")
        g_data.attrs["env_args"] = json.dumps({})  # keep empty as requested
        total_samples = 0

        for ep_idx, color_path in enumerate(tqdm(color_paths, desc="Packing episodes", unit="ep")):
            stem = color_path.name
            action_path  = action_dir  / stem.replace("color", "action")
            info_path    = info_dir    / stem.replace("color", "info")
            rewards_path = rewards_dir / stem.replace("color", "reward")

            with open(color_path, "rb") as fcol:
                color = pickle.load(fcol)  # (T, N, H, W, 3)
            with open(action_path, "rb") as fact:
                actions_dicts = pickle.load(fact)  # list of dicts
            with open(info_path, "rb") as finfo:
                infos = pickle.load(finfo)  # list of dicts
            with open(rewards_path, "rb") as frew:
                rewards = pickle.load(frew)  # np.array (T,)
                rewards = np.array(rewards)

            T, N, H, W, C = color.shape
            assert N >= 2, f"Expected at least 2 cameras, got N={N} in {color_path}"
            assert len(actions_dicts) == T, f"actions length {len(actions_dicts)} != T={T}"
            assert len(infos) == T, f"infos length {len(infos)} != T={T}"
            assert rewards.shape[0] == T, f"rewards length {rewards.shape[0]} != T={T}"

            side_imgs  = color[:, 0].astype(np.uint8)   # (T, H, W, 3)
            wrist_imgs = color[:, 1].astype(np.uint8)   # (T, H, W, 3)

            if resize_to_224:
                side_imgs  = _resize_frames_224(side_imgs)
                wrist_imgs = _resize_frames_224(wrist_imgs)

            ee_pose   = np.asarray([s["ee_pose"]   for s in infos], dtype=np.float32)  # (T, 7)
            goal_pose = np.asarray([s["goal_pose"] for s in infos], dtype=np.float32)  # (T, 7)
            obj_pose  = np.asarray([s["obj_pose"]  for s in infos], dtype=np.float32)  # (T, 7)
            grasp     = np.asarray([s["grasp"]     for s in infos], dtype=np.float32).reshape(T, 1)  # (T,1)

            # Actions: [x,y,z, qw,qx,qy,qz, suction]  -> (T, 8)
            actions = np.asarray([
                np.concatenate([
                    np.asarray(a["move_cmd"][0], dtype=np.float32),     # (3,)
                    np.asarray(a["move_cmd"][1], dtype=np.float32),     # (4,)
                    np.asarray([a["suction_cmd"]], dtype=np.float32),   # (1,)
                ], axis=0)
                for a in actions_dicts
            ], dtype=np.float32)

            rewards = rewards.astype(np.float32)
            dones = np.zeros((T,), dtype=bool); dones[-1] = True

            ep_name = f"demo_{ep_idx}"
            g_ep = g_data.create_group(ep_name)
            g_ep.attrs["num_samples"] = int(T)
            g_ep.attrs["model_file"] = ""  # empty

            g_obs = g_ep.create_group("obs")
            g_obs.create_dataset("side_camera_image",  data=side_imgs,  dtype=np.uint8)
            g_obs.create_dataset("wrist_camera_image", data=wrist_imgs, dtype=np.uint8)
            g_obs.create_dataset("ee_pose",            data=ee_pose,    dtype=np.float32)
            g_obs.create_dataset("goal_pose",          data=goal_pose,  dtype=np.float32)
            g_obs.create_dataset("obj_pose",           data=obj_pose,   dtype=np.float32)
            g_obs.create_dataset("grasp",              data=grasp,      dtype=np.float32)

            g_ep.create_dataset("actions", data=actions, dtype=np.float32)
            g_ep.create_dataset("rewards", data=rewards, dtype=np.float32)
            g_ep.create_dataset("dones",   data=dones,   dtype=bool)

            if total_samples < 1:
                # print shapes
                print("Dataset structure:")
                print(f"  obs/side_camera_image:  {side_imgs.shape} uint8")
                print(f"  obs/wrist_camera_image: {wrist_imgs.shape} uint8")
                print(f"  obs/ee_pose:            {ee_pose.shape} float32")
                print(f"  obs/goal_pose:          {goal_pose.shape} float32")
                print(f"  obs/obj_pose:           {obj_pose.shape} float32")
                print(f"  obs/grasp:              {grasp.shape} float32")
                print(f"  actions:                {actions.shape} float32")
                print(f"  rewards:                {rewards.shape} float32")
                print(f"  dones:                  {dones.shape} bool")

            total_samples += T

        g_data.attrs["total"] = int(total_samples)

    print(f"[done] wrote {output_hdf5} with {len(color_paths)} episodes, total steps={total_samples}")

def main():
    parser = argparse.ArgumentParser(description="Pack Ravens-style pickle episodes into a robomimic HDF5.")
    parser.add_argument("--parent_dir", type=str, required=True,
                        help="Path containing color/, action/, info/, rewards/ subfolders.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 path, e.g., dataset.hdf5")
    parser.add_argument("--resize_to_224", action="store_true",
                        help="If set, resize side & wrist images to 224x224 before writing.")
    args = parser.parse_args()
    create_robomimic_hdf5(args.parent_dir, args.output, resize_to_224=args.resize_to_224)

if __name__ == "__main__":
    main()
