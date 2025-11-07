#!/usr/bin/env python3
"""
rollout_ravens.py — Evaluate a robomimic-trained policy inside a Ravens (PyBullet) env.

Requires:
  - robomimic >= 0.4 (tested with 0.5)
  - ravens (your codebase)
  - pybullet, imageio, tqdm

Example:
  python rollout_ravens.py \
    --checkpoint /path/to/policy_epoch_050.pth \
    --config /path/to/training_config.json \
    --num_episodes 10 \
    --video_dir ./videos \
    --task place-red-in-green \
    --assets_root ./ravens/environments/assets \
    --continuous \
    --steps_per_seg 3 \
    --fps 5
"""

import os
import json
import argparse
import numpy as np
import imageio
from pathlib import Path
from tqdm import trange
import cv2
import wandb

import pybullet as p
from ravens.environments.environment import Environment, ContinuousEnvironment
from ravens import tasks

# ---- robomimic imports ----
import torch
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils
# import robomimic.envs.env_base as EB

# ---------- Minimal Ravens -> robomimic observation adapter ----------

def get_obj_id_and_goal(env):
    goals = env.task.goals[0]
    objs, matches, targs, replace, rotations, _, _, _ = goals
    object_id = objs[0][0]  # Get the ID of the first object
    target_pose = np.array(targs[0][0]+targs[0][1])  # (x, y, z, qx, qy, qz, qw)
    return object_id, target_pose


class RavensEvalWrapper:
    """
    Makes Ravens env return an obs dict that matches your HDF5 / robomimic obs keys:
      - side_camera_image (H,W,3) uint8 : camera index 0
      - wrist_camera_image (H,W,3) uint8 : camera index 1
      - ee_pose, goal_pose, obj_pose: (7,) float32
    Exposes reset() and step(action_vec) where action_vec is (8,) = [xyz, quat(4), suction].
    """
    def __init__(self, env, camera_index_for_frame=0, use_pos_only=False, debug=False, resize_image=False):
        self.env = env
        self.cam_for_frame = camera_index_for_frame
        self.pos_only = use_pos_only
        self.obj_id = None
        self.goal_pose = None
        self.action_dimension = 8
        self.debug = debug
        self.max_steps = 100
        self.resize_image = resize_image
        self.task = env.task # Store task for later use
        self.base_obs = None

    def _pack_obs(self, raw_obs, action):
        # Camera frames: raw_obs['color'] is (N_cams, H, W, 3) uint8
        side = raw_obs['color'][0]
        wrist = raw_obs['color'][1] if len(raw_obs['color']) > 1 else raw_obs['color'][0]

        # EE pose (pos, quat)
        ee_ls = p.getLinkState(self.env.ur5, self.env.ee_tip)
        ee = np.array(ee_ls[0] + ee_ls[1], dtype=np.float32)  # (7,)

        # Object pose
        obj_pos, obj_quat = p.getBasePositionAndOrientation(self.obj_id)
        obj = np.array(obj_pos + obj_quat, dtype=np.float32)  # (7,)

        goal = self.goal_pose.astype(np.float32)
        grasp = np.array([self.env.ee.check_grasp()], dtype=np.float32)

        if self.pos_only:
            ee = ee[:3]
            obj = obj[:3]
            goal = goal[:3]
        
        if self.resize_image:
            side = cv2.resize(side, (224, 224), interpolation=cv2.INTER_AREA)
            wrist = cv2.resize(wrist, (224, 224), interpolation=cv2.INTER_AREA)

        im1 = raw_obs['color'][0]
        im2 = raw_obs['color'][1] # T, H, W, 3
        _frame = np.concatenate((im1, im2), axis=1)
        obs = {
            "side_camera_image": side,
            "wrist_camera_image": wrist,
            "ee_pose": ee,
            "obj_pose": obj,
            "goal_pose": goal,
            "_frame": _frame,
            "grasp": grasp,
            "prev_action": action,
        }
        return obs
    
    def compute_sparse_reward(self, obs):
        # goal_pos = self.goal_pose[:2]
        obj_pos = obs['obj_pose'][:2]
        goal_pos = obs['goal_pose'][:2]
        dist = np.linalg.norm(goal_pos - obj_pos)
        height = obs['obj_pose'][2]
        grasp = self.env.ee.check_grasp()
        # print(f"Distance to goal: {dist:.4f}, Height: {height:.4f}, Grasp: {grasp}")
        return float(dist < 0.05 and height < 0.04 and not grasp)

    def reset(self, seed=None):
        # if seed is not None:
            # np.random.seed(seed)
        if self.debug:
            np.random.seed(28)  # fixed seed for debug
        raw_obs = self.env.reset()
        self.base_obs = raw_obs
        self.obj_id, self.goal_pose = get_obj_id_and_goal(self.env)
        self.elapsed_steps = 0
        self.prev_action = np.zeros((8,))
        # print("-------- Reset Env -------")
        return self._pack_obs(raw_obs, np.zeros((8,)))

    def action_dict2vec(self, action):
        pos, quat = action['move_cmd']
        suction = action['suction_cmd']
        action_vec = np.concatenate((pos, quat, np.array([suction], dtype=np.float32)))
        return action_vec
    
    def step(self, action_vec, acts_left=0):
        """
        action_vec: np.ndarray, shape (>=4,) expected [x, y, z, qw, qx, qy, qz, suction]
        If only [x,y,z,suction] provided, we default quat to [0,0,0,1].
        """
        a = np.asarray(action_vec).astype(np.float32)
        # if a.shape[0] == 4:
        pos = a[:3]
        quat = a[3:7]
        # quat = np.array([0, 0, 0, 1], dtype=np.float32)
        suction = a[-1]
        # else:
        #     pos = a[:3]
        #     quat = a[3:7]
        #     suction = a[7] > 0.0

        action = {
            "move_cmd": (pos, quat),
            "suction_cmd": int(suction > 0.8),
            "acts_left": acts_left,
        }
        raw_obs, reward, done, info = self.env.step(action)
        self.base_obs = raw_obs
        obs = self._pack_obs(raw_obs, action_vec)
        info = dict(info or {})
        info["frame"] = obs["_frame"]

        reward = self.compute_sparse_reward(obs)
        self.elapsed_steps += 1
        done = (reward > 0.99) or self.elapsed_steps >= self.max_steps
        return obs, float(reward), bool(done), info


# ---------- Video util ----------

def write_video(frames, actions, rewards, success, fps=5):

    for t in range(len(frames)):
        action = actions[t]
        action_str = ", ".join([f"{a:.2f}" for a in action])
        frame = frames[t]
        frame = np.ascontiguousarray(frame).copy()
        frame = frame.astype(np.uint8)
        # print(frames[t])
        # breakpoint()
        cv2.putText(frame, action_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        success_str = "SUCCESS" if success else "FAILURE"
        cv2.putText(frame, success_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if success else (0, 0, 255), 2, cv2.LINE_AA)
        reward_str = f"Reward: {rewards[t]:.2f}"
        cv2.putText(frame, reward_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        frames[t] = frame
    return frames

def save_video(frames, out_path, fps=5):
    # frames: list of (H, W, 3) uint8
    imageio.mimwrite(out_path, frames, fps=fps, quality=8)




# ---------- Main rollout ----------

def build_ravens_env(args, task_mode="test"):
    # Your exact env construction (kept faithful to your snippet)
    env_cls = ContinuousEnvironment if args.continuous else Environment
    env = env_cls(
        args.assets_root,
        disp=args.render,
        shared_memory=True,
        hz=480,
    )
    task = tasks.names[args.task](continuous=args.continuous)
    task.mode = task_mode
    env.set_task(task)

    # Compute horizon similarly to your code
    agent = task.oracle(env, steps_per_seg=args.steps_per_seg)
    max_steps = task.max_steps
    if args.continuous:
        max_steps *= (args.steps_per_seg * agent.num_poses)
    return RavensEvalWrapper(env, camera_index_for_frame=args.camera_index, debug=args.debug, use_pos_only=args.pos_only,
                                resize_image=args.resize_image), int(max_steps)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a robomimic policy in Ravens (PyBullet).")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to robomimic policy .pth")
    # parser.add_argument("--config", required=True, type=str, help="Training config JSON used for this policy")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=0, help="Override rollout horizon (0 = use env-derived)")
    parser.add_argument("--seed", type=int, default=0)

    # Ravens args (mirror your env creation)
    parser.add_argument("--assets_root", type=str, default="./ravens/environments/assets")
    parser.add_argument("--task", type=str, default="place-red-in-green")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--steps_per_seg", type=int, default=10)
    parser.add_argument("--camera_index", type=int, default=0, help="Which camera to use for saved video frames")
    parser.add_argument("--render", action="store_true")

    # Output
    parser.add_argument("--video_dir", type=str, default="", help="If set, saves per-episode mp4s here")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument('--debug', action='store_true', help='If set, enables debug mode')
    parser.add_argument('--pos_only', action='store_true', help='If set, use position-only obs (no quats)')
    parser.add_argument('--resize_image', action='store_true', help='If set, resize images to 224x224')
    parser.add_argument('--exp_name', type=str, default='ravens-eval', help='WandB experiment name')
    args = parser.parse_args()

    # ---- Load training config + init obs utils ----
    # with open(args.config, "r") as f:
    #     train_cfg_json = json.load(f)  # This should be the same config used for training
    # Initialize observation utilities so policy encoders preprocess obs the same as train-time
    config, _ = FileUtils.config_from_checkpoint(ckpt_path=args.checkpoint)
    ObsUtils.initialize_obs_utils_with_config(config)  # doc: initialize from config :contentReference[oaicite:2]{index=2}

    # ---- Restore policy from checkpoint (official util) ----
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=True
    )  # recommended pattern in docs / tutorials :contentReference[oaicite:3]{index=3}
    # policy is a RolloutPolicy-like object with .start_episode() and .get_action(obs)

    # ---- Build env ----
    env, env_horizon = build_ravens_env(args)
    horizon = args.horizon if args.horizon > 0 else env_horizon

    env = EnvUtils.wrap_env_from_config(env, config)

    rng = np.random.RandomState(args.seed)
    returns = []
    lengths = []
    successes = 0

    # Initialize wandb (always on as requested)
    wandb.init(project="ravens", config=vars(args))

    # Local video saving is disabled; videos will be logged to wandb.

    for ep in trange(args.num_episodes, desc="Evaluating"):
        # per-episode seed for stochastic resets
        seed = int(rng.randint(0, 2**31 - 1))
        obs = env.reset()
        policy.start_episode()  # sets eval mode, clears hidden state if any

        frames = []
        actions = []
        rewards = []
        ep_ret = 0.0
        done = False
        reward = 0.0
        length = 0
        for t in range(horizon):
            # The obs dict keys here: side_camera_image, wrist_camera_image, ee_pose, obj_pose, goal_pose
            # Only the keys used by the policy's config will be consumed by the encoder.
            action = policy(obs)  # numpy action in env space
            obs, reward, done, info = env.step(action)
            actions.append(action)
            ep_ret += reward
            frames.append(info["frame"])
            rewards.append(reward)
            length += 1
            if done:
                break

        returns.append(ep_ret)
        lengths.append(length)
        # Simple Ravens success heuristic (adjust to your task spec):
        succ = float(ep_ret > 0.99)
        successes += int(succ)

        # Save/upload per-episode video either locally or to wandb (or both)
        episodes_done = len(returns)
        running_succ_rate = successes / float(max(1, episodes_done))
        avg_return_so_far = float(np.mean(returns)) if returns else 0.0
        avg_length_so_far = float(np.mean(lengths)) if lengths else 0.0

        # Log video and running metrics to wandb
        video_np = np.stack(frames, axis=0)  # (T, H, W, 3)
        video_np = write_video(video_np, actions, rewards, succ > 0.5, fps=args.fps)
        wandb.log({
            f"video": wandb.Video(video_np.transpose(0, 3, 1, 2), fps=args.fps, format="mp4"),
            "success_rate": running_succ_rate,
            "avg_return": avg_return_so_far,
            "avg_length": avg_length_so_far,
            "episode_return": ep_ret,
            "episode_success": succ,
        }, step=ep)

    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    succ_rate = successes / max(1, args.num_episodes)
    print(f"\nDone. Episodes: {args.num_episodes} | Avg return: {avg_return:.3f} | Avg length: {avg_length:.3f} | Success rate: {succ_rate:.3f}")

    # Final summary
    wandb.summary["final/avg_return"] = avg_return
    wandb.summary["final/avg_length"] = avg_length
    wandb.summary["final/success_rate"] = succ_rate

if __name__ == "__main__":
    main()
