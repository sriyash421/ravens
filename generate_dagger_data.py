#!/usr/bin/env python3
"""
generate_dagger_data.py — Generate DAgger data for a robomimic-trained policy inside a Ravens (PyBullet) env.

Requires:
  - robomimic >= 0.4 (tested with 0.5)
  - ravens (your codebase)
  - pybullet, imageio, tqdm

Example:
  python generate_dagger_data.py \
    --checkpoint /path/to/policy_epoch_050.pth \
    --seed 3 \
    --num_episodes 20
    --save_dir ./dagger_data
"""

import os
import json
import argparse
import numpy as np
import imageio
from pathlib import Path
from tqdm import trange
import cv2
# import wandb

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

from rollout import build_ravens_env  # reuse the wrapper from rollout.py
from robomimic_datacollector import Hdf5DataCollector
import timeout_decorator
import copy
from collections import deque

class EnvStackWrapper():
    def __init__(self, num_frames):
        assert num_frames >= 1, "num_frames must be >= 1"
        self.num_frames = num_frames
    
    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )
        return obs_history

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }

    def cache_obs_history(self):
        self.obs_history_cache = copy.deepcopy(self.obs_history)
    
    def uncache_obs_history(self):
        self.obs_history = self.obs_history_cache
        self.obs_history_cache = None
    
    def reset(self, obs):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def step(self, obs):
        self.update_obs(obs, reset=False)
        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
        obs_ret = self._get_stacked_obs_from_history()
        return obs_ret

    def update_obs(self, obs, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        
        # if reset:
        #     obs["actions"] = np.zeros(8)
        # else:
        #     self.timestep += 1
        #     obs["actions"] = action[: 8]

    def _to_string(self):
        """Info to pretty print."""
        return "num_frames={}".format(self.num_frames)

def write_video(frames, actions, rewards, expert_masks, success, fps=5):

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
        cv2.putText(frame, success_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if success else (0, 0, 255), 1, cv2.LINE_AA)
        reward_str = f"Reward: {rewards[t]:.2f}"
        cv2.putText(frame, reward_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        expert_str = "EXPERT" if expert_masks[t] else "POLICY"
        cv2.putText(frame, expert_str, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        frames[t] = frame
        # add timestep
        cv2.putText(frame, f"Timestep: {t}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frames

def main():
    parser = argparse.ArgumentParser(description="Evaluate a robomimic policy in Ravens (PyBullet).")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to robomimic policy .pth")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--exploration_horizon", type=int, default=50, help="Override rollout horizon (0 = use env-derived)")
    parser.add_argument("--horizon", type=int, default=100, help="Override episode horizon (0 = use env-derived)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save DAgger data")
    parser.add_argument("--save_name", type=str, default="dagger_data.hdf5", help="Filename for saved DAgger data")

    # Ravens args (mirror your env creation)
    parser.add_argument("--assets_root", type=str, default="./ravens/environments/assets")
    parser.add_argument("--task", type=str, default="place-red-in-green")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--steps_per_seg", type=int, default=10)
    parser.add_argument("--camera_index", type=int, default=0, help="Which camera to use for saved video frames")
    parser.add_argument("--render", action="store_true")

    # Output
    parser.add_argument('--debug', action='store_true', help='If set, enables debug mode')
    parser.add_argument('--pos_only', action='store_true', help='If set, use position-only obs (no quats)')
    parser.add_argument('--resize_image', action='store_true', help='If set, resize images to 224x224')

    # Expert
    parser.add_argument('--expert_ckpt', type=str, required=True, help='Path to expert policy checkpoint (if different from main policy)')
    args = parser.parse_args()

    # ---- Restore policy from checkpoint (official util) ----
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    expert_policy, expert_policy_ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.expert_ckpt,
        device=device,
        verbose=True
    )

    # expert_config 
    expert_config, _ = FileUtils.config_from_checkpoint(ckpt_path=args.expert_ckpt)
    expert_obs_manager = EnvStackWrapper(num_frames=expert_config.train.frame_stack)

    policy, policy_ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=True
    )

    config, _ = FileUtils.config_from_checkpoint(ckpt_path=args.checkpoint)
    ObsUtils.initialize_obs_utils_with_config(config)  # doc: initialize from config :contentReference[oaicite:2]{index=2}

    # ---- Build env ----
    env, env_horizon = build_ravens_env(args, task_mode="train")
    horizon = args.horizon if args.horizon > 0 else env_horizon
    # env = EnvUtils.wrap_env_from_config(env, config)
    rng = np.random.RandomState(args.seed)
    # np.random.seed(args.seed)
    returns = []
    successes = 0

    # # Generating task oracle
    # agent = env.task.oracle(env.env, steps_per_seg=args.steps_per_seg)

    data_collector = Hdf5DataCollector(
        env_name="RavensEnv",
        directory_path=args.save_dir,
        filename=args.save_name,
        num_demos=args.num_episodes,
    )
    data_collector.reset()

    # wandb.init(project="ravens-dagger", config=vars(args))

    pbar = trange(args.num_episodes, desc="Generating DAgger Data")
    ep = 0
    total_ep = 0
    while not data_collector.is_stopped():
        # per-episode seed for stochastic resets
        seed = int(rng.randint(0, 2**31 - 1))
        np.random.seed(seed)
        obs = env.reset()
        expert_obs = expert_obs_manager.reset(obs)
        policy.start_episode()  # sets eval mode, clears hidden state if any
        expert_policy.start_episode()
        data_collector.reset()

        frames = []
        actions = []
        expert_masks = []
        rewards = []
        ep_ret = 0.0
        done = False
        reward = 0.0
        succ = 0.0

        for t in range(horizon):
            # The obs dict keys here: side_camera_image, wrist_camera_image, ee_pose, obj_pose, goal_pose
            # Only the keys used by the policy's config will be consumed by the encoder.
            # breakpoint()
            try:
                expert_mask = False
                action = policy(obs)  # numpy action in env space
                acts_left = 0
                if t > args.exploration_horizon:
                    expert_mask = True
                    # expert_action = agent.act(env.base_obs, {})  # get expert action from oracle
                    # acts_left = expert_action['acts_left']
                    # action = env.action_dict2vec(expert_action)
                    expert_action = expert_policy({
                        "ee_pose": expert_obs["ee_pose"],
                        "obj_pose": expert_obs["obj_pose"],
                        "goal_pose": expert_obs["goal_pose"],
                        "grasp": expert_obs["grasp"],
                    })
                    action = expert_action

                
                for k,v in obs.items():
                    data_collector.add(f"obs/{k}", v[None])
                data_collector.add("actions", action[None])
                data_collector.add("expert_mask", np.array([expert_mask], dtype=np.uint8))

                obs, reward, done, info = env.step(action, acts_left=acts_left)
                data_collector.add("rewards", np.array([reward], dtype=np.float32))
                data_collector.add("dones", np.array([done], dtype=np.uint8))
                actions.append(np.concatenate((action, np.array(reward).reshape(-1))))  # append reward to action for video overlay
                ep_ret += reward
                expert_obs = expert_obs_manager.step(obs)

                rewards.append(reward)
                expert_masks.append(expert_mask)
                frames.append(info['frame'])

                # if expert_mask:
                #     done = done and (acts_left == 1)

                if done:
                    break
            
            except Exception as e:
                # print(f"Exception during rollout at step {t}: {e}")
                # print(action)
                # print(expert_action)
                break
        
        
        returns.append(ep_ret)
        # Simple Ravens success heuristic (adjust to your task spec):
        succ = float(ep_ret > 0.99)
        successes += int(succ)

        # Save/upload per-episode video either locally or to wandb (or both)
        # episodes_done = len(returns)
        # running_succ_rate = successes / float(max(1, episodes_done))
        # avg_return_so_far = float(np.mean(returns)) if returns else 0.0

        # # Log video and running metrics to wandb
        if args.debug:
            video_np = np.stack(frames, axis=0)  # (T, H, W, 3)
            video_np = write_video(video_np, actions, rewards, expert_masks, succ)
            os.makedirs("videos/debug_dagger_videos", exist_ok=True)
            imageio.mimwrite(f"videos/debug_dagger_videos/totalep_{total_ep:03d}_ep{ep:03d}_succ{succ:.0f}.mp4", video_np, fps=5, quality=8)
        # wandb.log({
        #     # f"video/ep_{ep:03d}": wandb.Video(video_np.transpose(0, 3, 1, 2), fps=10, format="mp4"),
        #     "success_rate": running_succ_rate,
        #     "avg_return": avg_return_so_far,
        #     "episode_return": ep_ret,
        #     "episode_success": succ,
        # }, step=ep)
        # ep += 1
        # imageio.mimwrite(f"debug_dagger_videos/ep_{ep:03d}.mp4", video_np, fps=5, quality=8)
        total_ep += 1
        if succ:
            data_collector.flush()
            pbar.update(1)
            ep += 1
        else:
            data_collector.reset()

    avg_return = float(np.mean(returns)) if returns else 0.0
    succ_rate = successes / max(1, args.num_episodes)
    print(f"\nDone. Episodes: {args.num_episodes} | Avg return: {avg_return:.3f} | Success rate: {succ_rate:.3f}")

    # Final summary
    # wandb.summary["final/avg_return"] = avg_return
    # wandb.summary["final/success_rate"] = succ_rate

if __name__ == "__main__":
    main()
