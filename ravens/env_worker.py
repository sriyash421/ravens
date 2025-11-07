# ravens/env_worker.py

import traceback
from ravens.environments.environment import Environment, ContinuousEnvironment
from ravens import tasks
import numpy as np
import pybullet as p

def get_obj_id_and_goal(env):
    goals = env.task.goals[0]
    objs, matches, targs, replace, rotations, _, _, _ = goals
    object_id = objs[0][0]  # Get the ID of the first object
    target_pose = np.array(targs[0][0]+targs[0][1])  # (x, y, z, qx, qy, qz, qw)
    return object_id, target_pose

def run_env_worker(worker_id, task_name, assets_root, mode,
                   continuous, steps_per_seg, task_queue, result_queue, noise, debug):
    """Worker process that reuses a single persistent environment."""
    print(f'[Worker {worker_id}] Starting.')

    # Create and initialize environment
    env_cls = ContinuousEnvironment if continuous else Environment
    # env = Environment(assets_root, disp=False, shared_memory=False)
    env = env_cls(
      assets_root,
      disp=False,
      shared_memory=True,
      hz=480)

    task = tasks.names[task_name](continuous=continuous)
    task.mode = mode
    env.set_task(task)

    agent = task.oracle(env, steps_per_seg=steps_per_seg)
    max_steps = task.max_steps
    if continuous:
        max_steps *= (steps_per_seg * agent.num_poses)
    
    # if noise > 0:
    #     max_steps *= 2  # Allow more steps for noisy actions

    while True:
        try:
            seed = task_queue.get()
            np.random.seed(seed)
            obs = env.reset()
            obj_id, goal_pose = get_obj_id_and_goal(env)

            episode = []
            done = False
            info = {}
            total_reward = 0
            reward = 0
            prev_action = np.zeros((8,))

            for step in range(max_steps):
                # print(f'[Worker {worker_id}] Step {step}/{max_steps}')
                # print(f'Total Reward: {total_reward} Done: {done}')
                # print("----")
                act = agent.act(obs, info)
                ee_pose = p.getLinkState(env.ur5, env.ee_tip)
                obj_pose = p.getBasePositionAndOrientation(obj_id)
                grasp = np.array([env.ee.check_grasp()])
                info.update({
                    "ee_pose": np.array(ee_pose[0] + ee_pose[1]),
                    "obj_pose": np.array(obj_pose[0] + obj_pose[1]),
                    "goal_pose": goal_pose,
                    "grasp": grasp,
                    "prev_action": prev_action,
                })
                # print()
                if noise > 0:
                    pos = act['move_cmd'][0] + np.random.normal(0, noise, size=3)
                    rot = act['move_cmd'][1]
                    # rot = np.array([0,0,0,1])#act['move_cmd'][1] * 0.0
                    act['move_cmd'] = (pos, rot)
                    # act['move_cmd'][0] += np.random.normal(0, noise, size=3)
                    # act['move_cmd'][1] += np.random.normal(0, noise, size=4)
                prev_action = np.concatenate((
                    act['move_cmd'][0],
                    act['move_cmd'][1],
                    np.array([act['suction_cmd']])
                ))

                episode.append((obs, act, reward, info))
                obs, reward, done, info = env.step(act)
                # print(f'Total Reward: {total_reward} Done: {done}')
                total_reward += reward
                if done:
                    break
            
            if debug:
                result_queue.put((seed, episode))
            elif total_reward > 0.99 and len(episode) > 1 and len(episode) < 100:
                result_queue.put((seed, episode))

        except Exception as e:
            print(e)

    print(f'[Worker {worker_id}] Shutting down.')