import os
import time
from multiprocessing import Process, Queue

from absl import app, flags
from ravens.dataset import Dataset
from ravens.env_worker import run_env_worker
from tqdm import tqdm  # Added tqdm import

flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')
flags.DEFINE_bool('continuous', False, '')
flags.DEFINE_integer('steps_per_seg', 3, '')
flags.DEFINE_integer('num_workers', 4, 'Number of parallel environment workers.')
flags.DEFINE_float('noise', 0.0, 'Action noise standard deviation.')
flags.DEFINE_bool('debug', False, 'If true, save all episodes including failures.')
flags.DEFINE_string('exp_name', '', 'Optional experiment name to append to data directory.')

FLAGS = flags.FLAGS

def main(_):
    path = os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{FLAGS.mode}')
    if FLAGS.exp_name:
        path += f'-{FLAGS.exp_name}'
    if FLAGS.noise > 0:
        path += f'-noise{FLAGS.noise}'
    if FLAGS.debug:
        path += '-debug'
    dataset = Dataset(path)

    # Setup seeds
    seed = dataset.max_seed
    if seed < 0:
        seed = -1 if (FLAGS.mode == 'test') else -2

    n_total = FLAGS.n
    n_collected = dataset.n_episodes

    # Initialize tqdm progress bar
    pbar = tqdm(total=n_total, initial=n_collected, desc='Collecting episodes')

    task_queue = Queue()
    result_queue = Queue()
    workers = []

    # Spawn fixed worker processes
    for i in range(FLAGS.num_workers):
        proc = Process(
            target=run_env_worker,
            args=(i, FLAGS.task, FLAGS.assets_root, FLAGS.mode,
                  FLAGS.continuous, FLAGS.steps_per_seg, task_queue, result_queue, FLAGS.noise, FLAGS.debug))
        proc.start()
        workers.append(proc)

    # Dispatch seeds to workers
    next_seed = seed
    while n_collected < n_total:
        next_seed += 2
        task_queue.put(next_seed)

        # Collect finished episodes
        while not result_queue.empty():
            s, episode = result_queue.get()
            dataset.add(s, episode)
            n_collected += 1
            pbar.update(1)

        time.sleep(0.1)

    # Wait for any remaining results
    while n_collected < n_total:
        s, episode = result_queue.get()
        dataset.add(s, episode)
        n_collected += 1
        pbar.update(1)

    pbar.close()

    # Shutdown all workers
    # for _ in workers:
    #     task_queue.put(None)  # Shutdown signal

    for proc in workers:
        proc.terminate()
        proc.join()

if __name__ == '__main__':
    app.run(main)
