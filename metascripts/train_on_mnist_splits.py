'''
Train multiple DRNets on multiple slices of Moving MNIST. This yields trained models that can be
run on corresponding Moving MNIST validation slices (diagonal experiments) or on other validation
slices (for generalization experiments).
'''

import os
import sys
import glob
from pprint import pprint
import re
from filelock import FileLock, Timeout
import time
from multiprocessing import Pool
import numpy as np
import traceback
import subprocess
import argparse
from functools import partial


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'new_mnist'))
TRAIN_TORONTO_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'train_new_mnist.lua'))
TRAIN_LSTM_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'train_lstm_new_mnist.lua'))
DRNET_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def launch_job(t, num_gpus):
    i, cmd = t
    gpu_id = 0
    launched_job = False

    while not launched_job:
        # Try to acquire lock for current GPU
        lock = FileLock('/tmp/gpu_%d.lck' % gpu_id, timeout=0)
        try:
            with lock.acquire():
                # # Test dummy "process"
                # try:
                #     print('%d %s' % (gpu_id, cmd))
                #     np.random.seed(i)
                #     time.sleep(np.random.randint(3))
                # except KeyboardInterrupt:
                #     raise
                # finally:
                #     launched_job = True

                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                try:
                    subprocess.check_call(cmd, shell=True, env=env)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    traceback.print_exc()
                    # Log failed command
                    with FileLock('failed_cmds_log.lck'):
                        with open('failed_cmds.log', 'a') as f:
                            f.write(cmd + '\n')
                finally:
                    launched_job = True
        except Timeout:
            # Try the next GPU if current GPU is used
            gpu_id = gpu_id + 1 % num_gpus


def main(num_gpus, slice_names_file, lstm):
    os.chdir(DRNET_ROOT_DIR)
    if slice_names_file is None:
        raise ValueError('Path to slice names file must be provided')
    else:
        with open(slice_names_file, 'r') as f:
            slice_names = [line.strip() for line in f.readlines()]
        # Filter out empty lines or commented lines
        slice_names = filter(lambda x: len(x) > 0 and not x.startswith('#'), slice_names)
        video_file_paths = [os.path.join(MNIST_DATA_DIR, '%s_videos.h5' % slice_name) for slice_name in slice_names]

    if lstm:
        cmd_fmt = 'th %s --sliceName %%s --nPast 10 --nFuture 5' % TRAIN_LSTM_PATH
    else:
        cmd_fmt = 'th %s --sliceName %%s' % TRAIN_TORONTO_PATH
    dataset_labels = [re.search('.*/(.*)_videos\.h5', path).group(1) for path in video_file_paths]
    cmds = [cmd_fmt % dataset_label for dataset_label in dataset_labels][::-1]

    # Start the jobs
    pool = Pool(num_gpus)
    fn = partial(launch_job, num_gpus=num_gpus)
    res = pool.map_async(fn, enumerate(cmds))

    try:
        # Set timeout to avoid hanging on interrupt
        res.get(9999999)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, help='Number of GPUs on this machine')
    parser.add_argument('--slice_names_file', type=str,
                        default=os.path.join(SCRIPT_DIR, 'slice_names.txt'),
                        help='File path to list of MNIST slice names')
    parser.add_argument('--lstm', action='store_true',
                        help='Flag to train LSTM on top of trained DRNet')
    args = parser.parse_args()
    main(**vars(args))
