import os
import numpy as np
import h5py as h5
import re
import csv
from glob import glob
from tqdm import trange
import shutil
import time

from multiprocessing import Process, Queue, current_process, freeze_support


def worker(input, output):
    prog = re.compile('MicrosoftGestureDataset-RC/data/P(\d)_(\d)_(\d*).*_p(\d*).tagstream',re.IGNORECASE)
    for found_file in iter(input.get, 'STOP'):
        confpars = prog.findall(found_file)[0]
        instruction = (int(confpars[0]), int(confpars[1]))
        action = int(confpars[2])
        subject = int(confpars[3])

        with open(found_file) as csvfile:
            tags_reader = csv.reader(csvfile, delimiter=';')
            tags = []
            for r, row in enumerate(tags_reader):
                if r == 0:
                    assert(row[0] == 'XQPCTick')
                else:
                    tag = (int(row[0])*1000 + (49875/2))/49875
                    tags.append(tag)

        frame_count = 0
        pose_arrays = []
        data_file = found_file[:-10]+'.csv'
        with open(data_file) as csvfile:
            skelreader = csv.reader(csvfile, delimiter=' ')
            for tag in tags:
                current_frame = 0
                skels = []
                while current_frame < tag:
                    row = next(skelreader)
                    current_frame = int(row[0])
                    skel = np.reshape(np.array(row[1:], dtype=np.float32), [20, 4, 1])
                    skels.append(skel)
                pose_arrays.append(np.concatenate(skels,axis=2))
                frame_count = max(frame_count, len(skels))

        output.put((subject, action, pose_arrays, frame_count))


if __name__ == '__main__':

    found_dirs = [file for file in glob('MicrosoftGestureDataset-RC/data/*.tagstream')]
    print('Processing %d files...' % (len(found_dirs)))

    data_set = 'MSRC12'
    h5file = h5.File(data_set+"v1.h5", "w")

    subjects = set()
    actions = set()
    max_frame_count = 0

    num_procs = 4

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for found_dir in found_dirs:
        task_queue.put(found_dir)

    # Start worker processes
    print('Spawning processes...')
    for _ in range(num_procs):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Processed Files:')
    t = trange(len(found_dirs), dynamic_ncols=True)
    seq_num = 0
    for _ in t:
        subject, action, pose_arrays, frame_count = done_queue.get()

        subjects.add(subject)
        actions.add(action)
        sub_array = np.array(subject)
        act_array = np.array(action)
        max_frame_count = max(frame_count, max_frame_count)

        # v1 split (cross subject protocol)
        data_split = 'Train' if (subject % 2) == 1 else 'Validate'

        for pose_array in pose_arrays:
            data_path = '{}/{}/SEQ{}/'.format(data_set, data_split, seq_num)
            h5file.create_dataset(
                data_path + 'Subject', np.shape(sub_array),
                dtype='int32', data=sub_array
            )
            h5file.create_dataset(
                data_path + 'Action', np.shape(act_array),
                dtype='int32', data=act_array
            )
            h5file.create_dataset(
                data_path + 'Pose', np.shape(pose_array),
                dtype='float32', data=pose_array
            )
            seq_num += 1

    # Tell child processes to stop
    print('Stopping processes...')
    for _ in range(num_procs):
        task_queue.put('STOP')

    h5file.flush()
    h5file.close()

    print("")
    print("done.")
    print("Subjects: ", subjects)
    print("Actions: ", actions)
    print("Max Frame Count:", max_frame_count)
