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
                    tag = (int(row[0])*1000 + 49875/2)/49875
                    tags.append(tag)

        framecount = 0
        posearrays = []
        data_file = found_file[:-10]+'.csv'
        with open(data_file) as csvfile:
            skelreader = csv.reader(csvfile, delimiter=' ')
            for tag in tags:
                current_frame = 0
                skels = []
                while(current_frame < tag):
                    row = next(skelreader)
                    current_frame = int(row[0])
                    skel = np.reshape(np.array(row[1:], dtype=np.float32),[20,4,1])
                    skels.append(skel)
                posearrays.append(np.concatenate(skels,axis=2))
                framecount = max(framecount, len(skels))

        output.put((subject, action, posearrays, framecount))

if __name__ == '__main__':

    found_dirs = [file for file in glob('MicrosoftGestureDataset-RC/data/*.tagstream')]
    print('Processing %d files...' % (len(found_dirs)))

    dataset = 'MSRC12'
    h5file = h5.File(dataset+".h5", "w")

    subjects = set()
    actions = set()
    maxframecount = 0

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
    seqnum = 0
    for _ in t:
        subject, action, posearrays, framecount = done_queue.get()

        subjects.add(subject)
        actions.add(action)
        subarray = np.array(subject)
        actarray = np.array(action)
        maxframecount = max(framecount, maxframecount)

        # v1 split (cross subject protocol)
        datasplit = 'Train' if (subject % 2) == 1 else 'Validate'

        for posearray in posearrays:
            datapath = '{}/{}/SEQ{}/'.format(dataset,datasplit,seqnum)
            h5file.create_dataset(
                datapath+'Subject', np.shape(subarray),
                dtype='int32', data=subarray
            )
            h5file.create_dataset(
                datapath+'Action', np.shape(actarray),
                dtype='int32', data=actarray
            )
            h5file.create_dataset(
                datapath+'Pose', np.shape(posearray),
                dtype='float32', data=posearray
            )
            seqnum += 1

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
    print("Max Frame Count:", maxframecount)
