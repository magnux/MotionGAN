from __future__ import absolute_import, division, print_function

import os

import numpy as np
import h5py as h5
import re
from glob import glob
from tqdm import tqdm

dataset = 'Human36_expmaps'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cameras = ['54138969', '55011271', '58860488', '60457274']
actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning',
           'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',
           'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']


def read_pose(x):
    return np.reshape(np.transpose(np.array(x), [1, 0]), [33, 3, -1])


if __name__ == "__main__":
    found_files = [file for file in glob('extracted_expmaps/S*/*.txt')]
    print('Processing {} files...'.format(len(found_files)))

    prog = re.compile('(S\d+)/([^_]+)_(\d).txt')
    h5file = h5.File(dataset + "v1.h5", "w")
    max_len = 0
    for f, found_file in enumerate(tqdm(found_files)):
        confpars = prog.findall(found_file)[0]
        subject = [i for i, x in enumerate(subjects) if x == confpars[0]][0]
        action = [i for i, x in enumerate(actions) if x in confpars[1]][0]
        subaction = int(confpars[2])

        # print(found_file)
        # print(subject, action, subaction)

        subarray = np.array(subject + 1)
        actarray = np.array(action + 1)

        posearray = []
        lines = open(found_file).readlines()
        for line in lines:
            line = line.strip().split(',')
            if len(line) > 0:
                posearray.append(np.array([np.float32(x) for x in line]))

        posearray = read_pose(posearray)

        # S5 will be the Validate split the rest of the subjects are the training set
        datasplit = 'Validate' if subjects[subject] == 'S5' else 'Train'

        datapath = '{}/{}/SEQ{}/'.format(dataset, datasplit, f)
        h5file.create_dataset(
            datapath + 'Subject', np.shape(subarray),
            dtype='int32', data=subarray
        )
        h5file.create_dataset(
            datapath + 'Action', np.shape(actarray),
            dtype='int32', data=actarray
        )
        h5file.create_dataset(
            datapath + 'Pose', np.shape(posearray),
            dtype='float32', data=posearray
        )
        max_len = max(max_len, posearray.shape[2])

    print('Dataset sample: ')
    print(h5file.get(dataset + '/Validate/'))
    print('max length', max_len)
    h5file.flush()
    h5file.close()