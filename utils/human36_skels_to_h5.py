from __future__ import absolute_import, division, print_function

import os

os.environ["CDF_LIB"] = "/usr/local/cdf"
from spacepy import pycdf
import numpy as np
import h5py as h5
import re
from glob import glob
from tqdm import tqdm

found_files = [file for file in glob('extracted/S*/MyPoseFeatures/D3_Positions/*.cdf')]

print('Processing {} files...'.format(len(found_files)))

subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cameras = ['54138969', '55011271', '58860488', '60457274']
actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
           'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
           'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

prog = re.compile('(S\d+)/MyPoseFeatures/D3_Positions/([^.]+).cdf')


def read_pose(x):
    return np.reshape(np.transpose(np.array(x), [2, 0, 1]), [32, 3, -1])


dataset = 'Human36'
h5file = h5.File(dataset + "v1.h5", "w")
max_len = 0
for f, found_file in enumerate(tqdm(found_files)):
    confpars = prog.findall(found_file)[0]
    subject = [i for i, x in enumerate(subjects) if x == confpars[0]][0]
    action = [i for i, x in enumerate(actions) if x in confpars[1]][0]

    # print(found_file)
    # print(subject, action)

    subarray = np.array(subject + 1)
    actarray = np.array(action + 1)

    pose3dcdf = pycdf.CDF(found_file)

    posearray = read_pose(pose3dcdf['Pose'])
    pose3dcdf.close()

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