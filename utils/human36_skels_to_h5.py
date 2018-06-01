from __future__ import absolute_import, division, print_function

import os

os.environ["CDF_LIB"] = "/usr/local/cdf"
from spacepy import pycdf
import numpy as np
import h5py as h5
import re
from glob import glob
from tqdm import tqdm

dataset = 'Human36'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cameras = ['54138969', '55011271', '58860488', '60457274']
actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
           'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
           'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']


def read_pose(x):
    return np.reshape(np.transpose(np.array(x), [2, 0, 1]), [32, 3, -1])


if __name__ == "__main__":
    prog = re.compile('(S\d+)/MyPoseFeatures/D3_Positions/([^ ]+)[ ]*(\d)*.cdf')

    # First renaming files that need a 0
    found_files = sorted([file for file in glob('extracted/S*/MyPoseFeatures/D3_Positions/*.cdf')])
    for f, found_file in enumerate(tqdm(found_files)):
        confpars = prog.findall(found_file)[0]
        if confpars[2] == '':
            os.rename(found_file, found_file[:-4] + ' 0.cdf')

    # Then renaming files to 1, 2
    found_files = sorted([file for file in glob('extracted/S*/MyPoseFeatures/D3_Positions/*.cdf')])
    prev_subject = ''
    prev_action = ''
    for f, found_file in enumerate(tqdm(found_files)):
        confpars = prog.findall(found_file)[0]
        if confpars[0] == prev_subject and confpars[1] == prev_action:
            subaction = 2
        else:
            subaction = 1
        os.rename(found_file, found_file[:-5] + str(subaction) + '.cdf.tmp')
        prev_subject = confpars[0]
        prev_action = confpars[1]

    found_files = sorted([file for file in glob('extracted/S*/MyPoseFeatures/D3_Positions/*.cdf.tmp')])
    for f, found_file in enumerate(tqdm(found_files)):
        os.rename(found_file, found_file[:-4])

    found_files = sorted([file for file in glob('extracted/S*/MyPoseFeatures/D3_Positions/*.cdf')])
    print('Processing {} files...'.format(len(found_files)))

    h5file = h5.File(dataset + "v1.h5", "w")
    max_len = 0
    for f, found_file in enumerate(tqdm(found_files)):
        confpars = prog.findall(found_file)[0]
        subject = [i for i, x in enumerate(subjects) if x == confpars[0]][0]
        action = [i for i, x in enumerate(actions) if x in confpars[1]][0]
        subaction = int(confpars[2])

        # print(found_file)
        # print(subject, action)

        subarray = np.array(subject + 1)
        actarray = np.array(action + 1)
        sactarray = np.array(subaction)

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
            datapath + 'Subaction', np.shape(sactarray),
            dtype='int32', data=sactarray
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