import os
import numpy as np
import h5py as h5
import re
import json
from glob import glob
from tqdm import trange
import shutil
import time

from multiprocessing import Process, Queue, current_process, freeze_support

class Frame(object):
    pass

class Body(object):
    pass

class Joint(object):
    pass

def read_skeleton_file(filename):
    skfile = open(filename)
    framecount = int(skfile.readline())# no of the recorded frames
    frameinfo = [Frame() for _ in range(framecount)]# [Frame()] * framecount, uhh not the same, careful with ptrs
    for f in range(framecount):
        bodycount = int(skfile.readline())# no of observerd skeletons in current frame
        frameinfo[f].bodies = []# to store multiple skeletons per frame
        if bodycount > 0:
            for b in range(bodycount):
                body = Body()
                confline = skfile.readline().split()
                assert len(confline) == 10
                body.bodyID = int(confline[0])# tracking id of the skeleton
                # read 6 int
                body.clipedEdges = int(confline[1])
                body.handLeftConfidence = int(confline[2])
                body.handLeftState = int(confline[3])
                body.handRightConfidence = int(confline[4])
                body.handRightState = int(confline[5])
                body.isResticted = int(confline[6])
                # read 2 floats
                body.leanX = float(confline[7])
                body.leanY = float(confline[8])
                # read 1 int
                body.trackingState = int(confline[9])

                body.jointCount = int(skfile.readline())# no of joints (25)
                assert body.jointCount == 25
                body.joints = []
                for j in range(body.jointCount):
                    jointinfo = [float(s) for s in skfile.readline().split()]
                    assert len(jointinfo) == 12
                    joint = Joint()

                    # 3D location of the joint j
                    joint.x = jointinfo[0]
                    joint.y = jointinfo[1]
                    joint.z = jointinfo[2]

                    # 2D location of the joint j in corresponding depth/IR frame
                    joint.depthX = jointinfo[3]
                    joint.depthY = jointinfo[4]

                    # 2D location of the joint j in corresponding RGB frame
                    joint.colorX = jointinfo[5]
                    joint.colorY = jointinfo[6]

                    # The orientation of the joint j
                    joint.orientationW = jointinfo[7]
                    joint.orientationX = jointinfo[8]
                    joint.orientationY = jointinfo[9]
                    joint.orientationZ = jointinfo[10]

                    # The tracking state of the joint j
                    joint.trackingState = int(jointinfo[11])

                    body.joints.append(joint)

                frameinfo[f].bodies.append(body)

    skfile.close()
    return frameinfo, framecount

def worker(input, output):
    prog = re.compile('[^S]*S(\d+)C(\d+)P(\d+)R(\d+)A(\d+).skeleton')
    for found_file in iter(input.get, 'STOP'):
        confpars = prog.findall(found_file)[0]
        setup = int(confpars[0])
        camera = int(confpars[1])
        subject = int(confpars[2])# AKA Performer
        replication = int(confpars[3])# AKA Repetition
        action = int(confpars[4])

        frames, framecount = read_skeleton_file(found_file)

        skeletons = {}
        for f, frame in enumerate(frames):
            for body in frame.bodies:
                if body.bodyID not in skeletons:
                    skeletons[body.bodyID] = np.zeros([25,9,len(frames)])
                for j, joint in enumerate(body.joints):
                    skeletons[body.bodyID][j,0,f] = joint.x
                    skeletons[body.bodyID][j,1,f] = joint.y
                    skeletons[body.bodyID][j,2,f] = joint.z

                    skeletons[body.bodyID][j,3,f] = joint.orientationW
                    skeletons[body.bodyID][j,4,f] = joint.orientationX
                    skeletons[body.bodyID][j,5,f] = joint.orientationY
                    skeletons[body.bodyID][j,6,f] = joint.orientationZ
                    
                    skeletons[body.bodyID][j,7,f] = joint.colorX
                    skeletons[body.bodyID][j,8,f] = joint.colorY

        poses = []
        # cleaning artifacts
        for skey, skel in skeletons.items():
            minX = np.min(skel[:,0,:])
            maxX = np.max(skel[:,0,:])
            minY = np.min(skel[:,1,:])
            maxY = np.max(skel[:,1,:])
            skSpr = (maxY - minY) / (maxX - minX)
            if not (skSpr < 0.5 or skSpr > 10):
                varX = np.var(skel[:,0,:])
                varY = np.var(skel[:,1,:])
                varZ = np.var(skel[:,2,:])
                skelVar = varX + varY + varZ
                if not (skelVar > 10 or skelVar < 1e-2):
                    poses.append((skel, skelVar))

        # ordering principal subject
        if len(poses) > 1:
            poses.sort(key=lambda tup: tup[1], reverse=True)

        posearray = None
        if len(poses)  == 1:
            posearray = np.empty([50,9,len(frames)],dtype=float)
            posearray[0:25,:,:] = poses[0][0]
            posearray[25:50,:,:] = np.zeros([25,9,len(frames)])
        elif len(poses) > 1:
            posearray = np.empty([50,9,len(frames)],dtype=float)
            posearray[0:25,:,:] = poses[0][0]
            posearray[25:50,:,:] = poses[1][0]
        else:
            print("%s file will be skipped... skeletons %d" % (found_file, len(skeletons)))

        output.put((subject, action, posearray, framecount, camera))

if __name__ == '__main__':

    found_files = [file for file in glob('skeletons/*.skeleton')]
    print('Processing %d files...' % (len(found_files)))

    dataset = 'NTURGBD'
    train_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
    train_cameras = {2, 3}
    h5file_v1 = h5.File(dataset+".h5", "w")
    h5file_v2 = h5.File(dataset+"v2.h5", "w")

    subjects = set()
    actions = set()
    maxframecount = 0

    num_procs = 8

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for found_file in found_files:
        task_queue.put(found_file)

    # Start worker processes
    print('Spawning processes...')
    for _ in range(num_procs):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Processed Files:')
    t = trange(len(found_files), dynamic_ncols=True)
    for seqnum in t:
        subject, action, posearray, framecount, camera = done_queue.get()

        if posearray is not None:
            subjects.add(subject)
            actions.add(action)
            maxframecount = max(framecount, maxframecount)

            subarray = np.array(subject)
            actarray = np.array(action)
            camarray = np.array(camera)

            # v1 split (cross subject protocol)
            datasplit_v1 = 'Train' if subject in train_subjects else 'Validate'

            # v2 split (cross view protocol)
            datasplit_v2 = 'Train' if camera in train_cameras else 'Validate'

            def write_data(h5file, datasplit):
                datapath = '{}/{}/SEQ{}/'.format(dataset,datasplit,seqnum)
                h5file.create_dataset(
                    datapath+'Camera', np.shape(camarray),
                    dtype='int32', data=camarray
                )
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

            write_data(h5file_v1, datasplit_v1)
            write_data(h5file_v2, datasplit_v2)

    # Tell child processes to stop
    print('Stopping processes...')
    for _ in range(num_procs):
        task_queue.put('STOP')

    def fac(hfile):
        hfile.flush()
        hfile.close()

    fac(h5file_v1)
    fac(h5file_v2)

    print("")
    print("done.")
    print("Subjects: ", subjects)
    print("Actions: ", actions)
    print("Max Frame Count:", maxframecount)
