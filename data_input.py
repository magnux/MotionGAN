from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import h5py as h5
import os
from glob import glob
from tqdm import trange
from utils.threadsafe_iter import threadsafe_generator
from utils.seq_utils import get_swap_list
import re


class DataInput(object):
    """The input data."""
    def __init__(self, config):
        self.data_path = config.data_path
        self.data_set = config.data_set
        self.batch_size = config.batch_size
        self.pick_num = config.pick_num
        self.crop_len = config.crop_len
        self.only_val = config.only_val
        self.data_set_version = config.data_set_version
        self.normalize_data = config.normalize_data
        self.normalize_per_joint = config.normalize_per_joint
        self.epoch_factor = config.epoch_factor
        self.augment_data = config.augment_data
        self.body_members = config.body_members

        self.swap_list = get_swap_list(self.body_members)

        if "Human36" in self.data_set:
            self.used_joints = config.used_joints

        file_path = os.path.join(self.data_path, self.data_set + self.data_set_version + '.h5')
        self.h5file = h5.File(file_path, 'r')
        self.train_keys = [self.data_set + '/Train/' + k
                           for k in self.h5file.get(self.data_set + '/Train').keys()]
        self.val_keys = [self.data_set + '/Validate/' + k
                         for k in self.h5file.get(self.data_set + '/Validate').keys()]

        self.key_pattern = re.compile(".*SEQ(\d+).*")

        # Remove two skel seqs
        if self.data_set == "NTURGBD":
            self.train_keys = [
                key for key in self.train_keys if np.int32(self.h5file[key + '/Action']) < 50
            ]
            self.val_keys = [
                key for key in self.val_keys if np.int32(self.h5file[key + '/Action']) < 50
            ]

        self.len_train_keys = len(self.train_keys)
        self.len_val_keys = len(self.val_keys)

        self.train_epoch_size = (self.len_train_keys // self.batch_size) + 1
        self.val_epoch_size = (self.len_val_keys // self.batch_size) + 1

        self.pshape = [config.njoints, None, 4]
        self.max_plen = config.max_plen

        self.pshape[1] = self.pick_num if self.pick_num > 0 else (
                         self.crop_len if self.crop_len > 0 else None)

        if not self.only_val:
            self.train_batches = self.pre_comp_batches(True)
            self.train_batches *= self.epoch_factor
            self.train_epoch_size *= self.epoch_factor
        self.val_batches = self.pre_comp_batches(False)

    def pre_comp_batches(self, is_training):
        epoch_size = self.train_epoch_size if is_training else self.val_epoch_size
        labs, poses = self.load_to_ram(is_training)

        batches = []
        for slice_idx in range(epoch_size):
            slice_start = slice_idx * self.batch_size
            slice_len = min(slice_start + self.batch_size, labs.shape[0])
            labs_batch = labs[slice_start:slice_len, ...]
            poses_batch = poses[slice_start:slice_len, ...]
            if labs_batch.shape[0] < self.batch_size:
                rand_indices = np.random.random_integers(0, poses.shape[0] - 1, self.batch_size - labs_batch.shape[0])
                labs_batch_extra = labs[rand_indices, ...]
                labs_batch = np.concatenate([labs_batch, labs_batch_extra], axis=0)
                poses_batch_extra = poses[rand_indices, ...]
                poses_batch = np.concatenate([poses_batch, poses_batch_extra], axis=0)
            batches.append((labs_batch, poses_batch))

        del labs
        del poses

        return batches

    def load_to_ram(self, is_training):
        len_keys = self.len_train_keys if is_training else self.len_val_keys
        labs = np.empty([len_keys, 4], dtype=np.int32)
        poses = np.zeros([len_keys, self.pshape[0], self.max_plen, self.pshape[2]], dtype=np.float32)
        splitname = 'train' if is_training else 'val'
        print('Loading "%s" data to ram...' % splitname)
        t = trange(len_keys, dynamic_ncols=True)
        for k in t:
            seq_idx, subject, action, pose, plen = self.read_h5_data(k, is_training)
            pose = pose[:, :, :self.max_plen] if plen > self.max_plen else pose
            plen = self.max_plen if plen > self.max_plen else plen
            labs[k, :] = [seq_idx, subject, action, plen]
            poses[k, :, :plen, :] = pose

        stat_type = '_perjoint' if self.normalize_per_joint else '_global'
        mean_file_path = os.path.join(self.data_path, self.data_set + self.data_set_version + stat_type + '_poses_mean.npy')
        std_file_path = os.path.join(self.data_path, self.data_set + self.data_set_version + stat_type + '_poses_std.npy')

        if tf.gfile.Exists(mean_file_path) and tf.gfile.Exists(std_file_path):
            self.poses_mean = np.load(mean_file_path)
            self.poses_std = np.load(std_file_path)
        else:
            print('Computing mean and std of skels')
            norm_dims = (0, 2) if self.normalize_per_joint else (0, 1, 2)
            self.poses_mean = np.mean(poses[..., :3], axis=norm_dims, keepdims=True)
            self.poses_std = np.std(poses[..., :3], axis=norm_dims, keepdims=True)
            print(self.poses_mean, self.poses_std)
            np.save(mean_file_path, self.poses_mean)
            np.save(std_file_path, self.poses_std)

            zero_std = [i for i in range(self.poses_std.shape[1]) if np.sum(self.poses_std[:, i, ...], axis=-1) < 1e-4]
            if len(zero_std) > 0:
                print('Warning: the following joints have zero std:', zero_std)

        # print(np.min(poses, (0, 1, 2)), np.max(poses, (0, 1, 2)))
        # print(np.std(poses[..., :3], axis=(0, 1, 2), keepdims=True))
        if self.normalize_data:
            poses[..., :3] = self.normalize_poses(poses[..., :3])

        # print(np.min(poses, (0, 1, 2)), np.max(poses, (0, 1, 2)))
        # print(np.std(poses[..., :3], axis=(0, 1, 2), keepdims=True))

        return labs, poses

    def read_h5_data(self, key_idx, is_training):
        if is_training:
            key = self.train_keys[key_idx]
        else:
            key = self.val_keys[key_idx]

        subject = np.int32(self.h5file[key+'/Subject']) - 1  # Small hack to reindex the classes from 0
        action = np.int32(self.h5file[key+'/Action']) - 1  # Small hack to reindex the classes from 0
        pose = np.array(self.h5file[key+'/Pose'], dtype=np.float32)

        pose, plen = self.process_pose(pose)

        seq_idx = np.int32(re.match(self.key_pattern, key).group(1))

        return seq_idx, subject, action, pose, plen

    def process_pose(self, pose):
        plen = np.int32(pose.shape[2])

        if pose.shape[1] > 3:
            pose[:, 3, :] = (pose[:, 3, :] > 0).astype('float32')  # tracking state
        else:
            pose = np.concatenate([pose, np.ones((pose.shape[0], 1, pose.shape[2]))], axis=1)
        pose[np.isnan(pose)] = 0

        if self.data_set == 'NTURGBD':
            pose = pose[:25, :, :]  # Warning: only taking first skeleton
            pose[:, :3, :] = pose[:, :3, :] * 1.0e3  # Rescale to mm
            pose_1 = pose[:, 1, :].copy()
            pose[:, 1, :] = pose[:, 2, :]  # Swapping Y-Z coords
            pose[:, 2, :] = pose_1
        elif self.data_set == 'MSRC12':
            pose[:, :3, :] = pose[:, :3, :] * 1.0e3  # Rescale to mm
            pose_1 = pose[:, 1, :].copy()
            pose[:, 1, :] = pose[:, 2, :]  # Swapping Y-Z coords
            pose[:, 2, :] = pose_1
        elif self.data_set == 'Human36':
            pose = pose[self.used_joints, ...]
            # pose[:, :3, :] = pose[:, :3, :] / 1.0e3 # Rescale to meters
            pose = pose[:, :, range(0, plen, 2)]  # Subsampling to 25hz
            plen = np.int32(pose.shape[2])
        elif self.data_set == 'Human36_expmaps':
            pose = pose[self.used_joints, ...]
            pose = pose[:, :, range(0, plen, 2)]  # Subsampling to 25hz
            plen = np.int32(pose.shape[2])
            # pose[:, :3, :] = (pose[:, :3, :] + 90) / 180

        pose = np.transpose(pose, (0, 2, 1))

        return pose, plen

    def sub_sample_pose(self, pose, plen):

        if self.crop_len > 0:
            if self.crop_len >= plen:
                pose = pose[:, :self.crop_len, :]
            elif self.crop_len < plen:
                indx = np.random.randint(0, plen - self.crop_len)
                pose = pose[:, indx:indx + self.crop_len, :]
            plen = np.int32(self.crop_len)

        if self.pick_num > 0:
            if self.pick_num >= plen:
                pose = pose[:, :self.pick_num, :]
            elif self.pick_num < plen:
                subplen = plen / self.pick_num
                picks = np.random.randint(0, subplen, size=(self.pick_num)) + \
                        np.arange(0, plen, subplen, dtype=np.int32)
                pose = pose[:, picks, :]
            plen = np.int32(self.pick_num)

        return pose  #, plen

    def sub_sample_batch(self, batch, is_training):
        labs_batch, poses_batch = batch

        if self.pshape[1] is not None:
            new_labs_batch = np.empty([self.batch_size, 4], dtype=np.int32)
            new_poses_batch = np.empty([self.batch_size] + self.pshape, dtype=np.float32)
            new_labs_batch[:, :3] = labs_batch[:, :3]
            new_labs_batch[:, 3] = self.pshape[1]
            for i in range(self.batch_size):
                new_poses_batch[i, ...] = self.sub_sample_pose(poses_batch[i, ...], labs_batch[i, 3])

            labs_batch = new_labs_batch
            poses_batch = new_poses_batch

            if self.augment_data and is_training:
                poses_batch = self.data_augmentation(poses_batch)

        return labs_batch, poses_batch

    def data_augmentation(self, poses):
        def _jitter_height(poses):
            jitter_tensor = np.random.uniform(0.7, 1.3, (self.batch_size, 1, 1, 1))
            poses[..., 2:] = poses[..., 2:] * jitter_tensor
            return poses

        def _swap_sides(poses):
            if np.random.rand() > 0.5:
                poses[..., :1] = poses[..., :1] * -1.0
                for swap_tup in self.swap_list:
                    poses_tmp = poses[:, swap_tup[0], :, :].copy()
                    poses[:, swap_tup[0], :, :] = poses[:, swap_tup[1], :, :]
                    poses[:, swap_tup[1], :, :] = poses_tmp
            return poses

        poses = _jitter_height(poses)
        poses = _swap_sides(poses)
        return poses

    @threadsafe_generator
    def batch_generator(self, is_training):
        epoch_size = self.train_epoch_size if is_training else self.val_epoch_size
        batches = self.train_batches if is_training else self.val_batches

        while True:
            rand_indices = np.random.permutation(epoch_size)
            for slice_idx in range(epoch_size):
                if not self.only_val:
                    yield self.sub_sample_batch(batches[rand_indices[slice_idx]], is_training)
                else:
                    yield self.sub_sample_batch(batches[slice_idx], is_training)

    def normalize_poses(self, poses):
        return (poses - self.poses_mean) / (self.poses_std + 1e-8)

    def unnormalize_poses(self, poses):
        return (poses * (self.poses_std + 1e-8)) + self.poses_mean