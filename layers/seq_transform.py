from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras.layers import Lambda
from utils.scoping import Scoping
from utils.tfangles import quaternion_between, quaternion_to_expmap, expmap_to_rotmat, rotmat_to_euler, \
    vector3d_to_quaternion, quaternion_conjugate, rotate_vector_by_quaternion, rotmat_to_quaternion, \
    expmap_to_quaternion, quaternion_to_rotmat
from utils.seq_utils import get_body_graph


def remove_hip_in(x, x_mask, data_set):
    scope = Scoping.get_global_scope()
    with scope.name_scope('remove_hip'):

        if 'expmaps' in data_set:
            hip_info = Lambda(lambda arg: arg[:, :2, :, :], name=scope+'hip_expmaps')(x)

            x = Lambda(lambda arg: arg[:, 2:, ...], name=scope+'remove_hip_in')(x)
            x_mask = Lambda(lambda arg: arg[:, 2:, ...], name=scope+'remove_hip_mask_in')(x_mask)
        else:
            def _get_hips(arg):
                return K.reshape(arg[:, 0, :, :], (arg.shape[0], 1, arg.shape[2], 3))

            hip_info = Lambda(_get_hips, name=scope+'hip_coords')(x)

            x = Lambda(lambda args: (args[0] - args[1])[:, 1:, ...], name=scope+'remove_hip_in')([x, hip_info])
            x_mask = Lambda(lambda arg: arg[:, 1:, ...], name=scope+'remove_hip_mask_in')(x_mask)
    return x, x_mask, hip_info


def remove_hip_out(x, hip_info, data_set):
    scope = Scoping.get_global_scope()
    with scope.name_scope('remove_hip'):

        if 'expmaps' in data_set:
            x = Lambda(lambda args: K.concatenate([args[1], args[0]], axis=1),
                       name=scope+'remove_hip_out')([x, hip_info])
        else:
            x = Lambda(lambda args: K.concatenate([args[1], args[0] + args[1]], axis=1),
                       name=scope+'remove_hip_out')([x, hip_info])
    return x


def translate_start_in(x):
    scope = Scoping.get_global_scope()
    with scope.name_scope('translate_start'):
        def _get_start(arg):
            return K.reshape(arg[:, 0, 0, :], (arg.shape[0], 1, 1, 3))

        start_coords = Lambda(_get_start, name=scope+'start_coords')(x)

        x = Lambda(lambda args: args[0] - args[1], name=scope+'translate_start_in')([x, start_coords])
    return x, start_coords


def translate_start_out(x, start_coords):
    scope = Scoping.get_global_scope()
    with scope.name_scope('translate_start'):
        x = Lambda(lambda args: args[0] + args[1], name=scope+'translate_start_out')([x, start_coords])
    return x


def rotate_start_in(x, body_members):
    scope = Scoping.get_global_scope()
    with scope.name_scope('rotate_start'):
        left_shoulder = body_members['left_arm']['joints'][1]
        right_shoulder = body_members['right_arm']['joints'][1]
        hip = body_members['torso']['joints'][0]
        head_top = body_members['head']['joints'][-1]

        base_shape = [int(d) for d in x.shape]
        base_shape[1] = 1
        base_shape[2] = 1

        def _get_rotation(arg):
            coords_list = tf.unstack(arg[:, :, 0, :], axis=1)
            torso_rot = tf.cross(coords_list[left_shoulder] - coords_list[hip],
                                 coords_list[right_shoulder] - coords_list[hip])
            side_rot = K.reshape(tf.cross(coords_list[head_top] - coords_list[hip], torso_rot), base_shape)
            theta_diff = ((np.pi / 2) - tf.atan2(side_rot[..., 1], side_rot[..., 0])) / 2
            cos_theta_diff = tf.cos(theta_diff)
            sin_theta_diff = tf.sin(theta_diff)
            zeros_theta = K.zeros_like(sin_theta_diff)
            return K.stack([cos_theta_diff, zeros_theta, zeros_theta, sin_theta_diff], axis=-1)

        start_rotation = Lambda(_get_rotation, name=scope+'start_rotation')(x)

        x = Lambda(lambda args: rotate_vector_by_quaternion(args[1], args[0]),
                   name=scope+'rotate_start_in')([x, start_rotation])
    return x, start_rotation


def rotate_start_out(x, start_rotation):
    scope = Scoping.get_global_scope()
    with scope.name_scope('rotate_start'):
        x = Lambda(lambda args: rotate_vector_by_quaternion(quaternion_conjugate(args[1]), args[0]),
                   name=scope+'rotate_start_out')([x, start_rotation])
    return x


def rescale_body_in(x, body_members):
    scope = Scoping.get_global_scope()
    with scope.name_scope('rescale'):
        members_from, members_to, _ = get_body_graph(body_members)

        def _get_avg_bone_len(arg):
            bone_list = tf.unstack(arg[:, :, 0, :], axis=1)
            bones = [bone_list[j] - bone_list[i] for i, j in zip(members_from, members_to)]
            bones = K.expand_dims(K.stack(bones, axis=1), axis=2)
            bone_len = K.sqrt(K.sum(K.square(bones), axis=-1, keepdims=True) + K.epsilon())
            return K.mean(bone_len, axis=1, keepdims=True)

        bone_len = Lambda(_get_avg_bone_len, name=scope+'bone_len')(x)

        x = Lambda(lambda args: args[0] / args[1], name=scope+'rescale_body_in')([x, bone_len])
    return x, bone_len


def rescale_body_out(x, bone_len):
    scope = Scoping.get_global_scope()
    with scope.name_scope('rescale'):
        x = Lambda(lambda args: args[0] * args[1], name=scope+'rescale_body_out')([x, bone_len])
    return x


def seq_to_diff_in(x, x_mask=None):
    scope = Scoping.get_global_scope()
    with scope.name_scope('seq_to_diff'):
        start_pose = Lambda(lambda arg: arg[:, :, 0, :], name=scope+'start_pose')(x)

        x = Lambda(lambda arg: arg[:, :, 1:, :] - arg[:, :, :-1, :], name=scope+'seq_to_diff_in')(x)

        if x_mask is not None:
            x_mask = Lambda(lambda arg: arg[:, :, 1:, :] * arg[:, :, :-1, :], name=scope+'seq_mask_to_diff_in')(x_mask)
    return x, x_mask, start_pose


def seq_to_diff_out(x, start_pose):
    scope = Scoping.get_global_scope()
    with scope.name_scope('seq_to_diff'):
        def _diff_to_seq(args):
            diffs, start_pose = args
            diffs_list = tf.unstack(diffs, axis=2)
            poses = [start_pose]
            for p in range(diffs.shape[2]):
                poses.append(poses[p] + diffs_list[p])
            return K.stack(poses, axis=2)

        x = Lambda(_diff_to_seq, name=scope+'seq_to_diff_out')([x, start_pose])
    return x


def seq_to_angles_in(x, x_mask, body_members):
    scope = Scoping.get_global_scope()
    with scope.name_scope('seq_to_angles'):

        members_from, members_to, body_graph = get_body_graph(body_members)

        def _get_hips(arg):
            return K.reshape(arg[:, 0, :, :], (arg.shape[0], 1, arg.shape[2], 3))

        hip_coords = Lambda(_get_hips, name=scope+'hip_coords')(x)

        def _get_bone_len(arg):
            bone_list = tf.unstack(arg[:, :, 0, :], axis=1)
            bones = [bone_list[j] - bone_list[i] for i, j in zip(members_from, members_to)]
            bones = K.stack(bones, axis=1)
            return K.sqrt(K.sum(K.square(bones), axis=-1) + K.epsilon())

        bone_len = Lambda(_get_bone_len, name=scope+'bone_len')(x)

        def _get_angles(coords):
            base_shape = [int(dim) for dim in coords.shape]
            base_shape.pop(1)
            base_shape[-1] = 1

            coords_list = tf.unstack(coords, axis=1)

            def _get_angle_for_joint(joint_idx, parent_idx, angles):
                if parent_idx is None:  # joint_idx should be 0
                    parent_bone = K.constant(np.concatenate([np.ones(base_shape),
                                                             np.zeros(base_shape),
                                                             np.zeros(base_shape)], axis=-1))
                else:
                    parent_bone = coords_list[parent_idx] - coords_list[joint_idx]

                for child_idx in body_graph[joint_idx]:
                    child_bone = coords_list[child_idx] - coords_list[joint_idx]
                    angle = quaternion_between(parent_bone, child_bone)
                    angle = quaternion_to_expmap(angle)
                    angles.append(angle)

                for child_idx in body_graph[joint_idx]:
                    angles = _get_angle_for_joint(child_idx, joint_idx, angles)

                return angles

            angles = _get_angle_for_joint(0, None, [])
            return K.stack(angles, axis=1)

        x = Lambda(_get_angles, name=scope+'angles')(x)

        def _get_angles_mask(coord_masks):
            base_shape = [int(dim) for dim in coord_masks.shape]
            base_shape.pop(1)
            base_shape[-1] = 1

            coord_masks_list = tf.unstack(coord_masks, axis=1)

            def _get_angle_mask_for_joint(joint_idx, angles_mask):
                for child_idx in body_graph[joint_idx]:
                    angles_mask.append(coord_masks_list[child_idx])  # * coord_masks_list[joint_idx]

                for child_idx in body_graph[joint_idx]:
                    angles_mask = _get_angle_mask_for_joint(child_idx, angles_mask)

                return angles_mask

            angles_mask = _get_angle_mask_for_joint(0, [])
            return K.stack(angles_mask, axis=1)

        x_mask = Lambda(_get_angles_mask, name=scope+'angles_mask')(x_mask)

        fixed_angles = len(body_graph[0])
        fixed_angles = Lambda(lambda args: args[:, :fixed_angles, ...],
                                                    name=scope+'fixed_angles')(x)
        x = Lambda(lambda args: args[:, fixed_angles:, ...], name=scope+'motion_angles')(x)
        x_mask = Lambda(lambda args: args[:, fixed_angles:, ...], name=scope+'motion_angles_mask')(x_mask)

    return x, x_mask, hip_coords, bone_len, fixed_angles


def seq_to_angles_out(x, body_members, hip_coords, bone_len, fixed_angles):
    scope = Scoping.get_global_scope()
    with scope.name_scope('seq_to_angles'):

        members_from, members_to, body_graph = get_body_graph(body_members)

        x = Lambda(lambda args: K.concatenate(args, axis=1), name=scope+'concat_angles')([fixed_angles, x])

        x = Lambda(lambda arg: expmap_to_rotmat(arg), name=scope+'rotmat')(x)
        # euler_out = Lambda(lambda arg: rotmat_to_euler(arg), name=scope+'euler')(x)

        def _get_coords(args):
            rotmat, bone_len = args
            rotmat_list = tf.unstack(rotmat, axis=1)
            bone_len_list = tf.unstack(bone_len, axis=1)

            base_shape = [int(d) for d in rotmat.shape]
            base_shape.pop(1)
            base_shape[-2] = 1
            base_shape[-1] = 1
            bone_idcs = {idx_tup: i for i, idx_tup in enumerate([idx_tup for idx_tup in zip(members_from, members_to)])}
            trans_dims = range(len(base_shape))
            trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]

            def _get_coords_for_joint(joint_idx, parent_idx, child_angle_idx, coords):
                if parent_idx is None:  # joint_idx should be 0
                    coords[joint_idx] = K.zeros(base_shape[:-2] + [3, 1])
                    parent_bone = K.constant(np.concatenate([np.ones(base_shape),
                                                             np.zeros(base_shape),
                                                             np.zeros(base_shape)], axis=-2))
                else:
                    parent_bone = coords[parent_idx] - coords[joint_idx]
                    parent_bone_norm = K.sqrt(K.sum(K.square(parent_bone), axis=-2, keepdims=True) + K.epsilon())
                    parent_bone = parent_bone / parent_bone_norm

                for child_idx in body_graph[joint_idx]:
                    child_bone = K.batch_dot(tf.transpose(rotmat_list[child_angle_idx], trans_dims), parent_bone,
                                             axes=[-1, -2])
                    child_bone_idx = bone_idcs[(joint_idx, child_idx)]
                    child_bone = child_bone * K.reshape(bone_len_list[child_bone_idx], (child_bone.shape[0], 1, 1, 1))
                    coords[child_idx] = child_bone + coords[joint_idx]
                    child_angle_idx += 1

                for child_idx in body_graph[joint_idx]:
                    child_angle_idx, coords = _get_coords_for_joint(child_idx, joint_idx, child_angle_idx, coords)

                return child_angle_idx, coords

            child_angle_idx, coords = _get_coords_for_joint(0, None, 0, {})
            coords = K.stack([t for i, t in sorted(coords.iteritems())], axis=1)
            coords = K.squeeze(coords, axis=-1)
            return coords

        x = Lambda(_get_coords, name=scope+'coords')([x, bone_len])
        x = Lambda(lambda args: args[0] + args[1], name=scope+'add_hip_coords')([x, hip_coords])
    return x

