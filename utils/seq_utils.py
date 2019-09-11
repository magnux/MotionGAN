from __future__ import absolute_import, division, print_function
import numpy as np
import time
import copy
from utils.npangles import quaternion_between, quaternion_to_expmap, expmap_to_rotmat, rotmat_to_euler, rotmat_to_quaternion, rotate_vector_by_quaternion

MASK_MODES = ('No mask', 'Future Prediction', 'Missing Frames', 'Occlusion Simulation', 'Structured Occlusion', 'Noisy Transmission')


def gen_mask(mask_type, keep_prob, batch_size, njoints, seq_len, body_members, baseline_mode=False):
    # Default mask, no mask
    mask = np.ones(shape=(batch_size, njoints, seq_len, 1))
    if mask_type == 1:  # Future Prediction
        mask[:, :, np.int(seq_len * keep_prob):, :] = 0.0
    elif mask_type == 2:  # Missing Frames
        occ_frames = np.random.randint(seq_len - 1, size=np.int(seq_len * (1.0 - keep_prob)))
        mask[:, :, occ_frames, :] = 0.0
    elif mask_type == 3:  # Occlusion Simulation
        rand_joints = np.random.randint(njoints, size=np.int(njoints * (1.0 - keep_prob)))
        mask[:, rand_joints, :, :] = 0.0
    elif mask_type == 4:  # Structured Occlusion Simulation
        rand_joints = set()
        while ((njoints - len(rand_joints)) >
               (njoints * keep_prob)):
            joints_to_add = (body_members.values()[np.random.randint(len(body_members))])['joints']
            for joint in joints_to_add:
                rand_joints.add(joint)
        mask[:, list(rand_joints), :, :] = 0.0
    elif mask_type == 5:  # Noisy transmission
        mask = np.random.binomial(1, keep_prob, size=mask.shape)

    if baseline_mode:
        # This unmasks first and last frame for all sequences (required for baselines)
        mask[:, :, [0, -1], :] = 1.0
    return mask


def gen_latent_noise(batch_size, latent_cond_dim):
    return np.random.uniform(size=(batch_size, latent_cond_dim))


def linear_baseline(real_seq, mask):
    linear_seq = real_seq * mask
    for j in range(real_seq.shape[0]):
        for f in range(1, real_seq.shape[1] - 1):
            if mask[j, f, 0] == 0:
                prev_f = f - 1
                for g in range(f - 1, -1, -1):
                    if mask[j, g, 0] == 1:
                        prev_f = g
                        break
                next_f = f + 1
                for g in range(f + 1, real_seq.shape[1]):
                    if mask[j, g, 0] == 1:
                        next_f = g
                        break
                blend_factor = (f - prev_f) / (next_f - prev_f)
                linear_seq[j, f, :] = ((linear_seq[j, prev_f, :] * (1 - blend_factor)) +
                                       (linear_seq[j, next_f, :] * blend_factor))
    return linear_seq


def burke_baseline(rawdata, mask, tol=0.0025, sigR=1e-3, keepOriginal=True):
    """Low-Rank smoothed Kalman filter, based in Burke et. al"""
    rawdata = np.transpose(rawdata.copy(), (1, 0, 2))
    raw_shape = [int(dim) for dim in rawdata.shape]
    rawdata = np.reshape(rawdata, (raw_shape[0], raw_shape[1] * raw_shape[2]))

    mask = np.tile(mask.copy(), (1, 1, raw_shape[2]))
    mask = np.transpose(mask, (1, 0, 2))
    mask = np.reshape(mask, (raw_shape[0], raw_shape[1] * raw_shape[2]))

    rawdata[mask == 0] = np.nan

    X = rawdata[~np.isnan(rawdata).any(axis=1)]
    if X.size == 0 or np.product(X.shape[-2:]) == 0:
        return np.zeros((raw_shape[1], raw_shape[0], raw_shape[2]))

    m = np.mean(X, axis=0)

    U, S, V = np.linalg.svd(X - m)

    d = np.nonzero(np.cumsum(S) / np.sum(S) > (1 - tol))
    if len(d[0]) == 0:
        return np.zeros((raw_shape[1], raw_shape[0], raw_shape[2]))
    d = d[0][0]

    Q = np.dot(np.dot(V[0:d, :], np.diag(np.std(np.diff(X, axis=0), axis=0))), V[0:d, :].T)

    state = []
    state_pred = []
    cov_pred = []
    cov = []
    cov.insert(0, 1e12 * np.eye(d))
    state.insert(0, np.random.normal(0.0, 1.0, d))
    cov_pred.insert(0, 1e12 * np.eye(d))
    state_pred.insert(0, np.random.normal(0.0, 1.0, d))
    for i in range(1, rawdata.shape[0] + 1):
        z = rawdata[i - 1, ~np.isnan(rawdata[i - 1, :])]
        H = np.diag(~np.isnan(rawdata[i - 1, :]))
        H = H[~np.all(H == 0, axis=1)]
        Ht = np.dot(H, V[0:d, :].T)

        R = sigR * np.eye(H.shape[0])

        state_pred.insert(i, state[i - 1])
        cov_pred.insert(i, cov[i - 1] + Q)

        K = np.dot(np.dot(cov_pred[i], Ht.T), np.linalg.inv(np.dot(np.dot(Ht, cov_pred[i]), Ht.T) + R))

        state.insert(i, state_pred[i] + np.dot(K, (z - (np.dot(Ht, state_pred[i]) + np.dot(H, m)))))
        cov.insert(i, np.dot(np.eye(d) - np.dot(K, Ht), cov_pred[i]))

    y = np.zeros(rawdata.shape)
    y[-1, :] = np.dot(V[0:d, :].T, state[-1]) + m
    for i in range(len(state) - 2, 0, -1):
        state[i] = state[i] + np.dot(np.dot(cov[i], np.linalg.inv(cov_pred[i])),
                                     (state[i + 1] - state_pred[i + 1]))
        cov[i] = cov[i] + np.dot(np.dot(np.dot(cov[i], np.linalg.inv(cov_pred[i])),
                                        (cov[i + 1] - cov_pred[i + 1])), cov[i])

        y[i - 1, :] = np.dot(V[0:d, :].T, state[i]) + m

    if (keepOriginal):
        y[~np.isnan(rawdata)] = rawdata[~np.isnan(rawdata)]

    y = np.reshape(y, (raw_shape[0], raw_shape[1], raw_shape[2]))
    y = np.transpose(y, (1, 0, 2))

    return y


def get_body_graph(body_members):
    members_from = []
    members_to = []
    for member in body_members.values():
        for j in range(len(member['joints']) - 1):
            members_from.append(member['joints'][j])
            members_to.append(member['joints'][j + 1])

    members_lst = zip(members_from, members_to)

    graph = {name: set() for tup in members_lst for name in tup}
    has_parent = {name: False for tup in members_lst for name in tup}
    for parent, child in members_lst:
        graph[parent].add(child)
        has_parent[child] = True

    roots = [name for name, parents in has_parent.items() if not parents]  # assuming 0 (hip)

    def traverse(hierarchy, graph, names):
        for name in names:
            hierarchy[name] = traverse({}, graph, graph[name])
        return hierarchy
    # print(traverse({}, graph, roots))

    for key, value in graph.items():
        graph[key] = sorted(list(graph[key]))

    return members_from, members_to, graph


def get_swap_list(body_members):
    swap_list = []
    for member in [member for member in body_members.keys() if 'left' in member]:
        left_joints = body_members[member]['joints']
        right_joints = body_members['right' + member[4:]]['joints']
        swap_list.append((left_joints, right_joints))

    return swap_list


def post_process(real_seq, gen_seq, mask, body_members):
    _, _, graph = get_body_graph(body_members)

    blend_seq = real_seq * mask

    def _post_process_joint(frame, joint_idx, parent_idx):
        if mask[joint_idx, frame, 0] == 0:
            # Blend in time
            blend_seq[joint_idx, frame, :] = (blend_seq[joint_idx, frame - 1, :]
                                              + gen_seq[joint_idx, frame, :] - gen_seq[joint_idx, frame - 1, :])
            # Blend in space
            if parent_idx is not None:
                space_blend = (gen_seq[joint_idx, frame, :] - gen_seq[parent_idx, frame, :]
                               + blend_seq[parent_idx, frame, :])
                blend_seq[joint_idx, frame, :] = (blend_seq[joint_idx, frame, :] + space_blend) / 2

        for child_idx in graph[joint_idx]:
            _post_process_joint(frame, child_idx, joint_idx)

    for f in range(1, real_seq.shape[1] - 1):
        _post_process_joint(f, 0, None)

    return blend_seq


def seq_to_angles_transformer(body_members):
    _, _, body_graph = get_body_graph(body_members)

    def _get_angles(coords):
        base_shape = [int(dim) for dim in coords.shape]
        base_shape.pop(1)
        base_shape[-1] = 1

        coords_list = np.split(coords, int(coords.shape[1]), axis=1)
        coords_list = [np.squeeze(elem, axis=1) for elem in coords_list]

        def _get_angle_for_joint(joint_idx, parent_idx, angles):
            if parent_idx is None:  # joint_idx should be 0
                parent_bone = np.concatenate([np.ones(base_shape),
                                              np.zeros(base_shape),
                                              np.zeros(base_shape)], axis=-1)
            else:
                parent_bone = coords_list[parent_idx] - coords_list[joint_idx]

            for child_idx in body_graph[joint_idx]:
                child_bone = coords_list[child_idx] - coords_list[joint_idx]
                angle = quaternion_between(parent_bone, child_bone)
                angle = quaternion_to_expmap(angle)
                angle = expmap_to_rotmat(angle)
                angle = rotmat_to_euler(angle)
                angles.append(angle)

            for child_idx in body_graph[joint_idx]:
                angles = _get_angle_for_joint(child_idx, joint_idx, angles)

            return angles

        angles = _get_angle_for_joint(0, None, [])
        fixed_angles = len(body_graph[0])
        angles = angles[fixed_angles:]
        return np.stack(angles, axis=1)

    return _get_angles


def get_angles_mask(coord_masks, body_members):

    _, _, body_graph = get_body_graph(body_members)

    base_shape = [int(dim) for dim in coord_masks.shape]
    base_shape.pop(1)
    base_shape[-1] = 1

    coord_masks_list = np.split(coord_masks, int(coord_masks.shape[1]), axis=1)
    coord_masks_list = [np.squeeze(elem, axis=1) for elem in coord_masks_list]

    def _get_angle_mask_for_joint(joint_idx, angles_mask):
        for child_idx in body_graph[joint_idx]:
            angles_mask.append(coord_masks_list[child_idx])

        for child_idx in body_graph[joint_idx]:
            angles_mask = _get_angle_mask_for_joint(child_idx, angles_mask)

        return angles_mask

    angles_mask = _get_angle_mask_for_joint(0, [])
    fixed_angles = len(body_graph[0])
    angles_mask = angles_mask[fixed_angles:]
    return np.stack(angles_mask, axis=1)


def fkl(angles, parent, offset, rotInd, expmapInd):
    """
    Convert joint angles and bone lenghts into the 3d points of a person.
    Based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

    Args
      angles: 99-long vector with 3d position and 3d joint angles in expmap format
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    Returns
      xyz: 32x3 3d points that represent a person in 3d space
    """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        if not rotInd[i]:  # If the list is empty
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[rotInd[i][0] - 1]
            yangle = angles[rotInd[i][1] - 1]
            zangle = angles[rotInd[i][2] - 1]

        r = angles[expmapInd[i]]

        thisRotation = expmap_to_rotmat(r)
        thisPosition = np.array([xangle, yangle, zangle])

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :],
                                             (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i, :] + thisPosition).dot(
                xyzStruct[parent[i]]['rotation']) + xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(
                xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    xyz = xyz[:, [0, 2, 1]]
    # xyz = xyz[:,[2,0,1]]
    # xyz = np.reshape(xyz, [-1])

    return xyz


def revert_coordinate_space(channels, R0, T0):
    """
    Bring a series of poses to a canonical form so they are facing the camera when they start.
    Adapted from
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

    Args
      channels: n-by-99 matrix of poses
      R0: 3x3 rotation for the first frame
      T0: 1x3 position for the first frame
    Returns
      channels_rec: The passed poses, but the first has T0 and R0, and the
                    rest of the sequence is modified accordingly.
    """
    n, d = channels.shape

    channels_rec = copy.copy(channels)
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for ii in range(n):
        R_diff = expmap_to_rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[ii, rootRotInd] = quaternion_to_expmap(rotmat_to_quaternion(R))
        T = T_prev + (
        (R_prev.T).dot(np.reshape(channels[ii, :3], [3, 1]))).reshape(-1)
        channels_rec[ii, :3] = T
        T_prev = T
        R_prev = R

    return channels_rec


def _some_variables():
    """
    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28,
                       31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000,
         0.000000, -442.894612, 0.000000, 0.000000, -454.206447, 0.000000,
         0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437,
         132.948826, 0.000000, 0.000000, 0.000000, -442.894413, 0.000000,
         0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000,
         233.383263, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000,
         121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 151.034226, 0.000000, 0.000000,
         278.882773, 0.000000, 0.000000, 251.733451, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000,
         100.000188, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000,
         278.892924, 0.000000, 0.000000, 251.728680, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 99.999888, 0.000000,
         137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd


def rotate_start(x, body_members):
    left_shoulder = body_members['left_arm']['joints'][1]
    right_shoulder = body_members['right_arm']['joints'][1]
    hip = body_members['torso']['joints'][0]
    head_top = body_members['head']['joints'][-1]

    base_shape = [int(d) for d in x.shape]
    base_shape[1] = 1
    base_shape[2] = 1

    coords_list = np.split(x[:, :, 0, :], x.shape[1], axis=1)
    torso_rot = np.cross(coords_list[left_shoulder] - coords_list[hip],
                         coords_list[right_shoulder] - coords_list[hip])
    side_rot = np.reshape(np.cross(coords_list[head_top] - coords_list[hip], torso_rot), base_shape)
    # theta_diff = ((np.pi / 2) - np.arctan2(side_rot[..., 1], side_rot[..., 0])) / 2
    theta_diff = -np.arctan2(side_rot[..., 1], side_rot[..., 0]) / 2
    cos_theta_diff = np.cos(theta_diff)
    sin_theta_diff = np.sin(theta_diff)
    zeros_theta = np.zeros_like(sin_theta_diff)
    start_rotation = np.stack([cos_theta_diff, zeros_theta, zeros_theta, sin_theta_diff], axis=-1)

    x = rotate_vector_by_quaternion(start_rotation, x)
    
    return x, start_rotation