from __future__ import absolute_import, division, print_function
import numpy as np
from utils.npangles import quaternion_between, quaternion_to_expmap, expmap_to_rotmat, rotmat_to_euler

MASK_MODES = ('No mask', 'Future Prediction', 'Missing Frames', 'Occlusion Simulation', 'Structured Occlusion', 'Noisy Transmission')


def gen_mask(mask_type, keep_prob, batch_size, njoints, seq_len, body_members, test_mode=False):
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

    if test_mode:
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
    return np.stack(angles_mask, axis=1)