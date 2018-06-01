from __future__ import absolute_import, division, print_function
import numpy as np


def vector3d_to_quaternion(x):
    """
    Convert a tensor of 3D vectors to a quaternion.
    Prepends a 0 to the last dimension, i.e. [[1,2,3]] -> [[0,1,2,3]].
    Args:
        x: A `np.array` of rank R, the last dimension must be 3.
    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
    Raises:
        ValueError, if the last dimension of x is not 3.
    """
    if x.shape[-1] != 3:
        raise ValueError("The last dimension of x must be 3.")
    return np.pad(x, (len(x.shape) - 1) * [[0, 0]] + [[1, 0]])


def quaternion_to_vector3d(q):
    """Remove the w component(s) of quaternion(s) q."""
    return q[..., 1:]


def rotate_vector_by_quaternion(q, v, q_ndims=None, v_ndims=None):
    """
    Rotate a vector (or tensor with last dimension of 3) by q.
    This function computes v' = q * v * conjugate(q) but faster.
    Fast version can be found here:
    https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    https://github.com/PhilJd/tf-quaternion/blob/master/tfquaternion/tfquaternion.py
    Args:
        q: A `np.array` with shape (..., 4)
        v: A `np.array` with shape (..., 3)
        q_ndims: The number of dimensions of q. Only necessary to specify if
            the shape of q is unknown.
        v_ndims: The number of dimensions of v. Only necessary to specify if
            the shape of v is unknown.
    Returns: A `np.array` with the broadcasted shape of v and q.
    """
    qnorm = np.sqrt(np.sum(np.square(q), axis=-1, keepdims=True) + 1e-8)
    q = q / qnorm
    w = q[..., 0]
    q_xyz = q[..., 1:]
    if q_xyz.shape.ndims is not None:
        q_ndims = q_xyz.shape.ndims
    if v.shape.ndims is not None:
        v_ndims = v.shape.ndims
    for _ in range(v_ndims - q_ndims):
        q_xyz = np.expand_dims(q_xyz, axis=0)
    for _ in range(q_ndims - v_ndims):
        v = np.expand_dims(v, axis=0) + np.zeros_like(q_xyz)
    q_xyz += np.zeros_like(v)
    v += np.zeros_like(q_xyz)
    t = 2 * np.cross(q_xyz, v)
    return v + np.expand_dims(w, axis=-1) * t + np.cross(q_xyz, t)


def quaternion_conjugate(q):
    """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    return np.multiply(q, [1.0, -1.0, -1.0, -1.0])


def quaternion_between(u, v):
    """
    Finds the quaternion between two tensor of 3D vectors.
    See:
    http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
    Args:
        u: A `np.array` of rank R, the last dimension must be 3.
        v: A `np.array` of rank R, the last dimension must be 3.

    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
        returns 1, 0, 0, 0 quaternion if either u or v is 0, 0, 0

    Raises:
        ValueError, if the last dimension of u and v is not 3.
    """
    if u.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError("The last dimension of u and v must be 3.")

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    def _vector_batch_dot(a, b):
        return np.sum(np.multiply(a, b), axis=-1, keepdims=True)

    def _length_2(a):
        return np.sum(np.square(a), axis=-1, keepdims=True)

    def _normalize(a):
        return a / np.sqrt(_length_2(a) + 1e-8)

    base_shape = [int(d) for d in u.shape]
    base_shape[-1] = 1
    zero_dim = np.zeros(base_shape)
    one_dim = np.ones(base_shape)
    w = np.sqrt(_length_2(u) * _length_2(v)) + _vector_batch_dot(u, v)

    q = np.where(
            np.tile(np.equal(np.sum(u, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
            np.concatenate([one_dim, u], axis=-1),
            np.where(
                np.tile(np.equal(np.sum(v, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
                np.concatenate([one_dim, v], axis=-1),
                np.where(
                    np.tile(np.less(w, 1e-4), [1 for _ in u.shape[:-1]] + [4]),
                    np.concatenate([zero_dim, np.stack([-u[..., 2], u[..., 1], u[..., 0]], axis=-1)], axis=-1),
                    np.concatenate([w, np.cross(u, v)], axis=-1)
                )
            )
        )

    return _normalize(q)


def quaternion_to_expmap(q):
    """
    Converts a quaternion to an exponential map
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
        q: (..., 4) quaternion Tensor
    Returns:
        r: (..., 3) exponential map Tensor
    Raises:
        ValueError if the l2 norm of the quaternion is not close to 1
    """
    # if (np.abs(np.linalg.norm(q)-1)>1e-3):
    # raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.sqrt(np.sum(np.square(q[..., 1:]), axis=-1, keepdims=True) + 1e-8)
    coshalftheta = np.expand_dims(q[..., 0], axis=-1)

    r0 = q[..., 1:] / sinhalftheta
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2*np.pi, 2*np.pi)

    condition = np.greater(theta, np.pi)
    theta = np.where(condition, 2 * np.pi - theta, theta)
    r0 = np.where(np.tile(condition, [1 for _ in condition.shape[:-1]] + [3]), -r0, r0)
    r = r0 * theta

    return r


def rotmat_to_quaternion(R):
    """
    Converts a rotation matrix to a quaternion
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: (..., 3, 3) rotation matrix Tensor
    Returns:
      q: (..., 4) quaternion Tensor
    """
    trans_dims = range(len(R.shape))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    rotdiff = R - np.transpose(R, trans_dims)

    r = np.stack([-rotdiff[..., 1, 2], rotdiff[..., 0, 2], -rotdiff[..., 0, 1]], axis=-1)
    rnorm = np.sqrt(np.sum(np.square(r), axis=-1, keepdims=True) + 1e-8)
    sintheta = rnorm / 2.0
    r0 = r / rnorm

    costheta = np.expand_dims((np.trace(R) - 1.0) / 2.0, axis=-1)

    theta = np.arctan2(sintheta, costheta)

    q = np.concatenate([np.cos(theta / 2),  r0 * np.sin(theta / 2)], axis=-1)

    return q


def expmap_to_rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      r: (..., 3) exponential map Tensor
    Returns:
      R: (..., 3, 3) rotation matrix Tensor
    """
    base_shape = [int(d) for d in r.shape][:-1]
    zero_dim = np.zeros(base_shape)

    theta = np.sqrt(np.sum(np.square(r), axis=-1, keepdims=True) + 1e-8)
    r0 = r / theta

    r0x = np.reshape(
        np.stack([zero_dim, -1.0 * r0[..., 2], r0[..., 1],
                  zero_dim, zero_dim, -1.0 * r0[..., 0],
                  zero_dim, zero_dim, zero_dim], axis=-1),
        base_shape + [3, 3]
    )
    trans_dims = range(len(r0x.shape))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    r0x = r0x - np.transpose(r0x, trans_dims)

    tile_eye = np.tile(np.reshape(np.eye(3), [1 for _ in base_shape] + [3, 3]), base_shape + [1, 1])
    theta = np.expand_dims(theta, axis=-1)

    R = tile_eye + np.sin(theta) * r0x + (1.0 - np.cos(theta)) * np.matmul(r0x, r0x)
    return R


def expmap_to_quaternion(r):
    """
    Converts an exponential map angle to a quaternion
    See:
    http://www.cs.cmu.edu/~spiff/exp-map/
    Args:
        r: a (..., 3) exponential map tensor
    Returns:
        q: A `np.array` with R+1 dimensions and
        shape [d_1, ..., d_(R-1), 4], the rotation matrix
    """

    theta = np.sqrt(np.sum(np.square(r), axis=-1, keepdims=True) + 1e-8)

    scl = theta
    condition = np.greater(theta, 2 * np.pi)
    theta = np.where(condition, np.mod(theta, 2 * np.pi), theta)
    scl = theta / scl
    r = np.where(np.tile(condition, [1 for _ in condition.shape[:-1]] + [3]), r * scl, r)

    scl = theta
    condition = np.greater(theta, np.pi)
    theta = np.where(condition, 2 * np.pi - theta, theta)
    scl = 1.0 - 2 * np.pi / scl
    r = np.where(np.tile(condition, [1 for _ in condition.shape[:-1]] + [3]), r * scl, r)

    cosp = np.cos(.5 * theta)
    sinp = np.sin(.5 * theta)

    q = np.concatenate([cosp, np.where(np.tile(np.less(theta, 1e-7), [1 for _ in r.shape[:-1]] + [3]),
                                  r * (.5 - theta * theta / 48.0),
                                  r * (sinp / theta))], axis=-1)
    return q


def quaternion_to_rotmat(q):
    """
    Calculate the corresponding rotation matrix.
    See:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    https://github.com/PhilJd/tf-quaternion/blob/master/tfquaternion/tfquaternion.py
    Args:
        q: a (..., 4) quaternion tensor
    Returns:
        A `np.array` with R+1 dimensions and
        shape [d_1, ..., d_(R-1), 3, 3], the rotation matrix
    """

    # helper functions
    def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
        return 1 - 2 * np.power(a, 2) - 2 * np.power(b, 2)

    def tr_add(a, b, c, d):  # computes triangle entries with addition
        return 2 * a * b + 2 * c * d

    def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
        return 2 * a * b - 2 * c * d

    qnorm = np.sqrt(np.sum(np.square(q), axis=-1, keepdims=True) + 1e-8)
    normed_q = np.split(q / qnorm, int(q.shape[-1]), axis=-1)
    w, x, y, z = [np.squeeze(comp, axis=1) for comp in normed_q]
    m = [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
         [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
         [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]
    return np.stack([np.stack(m[i], axis=-1) for i in range(3)], axis=-2)


def rotmat_to_euler(R):
    """
    Converts a rotation matrix to Euler angles
    Tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: a (..., 3, 3) rotation matrix Tensor
    Returns:
      eul: a (..., 3) Euler angle representation of R
    """
    base_shape = [int(d) for d in R.shape][:-2]
    zero_dim = np.zeros(base_shape)
    one_dim = np.ones(base_shape)

    econd0 = np.equal(R[..., 0, 2], one_dim)
    econd1 = np.equal(R[..., 0, 2], -1.0 * one_dim)
    econd = np.logical_or(econd0, econd1)

    e2 = np.where(
        econd,
        np.where(econd1, one_dim * np.pi / 2.0, one_dim * -np.pi / 2.0),
        -np.arcsin(R[..., 0, 2])
    )
    e1 = np.where(
        econd,
        np.arctan2(R[..., 1, 2], R[..., 0, 2]),
        np.arctan2(R[..., 1, 2] / np.cos(e2), R[..., 2, 2] / np.cos(e2))
    )
    e3 = np.where(
        econd,
        zero_dim,
        np.arctan2(R[..., 0, 1] / np.cos(e2), R[..., 0, 0] / np.cos(e2))
    )

    eul = np.stack([e1, e2, e3], axis=-1)
    return eul
