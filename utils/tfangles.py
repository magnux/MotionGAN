from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.keras.api.keras.backend as K
import numpy as np


def vector3d_to_quaternion(x):
    """
    Convert a tensor of 3D vectors to a quaternion.
    Prepends a 0 to the last dimension, i.e. [[1,2,3]] -> [[0,1,2,3]].
    Args:
        x: A `tf.Tensor` of rank R, the last dimension must be 3.
    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
    Raises:
        ValueError, if the last dimension of x is not 3.
    """
    x = tf.convert_to_tensor(x)
    if x.shape[-1] != 3:
        raise ValueError("The last dimension of x must be 3.")
    return tf.pad(x, (len(x.shape) - 1) * [[0, 0]] + [[1, 0]])


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
        q: A `tf.Tensor` with shape (..., 4)
        v: A `tf.Tensor` with shape (..., 3)
        q_ndims: The number of dimensions of q. Only necessary to specify if
            the shape of q is unknown.
        v_ndims: The number of dimensions of v. Only necessary to specify if
            the shape of v is unknown.
    Returns: A `tf.Tensor` with the broadcasted shape of v and q.
    """
    qnorm = tf.sqrt(tf.reduce_sum(tf.square(q), axis=-1, keep_dims=True) + 1e-8)
    q = q / qnorm
    w = q[..., 0]
    q_xyz = q[..., 1:]
    if q_xyz.shape.ndims is not None:
        q_ndims = q_xyz.shape.ndims
    if v.shape.ndims is not None:
        v_ndims = v.shape.ndims
    for _ in range(v_ndims - q_ndims):
        q_xyz = tf.expand_dims(q_xyz, axis=0)
    for _ in range(q_ndims - v_ndims):
        v = tf.expand_dims(v, axis=0) + tf.zeros_like(q_xyz)
    q_xyz += tf.zeros_like(v)
    v += tf.zeros_like(q_xyz)
    t = 2 * tf.cross(q_xyz, v)
    return v + tf.expand_dims(w, axis=-1) * t + tf.cross(q_xyz, t)


def quaternion_conjugate(q):
    """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    return tf.multiply(q, [1.0, -1.0, -1.0, -1.0])


def quaternion_between(u, v):
    """
    Finds the quaternion between two tensor of 3D vectors.
    See:
    http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
    Args:
        u: A `tf.Tensor` of rank R, the last dimension must be 3.
        v: A `tf.Tensor` of rank R, the last dimension must be 3.

    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
        returns 1, 0, 0, 0 quaternion if either u or v is 0, 0, 0

    Raises:
        ValueError, if the last dimension of u and v is not 3.
    """
    u = tf.convert_to_tensor(u, dtype=tf.float32)
    v = tf.convert_to_tensor(v, dtype=tf.float32)
    if u.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError("The last dimension of u and v must be 3.")

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    def _vector_batch_dot(a, b):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1, keep_dims=True)

    def _length_2(a):
        return tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True)

    def _normalize(a):
        return a / tf.sqrt(_length_2(a) + 1e-8)

    base_shape = [int(d) for d in u.shape]
    base_shape[-1] = 1
    zero_dim = tf.zeros(base_shape)
    one_dim = tf.ones(base_shape)
    w = tf.sqrt(_length_2(u) * _length_2(v)) + _vector_batch_dot(u, v)

    return tf.where(
            tf.tile(tf.equal(tf.reduce_sum(u, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
            tf.concat([one_dim, u], axis=-1),
            tf.where(
                tf.tile(tf.equal(tf.reduce_sum(v, axis=-1, keepdims=True), zero_dim), [1 for _ in u.shape[:-1]] + [4]),
                tf.concat([one_dim, v], axis=-1),
                tf.where(
                    tf.tile(tf.less(w, 1e-4), [1 for _ in u.shape[:-1]] + [4]),
                    tf.concat([zero_dim, _normalize(tf.stack([-u[..., 2], u[..., 1], u[..., 0]], axis=-1))], axis=-1),
                    _normalize(tf.concat([w, tf.cross(u, v)], axis=-1))
                )
            )
        )


def quaternion_to_expmap(q):
    """
    Converts a quaternion to an exponential map
    Tensorflow port and tensorization of code in:
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

    sinhalftheta = tf.sqrt(tf.reduce_sum(tf.square(q[..., 1:]), axis=-1, keep_dims=True) + 1e-8)
    coshalftheta = tf.expand_dims(q[..., 0], axis=-1)

    r0 = q[..., 1:] / sinhalftheta
    theta = 2 * tf.atan2(sinhalftheta, coshalftheta)
    theta = tf.mod(theta + 2*np.pi, 2*np.pi)

    condition = tf.greater(theta, np.pi)
    theta = tf.where(condition, 2 * np.pi - theta, theta)
    r0 = tf.where(tf.tile(condition, [1 for _ in condition.shape[:-1]] + [3]), -r0, r0)
    r = r0 * theta

    return r


def rotmat_to_quaternion(R):
    """
    Converts a rotation matrix to a quaternion
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: (..., 3, 3) rotation matrix Tensor
    Returns:
      q: (..., 4) quaternion Tensor
    """
    trans_dims = range(len(R.shape))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    rotdiff = R - tf.transpose(R, trans_dims)

    r = tf.stack([-rotdiff[..., 1, 2], rotdiff[..., 0, 2], -rotdiff[..., 0, 1]], axis=-1)
    rnorm = tf.sqrt(tf.reduce_sum(tf.square(r), axis=-1, keep_dims=True) + 1e-8)
    sintheta = rnorm / 2.0
    r0 = r / rnorm

    costheta = tf.expand_dims((tf.trace(R) - 1.0) / 2.0, axis=-1)

    theta = tf.atan2(sintheta, costheta)

    q = tf.concat([tf.cos(theta / 2),  r0 * tf.sin(theta / 2)], axis=-1)

    return q


def expmap_to_rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      r: (..., 3) exponential map Tensor
    Returns:
      R: (..., 3, 3) rotation matrix Tensor
    """
    base_shape = [int(d) for d in r.shape][:-1]
    zero_dim = tf.zeros(base_shape)

    theta = tf.sqrt(tf.reduce_sum(tf.square(r), axis=-1, keep_dims=True) + 1e-8)
    r0 = r / theta

    r0x = tf.reshape(
        tf.stack([zero_dim, -1.0 * r0[..., 2], r0[..., 1],
                  zero_dim, zero_dim, -1.0 * r0[..., 0],
                  zero_dim, zero_dim, zero_dim], axis=-1),
        base_shape + [3, 3]
    )

    trans_dims = range(len(r0x.shape))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    r0x = r0x - tf.transpose(r0x, trans_dims)

    tile_eye = tf.constant(np.tile(np.reshape(np.eye(3), [1 for _ in base_shape] + [3, 3]), base_shape + [1, 1]), dtype=tf.float32)
    theta = tf.expand_dims(theta, axis=-1)

    R = tile_eye + tf.sin(theta) * r0x + (1.0 - tf.cos(theta)) * K.batch_dot(r0x, r0x, axes=[-1, -2])
    return R


def expmap_to_quaternion(r):
    """
    Converts an exponential map angle to a quaternion
    See:
    http://www.cs.cmu.edu/~spiff/exp-map/
    Args:
        r: a (..., 3) exponential map tensor
    Returns:
        q: A `tf.Tensor` with R+1 dimensions and
        shape [d_1, ..., d_(R-1), 4], the rotation matrix
    """

    theta = tf.sqrt(tf.reduce_sum(tf.square(r), axis=-1, keep_dims=True) + 1e-8)

    scl = theta
    condition = tf.greater(theta, 2 * np.pi)
    theta = tf.where(condition, tf.mod(theta, 2 * np.pi), theta)
    scl = theta / scl
    r = tf.where(condition, r * scl, r)

    scl = theta
    condition = tf.greater(theta, np.pi)
    theta = tf.where(condition, 2 * np.pi - theta, theta)
    scl = 1.0 - 2 * np.pi / scl
    r = tf.where(condition, r * scl, r)

    cosp = tf.cos(.5 * theta)
    sinp = tf.sin(.5 * theta)

    q = tf.concat([cosp, tf.where(tf.tile(tf.less(theta, 1e-7), [1 for _ in r.shape[:-1]] + [3]),
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
        A `tf.Tensor` with R+1 dimensions and
        shape [d_1, ..., d_(R-1), 3, 3], the rotation matrix
    """

    # helper functions
    def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
        return 1 - 2 * tf.pow(a, 2) - 2 * tf.pow(b, 2)

    def tr_add(a, b, c, d):  # computes triangle entries with addition
        return 2 * a * b + 2 * c * d

    def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
        return 2 * a * b - 2 * c * d

    qnorm = tf.sqrt(tf.reduce_sum(tf.square(q), axis=-1, keep_dims=True) + 1e-8)
    w, x, y, z = tf.unstack(q / qnorm, axis=-1)
    m = [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
         [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
         [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]
    return tf.stack([tf.stack(m[i], axis=-1) for i in range(3)], axis=-2)


def rotmat_to_euler(R):
    """
    Converts a rotation matrix to Euler angles
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: a (..., 3, 3) rotation matrix Tensor
    Returns:
      eul: a (..., 3) Euler angle representation of R
    """
    base_shape = [int(d) for d in R.shape][:-2]
    zero_dim = tf.zeros(base_shape)
    one_dim = tf.ones(base_shape)

    econd0 = tf.equal(R[..., 0, 2], one_dim)
    econd1 = tf.equal(R[..., 0, 2], -1.0 * one_dim)
    econd = tf.logical_or(econd0, econd1)

    e2 = tf.where(
        econd,
        tf.where(econd1, one_dim * np.pi / 2.0, one_dim * -np.pi / 2.0),
        -tf.asin(R[..., 0, 2])
    )
    e1 = tf.where(
        econd,
        tf.atan2(R[..., 1, 2], R[..., 0, 2]),
        tf.atan2(R[..., 1, 2] / tf.cos(e2), R[..., 2, 2] / tf.cos(e2))
    )
    e3 = tf.where(
        econd,
        zero_dim,
        tf.atan2(R[..., 0, 1] / tf.cos(e2), R[..., 0, 0] / tf.cos(e2))
    )

    eul = tf.stack([e1, e2, e3], axis=-1)
    return eul
