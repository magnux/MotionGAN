from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np


def quaternion_between(u, v):
    """Finds the quaternion between two tensor of 3D vectors.

    Args:
        u: A `tf.Tensor` of rank R, the last dimension must be 3.
        v: A `tf.Tensor` of rank R, the last dimension must be 3.

    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.
        returns 1, 0, 0, 0 quaternion if either u or v is 0, 0, 0

    Raises:
        ValueError, if the last dimension of u and v is not 3.
    """
    u = tf.convert_to_tensor(u)
    v = tf.convert_to_tensor(v)
    if u.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError("The last dimension of u and v must be 3.")

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    base_shape = [int(d) for d in u.shape]
    base_shape[-1] = 1
    zero_dim = tf.zeros(base_shape)
    one_dim = tf.ones(base_shape)

    def _batch_dot(a, b):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1, keep_dims=True)

    def _length_2(a):
        return tf.reduce_sum(tf.square(a), axis=-1, keep_dims=True)

    def _normalize(a):
        return a / tf.sqrt(_length_2(a) + 1e-8)

    def _perpendicular_vector(a):
        """ Finds an arbitrary perpendicular vector to *a*.
            returns 0, 0, 0 for the all zeros singular case
        """

        return tf.where(
            tf.reduce_sum(a, axis=-1, keepdims=True) == 0.0, a,
            tf.where(
                tf.expand_dims(tf.where(a[..., 0]), axis=-1) == 0.0,
                tf.stack([one_dim, zero_dim, zero_dim]),
                tf.where(
                    tf.expand_dims(tf.where(a[..., 1]), axis=-1) == 0.0,
                    tf.stack([zero_dim, one_dim, zero_dim]),
                    tf.where(
                        tf.expand_dims(tf.where(a[..., 2]), axis=-1) == 0.0,
                        tf.stack([zero_dim, zero_dim, one_dim]),
                        tf.stack([one_dim, one_dim,
                                  -1.0 * tf.reduce_sum(u[..., :2], axis=-1, keepdims=True) / u[..., 2]], axis=-1)
                    )
                )
            )
        )

    # w = tf.dot(u, v) + sqrt(length_2(u) * length_2(v))
    # xyz = cross(u, v)

    k_cos_theta = _batch_dot(u, v)
    k = tf.sqrt(_length_2(u) * _length_2(v))

    return tf.where(
            tf.reduce_sum(u, axis=-1, keepdims=True) == 0.0, tf.stack([one_dim, u], axis=-1),
            tf.where(
                tf.reduce_sum(v, axis=-1, keepdims=True) == 0.0, tf.stack([one_dim, v], axis=-1),
                tf.where(
                    (k_cos_theta / k) == -1,
                    tf.stack([zero_dim, _normalize(_perpendicular_vector(u))], axis=-1),
                    _normalize(tf.stack([k_cos_theta + k, tf.cross(u, v)], axis=-1))
                )
            )

        )


def quat2expmap(q):
    """Converts a quaternion to an exponential map
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
        q: 1x4 quaternion
    Returns:
        r: 1x3 exponential map
    Raises:
        ValueError if the l2 norm of the quaternion is not close to 1
    """
    # if (np.abs(np.linalg.norm(q)-1)>1e-3):
    # raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = tf.sqrt(tf.reduce_sum(tf.square(q[..., 1:]), axis=-1, keep_dims=True) + 1e-8)
    coshalftheta = q[..., 0]

    r0 = q[..., 1:] / sinhalftheta
    theta = 2 * tf.atan2(sinhalftheta, coshalftheta)
    theta = tf.mod(theta + 2*np.pi, 2*np.pi)

    theta = tf.where(theta > np.pi, 2 * np.pi - theta, theta)
    r0 = tf.where(theta > np.pi, -r0, r0)
    r = r0 * theta

    return r


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: 3x3 rotation matrix
    Returns:
      q: 1x4 quaternion
    """
    trans_dims = range(len(R.shape))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]
    rotdiff = R - tf.transpose(R, trans_dims)

    r = tf.stack([-rotdiff[..., 1, 2], rotdiff[..., 0, 2], -rotdiff[..., 0, 1]], axis=-1)
    rnorm = tf.sqrt(tf.reduce_sum(tf.square(r), axis=-1, keep_dims=True) + 1e-8)
    sintheta = rnorm / 2.0
    r0 = r / rnorm

    costheta = (tf.trace(R) - 1.0) / 2.0

    theta = tf.atan2(sintheta, costheta)

    q = tf.stack([tf.cos(theta / 2),  r0 * tf.sin(theta / 2)], axis=-1)

    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      r: 1x3 exponential map
    Returns:
      R: 3x3 rotation matrix
    """
    base_shape = [int(d) for d in r.shape]
    base_shape[-1] = 1
    zero_dim = tf.zeros(base_shape)

    theta = tf.sqrt(tf.reduce_sum(tf.square(r), axis=-1, keep_dims=True) + 1e-8)
    r0 = r / theta

    r0x = tf.reshape(
        tf.stack([zero_dim, -1.0 * r0[..., 2], r0[..., 1],
                  zero_dim, zero_dim, -1.0 * r0[..., 0],
                  zero_dim, zero_dim, zero_dim], axis=-1),
        base_shape[:-2] + [3, 3]
    )

    trans_dims = range(len(r.shape))
    trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]

    r0x = r0x - tf.transpose(r0x, trans_dims)
    tile_eye = tf.constant(np.tile(np.eye(3), base_shape[:-2] + [1, 1]))

    R = tile_eye + tf.sin(theta) * r0x + (1.0 - tf.cos(theta)) * \
        tf.reduce_sum(tf.multiply(r0x, r0x), axis=-1, keep_dims=True)
    return R


def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Tensorflow port and tensorization of code in:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py
    Args:
      R: a 3x3 rotation matrix
    Returns:
      eul: a 3x1 Euler angle representation of R
    """
    
    base_shape = [int(d) for d in R.shape][:-1]
    base_shape[-1] = 1
    zero_dim = tf.zeros(base_shape)
    one_dim = tf.ones(base_shape)

    econd0 = R[..., 0, 2] == 1
    econd1 = R[..., 0, 2] == -1
    econd = tf.logical_or(econd0, econd1)

    E2 = tf.where(
        econd,
        tf.where(econd1, one_dim * np.pi / 2.0, one_dim * -np.pi / 2.0),
        -tf.asin(R[..., 0, 2])
    )
    E1 = tf.where(
        econd,
        tf.atan2(R[..., 1, 2], R[..., 0, 2]),
        tf.atan2(R[..., 1, 2] / tf.cos(E2), R[..., 2, 2] / tf.cos(E2))
    )
    E3 = tf.where(
        econd,
        zero_dim,
        tf.atan2(R[..., 0, 1] / tf.cos(E2), R[..., 0, 0] / tf.cos(E2))
    )

    eul = tf.expand_dims(tf.stack([E1, E2, E3], axis=-1), axis=-1)
    return eul
