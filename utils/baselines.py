from __future__ import absolute_import, division, print_function
import numpy as np


def constant_baseline(X, mask):
    new_X = X * mask
    for j in range(X.shape[0]):
        for f in range(1, X.shape[1]):
            if mask[j, f, 0] == 0:
                new_X[j, f, :] = new_X[j, f - 1, :]
    return new_X


def burke_baseline(X, mask, gamma=0.99):
    """Low-Rank smoothed Kalman filter, based in Burke et. al"""
    X = np.transpose(X, (1, 0, 2))
    mask = np.transpose(mask, (1, 0, 2))
    m0 = np.mean(X[0, ...], -1, keepdims=True)
    U, s, V = np.linalg.svd(X[0, ...] - m0, full_matrices=True)
    d = np.searchsorted(np.cumsum(np.diag(s)), gamma) + 1
    print(U.shape, s.shape, V.shape, d)
    V = V[:, d]

    n_joints = int(X.shape[1])
    n_frames = int(X.shape[0])
    Z = X * mask
    masked = np.sum(mask, -1) == 0
    H = np.zeros((n_frames, n_joints, n_joints))
    for t in range(1, n_frames):
        H[t, ...] = np.eye(n_joints)
        for j in range(n_joints):
            if masked[t, j]:
                H[t, j, j] = 0
    H_hat = np.zeros((n_frames, n_joints, d))
    # Note this operation could be sped up if needed by removing the for loop
    for t in range(1, n_frames):
        H_hat[t, ...] = np.dot(H[t, ...], V)

    m_hat = np.random.uniform(size=(n_frames, d, 1))
    m = np.zeros_like(m_hat)
    P_hat = np.random.uniform(size=(n_frames, d, d))
    P = np.zeros_like(P_hat)
    K = np.zeros((n_frames, d, n_joints))
    # Q is obtained by determining the standard
    # deviation of the rate of change of marker positions
    # and projecting this into the low rank space.
    Q = 0
    # Noise covariance matrix R is diagonal,
    # with elements selected empirically
    R = np.zeros((n_joints, n_joints))
    I = np.eye(d)
    for t in range(1, n_frames):
        m_hat[t, ...] = m_hat[t - 1, ...]
        P_hat[t, ...] = P_hat[t - 1, ...] + Q
        K[t, ...] = np.dot(P_hat[t, ...],
                           np.dot(H_hat[t, ...].transpose(),
                                  np.linalg.pinv(
                                      np.dot(H_hat[t, ...],
                                             np.dot(P_hat[t, ...],
                                                    H_hat[t, ...].transpose())) + R)))
        m[t, ...] = m_hat[t, ...] + np.dot(K[t, ...], Z[t, ...] -
                                           np.dot(H_hat[t, ...], m_hat[t, ...]) +
                                           np.dot(H[t, ...], m0))
        P[t, ...] = np.dot((I - np.dot(K[t, ...], H_hat[t, ...])), P_hat[t, ...])

    m_tilde = np.zeros_like(m)
    m_tilde[-1, ...] = m[-1, ...]
    P_tilde = np.zeros_like(P)
    P_tilde[-1, ...] = P[-1, ...]
    y = np.zeros_like(Z)
    for t in range(n_frames -1, 1):
        m_tilde[t, ...] = m[t, ...] + np.dot(P[t, ...],
                                             np.dot(np.linalg.pinv(P_tilde[t, ...]),
                                                    m_tilde[t+1] - m_hat[t+1]))
        P_tilde = P[t, ...] + np.dot(P[t, ...],
                                     np.dot(np.linalg.pinv(P_tilde[t, ...]),
                                            np.dot(P_tilde[t + 1, ...] - P_hat[t + 1, ...],
                                                   np.dot(np.linalg.pinv(P_tilde[t, ...]).transpose(),
                                                          P[t, ...].transpose()))))
        y[t, ...] = np.dot(V, m_tilde[t, ...]) + m0

    return y





