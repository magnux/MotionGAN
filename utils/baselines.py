from __future__ import absolute_import, division, print_function
import numpy as np


def constant_baseline(X, mask):
    new_X = X * mask
    for j in range(X.shape[0]):
        for f in range(1, X.shape[1]):
            if mask[j, f, 0] == 0:
                new_X[j, f, :] = new_X[j, f - 1, :]
    return new_X


def burke_baseline(rawdata, mask, tol=0.0025, sigR=1e-3, keepOriginal=True):
    """Low-Rank smoothed Kalman filter, based in Burke et. al"""
    rawdata = np.transpose(rawdata.copy(), (1, 0, 2))
    raw_shape = [int(dim) for dim in rawdata.shape]
    rawdata = np.reshape(rawdata, (raw_shape[0], raw_shape[1] * raw_shape[2]))

    mask = np.tile(mask.copy(), (1, 1, raw_shape[2]))
    mask = np.transpose(mask, (1, 0, 2))
    mask = np.reshape(mask, (raw_shape[0], raw_shape[1] * raw_shape[2]))

    X = rawdata[(mask != 0).any(axis=1)]

    m = np.mean(X, axis=0)

    U, S, V = np.linalg.svd(X - m)

    d = np.nonzero(np.cumsum(S) / np.sum(S) > (1 - tol))[0][0]

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
        z = rawdata[i - 1, (mask[i - 1, :] != 0)]
        H = np.diag((mask[i - 1, :] != 0))
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
        y[(mask != 0)] = rawdata[(mask != 0)]

    y = np.reshape(y, (raw_shape[0], raw_shape[1], raw_shape[2]))
    y = np.transpose(y, (1, 0, 2))

    return y
