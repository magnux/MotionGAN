from __future__ import absolute_import, division, print_function
"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # NTU
    self.body_members = {
      'left_arm': {'joints': [20, 8, 9, 10, 11], 'side': 'left'},  # [21, 9, 10, 11, 12, 24, 25]
      'right_arm': {'joints': [20, 4, 5, 6, 7], 'side': 'right'},  # [21, 5, 6, 7, 8, 22, 23]
      'head': {'joints': [20, 2, 3], 'side': 'right'},
      'torso': {'joints': [20, 1, 0], 'side': 'right'},
      'left_leg': {'joints': [0, 16, 17, 18, 19], 'side': 'left'},
      'right_leg': {'joints': [0, 12, 13, 14, 15], 'side': 'right'},
    }
    self.njoints = 25

    # Human3.6
    # self.body_members = {
    #   'left_arm': {'joints': [16, 17, 18, 19, 20, 21, 20, 19, 22, 23, 22, 19, 18, 17, 16, 12], 'side': 'left'},
    #   'right_arm': {'joints': [24, 25, 26, 27, 28, 29, 28, 27, 30, 31, 30, 27, 26, 25, 24, 12], 'side': 'right'},
    #   'head': {'joints': [13, 14, 15, 14, 13, 12], 'side': 'right'},
    #   'torso': {'joints': [0, 11, 12], 'side': 'right'},
    #   'left_leg': {'joints': [0, 6, 7, 8, 9, 10, 9, 8, 7, 6], 'side': 'left'},
    #   'right_leg': {'joints': [0, 1, 2, 3, 4, 5, 4, 3, 2, 1], 'side': 'right'},
    # }
    # self.njoints = 32

    # OpenPose
    # self.body_members = {
    #       'left_arm': {'joints': [2, 3, 4, 3, 2], 'side': 'left'},
    #       'right_arm': {'joints': [5, 6, 7, 6, 5], 'side': 'right'},
    #       'head': {'joints': [1, 0, 1], 'side': 'right'},
    #       # 'ext_head': {'joints': [14, 15, 16, 17, 16, 15, 14], 'side': 'right'},
    #       'ears': {'joints': [14, 0, 15], 'side': 'right'},
    #       'torso': {'joints': [2, 1, 5, 1, 8, 1, 11], 'side': 'right'},
    #       'left_leg': {'joints': [8, 9, 10, 9, 8], 'side': 'left'},
    #       'right_leg': {'joints': [11, 12, 13, 12, 11], 'side': 'right'},
    # }
    # self.njoints = 16

    vals = np.zeros((self.njoints, 3))

    self.ax = ax

    # Make connection matrix
    self.plots = {}
    for member in self.body_members.values():
      for j in range(len(member['joints'])-1):
        j_idx_start = member['joints'][j]
        j_idx_end = member['joints'][j + 1]
        x = np.array([vals[j_idx_start, 0], vals[j_idx_end, 0]])
        y = np.array([vals[j_idx_start, 1], vals[j_idx_end, 1]])
        z = np.array([vals[j_idx_start, 2], vals[j_idx_end, 2]])
        self.plots[(j_idx_start, j_idx_end)] = self.ax.plot(x, y, z, lw=2, c=lcolor if member['side'] == 'left' else rcolor)

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: njoints * 3-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == self.njoints * 3, "channels should have %d entries, it has %d instead" % (self.njoints * 3, channels.size)
    vals = np.reshape(channels, (self.njoints, -1))

    for member in self.body_members.values():
      for j in range(len(member['joints'])-1):
        j_idx_start = member['joints'][j]
        j_idx_end = member['joints'][j + 1]
        x = np.array([vals[j_idx_start, 0], vals[j_idx_end, 0]])
        y = np.array([vals[j_idx_start, 1], vals[j_idx_end, 1]])
        z = np.array([vals[j_idx_start, 2], vals[j_idx_end, 2]])
        self.plots[(j_idx_start, j_idx_end)][0].set_xdata(x)
        self.plots[(j_idx_start, j_idx_end)][0].set_ydata(y)
        self.plots[(j_idx_start, j_idx_end)][0].set_3d_properties(z)
        self.plots[(j_idx_start, j_idx_end)][0].set_color(lcolor if member['side'] == 'left' else rcolor)

    r = 1  # 500;
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    self.ax.set_xlim3d([-r + xroot, r + xroot])
    self.ax.set_zlim3d([-r + zroot, r + zroot])
    self.ax.set_ylim3d([-r + yroot, r + yroot])

    self.ax.set_aspect('equal')