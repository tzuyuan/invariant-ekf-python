'''
 * ----------------------------------------------------------------------------
 * Copyright 2021, Tzu-Yuan Lin <tzuyaun@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

 * This is a python version of Ross Hartley's invariant EKF code.
 * Original C++ code can be found here: https://github.com/RossHartley/invariant-ekf

 **
 *  @file   Observation.py
 *  @author Tzu-Yuan Lin
 *  @brief  Source file for observations
 *  @date   April 1, 2021
 **
 '''


import numpy as np
from copy import copy

class Observation:
    def __init__(self, Y, b, H, N, PI):
        self.Y = Y.copy()
        self.b = b.copy()
        self.H = H.copy()
        self.N = N.copy()
        self.PI = PI.copy()


class Kinematics: 
    def __init__(self, id_in, pose_in, covariance_in):
        # id is a reserved word for python so we use ID here
        self.ID = copy(id_in)
        self.pose = pose_in.copy()
        self.covariance = covariance_in.copy()


class Landmark:
    def __init__(self, id_in, position_in, covariance_in):
        self.ID = copy(id_in)
        self.position = position_in.copy()
        self.covariance = covariance_in.copy()