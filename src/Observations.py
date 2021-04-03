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


class Observation:

    Y = None
    b = None 
    H = None 
    N = None 
    PI = None 

    def __init__(self, Y, b, H, N, PI):
        self.Y = Y
        self.b = b
        self.H = H
        self.N = N
        self.PI = PI


class Kinematics:
    
    # id is a reserved word for python so we use ID here
    ID = None 
    pose = None 
    covariance = None

    def __init__(self, id_in, pose_in, covariance_in):
        self.ID = id_in
        self.pose = pose_in
        self.covariance = covariance_in


class Landmark:

    ID = None 
    position = None 
    covariance = None 

    def __init__(self, id_in, position_in, covariance_in):
        self.ID = id_in
        self.position = position_in
        self.covariance = covariance_in