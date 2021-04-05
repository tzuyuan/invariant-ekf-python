'''
 * ----------------------------------------------------------------------------
 * Copyright 2021, Tzu-Yuan Lin <tzuyaun@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

 * This is a python version of Ross Hartley's invariant EKF code.
 * Original C++ code can be found here: https://github.com/RossHartley/invariant-ekf

 **
 *  @file   NoiseParams.py
 *  @author Tzu-Yuan Lin
 *  @brief  Source file for Invariant EKF noise parameter class 
 *  @date   April 1, 2021
 **
'''

import numpy as np

class NoiseParams:

    def __init__(self):
        self.Qg_ = None
        self.Qa_ = None
        self.Qbg_ = None
        self.Qba_ = None
        self.Ql_ = None
        self.Qc_ = None
        
        self.setGyroscopeNoise(0.01)
        self.setAccelerometerNoise(0.1)
        self.setGyroscopeBiasNoise(0.00001)
        self.setAccelerometerBiasNoise(0.0001)
        self.setContactNoise(0.1)

    
    def setGyroscopeNoise(self,std):
        if np.isscalar(std):
            self.Qg_ = std*std*np.eye(3)
        elif np.ndim(std) == 1:
            self.Qg_ = np.eye(3)
            for i in range(3):
                self.Qg_[i,i] = std[i]*std[i]
        elif np.ndim(std) == 2:
            self.Qg_ = std

            
    def setAccelerometerNoise(self,std):
        if np.isscalar(std):
            self.Qa_ = std*std*np.eye(3)
        elif np.ndim(std) == 1:
            self.Qa_ = np.eye(3)
            for i in range(3):
                self.Qa_[i,i] = std[i]*std[i]
        elif np.ndim(std) == 2:
            self.Qa_ = std

    def setGyroscopeBiasNoise(self,std):
        if np.isscalar(std):
            self.Qbg_ = std*std*np.eye(3)
        elif np.ndim(std) == 1:
            self.Qbg_ = np.eye(3)
            for i in range(3):
                self.Qbg_[i,i] = std[i]*std[i]
        elif np.ndim(std) == 2:
            self.Qbg_ = std


    def setAccelerometerBiasNoise(self,std):
        if np.isscalar(std):
            self.Qba_ = std*std*np.eye(3)
        elif np.ndim(std) == 1:
            self.Qba_ = np.eye(3)
            for i in range(3):
                self.Qba_[i,i] = std[i]*std[i]
        elif np.ndim(std) == 2:
            self.Qba_ = std

    
    def setContactNoise(self,std):
        if np.isscalar(std):
            self.Qc_ = std*std*np.eye(3)
        elif np.ndim(std) == 1:
            self.Qc_ = np.eye(3)
            for i in range(3):
                self.Qc_[i,i] = std[i]*std[i]
        elif np.ndim(std) == 2:
            self.Qc_ = std

    def getGyroscopeCov(self):
        return self.Qg_.copy()
    def getAccelerometerCov(self):
        return self.Qa_.copy()
    def getGyroscopeBiasCov(self):
        return self.Qbg_.copy()
    def getAccelerometerBiasCov(self):
        return self.Qba_.copy()
    def getContactCov(self):
        return self.Qc_.copy()


# def main():
#     std = np.array([[1,0,0],
#                     [0,2,0],
#                     [0,0,3]])
#     noiseparams = NoiseParams()
#     noiseparams.setGyroscopeNoise(std)

# if __name__ == '__main__':
#     main()