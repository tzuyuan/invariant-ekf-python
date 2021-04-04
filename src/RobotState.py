'''
 * ----------------------------------------------------------------------------
 * Copyright 2021, Tzu-Yuan Lin <tzuyaun@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

 * This is a python version of Ross Hartley's invariant EKF code.
 * Original C++ code can be found here: https://github.com/RossHartley/invariant-ekf

 **
 *  @file   RobotState.py
 *  @author Tzu-Yuan Lin
 *  @brief  Source file for RobotState
 *  @date   April 1, 2021
 **
'''

import numpy as np
from enum import Enum
from copy import deepcopy

from LieGroup import *

class StateType(Enum):
    WorldCentric = 0
    BodyCentric = 1

class RobotState:

    # class variables
    state_type_ = StateType.WorldCentric
    X_ = np.eye(5)
    Theta_ = np.zeros((6,1))
    P_ = np.eye(15)
    
    def __init__(self, X=None, Theta=None, P=None):
        if X is None: # no input to constructor
            pass
        elif Theta is None: # Only X is the input
            self.X_ = X.copy()
            self.P_ = np.eye(3*self.dimX()+self.dimTheta()-6, 3*self.dimX()+self.dimTheta()-6)
        elif P is None: # X and Theta are input
            self.X_ = X.copy()
            self.Theta_ = Theta.copy()
            self.P_ = np.eye(3*self.dimX()+self.dimTheta()-6, 3*self.dimX()+self.dimTheta()-6)
        else: # We have X, Theta, and P
            self.X_ = X.copy()
            self.Theta_ = Theta.copy()
            self.P_ = P.copy()

    
    # some getter functions
    def getX(self):
        return self.X_.copy()
    def getTheta(self):
        return self.Theta_.copy()
    def getP(self):
        return self.P_.copy()
    def getRotation(self):
        return self.X_[0:3,0:3].copy()
    def getVelocity(self):
        return self.X_[0:3,3].copy().reshape(3,1)
    def getPosition(self):
        return self.X_[0:3,4].copy().reshape(3,1)
    def getVector(self, idx):
        return self.X_[0:3,idx].copy().reshape(3,1)


    def getGyroscopeBias(self):
        return self.Theta_[0:3].copy()
    def getAccelerometerBias(self):
        return self.Theta_[3:6].copy()
    

    def getRotationCovariance(self):
        return self.P_[0:3,0:3].copy()
    def getVelocityCovariance(self):
        return self.P_[3:6,3:6].copy()
    def getPositionCovariance(self):
        return self.P_[6:9,6:9].copy()
    def getGyroscopeBiasCovariance(self):
        i = np.shape(self.P_)[0]
        return self.P_[i-6:i-3,i-6:i-3].copy()
    def getAccelerometerBiasCovariance(self):
        i = np.shape(self.P_)[0]
        return self.P_[i-3:i,i-3:i].copy()


    # get dimensions
    def dimX(self):
        return np.shape(self.X_)[1]
    def dimTheta(self):
        return np.shape(self.Theta_)[0]
    def dimP(self):
        return np.shape(self.P_)[1]

    # get states in World/Body frame
    def getStateType(self):
        return deepcopy(self.state_type_)
    def getWorldX(self):
        if self.state_type_.name == 'WorldCentric':
            return self.getX()
        else:
            return self.Xinv()
    def getWorldRotation(self):
        if self.state_type_.name == 'WorldCentric':
            return self.getRotation()
        else:
            return np.transpose(self.getRotation())
    def getWorldVelocity(self):
        if self.state_type_.name == 'WorldCentric':
            return self.getVelocity()
        else:
            return -self.getRotation().T @ self.getVelocity()
    def getWorldPosition(self):
        if self.state_type_.name == 'WorldCentric':
            return self.getPosition()
        else:
            return -np.transpose(self.getRotation())@self.getPosition()
    def getBodyX(self):
        if self.state_type_.name == 'BodyCentric':
            return self.getX()
        else:
            return self.Xinv()
    def getBodyRotation(self):
        if self.state_type_.name == 'BodyCentric':
            return self.getRotation()
        else:
            return np.transpose(self.getRotation())
    def getBodyVelocity(self):
        if self.state_type_.name == 'BodyCentric':
            return self.getVelocity()
        else:
            return -np.transpose(self.getRotation())@self.getVelocity()
    def getBodyPosition(self):
        if self.state_type_.name == 'BodyCentric':
            return self.getPosition()
        else:
            return -np.transpose(self.getRotation())@self.getPosition()
    
    

    def setX(self,X):
        self.X_ = X.copy()
    def setTheta(self,Theta):
        self.Theta_ = Theta.copy()
    def setP(self,P):
        self.P_ = P.copy()
    def setRotation(self,R):
        self.X_[0:3,0:3] = R.copy()
    def setVelocity(self,v):
        self.X_[0:3,3] = v.copy().reshape(3,)
    def setPosition(self,p):
        self.X_[0:3,4] = p.copy().reshape(3,)
    def setGyroscopeBias(self, bg):
        self.Theta_[0:3] = bg.copy()
    def setAccelerometerBias(self, ba):
        self.Theta_[3:6] = ba.copy()
    def setRotationCovariance(self,cov):
        self.P_[0:3,0:3] = cov.copy()
    def setVelocityCovariance(self,cov):
        self.P_[3:6,3:6] = cov.copy()
    def setPositionCovariance(self,cov):
        self.P_[6:9,6:9] = cov.copy()
    def setGyroscopeBiasCovariance(self,cov):
        i = np.shape(self.P_)[0]
        self.P_[i-6:i-3,i-6:i-3] = cov.copy()
    def setAccelerometerBiasCovariance(self,cov):
        i = np.shape(self.P_)[0]
        self.P_[i-3:i,i-3:i] = cov.copy()



    def copyDiagX(self, n, BigX):
        dimX = self.dimX()
        for i in range(n):
            startIndex = np.shape(BigX)[0]
            BigX = np.hstack((BigX,np.zeros((startIndex,dimX))))
            temp_bottom = np.hstack((np.zeros((dimX,startIndex)), self.X_))
            BigX = np.vstack((BigX,temp_bottom))

        return BigX.copy()

    def copyDiagXinv(self, n, BigXinv):
        dimX = self.dimX()
        Xinv = self.Xinv()
        for i in range(n):
            startIndex = np.shape(BigXinv)[0]
            BigXinv = np.hstack((BigXinv,np.zeros((startIndex,dimX))))
            temp_bottom = np.hstack((np.zeros((dimX,startIndex)), Xinv))
            BigXinv = np.vstack((BigXinv,temp_bottom))
            
        return BigXinv.copy()


    def Xinv(self):
        dimX = self.dimX()
        Xinv = np.eye(dimX)
        RT = self.X_[0:3,0:3].T
        Xinv[0:3,0:3] = RT
        for i in np.arange(3,dimX):
            Xinv[0:3,i] = -RT@self.X_[0:3,i]
        return Xinv

    def printState(self):
        # np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)
        print("-------- Robot State --------")
        print("X:\n", self.X_)
        print("Theta:\n", self.Theta_)
        print("-----------------------------")


# def main():
#     X = 5*np.eye(5)
#     X[0,1] = 33
#     X[1,0] = 44
#     cov = 6*np.eye(3)
#     robotstate = RobotState(X)
#     robotstate.getWorldVelocity()
#     robotstate.setAccelerometerBiasCovariance(cov)
#     cov_out = robotstate.getAccelerometerBiasCovariance()
#     # print(cov_out)

#     a = [[1,2,3],\
#         [4,5,6],\
#         [7,8,9]]
#     Test = np.array(a)
#     bigx = robotstate.copyDiagX(2,Test)

#     X_o = robotstate.getX()
#     print(X)
#     print(Test)
#     print(bigx)

# if __name__ == '__main__':
#     main()