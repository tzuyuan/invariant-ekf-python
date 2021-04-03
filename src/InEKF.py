'''
 * ----------------------------------------------------------------------------
 * Copyright 2021, Tzu-Yuan Lin <tzuyaun@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

 * This is a python version of Ross Hartley's invariant EKF code.
 * Original C++ code can be found here: https://github.com/RossHartley/invariant-ekf

 **
 *  @file   InEKF.py
 *  @author Tzu-Yuan Lin
 *  @brief  Source file for Invariant EKF
 *  @date   April 1, 2021
 **
 '''

import numpy as np
from enum import Enum
import sys
sys.path.append('.')

from LieGroup import *
from NoiseParams import *
from RobotState import *
from Observations import *


class ErrorType(Enum):
    LeftInvariant = 0
    RightInvariant = 1


class InEKF:

    error_type_ = ErrorType.LeftInvariant
    estimate_bias_ = True
    state_ = None
    noise_params_ = None
    
    # Gravity vector
    g_ = np.zeros((3,1))
    
    # contacts_ is a dict<int,bool>. 
    # The keys are kin.IDs, values are boolean indicator
    contacts_ = {}
    
    # estimated_contact_positions_ is a dict<int,int>
    # The keys are kin.IDs, values are startIndex
    estimated_contact_positions_ = {}
    
    prior_landmarks_ = {}
    estimated_landmarks_ = {}

    # Magnetic Field Vector
    magnetic_field_ = np.zeros((3,1))


    def __init__(self, state=None, params=None, error_type=None):
        self.g_[2] = -9.81
        self.magnetic_field_[0] = np.cos(1.2049)
        self.magnetic_field_[2] = np.sin(1.2049)

        if state is not None:
            self.state_ = state
        else:
            self.state_ = RobotState()

        if params is not None:
            self.noise_params_ = params
        else:
            self.noise_params_ = NoiseParams()

        if error_type is not None:
            self.error_type_ = error_type
        else:
            self.error_type_ = ErrorType.LeftInvariant

    # reset the filter
    # Initializes state matrix to identity, removes all augmented states, and assigns default noise parameters.
    def clear(self):
        self.state_ = RobotState()
        self.noise_params_ = NoiseParams()
        self.prior_landmarks_ = {}
        self.estimated_landmarks_ = {}
        self.contacts_ = {}
        self.estimated_contact_positions_ = {}

    def getErrorType(self):
        return self.error_type_

    def getState(self):
        return self.state_.copy()

    def setState(self, state):
        self.state_ = state

    def getNoiseParams(self):
        return self.noise_params_.copy()

    def setNoiseParams(self, params):
        self.noise_params_ = params

    def getPriorLandmarks(self):
        return self.prior_landmarks_.copy()
    
    def setPriorLandmarks(self, prior_landmarks_):
        self.prior_landmarks_ = prior_landmarks_.copy()

    def getEstimatedLandmarks(self):
        return self.estimated_landmarks_.copy()

    def getEstimatedContactPositions(self):
        return self.estimated_contact_positions_.copy()

    def setContacts(self, contacts):
        for c in contacts:
            self.contacts_[c[0]] = c[1]

    def getContacts(self):
        return self.contacts_.copy()

    def setMagneticField(self,true_magnetic_field):
        self.magnetic_field_ = true_magnetic_field

    def getMagneticField(self):
        return self.magnetic_field_.copy()

    # compute analytical state transition matrix
    def StateTransitionMatrix(self, w, a, dt):
        phi = w*dt
        G0 = Gamma_SO3(phi,0)
        G1 = Gamma_SO3(phi,1)
        G2 = Gamma_SO3(phi,2)
        G0t = np.transpose(G0)
        G1t = np.transpose(G1)
        G2t = np.transpose(G2)
        G3t = Gamma_SO3(-phi,3)

        dimX = self.state_.dimX()
        dimTheta = self.state_.dimTheta()
        dimP = self.state_.dimP()
        Phi = np.eye(dimP)

        ax = skew(a)
        wx = skew(w)
        wx2 = wx@wx

        dt2 = dt*dt
        dt3 = dt2*dt
        theta = np.linalg.norm(w)
        theta2 = theta*theta
        theta3 = theta2*theta
        theta4 = theta3*theta
        theta5 = theta4*theta
        theta6 = theta5*theta
        theta7 = theta6*theta
        thetadt = theta*dt
        thetadt2 = thetadt*thetadt
        thetadt3 = thetadt2*thetadt
        sinthetadt = np.sin(thetadt)
        costhetadt = np.cos(thetadt)
        sin2thetadt = np.sin(2*thetadt)
        cos2thetadt = np.cos(2*thetadt)
        thetadtcosthetadt = thetadt*costhetadt
        thetadtsinthetadt = thetadt*sinthetadt


        Phi25L = G0t@(ax@G2t*dt2\
                + ((sinthetadt-thetadtcosthetadt)/(theta3))*(wx@ax)\
                - ((cos2thetadt-4*costhetadt+3)/(4*theta4))*(wx@ax@wx)\
                + ((4*sinthetadt+sin2thetadt-4*thetadtcosthetadt-2*thetadt)/(4*theta5))*(wx@ax@wx2)\
                + ((thetadt2-2*thetadtsinthetadt-2*costhetadt+2)/(2*theta4))*(wx2@ax)\
                - ((6*thetadt-8*sinthetadt+sin2thetadt)/(4*theta5))*(wx2@ax@wx)\
                + ((2*thetadt2-4*thetadtsinthetadt-cos2thetadt+1)/(4*theta6))*(wx2@ax@wx2))

        Phi35L = G0t@(ax@G3t*dt3\
                - ((thetadtsinthetadt+2*costhetadt-2)/(theta4))*(wx@ax)\
                - ((6*thetadt-8*sinthetadt+sin2thetadt)/(8*theta5))*(wx@ax@wx)
                - ((2*thetadt2+8*thetadtsinthetadt+16*costhetadt+cos2thetadt-17)/(8*theta6))*(wx@ax@wx2)\
                + ((thetadt3+6*thetadt-12*sinthetadt+6*thetadtcosthetadt)/(6*theta5))*(wx2@ax)\
                - ((6*thetadt2+16*costhetadt-cos2thetadt-15)/(8*theta6))*(wx2@ax@wx)\
                + ((4*thetadt3+6*thetadt-24*sinthetadt-3*sin2thetadt+24*thetadtcosthetadt)/(24*theta7))*(wx2@ax@wx2))


        tol = 1e-6
        if theta < tol:
            Phi25L = (1/2)*ax*dt2
            Phi35L = (1/6)*ax*dt3


        if (self.state_.getStateType().name == 'WorldCentric' and self.error_type_.name == 'LeftInvariant') or \
           (self.state_.getStateType().name == 'BodyCentric' and self.error_type_.name == 'RightInvariant'):
            
            Phi[0:3,0:3] = G0t
            Phi[3:6,0:3] = -G0t@skew(G1@a)*dt
            Phi[6:9,0:3] = -G0t@skew(G2@a)*dt2
            Phi[3:6,3:6] = G0t
            Phi[6:9,3:6] = G0t*dt
            Phi[6:9,6:9] = G0t

            for i in np.arange(5,dimX):
                Phi[(i-2)*3:(i-2)*3+3,(i-2)*3:(i-2)*3+3] = G0t

            Phi[0:3,dimP-dimTheta:dimP-dimTheta+3] = -G1t*dt
            Phi[3:6,dimP-dimTheta:dimP-dimTheta+3] = Phi25L
            Phi[6:9,dimP-dimTheta:dimP-dimTheta+3] = Phi35L
            Phi[3:6,dimP-dimTheta+3:dimP-dimTheta+6] = -G1t*dt
            Phi[6:9,dimP-dimTheta+3:dimP-dimTheta+6] = -G0t@G2*dt2
        else:
            gx = skew(self.g_)
            R = self.state_.getRotation()
            v = self.state_.getVelocity()
            p = self.state_.getPosition()
            RG0 = R@G0
            RG1dt = R@G1*dt
            RG2dt2 = R@G2*dt2
            Phi[3:6,0:3] = gx*dt
            Phi[6:9,0:3] = 0.5*gx*dt2
            Phi[6:9,3:6] = np.eye(3)*dt
            Phi[0:3,dimP-dimTheta:dimP-dimTheta+3] = -RG1dt
            Phi[3:6,dimP-dimTheta:dimP-dimTheta+3] = -skew(v+RG1dt@a+self.g_*dt)@RG1dt + RG0@Phi25L
            Phi[6:9,dimP-dimTheta:dimP-dimTheta+3] = -skew(p+v*dt+RG2dt2@a+0.5*self.g_*dt2)@RG1dt + RG0@Phi35L
            for i in np.arange(5,dimX):
                Phi[(i-2)*3:(i-2)*3+3,dimP-dimTheta:dimP-dimTheta+3] = -skew(self.state_.getVector(i))@RG1dt

            Phi[3:6,dimP-dimTheta+3:dimP-dimTheta+6] = -RG1dt
            Phi[6:9,dimP-dimTheta+3:dimP-dimTheta+6] = -RG2dt2

        return Phi

    # Compute Discrete noise matrix
    def DiscreteNoiseMatrix(self, Phi, dt):
        dimX = self.state_.dimX()
        dimTheta = self.state_.dimTheta()
        dimP = self.state_.dimP()
        G = np.eye(dimP)

        # Compute G using Adjoint of Xk if needed, otherwise identity (Assumes unpropagated state)
        if (self.state_.getStateType().name == 'WorldCentric' and self.error_type_.name == 'LeftInvariant') or \
           (self.state_.getStateType().name == 'BodyCentric' and self.error_type_.name == 'RightInvariant'):
            G[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(self.state_.getWorldX())

        # Continuous noise covariance 
        Qc = np.zeros((dimP,dimP))  # Landmark noise terms will remain zero
        Qc[0:3,0:3] = self.noise_params_.getGyroscopeCov()
        Qc[3:6,3:6] = self.noise_params_.getAccelerometerCov()
        for _, c_val in self.estimated_contact_positions_.items():  # Contact noise terms
            Qc[3+3*(c_val-3):6+3*(c_val-3),3+3*(c_val-3):6+3*(c_val-3)] = self.noise_params_.getContactCov()

        Qc[dimP-dimTheta:dimP-dimTheta+3,dimP-dimTheta:dimP-dimTheta+3] = self.noise_params_.getGyroscopeBiasCov()
        Qc[dimP-dimTheta+3:dimP-dimTheta+6,dimP-dimTheta+3:dimP-dimTheta+6] = self.noise_params_.getAccelerometerBiasCov()

        # Noise Covariance Discretization
        PhiG = Phi@G
        Qd = PhiG @ Qc @ np.transpose(PhiG) * dt    # Approximated discretized noise matrix

        return Qd

    def Propagate(self,imu,dt):
        # Bias corrected IMU measurements
        w = imu[0:3]-self.state_.getGyroscopeBias()
        a = imu[3:6]-self.state_.getAccelerometerBias()

        # Get current state estimate and dimensions
        X = self.state_.getX()
        Xinv = self.state_.Xinv()
        P = self.state_.getP()
        dimX = self.state_.dimX()
        dimP = self.state_.dimP()
        dimTheta = self.state_.dimTheta()

        # ------------ Propagate Covariance ------------- #
        Phi = self.StateTransitionMatrix(w,a,dt)
        Qd = self.DiscreteNoiseMatrix(Phi,dt)
        P_pred = Phi @ P @ np.transpose(Phi) + Qd

        # If we don't want to estimate bias, remove correlation
        if not self.estimate_bias_:
            P_pred[0:dimP-dimTheta,dimP-dimTheta:dimP] = np.zeros((dimP-dimTheta,dimTheta))
            P_pred[dimP-dimTheta:dimP,0:dimP-dimTheta] = np.zeros((dimTheta,dimP-dimTheta))
            P_pred[dimP-dimTheta:dimP,dimP-dimTheta:dimP] = np.zeros((dimTheta,dimTheta))

        
        # ------------ Propagate Mean ------------- #
        R = self.state_.getRotation()
        v = self.state_.getVelocity()
        p = self.state_.getPosition()

        phi = w*dt
        G0 = Gamma_SO3(phi,0)
        G1 = Gamma_SO3(phi,1)
        G2 = Gamma_SO3(phi,2)

        X_pred = X.copy()
        if self.state_.getStateType().name == 'WorldCentric':
            # Propagate world-centric state estimate
            X_pred[0:3,0:3] = R @ G0
            X_pred[0:3,3] = (v + (R @ G1 @ a + self.g_)* dt).reshape(3)
            X_pred[0:3,4] = (p + v*dt + (R @ G2 @ a + 0.5*self.g_)*dt*dt).reshape(3)
        else:
            # Propagate body-centric state estimate
            X_pred = X.copy()
            G0t = np.transpose(G0)
            X_pred[0:3,0:3] = G0t @ R
            X_pred[0:3,3] = (G0t @ (v - (G1 @ a + R @ self.g_) * dt)).reshape(3)
            X_pred[0:3,4] = (G0t @ (p + v*dt - (G2 @ a + 0.5 * R @ self.g_)*dt*dt)).reshape(3)
            for i in np.arange(5,dimX):
                X_pred[0:3,i] = G0t @ X[0:3,i]

        # ------------ Update State ------------- #
        self.state_.setX(X_pred)
        self.state_.setP(P_pred)

    
    # Correct State: Right-Invariant Observation
    def CorrectRightInvariant(self,Z,H,N):
        # Get current state estimate
        X = self.state_.getX()
        Theta = self.state_.getTheta()
        P = self.state_.getP()
        dimX = self.state_.dimX()
        dimP = self.state_.dimP()
        dimTheta = self.state_.dimTheta()

        # Remove bias
        Theta = np.zeros((6))
        P[dimP-dimTheta:dimP-dimTheta+6,dimP-dimTheta:dimP-dimTheta+6] = 0.0001*np.eye(6)
        P[0:dimP-dimTheta,dimP-dimTheta:dimP] = np.zeros((dimP-dimTheta,dimTheta))
        P[dimP-dimTheta:dimP,0:dimP-dimTheta] = np.zeros((dimTheta,dimP-dimTheta))

        # Map from left invariant to right invariant error temporarily
        if self.error_type_.name == 'LeftInvariant':
            Adj = np.eye(dimP)
            Adj[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(X)
            P = (Adj @ P @ Adj.T)

        # Compute Kalman Gain
        PHT = P @ H.T
        S = H @ PHT + N
        K = PHT @ np.linalg.inv(S)

        # Compute state correction vector
        delta = K @ Z
        dX = Exp_SEK3(delta[0:np.shape(delta)[0]-dimTheta])
        dTheta = delta[np.shape(delta)[0]-dimTheta:]

        # Update State
        X_new = dX @ X
        Theta_new = Theta + dTheta

        # Set new state
        self.state_.setX(X_new)
        self.state_.setTheta(Theta_new)

        # Update Covariance
        IKH = np.eye(dimP) - K @ H
        P_new = IKH @ P @ IKH.T + K @ N @ K.T

        # Map from right invariant back to left invariant error
        if self.error_type_.name == 'LeftInvariant':
            AdjInv = np.eye(dimP)
            AdjInv[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(self.state_.Xinv())
            P_new = (AdjInv @ P_new @ AdjInv.T)

        # Set new covariance
        self.state_.setP(P_new)


    # Correct State: Left-Invariant Observation
    def CorrectLeftInvariant(self,Z,H,N):
        # Get current state estimate
        X = self.state_.getX()
        Theta = self.state_.getTheta()
        P = self.state_.getP()
        dimX = self.state_.dimX()
        dimP = self.state_.dimP()
        dimTheta = self.state_.dimTheta()

        # Map from right invariant to left invariant error temporarily
        if self.error_type_.name == 'RightInvariant':
            AdjInv = np.eye(dimP)
            AdjInv[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(self.state_.Xinv())
            P = (AdjInv @ P @ AdjInv.T)

        # Compute Kalman Gain
        PHT = P @ H.T
        S = H @ PHT + N
        K = PHT @ np.linalg.inv(S)

        # Compute state correction vector
        delta = K @ Z
        dX = Exp_SEK3(delta[0:np.shape(delta)[0]-dimTheta])
        dTheta = delta[np.shape(delta)[0]-dimTheta:]

        # Update State
        X_new = X @ dX
        Theta_new = Theta + dTheta

        # Set new state
        self.state_.setX(X_new)
        self.state_.setTheta(Theta_new)

        # Update Covariance
        IKH = np.eye(dimP) - K @ H
        P_new = IKH @ P @ IKH.T + K @ N @ K.T

        # Map from left invariant back to right invariant error
        if self.error_type_.name == 'RightInvariant':
            Adj = np.eye(dimP)
            Adj[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(X_new)
            P_new = (Adj @ P_new @ Adj.T)

        self.state_.setP(P_new)

    # Correct state using kinematics measured between imu and contact point
    def CorrectKinematics(self, measured_kinematics):
        
        Z = np.empty(shape=(0,1))
        H = np.empty(shape=(0,0))
        N = np.empty(shape=(0,0))
        PI = np.empty(shape=(0,0))


        # remove_contacts is a list of tuples: (kin.ID, est_contact_pos)
        remove_contacts = []
        # new_contacts is a list of kinematic objects.
        new_contacts = []
        # used_contact_ids is a list of int containing the kin.IDs
        used_contact_ids = []

        # measured_kinematics is a list of kinematics object
        for kin in measured_kinematics:
            # Detect and skip if an ID is not unique (this would cause singularity issues in InEKF::Correct)
            if kin.ID in used_contact_ids:
                print("Duplicate contact ID detected! Skipping measurement.")
                continue
            else:
                used_contact_ids.append(kin.ID)

            # Find contact indicator for the kinematics measurement
            # self.contacts_ is a dict<int,bool>. 
            # The keys are kin.IDs, values are boolean indicator
            if kin.ID not in self.contacts_:
                continue    # Skip if contact state is unknown
            # contact_indicated is a boolean value
            contact_indicated = self.contacts_[kin.ID]

           
            # See if we can find id estimated_contact_positions
            # self.estimated_contact_positions_ is a dict<int,int>
            # The keys are kin.IDs, values are startIndex
            found = False
            if kin.ID in self.estimated_contact_positions_:
                found = True
                est_contact_pos = self.estimated_contact_positions_[kin.ID]


            # If contact is not indicated and id is found in estimated_contacts_, then remove state
            if (not contact_indicated) and found:
                remove_contacts.append((kin.ID,est_contact_pos)) # Add id to remove list
            # If contact is indicated and id is not found i n estimated_contacts_, then augment state
            elif contact_indicated and (not found):
                new_contacts.append(kin) # Add to augment list
            # If contact is indicated and id is found in estimated_contacts_, then correct using kinematics
            elif contact_indicated and found:
                dimX = self.state_.dimx()
                dimTheta = self.state_.dimTheta()
                dimP = self.state_.dimP()

                # Fill out H
                startIndex = np.shape(H)[0]
                old_dimP = np.shape(H)[1]
                H = np.hstack((H,np.zeros((startIndex,dimP-old_dimP))))
                H = np.vstack((H,np.zeros((3,dimP))))
                if self.state_.getStateType().name == 'WorldCentric':
                    H[startIndex:startIndex+3,6:9] = -np.eye(3) # -I
                    H[startIndex:startIndex+3,3*est_contact_pos-dimTheta:3*est_contact_pos-dimTheta+3] = np.eye(3)  # I
                else:
                    H[startIndex:startIndex+3,6:9] = np.eye(3) # I
                    H[startIndex:startIndex+3,3*est_contact_pos-dimTheta:3*est_contact_pos-dimTheta+3] = -np.eye(3)  # -I
                

                # Fill out N
                startIndex = np.shape(N)[0]
                N = np.hstack((N,np.zeros((startIndex,3))))
                new_N = self.state_.getRotation() @ kin.covariance[0:3,0:3] @ self.state_.getRotation().T
                temp_bottom = np.hstack((np.zeros((3,startIndex)),new_N))
                N = np.vstack((N,temp_bottom))


                # Fill out Z
                startIndex = np.shape(Z)[0]
                R = self.state_.getRotation()
                p = self.state_.getPosition()
                d = self.state_.getVector(est_contact_pos)
                if self.state_.getStateType().name == 'WorldCentric':
                    new_Z = R @ kin.pose[3:1,3] - (d - p)
                    np.vstack((Z,new_Z))
                else:
                    new_Z = R.T @ (kin.pose[3:1,3] - (d - p))
                    np.vstack((Z,new_Z))

            # If contact is not indicated and id is found in estimated_contacts_, then skip
            else:
                continue

            # Correct state using stacked observation
            if np.shape(Z)[0]>0:
                if self.state_.getStateType().name == 'WorldCentric':
                    self.CorrectRightInvariant(Z,H,N)
                else:
                    self.CorrectLeftInvariant(Z,H,N)

            # Remove contacts from state if remove_contacts is not empty
            if remove_contacts:
                X_rem = self.state_.getX()
                P_rem = self.state_.getP()
                # rm_ct is a tuple: (kin.ID, est_contact_pos)
                for rm_ct in remove_contacts:
                    # Remove row and column from X
                    np.delete(X_rem,rm_ct[1],0)
                    np.delete(X_rem,rm_ct[1],1)

                    # Remove 3 rows and columns from P
                    startIndex = 3 + 3*(rm_ct[1]-3)
                    np.delete(P_rem, list(range(startIndex,startIndex+3)),0)
                    np.delete(P_rem, list(range(startIndex,startIndex+3)),1)




def main():
    contacts = [[1,0],[2,0],[3,0]]
    contacts2 = [[1,1],[2,1],[3,1],[4,1]]
    inekf = InEKF()
    print(inekf.getState())
    inekf.setContacts(contacts)
    print(inekf.getContacts())
    inekf.setContacts(contacts2)
    print(inekf.getContacts())


if __name__ == '__main__':
    main()
