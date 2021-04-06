import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

from InEKF import *
from LieGroup import *
from Observations import *

def main():

    begin_time = time.time()

    # ---- Initialize invariant extended Kalman filter ----- #
    initial_state = RobotState()

    # Initialize state mean
    R0 = np.array([[1, 0, 0],    # initial orientation
                   [0, -1, 0],   # IMU frame is rotated 90deg about the x-axis
                   [0, 0, -1]]) 
                    
    v0 = np.array([1,2,3])    # initial velocity
    p0 = np.array([4,5,6])    # initial position
    bg0 = np.zeros((3,1))   # initial gyroscope bias
    ba0 = np.zeros((3,1))   # initial accelerometer bias
    
    initial_state.setRotation(R0)
    initial_state.setVelocity(v0)
    initial_state.setPosition(p0)
    initial_state.setGyroscopeBias(bg0)
    initial_state.setAccelerometerBias(ba0)

    # Initialize state covariance
    noise_params = NoiseParams()
    noise_params.setGyroscopeNoise(0.0)
    noise_params.setAccelerometerNoise(0.0)
    noise_params.setGyroscopeBiasNoise(0.0)
    noise_params.setAccelerometerBiasNoise(0.0)
    noise_params.setContactNoise(0.0)

    # Initial Covariance and Adjoint
    P = np.eye(15)
    Adj = np.eye(initial_state.dimP())
    Adj[0:initial_state.dimP()-initial_state.dimTheta(),0:initial_state.dimP()-initial_state.dimTheta()] = Adjoint_SEK3(initial_state.getX())

    # print("Initial Adjoint: \n", Adj)
    # Left invariant filter
    initial_state.setP(P)
    LI_filter = InEKF(initial_state, noise_params, ErrorType.LeftInvariant)

    # Right invariant filter
    initial_state.setP(Adj @ P @ Adj.T)
    RI_filter = InEKF(initial_state, noise_params, ErrorType.RightInvariant)

    LI_state = LI_filter.getState()
    RI_state = RI_filter.getState()
    # print("Left Invariant Initial State: ")
    # LI_state.printState()
    # print("Right Invariant Initial State: ")
    # RI_state.printState()
    # print("LI init P: \n", LI_state.getP())
    # print("RI init P: \n", RI_state.getP())

    print("\n\n ------ Propagate using random data -------\n\n")
    NUM_PROPAGATE = 100

    for i in range(NUM_PROPAGATE):
        # print("------propagating ",i,"-----------")
        imu = np.random.normal(0,1,(6,1))
        dt = np.random.normal(0,1)
        # imu = np.array([0,1,2,3,4,5],dtype=np.float64)
        # imu = imu.reshape(6,1)
        # dt = 0.5

        LI_filter.Propagate(imu,dt)
        RI_filter.Propagate(imu,dt)

    LI_state = LI_filter.getState()
    RI_state = RI_filter.getState()


    print("Left Invariant State: ")
    # np.savetxt(sys.stdout,LI_state.getX())
    LI_state.printState()
    print("Right Invariant State: ")
    # np.savetxt(sys.stdout,RI_state.getX())
    RI_state.printState()
    print("Left Invariant Covariance: ")
    # print(LI_state.getP(),"\n \n")
    np.savetxt(sys.stdout,LI_state.getP())
    print("Right Invariant Covariance: ")
    # print(RI_state.getP(),"\n \n")
    np.savetxt(sys.stdout,RI_state.getP())

    Adj = np.eye(LI_state.dimP())
    Adj[0:LI_state.dimP()-LI_state.dimTheta(),0:LI_state.dimP()-LI_state.dimTheta()] = Adjoint_SEK3(LI_state.getX())
    print("Difference between right invariant covariance (left is mapped using adjoint): \n")
    print( np.linalg.norm(RI_state.getP() - (Adj @ LI_state.getP() @ Adj.T)), "\n \n")
   
    AdjInv = np.eye(RI_state.dimP())
    AdjInv[0:RI_state.dimP()-RI_state.dimTheta(),0:RI_state.dimP()-RI_state.dimTheta()] = Adjoint_SEK3(RI_state.Xinv())
    print("Difference between left invariant covariance (right is mapped using adjoint inverse): \n")
    print(np.linalg.norm(LI_state.getP() - (AdjInv @ RI_state.getP() @ AdjInv.T)), "\n \n")
    print("Difference between state estimates: \n")
    print(np.linalg.norm(LI_state.getX() - RI_state.getX()))


    print("\n\n ------ Correct using random data -------\n\n")
    # ----- Correct using random data ------
    contacts = [[0,1],[1,1]]
    LI_filter.setContacts(contacts)
    RI_filter.setContacts(contacts)

    NUM_CORRECT = 10
    for i in range(NUM_CORRECT):
        measured_kinematics = []
        pose = np.eye(4)
        p = np.zeros((3,1))
        covariance = np.eye(6)

        # p = np.array([i*0.3,i*0.4,i*0.5])
        p = np.random.normal(0,1,(3,1))
        pose[0:3,3] = p.reshape(3).copy()
        
        measured_kinematics.append(Kinematics(0,pose,covariance))
        
        # p = np.array([i*0.2,i*0.1,i*0.6])
        p = np.random.normal(0,1,(3,1))
        pose[0:3,3] = p.reshape(3).copy()
        measured_kinematics.append(Kinematics(1,pose,covariance))
        # print("\n ---------- LI filter ----------- \n")
        LI_filter.CorrectKinematics(measured_kinematics)
        # print("\n ---------- RI filter ----------- \n")
        RI_filter.CorrectKinematics(measured_kinematics)
        

    LI_state = LI_filter.getState()
    RI_state = RI_filter.getState()

    print("Left Invariant State: ")
    LI_state.printState()
    # np.savetxt(sys.stdout,LI_state.getX())
    print("\n \n")
    print("Right Invariant State: ")
    RI_state.printState()
    # np.savetxt(sys.stdout,RI_state.getX())
    print("\n \n")
    print("Left Invariant Covariance: ")
    np.savetxt(sys.stdout,LI_state.getP())
    print("\n \n")
    # print(LI_state.getP(),"\n \n")
    print("Right Invariant Covariance: ")
    # print(RI_state.getP(),"\n \n")
    np.savetxt(sys.stdout,RI_state.getP())
    print("\n \n")

    Adj = np.eye(LI_state.dimP())
    Adj[0:LI_state.dimP()-LI_state.dimTheta(),0:LI_state.dimP()-LI_state.dimTheta()] = Adjoint_SEK3(LI_state.getX())
    print("Difference between right invariant covariance (left is mapped using adjoint): \n")
    print(np.linalg.norm(RI_state.getP() - (Adj @ LI_state.getP() @ Adj.T)), "\n \n")
   
    AdjInv = np.eye(RI_state.dimP())
    AdjInv[0:RI_state.dimP()-RI_state.dimTheta(),0:RI_state.dimP()-RI_state.dimTheta()] = Adjoint_SEK3(RI_state.Xinv())
    print("Difference between left invariant covariance (right is mapped using adjoint inverse): \n")
    print(np.linalg.norm(LI_state.getP() - (AdjInv @ RI_state.getP() @ AdjInv.T)), "\n \n")
    print("Difference between state estimates: \n")
    print(np.linalg.norm(LI_state.getX() - RI_state.getX()))


if __name__ == '__main__':
    main()