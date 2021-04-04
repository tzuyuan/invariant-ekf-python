import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

from InEKF import *


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
    print("Left Invariant Initial State: ")
    LI_state.printState()
    print("Right Invariant Initial State: ")
    RI_state.printState()
    print("init P: \n", LI_state.getP())

    print("\n\n ------ Propagate using random data -------\n\n")
    NUM_PROPAGATE = 1

    for i in range(NUM_PROPAGATE):
        # imu = np.random.normal(0,1,(6,1))
        # dt = np.random.normal(0,1)
        imu = np.array([0,1,2,3,4,5])
        imu = imu.T.reshape(6,1)
        dt = 0.5
        LI_filter.Propagate(imu,dt)
        # RI_filter.Propagate(imu,dt)

    LI_state = LI_filter.getState()
    RI_state = RI_filter.getState()
    print("Left Invariant State: ")
    LI_state.printState()
    print("Right Invariant State: ")
    RI_state.printState()


if __name__ == '__main__':
    main()