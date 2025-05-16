import numpy as np
from scipy.linalg import expm
import time


def first_order_integrate(start, delta_y, delta_x):
    return start + delta_y*delta_x

def second_order_integrate(start, delta_y_new, delta_y_old, delta_x):
    return start + 0.5*(delta_y_old+ delta_y_new)*delta_x

# frequency should be in hertz
# data should be a a vector
def gen_motion_2D(gyro_data, accel_x1_data, accel_x2_data, x0, v0, a0, freq, int_order=1):
    # check that the shapes are correct
    assert gyro_data.shape == accel_x1_data.shape, 'gyro and acceleromater x have different shape'
    assert gyro_data.shape == accel_x2_data.shape, 'gyro and acceleromater y have different shape'
    assert accel_x1_data.shape == accel_x2_data.shape, 'acceleromater x and acceleromater y have different shape'

    # Create new matricies for the acceleration and gyro data that add one place
    # this is done for ease of for loop
    accel_x1 = np.empty(accel_x1_data.shape[0]+1)
    accel_x1[0] = 0
    accel_x1[1:] = accel_x1_data
    
    accel_x2 = np.empty(accel_x2_data.shape[0]+1)
    accel_x2[0] = 0
    accel_x2[1:] = accel_x2_data

    # Combine arrays
    accel_x_b = np.vstack([accel_x1,accel_x2])

    gyro = np.empty(gyro_data.shape[0]+1)
    gyro[0] = 0
    gyro[1:] = gyro_data

    # set up position vector, alpha vector, rotion matrix holder, velocity vector, attitude
    position = np.empty((2, gyro.shape[0]))
    Rm_b = np.empty((2,2,gyro.shape[0]))
    vel = np.empty((2, gyro.shape[0]))
    attitude = np.empty(gyro.shape[0])

    # set up matricies for accelormeter data in mapping frame
    accel_x_m = np.empty(accel_x_b.shape)
    accel_x_m[:,0] = np.array([0,0])

    # Set Inital Conditions
    position[0,0] = x0[0]
    position[1,0] = x0[1]

    Rm_b[:,:,0] = np.array([
        [np.cos(a0), -1*np.sin(a0)],
        [np.sin(a0), np.cos(a0)]])
    
    vel[0,0] = v0[0]
    vel[1,0] = v0[1]
    attitude[0] = a0

    # For each piece of available data, from t1
    start = time.time()
    for t in range(1,gyro.shape[0],1):
        # First we need to compute the rotation matrix from body to mapping
        omegam_b = np.array([
            [0, -1*gyro[t]],
            [gyro[t], 0]])
        
        Rm_b[:,:,t] = np.matmul(Rm_b[:,:,t-1], expm((1/freq)*omegam_b))



        # we can now project the acceleration into the mapping frame

        accel_x_m[:,t] = Rm_b[:,:,t] @ accel_x_b[:,t]

        # Now we can find the current velocity by integrating the acceleration and the
        # postion by integrating the velocity, this will be done with either first or second order
        # integration, which will be broken up by if statements
        # added conditions because higher approximatations cannot be done at the start
        if int_order == 1 or t < 2:
            vel[0,t] = first_order_integrate(vel[0,t-1], accel_x_m[0,t], 1/freq)
            vel[1,t] = first_order_integrate(vel[1,t-1], accel_x_m[1,t], 1/freq)
            position[0,t] = first_order_integrate(position[0,t-1], vel[0,t], 1/freq)
            position[1,t] = first_order_integrate(position[1,t-1], vel[1,t], 1/freq)
            attitude[t] = first_order_integrate(attitude[t-1], gyro[t], 1/freq)
        elif int_order == 2 or t < 3:
            vel[0,t] = second_order_integrate(vel[0,t-1], accel_x_m[0,t], accel_x_m[0,t-1], 1/freq)
            vel[1,t] = second_order_integrate(vel[1,t-1], accel_x_m[1,t], accel_x_m[1,t-1], 1/freq)
            position[0,t] = second_order_integrate(position[0,t-1], vel[0,t], vel[0,t-1], 1/freq)
            position[1,t] = second_order_integrate(position[1,t-1], vel[1,t], vel[1,t-1], 1/freq)
            attitude[t] = second_order_integrate(attitude[t-1], gyro[t], gyro[t-1], 1/freq)
        else:
            print('error in gen motion 2D. Integration order is not supported')
            return None
    end = time.time()

    time_elapsed = end-start
    return position, vel, accel_x_m[0:1,:], accel_x_m[1:2,:], time_elapsed, attitude




