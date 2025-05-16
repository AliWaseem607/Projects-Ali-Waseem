import numpy as np
from INU_functions import integrate
from scipy.linalg import expm
from tqdm import tqdm

def INU(nav_init, freq_gps, freq_KF, pos_gps, P_init, accel_x1_data, accel_x2_data, gyro_data):
    #################### first space out the gps measurements so they can be indexed ####################
    dt_gps = 1/freq_gps
    dt_KF = 1/freq_KF
    measurement_ratio = dt_gps/dt_KF
    total_length = int(np.floor((pos_gps.shape[1]-1)*(measurement_ratio)))

    gps_spread = np.full((pos_gps.shape[0],total_length+1),np.nan)
    count = 0
    for n in range(gps_spread.shape[1]):
        if n+1>= (count+1)*measurement_ratio:
            gps_spread[:,n] = pos_gps[:,count]
            count+=1
    
    # Make H matrix 
    H = np.array([[0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0]])
    
    # fix accel and gyro data so that they don't have a reading at t=0
    accel_x1 = np.zeros(accel_x1_data.shape[0]+1)
    accel_x1[0] = 0
    accel_x1[1:] = accel_x1_data
    
    accel_x2 = np.zeros(accel_x2_data.shape[0]+1)
    accel_x2[0] = 0
    accel_x2[1:] = accel_x2_data

    gyro = np.zeros(gyro_data.shape[0]+1)
    gyro[0] = 0
    gyro[1:] = gyro_data

    #################### Create variables for storing data ####################
    '''
    so here there are actually a lot of variables we need to store, namely we have the variables
    associated with the INU which will be called the nav_state, accel_n, accel_e (in the mapping
    frame) and then we will have the variables associated wit the Kalman filter, this is the 
    diff_state. Other variables we will have to store are the error sum to be used later if we 
    wish, P, and K
    '''
    iterations = gps_spread.shape[1]
    # there are 20001 iterations

    # navigation state
    nav_state = np.zeros((5, iterations))
    nav_state[:,0] = nav_init

    # accelerometer data in the mapping frame
    accel_b = np.vstack([accel_x1, accel_x2])
    accel_b_corrected = np.zeros(accel_b.shape)
    accel_m = np.zeros(accel_b.shape)
    # there are 20001 data points

    #gyro corrected holder
    gyro_corrected = np.zeros(gyro.shape)

    # rotation matrix holder
    Rm_b = np.zeros((2,2,iterations))
    a0 = nav_state[0,0]
    Rm_b[:,:,0] = np.array([
        [np.cos(a0), -1*np.sin(a0)],
        [np.sin(a0), np.cos(a0)]])
    
    # Differential state
    diff_state = np.zeros((9,iterations))
    diff_state_pred = np.zeros(diff_state.shape)

    # Covariance Matrix
    P = np.zeros((9,9,iterations))
    P[:,:,0] = P_init
    P_pred = np.zeros(P.shape)

    # Gain matrix
    K = np.zeros((9,2,iterations))

    # To sum errors
    errors = np.zeros(4)

    #################### Start the Navigation ####################
    '''
    So the process here will be to first integrate the next position, to find out 
    where we should be after that we use the Kalman Filter to predict how much
    our integrator is off by. Once there is a gps measurement we make an update
    in the KF and then with the predicted difference we move our nav state, if 
    if we are doing errors we also correct the gyro and accel errors
    '''

    for t in tqdm(range(1, iterations, 1)):
        # first "subtract" the error from the measurments
        accel_b_corrected[:,t] = accel_b[:,t] - errors[2:4]
        gyro_corrected[t] = gyro[t] - errors[0] - errors[1]

        # rotate the corrected accerlation into the mapping frame
        omegam_b = np.array([
            [0, -1*gyro_corrected[t]],
            [gyro_corrected[t], 0]])
        
        Rm_b[:,:,t] = np.matmul(Rm_b[:,:,t-1], expm((1/freq_KF)*omegam_b))

        accel_m[:,t] = Rm_b[:,:,t] @ accel_b_corrected[:,t]

        ###### integrate each quantity to get next navigation state ######
        # First we do attitude
        nav_state[0,t] = integrate(nav_state[0,t-1], gyro_corrected, t, freq_KF)
        # Then velocity
        nav_state[1,t] = integrate(nav_state[1,t-1], accel_m[0,:], t, freq_KF)
        nav_state[2,t] = integrate(nav_state[2,t-1], accel_m[1,:], t, freq_KF)
        # Then position 
        nav_state[3,t] = integrate(nav_state[3,t-1], nav_state[1,:], t, freq_KF)
        nav_state[4,t] = integrate(nav_state[4,t-1], nav_state[2,:], t, freq_KF)
    
    return nav_state