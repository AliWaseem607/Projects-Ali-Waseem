import numpy as np
from lab_4_updated import gen_theoretical_circle_PVA, gen_circle_data
from lab_1_A import white_noise, gauss_markhov
from helpers import *
from INU_functions import integrate
from scipy.linalg import expm
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from Other_navigation import INU


def Kalman_filter_final(nav_init, freq_gps, freq_KF, pos_gps, R, P_init,  W, F22, accel_x1_data, accel_x2_data, gyro_data, error_feedback_level=0, cut=None):
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

    # keep track of differences before reset
    dz = np.full(gps_spread.shape,np.nan)

    #################### Start the Navigation ####################
    '''
    So the process here will be to first integrate the next position, to find out 
    where we should be after that we use the Kalman Filter to predict how much
    our integrator is off by. Once there is a gps measurement we make an update
    in the KF and then with the predicted difference we move our nav state, if 
    if we are doing errors we also correct the gyro and accel errors
    '''
    ###### do cut if we want that #######
    
    if cut != None:
        print('cutting')
        x = cut[1]-cut[0]
        gps_spread = np.hstack([gps_spread[:,:cut[0]], np.full((2,x),np.nan), gps_spread[:,cut[1]:]])


    for t in tqdm(range(1, iterations, 1)):
        # first "subtract" the error from the measurments
        accel_b_corrected[:,t] = accel_b[:,t] + errors[2:4]
        gyro_corrected[t] = gyro[t] + errors[0] + errors[1]

        # rotate the corrected accerlation into the mapping frame
        omegam_b = np.array([
            [0, -1*gyro_corrected[t]],
            [gyro_corrected[t], 0]])
        
        #Rm_b[:,:,t] = np.matmul(Rm_b[:,:,t-1], expm((1/freq_KF)*omegam_b))
        nav_state[0,t] = integrate(nav_state[0,t-1], gyro_corrected, t, freq_KF)
        a = nav_state[0,t]

        Rm_b[:,:,t] = np.array([
                    [np.cos(a), -1*np.sin(a)],
                    [np.sin(a), np.cos(a)]])

        accel_m[:,t] = Rm_b[:,:,t] @ accel_b_corrected[:,t]

        ###### integrate each quantity to get next navigation state ######
        # First we do attitude
        #nav_state[0,t] = integrate(nav_state[0,t-1], gyro_corrected, t, freq_KF)
        # Then velocity
        nav_state[1,t] = integrate(nav_state[1,t-1], accel_m[0,:], t, freq_KF)
        nav_state[2,t] = integrate(nav_state[2,t-1], accel_m[1,:], t, freq_KF)
        # Then position 
        nav_state[3,t] = integrate(nav_state[3,t-1], nav_state[1,:], t, freq_KF)
        nav_state[4,t] = integrate(nav_state[4,t-1], nav_state[2,:], t, freq_KF)
        # Then attitude
        

        ###### Now we run the Kalman Filter ######
        # first step is get our F and G matrix
        # f2_m = accel_b_corrected[0,t]*np.sin(nav_state[0,t]) +accel_b_corrected[1,t]*np.cos(nav_state[0,t]) 
        # f1_m = accel_b_corrected[0,t]*np.cos(nav_state[0,t]) -accel_b_corrected[1,t]*np.sin(nav_state[0,t])
        F = get_F_mat(accel_m[0,t], accel_m[1,t], nav_state[0,t], F22)
        # F = get_F_mat(f1_m, f2_m, nav_state[0,t], F22)
        G = get_G_mat(nav_state[4,t]) # supposed to be the attitude not position change to nav_state[0,t]

        # Now to get our phi and Q matrix
        Phi, Q = get_Phi_Q_mat(F, G, W, 1/freq_KF)

        #predict next diff_state and P
        diff_state_pred[:,t] = Phi @ diff_state[:,t-1]
        P_pred[:,:,t] = Phi @ P[:,:,t-1] @ Phi.T + Q

        if np.any(np.isnan(gps_spread[:,t])): #if we do not have an update
            # update the state
            diff_state[:,t] = diff_state_pred[:,t]
            # update the covariance
            P[:,:,t] = P_pred[:,:,t]
        else:
            # caluclate z
            dz[:,t] = gps_spread[:,t] - nav_state[3:5, t] 
            # next calculate the gain weight
            K[:,:,t] = P_pred[:,:,t] @ H.T @ (np.linalg.inv(H @ P_pred[:,:,t] @ H.T + R))
            # next update the state
            diff_state[:,t] = diff_state_pred[:,t] + K[:,:,t] @ (dz[:,t] - H @ diff_state_pred[:,t])
            #update the covariance
            P[:,:,t] = (np.identity(P.shape[0]) - K[:,:,t] @ H) @ P_pred[:,:,t]

            # if we are doing an extended Kalman Filter
            if error_feedback_level>0:
                nav_state[:5, t] = nav_state[:5, t] + diff_state[:5,t]
                # reset the diff_state
                diff_state[:5,t] = np.zeros(5)
                
        if error_feedback_level>1:
            errors = errors + diff_state[5:, t] #check the error adding
            diff_state[5:,t] = np.zeros(4)


    '''
    To check you can run without error feedback ad take a look at if your biases are ending up as the same as the ones
    that are set in the function. Look at sensor fusion and how it is done
    '''

    print(errors)
    return {'nav_state':nav_state, 'diff_state': diff_state, 'errors':errors, 'P':P, 'K':K, 'dz': dz}









    

def main():
    print('------------------------')
    print('starting lab 10')
    print('------------------------')
    print()
    

    # create folder for plots
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    cmap = plt.get_cmap("tab10")
    ############ Set up Inital Data for IMU Signals ############
    print('Setting up signal variables...')
    # physical location data
    radius = 500 # m
    omega = np.pi/100 # rad/s
    psy_init = 0

    # signal data
    freq_instrument = 100 # Hz
    freq_gps = 0.5 # Hz

    # noise data
    gyro_bias = -400 /3600 /180*np.pi   # deg/h -> rad/s
    gyro_GM = 0.01 *np.pi/180              # deg/s/Hz -> rad/s/Hz
    gyro_GM_corr = 30                       # 1/s (1/beta)
    gyro_wn = 0.1 /np.sqrt(3600) *np.pi/180 # deg/h^0.5 -> rad/s^0.5

    accel_wn = 50 *9.81 *1e-6              # ug/Hz -> m/s^2/Hz
    accel_GM = 200 *9.81 *1e-6             # ug/Hz -> m/s^2/Hz
    accel_GM_corr = 60                      # 1/s (1/beta)
    accel_GM_init_1 = -100 *9.81 *1e-6     # ug -> m/s^2
    accel_GM_init_2 = 200 *9.81 *1e-6      # ug -> m/s^2

    gps_std = 1                             # m

    

    ############ Create Signal Data ############
    print('Creating signal data...')
    # generate the clean accelerometer, gyro, and GPS data
    gyro_clean, accel_x1_clean, accel_x2_clean = gen_circle_data(radius, omega, freq_instrument, revs=1)
    pos_ne, vel, attitude = gen_theoretical_circle_PVA(radius, omega, freq_gps, revs=1, a_init=psy_init, reverse=False)


    # generate the noise for each instrument
    gyro_bias_noise = random_bias(gyro_bias)
    gyro_GM_noise = gauss_markhov(white_noise(samples = len(gyro_clean),standard_dev= gyro_GM),
                                   cor_time=gyro_GM_corr,
                                   initial=0,
                                   delta= freq_instrument,
                                   freq=True)
    gyro_wn_noise = white_noise(samples= len(gyro_clean), standard_dev= gyro_wn)

    accel_x1_GM_noise = gauss_markhov(white_noise(samples=len(accel_x1_clean), standard_dev=accel_GM),
                                      cor_time= accel_GM_corr,
                                      initial=accel_GM_init_1,
                                      delta=freq_instrument,
                                      freq=True)
    accel_x1_wn = white_noise(samples=len(accel_x1_clean), standard_dev=accel_wn)

    accel_x2_GM_noise = gauss_markhov(white_noise(samples=len(accel_x2_clean), standard_dev=accel_GM),
                                      cor_time= accel_GM_corr,
                                      initial=accel_GM_init_2,
                                      delta=freq_instrument,
                                      freq=True)
    accel_x2_wn = white_noise(samples=len(accel_x2_clean), standard_dev=accel_wn)

    gps_wn_n = white_noise(samples=pos_ne.shape[1], standard_dev=gps_std)
    gps_wn_e = white_noise(samples=pos_ne.shape[1], standard_dev=gps_std)

    # generate the IMU data to be used
    gyro = gyro_clean + gyro_bias_noise + gyro_GM_noise + gyro_wn_noise
    accel_x1 = accel_x1_clean + accel_x1_GM_noise + accel_x1_wn
    accel_x2 = accel_x2_clean + accel_x2_GM_noise + accel_x2_wn
    gps = pos_ne + np.vstack([gps_wn_n,gps_wn_e])

    ############ Set up Inital Data for Navigation ############
    print('Setting up inital variables for navigation...')
    pos_init = np.array([radius, 0])
    v_init = np.array([0,omega*radius])
    gyro_misalign = 3 *np.pi/180 # deg -> rad
    vel_error_n = -2 # m/s
    vel_error_e = 1 # m/s
    nav_errors  = np.array([gyro_misalign, vel_error_n, vel_error_e, 0, 0]) 
    nav_init = np.hstack([np.array([psy_init+np.pi/2]), v_init, gps[:,0] ]) + nav_errors

    ############ Creating inital variables for Kalman Filter ############
    print('Setting up inital variables for Kalman Filter...')
    R = np.array([[gps_std**2, 0],
                  [0, gps_std**2]])

    # inital uncertainties            da,    dvn  dve  dpn  dpe        bc                bg             bA1                bA2
    P_init = np.square(np.diag([2*np.pi/180,  5,   5,  10,  10,   0.05*np.pi/180,  0.01*np.pi/180,  300*9.81*10e-6,   300*9.81*10e-6]))

    F22 = np.diag([0, -1/gyro_GM_corr, -1/accel_GM_corr, -1/accel_GM_corr ])

    cor_test = True
    if cor_test:
        F22 = F22/100

    W = np.diag([gyro_wn**2, accel_wn**2, accel_wn**2, (2/gyro_GM_corr)*gyro_GM**2, (2/accel_GM_corr)*accel_GM**2, (2/accel_GM_corr)*accel_GM**2])

    print('Beginning Kalman Filter...')
    error_level_1= Kalman_filter_final(nav_init=nav_init,
                                    freq_gps=freq_gps,
                                    freq_KF=freq_instrument,
                                    pos_gps=gps,
                                    R=R,
                                    P_init=P_init,
                                    W=W,
                                    F22=F22,
                                    accel_x1_data=accel_x1,
                                    accel_x2_data=accel_x2,
                                    gyro_data=gyro,
                                    error_feedback_level=1)
    
    error_level_2 = Kalman_filter_final(nav_init=nav_init,
                                    freq_gps=freq_gps,
                                    freq_KF=freq_instrument,
                                    pos_gps=gps,
                                    R=R,
                                    P_init=P_init,
                                    W=W,
                                    F22=F22,
                                    accel_x1_data=accel_x1,
                                    accel_x2_data=accel_x2,
                                    gyro_data=gyro,
                                    error_feedback_level=2)
    
    error_level_1_cut= Kalman_filter_final(nav_init=nav_init,
                                    freq_gps=freq_gps,
                                    freq_KF=freq_instrument,
                                    pos_gps=gps,
                                    R=R,
                                    P_init=P_init,
                                    W=W,
                                    F22=F22,
                                    accel_x1_data=accel_x1,
                                    accel_x2_data=accel_x2,
                                    gyro_data=gyro,
                                    error_feedback_level=1,
                                    cut=(10000,15000))
    
    error_level_2_cut = Kalman_filter_final(nav_init=nav_init,
                                    freq_gps=freq_gps,
                                    freq_KF=freq_instrument,
                                    pos_gps=gps,
                                    R=R,
                                    P_init=P_init,
                                    W=W,
                                    F22=F22,
                                    accel_x1_data=accel_x1,
                                    accel_x2_data=accel_x2,
                                    gyro_data=gyro,
                                    error_feedback_level=2,
                                    cut=(10000,15000))
    
    # error_level_0 = Kalman_filter_final(nav_init=nav_init,
    #                                 freq_gps=freq_gps,
    #                                 freq_KF=freq_instrument,
    #                                 pos_gps=gps,
    #                                 R=R,
    #                                 P_init=P_init,
    #                                 W=W,
    #                                 F22=F22,
    #                                 accel_x1_data=accel_x1,
    #                                 accel_x2_data=accel_x2,
    #                                 gyro_data=gyro,
    #                                 error_feedback_level=0)
    
    
    print('Finished filtering')

    print('Making reference trajectories')
    pos_ne_100, vel_100, attitude_100 = gen_theoretical_circle_PVA(radius, omega, freq_instrument, 1, psy_init)

    print('Beginning plotting...')
    plotting_time = True
    if plotting_time:
        plt.figure(figsize=(5,5))
        plt.plot(pos_ne[1,:], pos_ne[0,:], 'o', label='Reference Trajectory', alpha = 1)
        plt.plot(gps[1,:], gps[0,:], 'o', label='GPS postitions', color = 'k', alpha=0.5)
        plt.plot(error_level_1['nav_state'][4,:], error_level_1['nav_state'][3,:], color ='r')
        plt.legend()
        plt.ylabel('North Axis')
        plt.xlabel('East Axis')
        plt.title('Trajectory for Kalman Filter fixing\nattitude, velocity, and position')
        plt.tight_layout()
        plt.savefig('./KF_1.png', dpi=400)


        plt.figure(figsize=(5,5))
        plt.plot(pos_ne[1,:], pos_ne[0,:], 'o', label='Reference Trajectory', alpha = 1)
        plt.plot(gps[1,:], gps[0,:], 'o', label='GPS postitions', color = 'k', alpha=0.5)
        plt.plot(error_level_2['nav_state'][4,:], error_level_2['nav_state'][3,:], color ='r')
        plt.ylabel('North Axis')
        plt.xlabel('East Axis')
        plt.title('Trajectory for Kalman Filter fixing\nattitude, velocity, position, and errors')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./KF_2.png', dpi=400)


        plt.figure(figsize=(7,5))
        plt.plot(pos_ne[1,:], pos_ne[0,:], 'o', label='Reference Trajectory', alpha = 1)
        plt.plot(gps[1,:50], gps[0,:50], 'o', label='GPS postitions', color = 'k', alpha=0.5)
        plt.plot(gps[1,75:], gps[0,75:], 'o', label='GPS postitions', color = 'k', alpha=0.5)
        plt.plot(error_level_1_cut['nav_state'][4,:], error_level_1_cut['nav_state'][3,:], color ='r')
        plt.legend()
        plt.ylabel('North Axis')
        plt.xlabel('East Axis')
        plt.title('Trajectory for Kalman Filter fixing\nattitude, velocity, and position')
        plt.tight_layout()
        plt.savefig('./KF_1_cut.png', dpi=400)


        plt.figure(figsize=(7,5))
        plt.plot(pos_ne[1,:], pos_ne[0,:], 'o', label='Reference Trajectory', alpha = 1)
        plt.plot(gps[1,:50], gps[0,:50], 'o', label='GPS postitions', color = 'k', alpha=0.5)
        plt.plot(gps[1,75:], gps[0,75:], 'o', label='GPS postitions', color = 'k', alpha=0.5)
        plt.plot(error_level_2_cut['nav_state'][4,:], error_level_2_cut['nav_state'][3,:], color ='r')
        plt.ylabel('North Axis')
        plt.xlabel('East Axis')
        plt.title('Trajectory for Kalman Filter fixing\nattitude, velocity, position, and errors')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./KF_2_cut.png', dpi=400)

    print('Beginning numerical analysis...')
    analysis(error_level_1, 'error level 1', pos_ne_100, vel_100, attitude_100)
    analysis(error_level_2, 'error level 2', pos_ne_100, vel_100, attitude_100)



def analysis(dict_arrs, name, pos_ne, vel, attitude):
    print('Beginning numerical analysis...')
    print(name)
    # gps_1 = 199
    # gps_2 = 199+200
    # gps_10 = 2000-1
    # gps_9 = gps_10-200
    # gps_11 = gps_10+200

    nav_state = dict_arrs['nav_state']
    pos_error = nav_state[3:5,:]

    print('error before 1')
    accuracy = np.mean(np.square(pos_ne[:,:199] - pos_error[:,:199]))
    print(accuracy)
    print('error after 1')
    accuracy = np.mean(np.square(pos_ne[:,199:399] - pos_error[:,199:399]))
    print(accuracy)

    print('error before 10')
    accuracy = np.mean(np.square(pos_ne[:,1799:1999] - pos_error[:,1799:1999]))
    print(accuracy)

    print('error after 10')
    accuracy = np.mean(np.square(pos_ne[:,1999:2199] - pos_error[:,1999:2199]))
    print(accuracy)

    print()
    print('def2')
    print('error before 1')
    accuracy = np.mean(np.square(pos_ne[:,:199] - pos_error[:,:199]))
    print(accuracy)
    print('error after 1')
    accuracy = np.mean(np.square(pos_ne[:,199:] - pos_error[:,199:]))
    print(accuracy)

    print('error before 10')
    accuracy = np.mean(np.square(pos_ne[:,:1999] - pos_error[:,:1999]))
    print(accuracy)

    print('error after 10')
    accuracy = np.mean(np.square(pos_ne[:,1999:] - pos_error[:,1999:]))
    print(accuracy)
    print()
    print()
    

    




if __name__ == '__main__':
    main()










