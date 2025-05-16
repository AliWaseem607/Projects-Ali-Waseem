import numpy as np


def gen_circle_data(radius, omega, freq, revs):

    # find out how many data points we shoud have
    sec_per_rev = 2*np.pi/omega
    num_data_points = int(np.ceil(sec_per_rev*freq*revs))

    # create the data
    gyro = np.ones(num_data_points)*omega
    accel_x1 = np.zeros(num_data_points)
    accel_x2 = np.ones(num_data_points)*radius*(omega*omega)

    #return the data
    return gyro, accel_x1, accel_x2


def gen_theoretical_circle_PVA(radius, omega, freq, revs, a_init, reverse=False):

    # Find out how long of an array we need
    sec_per_rev = 2*np.pi/omega
    num_data_points = int(np.ceil(sec_per_rev*freq*revs))
    rad_init = a_init
    
    # create an array with the radian locations of the datapoints
    if not reverse:
        rad_end = rad_init + (2*np.pi*revs)
    else:
        rad_end = rad_init - (2*np.pi*revs)
    radians = np.linspace(rad_init, rad_end, num_data_points+1)

    # Use radian array to find x and y positions
    pos_n = np.cos(radians)*radius
    pos_e = np.sin(radians)*radius

    pos = np.vstack([pos_n,pos_e])

    vel_n = -radius*omega*np.sin(radians)
    vel_e = radius*omega*np.cos(radians)
    vel = np.vstack([vel_n, vel_e])

    attitude = np.linspace(a_init,2*np.pi*revs, num_data_points+1)

    return pos, vel, attitude