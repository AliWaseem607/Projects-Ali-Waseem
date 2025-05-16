import numpy as np




def first_order_integrate(start, delta_y, delta_x):
    return start + delta_y*delta_x

def second_order_integrate(start, delta_y_new, delta_y_old, delta_x):
    return start + 0.5*(delta_y_old+ delta_y_new)*delta_x


def integrate(start, slope_arr, k, freq, int_order=2):

    if int_order == 1 or k < 2:
        return first_order_integrate(start, slope_arr[k], 1/freq)
    elif int_order == 2 or k < 3:
        return second_order_integrate(start, slope_arr[k], slope_arr[k-1], 1/freq)
    else:
        print('error in Integration. Integration order is not supported')
