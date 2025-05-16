# This is the main document for lab 1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def white_noise(samples=200000, standard_dev=2):
     return np.random.normal(0,standard_dev, samples)

def random_walk(white_noise, initial=0):
    walker = np.zeros(white_noise.shape)
    walker[0]=initial +white_noise[0]
    for i in range(1, len(white_noise)):
        walker[i] = walker[i-1] + white_noise[i]
    return walker

def gauss_markhov(white_noise, cor_time, initial=0, delta=1, freq=True):
    # delta will be representative of the delta t in the formula for the Gauss Markhov
    # A frequency can also be put in in which case the delta t will be calculated below

    # if a direct delta t is entered then 
    if freq:
        delta_t = 1/delta
    else:
        delta_t = delta
    
    walker = np.zeros(white_noise.shape)

    walker[0] = initial
    for i in range(1, len(white_noise)):
        walker[i] = walker[i-1]*np.exp(-delta_t/cor_time) + white_noise[i]
    
    return walker

#Plotting function for noises
def plot_noise(noise, title, x_ax, y_ax, legend_labels, file_name, figure_size=(9,4.5)):
    plt.figure(figsize=figure_size)
    for i in range(len(noise)):
        plt.plot(noise[i], alpha=0.75, label=legend_labels[i])
    plt.title(title, size=18)
    plt.xlabel(x_ax, size=14)
    plt.ylabel(y_ax, size=14)
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(file_name)

def main(plot=False):
    #create lists to store all of the generated signals
    white_noises = []
    random_walks = []
    gauss_markhovs_500 = []
    gauss_markhovs_2000 = []

    #create the initial white noise
    for i in range(3):
        white_noises.append(white_noise(samples=200000, standard_dev= 2))
    
    #use those noises to then make the associated random walks and Gauss Markhovs
    for noise in white_noises:
        random_walks.append(random_walk(noise))
        gauss_markhovs_500.append(gauss_markhov(noise, 500))
        gauss_markhovs_2000.append(gauss_markhov(noise, 2000))
    
    #Collect all of the signals and save it as a text files to 8 decimal places
    all_files = []

    for noise_type in (white_noises, random_walks, gauss_markhovs_500, gauss_markhovs_2000):
        for noise in noise_type:
            all_files.append(noise)
    
    to_save = np.stack(all_files,axis=1)
    np.round(to_save, decimals=8)
    
    #This file is for the lab_1_B.py file to use
    np.savetxt('./generated_signals.txt', to_save)
    #This file is for the R_analysis.R file to use
    pd.DataFrame(to_save, columns=['WN_1', 'WN_2', 'WN_3', 'RW_1', 'RW_2', 'RW_3', 'GM_500_1', 'GM_500_2',
                                   'GM_500_3', 'GM_2000_1', 'GM_2000_2', 'GM_2000_3'] ).to_csv('./generated_signals.csv', sep=',', header=True)

    if plot:
        plot_noise(white_noises, 'White Noises', '', 'Amplitude', ['White Noise 1', 'White Noise 2', 'White Noise 3'],'./white_noise.png')
        plot_noise(random_walks, 'Random Walk Signal', '', 'Amplitude', ['Random Walk 1', 'Random Walk 2', 'Random Walk 3'],'./random_walk.png')
        plot_noise(gauss_markhovs_2000,'Gauss Markhovs with 2000 Correlation Time', '', 'Amplitude', ['Signal 1', 'Signal 2', 'Signal 3'],'./gm_2000.png')
        plot_noise(gauss_markhovs_500,'Gauss Markhovs with 500 Correlation Time', '', 'Amplitude', ['Signal 1', 'Signal 2', 'Signal 3'],'./gm_500.png')
    
if __name__ == "__main__":
    main(plot=True)