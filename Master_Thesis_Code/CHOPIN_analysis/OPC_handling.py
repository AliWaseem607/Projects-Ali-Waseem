# %%
import numpy as np
import pandas as pd

# %%

OPC = pd.read_csv("data/OPC_data_1.csv")
channel_sizes = np.zeros((30,))
for i in range(1, 31):
    channel_sizes[i - 1] = float(OPC.columns[i])  # Assume microns?

paer = 2.0
x = 1.1

# Calculate aerodynamic diameter

daer = channel_sizes * np.sqrt(paer / x)

startBin1 = 4
startBin2 = 10
endBin = 26  # 12; based on Georgia's paper GRIMM data

# Extract datetime information from original timetable
myDatetime = OPC.Time

# Compute the sum over the desired range of bins
sumVals = np.sum(OPC[:, startBin1:endBin], axis=0)
sumVals2 = np.sum(OPC[:, startBin2:endBin], axis=0)
