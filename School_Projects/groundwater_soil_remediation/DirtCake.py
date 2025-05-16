import numpy as np
import pandas as pd

def flow_rate(H, k, u, Pw, Rw, RI, PRI):
    a = H*np.pi*k/u
    b = Pw/(np.log(Rw/RI))
    c = 1-(PRI/Pw)**2
    return a*b*c

class DirtLayer:
    def __init__(self, R_inf, R_well, H, depth, density, por, por_water, temp, k, u, Pw, intial_concs, compound_data):
        self.volume = depth*(R_inf**2-R_well**2)
        self.soil_mass = self.volume * density
        self.water_volume = self.volume * por * por_water
        self.gas_volume = self.volume * por *(1-por_water)
        self.temp = temp # [K]

        # handle the dictionaries
        self.C_init = intial_concs
        P_vap_arr_temp = []
        KH_arr_temp = []
        KD_arr_temp = []
        molar_arr_temp = []
        for key in self.C_init.keys():
            P_vap_arr_temp.append(compound_data[key]['P_vap'])
            KH_arr_temp.append(compound_data[key]['KH'])
            KD_arr_temp.append(compound_data[key]['KD'])
            molar_arr_temp.append(compound_data[key]['molar'])
        self.P_vap = np.array(P_vap_arr_temp)
        self.KH = np.array(KH_arr_temp)
        self.KD = np.array(KD_arr_temp)
        self.M = np.array(molar_arr_temp)
        self.R = 8.3145 # [J/mol/K]
        self.Q = flow_rate(H, k, u, Pw, R_well, R_inf, 101325) * 60*60*24 # [m^3/day]

    def SVE(self, time, dt):
        self.dt = dt
        # time and dt are in days
        # create numpy arrays to hold data, (data type, contamiants, data type)
        # data types: 0:m_tot, 1:C_s, 2:C_l, 3:C_g, 4:n, 5:mole fraction, 6:max concetration in gas phase, 7:measured conc in gas, 8:m_g_removed
        loops = int(np.ceil(time/dt))
        self.results = np.zeros((9, len(self.C_init), loops+1)) # set up holder for data
        self.results[1, :, 0] = list(self.C_init.values()) # inital solid concentrations [mg/kg]
        self.results[2, :, 0] = np.divide(self.results[1, :, 0], self.KD) # initial water concentrations
        self.results[3, :, 0] = np.multiply(self.results[2, :, 0 ], self.KH) # initial air concentrations

        # get intial total mass of contaminants [mg]
        self.results[0, :, 0] = self.results[1,:,0]*self.soil_mass + self.results[2,:,0]*self.water_volume + self.results[3,:,0]*self.gas_volume
        
        for t in range(loops):
            # get inital moles of contaminants
            self.results[4,:,t] = np.divide(self.results[0,:,t]/1000,self.M) # convert mg to g

            # get mole fractions
            n_tot = np.sum(self.results[4,:,t])
            self.results[5,:,t] = self.results[4,:,t]/n_tot

            # get max concentrations by using ideal gas law m/V = P_vap*n_frac*molar/R/T*1000[mg/g] ideal gas law gives 
            self.results[6,:,t] = np.multiply(np.multiply(self.P_vap, self.results[5,:,t]), self.M)/self.R/self.temp*1000

            # compare equilibrium concentration and max concentration to get actual concentration in gas phase [mg]
            self.results[7,:,t] = np.min(self.results[[3,6],:,t],axis=0)
            self.results[8,:,t] = self.results[7,:,t] *self.Q *dt # [mg/m^3]*[m^3/day]*[day] ->[mg]
            # for idx in range(self.results.shape[1]):
            #     self.results[7,idx,t] = np.min(self.results[3,idx,t],self.results[6,idx,t]) * self.Q *dt
            
            # update mass for next time step
            self.results[0,:,t+1] = self.results[0,:,t] - self.results[8,:,t]

            # update concentrations for new mass starting with liquid
            denom = self.KD*self.soil_mass + self.KH*self.gas_volume +self.water_volume # formula for C_i_aq
            self.results[2,:,t+1] = np.divide(self.results[0,:,t+1], denom) # liquid concentrations
            self.results[1,:,t+1] = np.multiply(self.results[2,:,t+1],self.KD) # solid concentrations
            self.results[3,:,t+1] = np.multiply(self.results[2,:,t+1], self.KH) # gas concentrations

    def describe(self):
        print('The concentrations are in the following order')
        i= 1
        for key in self.C_init.keys():
            print(f'{i}: {key}')
            i+1
        print()
        print('DataFrame indicies are: ')
        print('0: total mass\n1:')

    def to_df(self, index):
        cols = self.C_init.keys()
        return pd.DataFrame(self.results[index,:,:],columns=cols)
    
    def get_keys(self):
        return list(self.C_init.keys())