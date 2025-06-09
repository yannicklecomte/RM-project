# -*- coding: utf-8 -*-
"""
@author: Y. Lecomte
"""

#%% Packages, initialization and functions

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.signal import butter, filtfilt
from scipy import signal
import matplotlib.cm as cm
from scipy.integrate import odeint
from scipy.optimize import minimize

# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=18)   # X and Y axis label size

long_fig_size = (8,3)
square_fig_size = (5,5)
fig_size = (6,3.5)

auto_fitting = False
cutoff = 1

# ====== Define the model's constant ======
# ---- T vs PWM ----
Kt = 0.10083599643284652
T0 = -0.5338394640385178

# ---- RPM vs PWM ----
def pwm_to_RPM(pwm_map, coeffs=(0.01349794364264695, -1.8672816543885395, 151.8771526138152, 746.4757887495163)):
    return np.polyval(coeffs, pwm_map)

# ---- for the RPM ----
tau_RPM_UP = 0.1305
tau_RPM_DOWN = 0.1733
tau_d_RPM_UP = 0.035
tau_d_RPM_DOWN = 0.00001

# ---- for the Thrust ----
tau_T_UP = 0.0980
tau_T_DOWN = 0.1549
tau_d_T_UP = 0.06
tau_d_T_DOWN = 0.02

# ====== Selection of the constants ======
val_type = 'thrust'
if val_type == 'thrusdt':
    tau_UP = tau_T_UP
    tau_DOWN = tau_T_DOWN
    tau_d_UP = tau_d_T_UP 
    tau_d_DOWN = tau_d_T_DOWN 
else:
    tau_UP = tau_RPM_UP
    tau_DOWN = tau_RPM_DOWN
    tau_d_UP = tau_d_RPM_UP 
    tau_d_DOWN = tau_d_RPM_DOWN 

ref_pwm = 40

#%% FUNCTIONS

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)  # zero-phase distortion -> SP ;)

def map_pwm(x):
    return ((x-5)/5)*100

def motor_model_RPM(y, t, tau, gamma):
    dydt = (pwm_to_RPM(gamma) - y)/tau
    return dydt

def motor_model_thrust(y, t, tau, gamma):
    dydt = ((Kt*gamma+T0) - y)/tau
    return dydt

def gamma_model(y, t, tau_d, gamma_cmd):
    dydt = (gamma_cmd - y)/tau_d
    return dydt

def error_RPM(tau, t, u, y_real):
    y0 = y_real[0]
    y_pred = odeint(motor_model_RPM, y0, t, args=(u, tau)).flatten()
    err = np.mean((y_real - y_pred) ** 2)
    return err

def error_RPM_thrust(tau, t, u, y_real):
    y0 = y_real[0]
    y_pred = odeint(motor_model_thrust, y0, t, args=(u, tau)).flatten()
    err = np.mean((y_real - y_pred) ** 2)
    return err

#%% Load the data + filtering
data_dict = {}
test_dict = {}

DATA = "Motor_Model"
DATA_NAMES = os.listdir(DATA)

#extract the values of PWM from the files name in the folder
campaign_list = [re.findall(r"\d+", file) for file in DATA_NAMES]
campaign_list = [float(num) for sublist in campaign_list for num in sublist]


for campaign in campaign_list:
    print(f'Campaign {campaign:.0f}')
    data_dict[campaign] = {}
    CASE_PATH = DATA + os.sep + f"campaign_{campaign:.0f}"
    TEST_NAMES = os.listdir(CASE_PATH)
    PLT_PATH = CASE_PATH + os.sep + 'plots'
    if not os.path.exists(PLT_PATH):
        os.mkdir(PLT_PATH)
    
    #extract the values of PWM from the files name in the folder
    test_list = [re.findall(r"\d+", file) for file in TEST_NAMES]
    test_list = [float(num) for sublist in test_list for num in sublist]
    test_dict[campaign] = test_list
    
    # Add each data case of the campaign in a dictionnary
    for test in test_list:
        data_dict[campaign][test] = {}
        TEST_PATH = os.path.join(CASE_PATH,  f'test_{test:.0f}')
        print(TEST_PATH)
        filepath = os.path.join(TEST_PATH, "data.dat")
        data = np.loadtxt(filepath, skiprows=1) 
        # Extract values
        time = data[:,0] 
        thrust = data[:,1] 
        rpm = data[:,2]
        gamma = data[:,3] 
        # save all in the dictionnary
        data_dict[campaign][test] = np.column_stack((time, thrust, rpm, gamma))
        
#%% Plots

for campaign in campaign_list:
    for test in test_dict[campaign]:
        t =  data_dict[campaign][test][:, 0]
        thrust =  data_dict[campaign][test][:, 1]
        rpm =  data_dict[campaign][test][:, 2]
        gamma =  data_dict[campaign][test][:, 3]
        
        time_diffs = np.diff(t)
        fs = 1/np.mean(time_diffs); dt = 1/fs
        
        pred_thrust = np.zeros_like(rpm)
        pred_RPM = np.zeros_like(rpm)
        pred_gamma = np.zeros_like(rpm)

        if val_type == 'thrust':
            target_thrust = Kt*gamma + T0
            T_init = Kt*gamma[0] + T0
            gamma_init = gamma[0]
            tau = tau_UP
            tau_d = tau_d_UP
            for i in range(len(gamma)):
                t_span = [0, dt]  
                
                sol_g = odeint(gamma_model, gamma_init, t_span, args=(tau_d, gamma[i]))
                pred_gamma[i] = sol_g[-1]
                gamma_init = pred_gamma[i]
                
                sol = odeint(motor_model_thrust, T_init, t_span, args=(tau, pred_gamma[i]))
                pred_thrust[i] = sol[-1]  
                T_init = pred_thrust[i] 
                
                if pred_RPM[i] > pwm_to_RPM(gamma[i]):
                    tau = tau_DOWN
                    tau_d = tau_d_DOWN
                else: 
                    tau = tau_UP
                    tau_d = tau_d_UP
                    
            plt.figure(figsize=(10,6))
            plt.plot(t, thrust, label='Measured')
            plt.plot(t, pred_thrust, label='Predicted')
            plt.plot(t, target_thrust, label='Target')
            plt.title(f'For test nbr {test:.0f}')
            plt.xlabel('Time [s]')
            plt.ylabel('Thrust $T$ [N]')
            plt.grid(); plt.legend()
            fig_name = PLT_PATH + os.sep + f'motor_model_test_THRUST_{test:.0f}.png'
            plt.savefig(fig_name, dpi=400, bbox_inches='tight')
            plt.show()
            
            
        else:
            target_RPM = pwm_to_RPM(gamma)
            RPM_init = pwm_to_RPM(gamma[0])
            gamma_init = gamma[0]
            tau = tau_UP
            tau_d = tau_d_UP
            for i in range(len(gamma)):
                t_span = [0, dt]  
                
                sol_g = odeint(gamma_model, gamma_init, t_span, args=(tau_d, gamma[i]))
                pred_gamma[i] = sol_g[-1]
                gamma_init = pred_gamma[i]
                
                sol = odeint(motor_model_RPM, RPM_init, t_span, args=(tau, pred_gamma[i]))
                pred_RPM[i] = sol[-1]  
                RPM_init = pred_RPM[i] 
                
                if pred_RPM[i] > pwm_to_RPM(gamma[i]):
                    tau = tau_DOWN
                    tau_d = tau_d_DOWN
                else: 
                    tau = tau_UP
                    tau_d = tau_d_UP
                    
            plt.figure(figsize=(10,6))
            plt.plot(t, rpm, label='Measured')
            plt.plot(t, pred_RPM, label='Predicted')
            plt.plot(t, target_RPM, label='Target')
            plt.title(f'For test nbr {test:.0f}')
            plt.xlabel('Time [s]')
            plt.ylabel('RPM')
            plt.grid(); plt.legend()
            fig_name = PLT_PATH + os.sep + f'motor_model_RPM_test_{test:.0f}.png'
            plt.savefig(fig_name, dpi=400, bbox_inches='tight')
            plt.show()

#%%

T_init = Kt*ref_pwm + T0
tau = tau_UP
for i in range(len(u)):
    t_span = [0, dt]  
    sol = odeint(motor_model, T_init, t_span, args=(tau, u[i]))
    T[i] = sol[-1]  
    T_init = T[i] 
    if T[i] > (Kt * u[i] + T0):
        tau = tau_DOWN
    else: 
        tau = tau_UP
        
#%%

u_theo = Kt*u + T0
        
plt.figure(figsize=(5,4))
plt.plot(t, T)
plt.plot(t, u_theo)
plt.xlabel('Time [s]')
plt.ylabel('Predicted Thrust [N]')
plt.grid()
# fig_name = PLT_PATH + os.sep + f'.png'
# plt.savefig(fig_name, dpi=400, bbox_inches='tight')
plt.show()

