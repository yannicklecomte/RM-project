# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:16:38 2025

@author: Y. Lecomte
"""
#%% Initialization and packages
import numpy as np
import matplotlib.pyplot as plt
import os 
import re


# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=14)  
long_fig_size = (8,3)
square_fig_size = (5,5)
fig_size = (7,3)


#%% Target ====================================================================

def control_sequence(target_val, fs, T_step):
    # Extract the number of steps for the target list
    N_steps = len(target_val)
    
    # Define the size the target list
    samples_per_step = int(fs * T_step)
    total_samples = samples_per_step * N_steps

    # Time vector for initial step sequence
    time_list = np.linspace(0, T_step * N_steps, total_samples)
    target_list = np.zeros_like(time_list)

    # Step input
    for i in range(N_steps):
        target_list[i*samples_per_step:(i+1)*samples_per_step] = target_val[i]
        
    return target_list

# Step sequence definition
target_val = [0, 10, -10, 5, -5]  

# Sequence parameters
fs = 400      # Sampling frequency (Hz)
dt = 1 / fs   # Time step
T_step = 5    # Duration of each step (s)

# Create the full sequence
target_list = control_sequence(target_val, fs, T_step)

#%% Load the data

# Define the case parameters
Kp, Kd = 0.89, 0.52
Ne = 10 # number of episodes
alpha_p, alpha_d = 0.05, 0.5
data_path = os.path.join("Model_Opti_Controller_Sequences", f'Kp_{Kp}_Kd_{Kd}_Ne_{Ne}')

# Define empty list of savinf
all_data_list = []
err_list = []

# Over the number of episode per samples
for eps in range(Ne):
    file_path = os.path.join(data_path, f"episode_{eps}.csv")
   
    time_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 0]
    theta_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1]
    
    theta_list[0] = theta_list[1]
    
    all_data_list.append(theta_list)
    Ns = len(theta_list)
    err = np.linalg.norm(theta_list - target_list) / np.sqrt(Ns)
    err_list.append(err + alpha_p*Kp**2 + alpha_d*Kd**2)

# Average cost
err_mean = np.mean(err_list)

# Convert to numpy array of shape (Ne, time_points)
all_data_array = np.array(all_data_list)

# Compute ensemble average and standard deviation along axis 0 (episodes)
ensemble_avg = np.mean(all_data_array, axis=0)
ensemble_std = np.std(all_data_array, axis=0)

                             
# -------------------------------------------------------------------------
# PLOT 1: Current time serie
# -------------------------------------------------------------------------

plt.figure(figsize=(9, 3))
if Ne > 1:
    plt.fill_between(time_list, 
                     ensemble_avg - ensemble_std, 
                     ensemble_avg + ensemble_std, 
                     color='cyan', 
                     alpha=0.5, 
                     label='Standard deviation')

plt.plot(time_list, ensemble_avg, color='blue', label='Controlled balance', linewidth=2)
plt.plot(time_list, target_list, c='r', linestyle='--', label='Target')
plt.ylim(-20, 20)
plt.title('$\\mathcal{J}$' + f' = {err_mean:.2f}' + f' — Kp = {Kp:.2f}, Kd = {Kd:.2f}')
plt.xlabel('Time [s]')
plt.ylabel('$\\theta$ [deg]')
plt.legend()
plt.grid(True)
plt.tight_layout()
filename = os.path.join(data_path, 'mean_time_serie.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()

                         
# -------------------------------------------------------------------------
# PLOT 2: Plot the error evolution
# -------------------------------------------------------------------------

plt.figure(figsize=(4, 4))
plt.plot(err_list, linestyle='-', marker='*', c='blue', label='Cost')
plt.axhline(err_mean, linestyle='--', c='k',  label='Mean cost: $\\mathcal{J}$' + f' = {err_mean:.2f}')
plt.xlabel('Iteration')
plt.ylabel('Cost $\\mathcal{J}$')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
filename = os.path.join(data_path, 'cost_over_iter.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()