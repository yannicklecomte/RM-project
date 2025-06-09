# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:51:27 2025

@author: lyann
"""

#%% Import packages ===========================================================

import numpy as np
import matplotlib.pyplot as plt
import os  

# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=14)  
long_fig_size = (8,3)
square_fig_size = (5,5)
fig_size = (7,3)

single_data_path = os.path.join('DATA_time_series', 'single', 'iteration_12')
multi_data_path = os.path.join('DATA_time_series', 'multi', 'iteration_18')
Ne = 4
alpha_p, alpha_d = 0.05, 0.5

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


#%% Load the time serie data of the SINGLE ==================================== 

BO_input_path = os.path.join(single_data_path, 'BO_inputs.npz')
BO_inputs = np.load(BO_input_path)
x_next = BO_inputs['x_next']
Kp, Kd = x_next

all_single_list = []
err_list_single = []

for eps in range(Ne):
    file_path = os.path.join(single_data_path, f"episode_{eps}.csv")
   
    time_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 0]
    theta_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1]
    
    theta_list[0] = theta_list[1]
    
    all_single_list.append(theta_list)
    Ns = len(theta_list)
    err = np.linalg.norm(theta_list - target_list) / np.sqrt(Ns)
    err_list_single.append(err + alpha_p*Kp**2 + alpha_d*Kd**2)
    
# Compute the mean cost
mean_err_single = np.mean(err_list_single)

# Convert to numpy array of shape (Ne, time_points)
all_single_array = np.array(all_single_list)

# Compute ensemble average and standard deviation along axis 0 (episodes)
ensemble_avg_single = np.mean(all_single_array, axis=0)
ensemble_std_single = np.std(all_single_array, axis=0)

#%% Load the time serie data of the MULTI ==================================== 

BO_input_path = os.path.join(multi_data_path, 'BO_inputs.npz')
BO_inputs = np.load(BO_input_path)
x_next = BO_inputs['x_next']
Kp, Kd = x_next

all_multi_list = []
err_list_multi = []

for eps in range(Ne):
    file_path = os.path.join(multi_data_path, f"episode_{eps}.csv")
   
    time_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 0]
    theta_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1]
    
    theta_list[0] = theta_list[1]
    
    all_multi_list.append(theta_list)
    Ns = len(theta_list)
    err = np.linalg.norm(theta_list - target_list) / np.sqrt(Ns)
    err_list_multi.append(err + alpha_p*Kp**2 + alpha_d*Kd**2)
    
# Compute the mean cost
mean_err_multi = np.mean(err_list_multi)

# Convert to numpy array of shape (Ne, time_points)
all_multi_array = np.array(all_multi_list)

# Compute ensemble average and standard deviation along axis 0 (episodes)
ensemble_avg_multi = np.mean(all_multi_array, axis=0)
ensemble_std_multi = np.std(all_multi_array, axis=0)


#%% Plot ======================================================================

# Initialize the figure
plt.figure(figsize=(8, 3)) 

# Plot the standard deviation band if needed
plt.plot(time_list, ensemble_avg_multi, c='blue', label='Multi-Fidelity: $\\mathcal{J}$' + f' = {mean_err_multi:.2f}')
plt.fill_between(time_list, 
                ensemble_avg_multi - ensemble_std_multi, 
                ensemble_avg_multi + ensemble_std_multi, 
                color='cyan', 
                alpha=0.5, 
                label='Standard deviation')

plt.plot(time_list, ensemble_avg_single, c='red', label='Single-Fidelity:  $\\mathcal{J}$' + f' = {mean_err_single:.2f}')
plt.fill_between(time_list, 
                ensemble_avg_single - ensemble_std_single, 
                ensemble_avg_single + ensemble_std_single, 
                color='salmon', 
                alpha=0.5, 
                label='Standard deviation')

# Plot main lines
plt.plot(time_list, target_list, c='red', linestyle='--')

# ax.set_title(r'$\mathcal{J}$' + f' = {err:.2f} — Kp = {x_next[0]:.2f}, Kd = {x_next[1]:.2f}')
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta$ [deg]')
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
plt.legend()

# Save and show
plt.tight_layout()
filename = os.path.join('DATA_time_series', 'single_multi_comp.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
     