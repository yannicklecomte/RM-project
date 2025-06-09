# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:00:21 2025

@author: Y. Lecomte
"""

#%% Import packages ===========================================================

import numpy as np
import matplotlib.pyplot as plt
import os 
import MFGP_BO_functions as fct
from scipy.stats import norm


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

    
#%% Load AND plot the data =============================================================    

# -------------------------------------------------------------------------
# Define the case to post process
# -------------------------------------------------------------------------

# Initial params
Ne_init = 4   # Number of episodes
Ns_init_HF = 5
Ns_init_LF = 200

# Exploration params
Ne_iter = 4
Ns_iter = 20

# Exploration/Exploitation factor
xi = 0.01

campaign = f"INIT_Ns_L_{Ns_init_LF}_Ns_H_{Ns_init_HF}_Ne_{Ne_init}_ITER_Ns_{Ns_iter}_Ne_{Ne_iter}_xi_{xi}"
data_path = os.path.join("PD_SEQUENCE", f'{campaign}', 'data')


# Define the path where the time series plots will be saved
seq_path = os.path.join('PD_SEQUENCE', f'{campaign}', 'plots', 'Sequences')
os.makedirs(seq_path, exist_ok = True)

# Define the path where the cost evolution plots will be saved
cost_path = os.path.join('PD_SEQUENCE', f'{campaign}', 'plots', 'Cost')
os.makedirs(cost_path, exist_ok = True)
    
# Define the path where the EI plots will be saved
ei_path = os.path.join('PD_SEQUENCE', f'{campaign}', 'plots', 'EI')
os.makedirs(ei_path, exist_ok = True)
    
# Penalty coefficients for the cost evaluation
alpha_p, alpha_d = 0.05, 0.5

# -----------------------------------------------------------------------------
# Boundaries of the domain
xmin_kp, xmax_kp = 0.1, 2 
xmin_kd, xmax_kd = 0.1, 1.5 
bounds = ([xmin_kp, xmax_kp], [xmin_kd, xmax_kd])
dx = xmax_kp - xmin_kp
dy = xmax_kd - xmin_kd
area = (xmax_kp - xmin_kp) * (xmax_kd - xmin_kd)
# Resolution of grid for the evaluation of the EI
resolution = 80
# Generate a dense gid data set for the plot
kp = np.linspace(bounds[0][0], bounds[0][1], resolution)
kd = np.linspace(bounds[1][0], bounds[1][1], resolution)
KP, KD = np.meshgrid(kp, kd)
X_grid = np.column_stack((KP.ravel(), KD.ravel()))

# Iinitialize saving list
EI_int_list = []
cost_over_time = []

for iter_ in range(Ns_iter):
    # List to store the all of all epsiode for a given iteration
    all_data_list = []
    err_list = []
    
    # -------------------------------------------------------------------------
    # Load of the data of the current iteration
    # -------------------------------------------------------------------------
    
    # Import all saved data from the BO iterations
    BO_input_path = os.path.join(data_path, f'iteration_{iter_}', 'BO_inputs.npz')
    BO_inputs = np.load(BO_input_path)
    # New evaluation location
    x_next = BO_inputs['x_next']
    Kp, Kd = x_next
    # Load of GPR 
    L, alpha = BO_inputs['L'], BO_inputs['alpha'] 
    X_H = BO_inputs['X_H']; X_L = BO_inputs['X_L']
    y_best = BO_inputs['y_best']
    # Load the hyper-parameters
    params_path = os.path.join(data_path, 'HPO_data', f'Iteration_{iter_+1:02d}_param_history.npy')
    params_opt = np.load(params_path)[-1, :]
    rho, theta = params_opt[0], params_opt[1:]

    # Over the number of episode per samples
    for eps in range(Ne_iter):
        file_path = os.path.join(data_path, f'iteration_{iter_}', f"episode_{eps}.csv")
       
        time_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 0]
        theta_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1]
        
        theta_list[0] = theta_list[1]
        
        all_data_list.append(theta_list)
        Ns = len(theta_list)
        err = np.linalg.norm(theta_list - target_list) / np.sqrt(Ns)
        err_list.append(err + alpha_p*Kp**2 + alpha_d*Kd**2)


    # Convert to numpy array of shape (Ne, time_points)
    all_data_array = np.array(all_data_list)

    # Compute ensemble average and standard deviation along axis 0 (episodes)
    ensemble_avg = np.mean(all_data_array, axis=0)
    ensemble_std = np.std(all_data_array, axis=0)
    
    err = np.mean(err_list)
    cost_over_time.append(err)
                          
    # Posterior prediction on the dense grid
    mu, std = fct.mf_gp_predict(X_grid, X_L, X_H, rho, theta, alpha, L)

    # EI computation over the dense grid
    Z = (y_best - mu - xi) / std
    EI = (y_best - mu - xi) * norm.cdf(Z) + std * norm.pdf(Z)
    EI[std == 0.0] = 0.0
    EI_integral = np.sum(EI) * dx * dy / (resolution*resolution)

    EI_int_list.append(EI_integral)
                             
    # -------------------------------------------------------------------------
    # PLOT 1: Current time serie
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(9, 3))
    if Ne_iter > 1:
        plt.fill_between(time_list, 
                         ensemble_avg - ensemble_std, 
                         ensemble_avg + ensemble_std, 
                         color='cyan', 
                         alpha=0.5, 
                         label='Standard deviation')

    plt.plot(time_list, ensemble_avg, color='blue', label='Controlled balance', linewidth=2)
    plt.plot(time_list, target_list, c='r', linestyle='--', label='Target')
    plt.ylim(-20, 20)
    plt.title('$\\mathcal{J}$' + f' = {err:.2f}' + f' — Kp = {x_next[0]:.2f}, Kd = {x_next[1]:.2f}')
    plt.xlabel('Time [s]')
    plt.ylabel('$\\theta$ [deg]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(seq_path, f'mean_sequence_iter_{iter_:03d}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # -------------------------------------------------------------------------
    # PLOT 2: Cost value evolution
    # -------------------------------------------------------------------------

    plt.figure(figsize=(4, 4))
    plt.plot(cost_over_time, linestyle='-', marker='*', c='blue', label='Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost $\\mathcal{J}$')
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
    filename = os.path.join(cost_path, f'cost_over_time_iter_{iter_}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # -------------------------------------------------------------------------
    # PLOT 3: EI value evolution
    # -------------------------------------------------------------------------
    
    plt.figure(figsize=(4, 4))
    plt.plot(EI_int_list, marker='*', linestyle='-', color='blue', linewidth=1.8)
    plt.xlabel('Iterations')
    plt.ylabel(r'$\int \mathrm{EI} \, dxdy$')
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
    filename = os.path.join(ei_path, f'iteration_{iter_}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    

# -------------------------------------------------------------------------
# Save the list of EI and Cost
# -------------------------------------------------------------------------  

EI_array = np.array(EI_int_list)
Cost_array = np.array(cost_over_time)

np.save(os.path.join(data_path, 'EI_array.npy'), EI_array)
np.save(os.path.join(data_path, 'Cost_array.npy'), Cost_array)
