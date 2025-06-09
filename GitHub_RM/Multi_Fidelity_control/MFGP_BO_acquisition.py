# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:35:19 2025

@author: Y. Lecomte
"""

#%% PACKAGES ==================================================================
import MFGP_BO_functions as fct
import time
import smbus2
from rpi_hardware_pwm import HardwarePWM
import math
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os
import struct
from tqdm import tqdm
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
from skopt import load
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import itertools
import pickle  
from scipy.linalg import cholesky, cho_solve

#%% INITIALIZATION ============================================================

# Define the reference power input
gamma_ref = 40

# Define how to compute the cost
cost_type = 'penalty'

# Define the cost parameter in case of penalty
alpha_p, alpha_d =  0.5, 1
cost_params = [alpha_p, alpha_d]


# Define the loading path for the initial training set
case_training = 3
training_path = os.path.join('training_data', f'Case_{case_training}')

#%% TARGET SEQUENCE ===========================================================

# Step parameters
target_val = [0, 10, -10, 5, -5]
fs = 400
T_step = 5
N_steps = len(target_val)
dt = 1/fs

samples_per_step = int(fs * T_step)
total_samples = samples_per_step * N_steps

# Time vector for initial step sequence
t_step = np.linspace(0, T_step * N_steps, total_samples)
target_list = np.zeros_like(t_step)

# Step input
for i in range(N_steps):
    target_list[i*samples_per_step:(i+1)*samples_per_step] = target_val[i]
    

#%% BO INITIALIZATION =========================================================

# Boundaries of the domain
xmin_kp, xmax_kp = 0.05, 2
xmin_kd, xmax_kd = 0.05, 1.5
# Domain 
bounds = ([xmin_kp, xmax_kp], [xmin_kd, xmax_kd])



# -----------------------------------------
# STEP 1: Import the training set
# -----------------------------------------
# Low fidelity data (from digital twin)
X_L_train = np.load(os.path.join(training_path, 'X_L_train.npy'))
y_L_train = np.load(os.path.join(training_path, 'y_L_train.npy'))
# High fidelity data (from balance bench)
X_H_train = np.load(os.path.join(training_path, 'X_H_train.npy'))
y_H_train_full = np.load(os.path.join(training_path, 'y_H_train_full.npy'))
mask = (
        (X_H_train[:,0] >= xmin_kp) & ( X_H_train[:,0] <= xmax_kp) & 
        (X_H_train[:,1] >= xmin_kd) & ( X_H_train[:,1] <= xmax_kd)
)

X_H_train = X_H_train[mask]
y_H_train_full = y_H_train_full[mask]
# -----------------------------------------
# SETP 2: Define BO parameters
# -----------------------------------------
# Total number of samples in each data set
Nt_lf = X_L_train.shape[0]; Nt_hf = X_H_train.shape[0]
# Define the numebr of training sample of each system
Ns_train_lf, Ns_train_hf = 200, Nt_lf
# Number of sequences to average for each sample (for high-fidelity)
Ne_init = 4

# Extract the specified amount of info from the training set
y_H_train = np.mean(y_H_train_full[:, :Ne_init], axis=1)

# Select Ns random index for each fidelity system
random_idx_lf = np.random.choice(Nt_lf, size = Ns_train_lf, replace=False)
random_idx_hf = np.random.choice(Nt_hf, size=Ns_train_hf, replace=False)

# Extract these random indexes from the inital data set
X_L = X_L_train[random_idx_lf]
y_L = y_L_train[random_idx_lf]


X_H = X_H_train[random_idx_hf]
y_H = y_H_train[random_idx_hf] 

# NOTE: For the moment we take only 1 episode and let the HPO find the 
# optimal noise level for each fidelity system
#y_H_mean = y_H_train_mean[random_idx_hf]
#y_H_std = y_H_train_std[random_idx_hf]

# Exploration factor
xi = 0.01
# Number of BO iterations
n_iter = 40
# Episode per acquisition
Ne = 4
# Re-optimization of the hyper-parameters
n_params = 5

# -----------------------------------------
# SETP 4: Define the saving path
# -----------------------------------------

# Folder initialization
Data = "PD_SEQUENCE"
campaign = f"INIT_Ns_L_{Ns_train_lf}_Ns_H_{Ns_train_hf}_Ne_{Ne_init}_ITER_Ns_{n_iter}_Ne_{Ne}_xi_{xi}_NEW"

# Create the corresponding campaign folder
CAMPAIGN_PATH = os.path.join(Data, f"{campaign}")
os.makedirs(CAMPAIGN_PATH, exist_ok = True)
    
# Create a folder to save the plots
PLT_PATH = os.path.join(CAMPAIGN_PATH, "plots")
os.makedirs(PLT_PATH, exist_ok = True)

# Create a folder to save the full sequence data
DATA_PATH = os.path.join(CAMPAIGN_PATH, "data")
os.makedirs(DATA_PATH, exist_ok = True)


# -----------------------------------------
# SETP 4: Hyper-parameters optimization
# -----------------------------------------

MLE_history = []
param_history = []

# The initial guess it filled with the HPO run separatly only on the model (LF)
params0 = np.log([0.37212993679243683, 17.558130895316797, 1.0, 1.0, 0.0011960904918658897, 0.1])
params0 = np.insert(params0, 0, 1.0)
res = minimize(fct.log_marginal_likelihood, params0, args=(X_L, y_L, X_H, y_H, MLE_history, param_history), method='L-BFGS-B')
opt_rho = res.x[0]
opt_theta = np.exp(res.x[1:])
last_params = res.x.copy()
# NOTE: For the moment we consider only 1 epsiode so no measure of a
# possible noise estimation. We let the HPO find it
# opt_theta[-1] = y_H_init_std


# Plot and save the results 
fct.plot_n_save_hpo(MLE_history, param_history, 0, PLT_PATH, DATA_PATH)


# -----------------------------------------
# SETP 5: Save all the paramters configuration in a .txt file
# -----------------------------------------
# Text that summerize all the parameter of the current case
case_infos = (
"================================\n"
f"Number of episodes = {Ne}\n"
f"Low-Fidelity training samples = {Ns_train_lf}\n"
f"High-Fidelity training samples = {Ns_train_hf}\n"
f"BO iteration = {n_iter}\n"
f"Exploration param, xi = {xi}\n"
f"Re-HPO frequency = {n_params}\n"
f"Cost function type = {cost_type}\n"
f"Alpha_p = {alpha_p}, Alpha_d = {alpha_d}\n"
"Domain boundaries = {bounds}\n"
"For training case 3 with quadratic penalty, large bounds and fitlered error value\n"
"================================"
)
# Save the info in a .txt file
np.savetxt(os.path.join(CAMPAIGN_PATH, 'INFO.txt'), np.empty((0,)), header=case_infos, comments='')                


#%% BO MAIN ===================================================================

if __name__ == '__main__':   # Program entrance
    print('Program is starting ... ')
    try:
        # Start all the hardware
        encoder = fct.AMT23AngleSensor()
        pL, pR = fct.setup()
        convergence_history = []
        
        # Bayesian Optimization
        for i in range(n_iter):
            print(f'Iteration {i+1}/{n_iter}')
            # -----------------------------------------------------------------
            # STEP 1: If requried, re-optimize the hyper-parameters
            # -----------------------------------------------------------------
            perform_hpo = (i % n_params == 0)
            if perform_hpo:
                MLE_history = []
                param_history = []
                res = minimize(fct.log_marginal_likelihood, last_params, args=(X_L, y_L, X_H, y_H, MLE_history, param_history), method='L-BFGS-B')
                opt_rho = res.x[0]
                opt_theta = np.exp(res.x[1:])
                # NOTE: For the moment we consider only 1 epsiode so no measure of a
                # possible noise estimation. We let the HPO find it
                # opt_theta[-1] = y_H_init_std
                last_params = res.x.copy()
                
                # Plot and save the results 
                fct.plot_n_save_hpo(MLE_history, param_history, i, PLT_PATH, DATA_PATH)

            # -----------------------------------------------------------------
            # STEP 2: Evaluate the next point to sample
            # -----------------------------------------------------------------
            x_next, *_ = fct.propose_location_hybrid(X_L, y_L, X_H, y_H, opt_rho, opt_theta, bounds, xi, grid_resolution=20, n_restarts=5)
                
            # Store the current best value for the convergence history
            convergence_history.append(np.min(y_H))
                
            # -----------------------------------------------------------------
            # STEP 3: Plot and save the current prediction and the nxt sampling
            # -----------------------------------------------------------------

            fct.plot_n_save_BO(X_L, y_L, X_H, y_H, opt_rho, opt_theta, Ns_train_hf, Ns_train_lf, x_next, bounds, PLT_PATH, DATA_PATH, xi, iteration=i)
            
            # -----------------------------------------------------------------
            # STEP 4: Run the sequences at the next point
            # -----------------------------------------------------------------
        
            new_evals = []
            # Path to save the full sequences
            iter_folder = os.path.join(DATA_PATH, f"iteration_{i}")
            os.makedirs(iter_folder, exist_ok=True)

            for j in range(Ne):
                # Run the full sequence and extract the error, the time list and the theta list
                cost, time_seq, theta_seq = fct.system_cost(x_next, target_list, cost_params, cost_type, encoder, total_samples)
				# Add the cost value to the current point list
                new_evals.append(cost)
				
				# File path for each episode, within a given iteration
                file_path = os.path.join(iter_folder, f"episode_{j}.csv")
                np.savetxt(file_path, np.column_stack((time_seq, theta_seq)), delimiter=',', header='time,theta', comments='', fmt='%1.5f')

            # Convert the cost evaluation list into a numpy array
            new_evals = np.array(new_evals).flatten()  # Ensures shape (Ne,)
            new_evals_mean = np.mean(new_evals)
            new_evals_std = np.std(new_evals)

            # NOTE: For the moment we only go for one episode per point, 
            # so the mean is done for only one value. 
            #y_H_mean = np.vstack((y_H_mean, new_evals_mean))
            #y_H_std = np.vstack((y_H_mean, new_evals_std))
            
            # -----------------------------------------------------------------
            # STEP 5: Add the new exploration in the data set (HF)
            # -----------------------------------------------------------------
            
            # Add this to the set of data
            X_H = np.vstack((X_H, x_next))
            y_H = np.hstack((y_H, new_evals_mean))          # Use hstack instead of vstack for 1D arrays            

            # Plot the convergence
            plt.figure(figsize = (5, 3))
            plt.plot(convergence_history, '--bo')
            plt.grid()
            plt.xlabel('Iterations')
            plt.ylabel('Cost $\\mathcal{J}$')
            plotname = os.path.join(CAMPAIGN_PATH, 'cost_conv.png')
            plt.savefig(plotname, dpi=300, bbox_inches='tight')
        print('Optimization done!')
        pL.stop()
        pR.stop()
    except KeyboardInterrupt:  # Press ctrl-c to end the program.
        print('PO')
        pL.stop()
        pR.stop()

