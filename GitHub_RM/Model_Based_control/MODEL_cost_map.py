# -*- coding: utf-8 -*-
"""
Created on Sun May 18 19:36:11 2025

@author: Y. Lecomte
"""

#%% Import packages ===========================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os 
import all_systems_RK4 as sys
from scipy.stats.qmc import LatinHypercube
from tqdm import trange
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm
import matplotlib.gridspec as gridspec


# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=14)  
coutour_size = (6,5)
fig_size = (5, 3)



#%% STEP 1: Define the low fidelity system ====================================

# -----------------------------------
# Define the target sequence 
# -----------------------------------
    
# Step sequence definition
target_val = [0, 10, -10, 5, -5]
# Sequence parameters
fs = 400; dt = 1/fs
T_step = 5    # Duration of each step (s)
# Create the full sequence
target_list = sys.control_sequence(target_val, fs, T_step)
# Limit of the actuation
gamma_clip = 8
x_min_kp, x_max_kp = 0.05, 2
x_min_kd, x_max_kd = 0.05, 1.5

# Number of sequence per location (nbre of episode)
Ne = 1
cost_type = 'penalty'
known_noise_std = 0


# Penalty values
penalty_params =  0.05, 0.5

def low_fidelity_function(X, Ne, noise_std):
    # Extract the input parameters
    Kp, Kd = X

    # -----------------------------------
    # Compute the sequence 
    # -----------------------------------
    y_ = []

    for eps in range(Ne): 
        if cost_type == 'classical':
            # Compute the cost of the system with random noise 
            y_.append(sys.cost_function(X, target_list, dt, gamma_clip, noise_std))
        elif cost_type == 'penalty':
            # Compute the cost of the system with random noise 
            y_.append(sys.cost_function_penalty(X, penalty_params, target_list, dt, gamma_clip, noise_std))
            
    # Average over the various episode
    y_mean  = np.mean(y_)
    y_std = np.std(y_)  
    return y_mean, y_std


#%% STEP 2: Generate the cost map =============================================

# Create a prediciton grid
x1 = np.linspace(x_min_kp, x_max_kp, 30)
x2 = np.linspace(x_min_kd, x_max_kd, 30)
X1g, X2g = np.meshgrid(x1, x2)
X_test = np.column_stack([X1g.ravel(), X2g.ravel()])

# --------------------------------------------------
# Uncomment to run a new set of point
# --------------------------------------------------

# y_true = np.array([low_fidelity_function(X_test[i], Ne=1, noise_std=0)[0] 
#                     for i in tqdm(range(X_test.shape[0]))])
# y_true_grid = y_true.reshape(X1g.shape)


#%% STEP 2bis: Load the prediction grid data
save_path = os.path.join('DATA_cost_map', f'alpha_p_{penalty_params[0]}_alpha_d_{penalty_params[1]}')

y_true = np.load(os.path.join(save_path, 'y_train.npy'))
X_test = np.load(os.path.join(save_path, 'X_train.npy'))

x1 = np.linspace(x_min_kp, x_max_kp, 30)
x2 = np.linspace(x_min_kd, x_max_kd, 30)
X1g, X2g = np.meshgrid(x1, x2)
y_true_grid = y_true.reshape(X1g.shape)

#%% STEP 3: Load the Nelder-mead opti tracking

# ==== Important parameters to define ====
######################################
# Control action limitations
gamma_clip = 8
# Optimization solver ('Nelder-Mead', 'BFGS', 'BO', ...)
opti_solver = 'Nelder-Mead'
# Cost function type ('classic', 'penalty', ...)
cost_type = 'penalty'
# Penalty parameters (for penalized cost function)
alpha_p, alpha_d = 0.5, 1
noise_std = 0
######################################

OUT_PATH = os.path.join('MB_optimization', f'{opti_solver}', f'{cost_type}_cost', f'g_clip_{gamma_clip}')
track_file= os.path.join(OUT_PATH, 'opti_tracking.dat')
tracking_points = np.loadtxt(track_file, skiprows=8)


#%% STEP 4: Plot and save =====================================================

# Define the save path
save_path = os.path.join('DATA_cost_map', f'alpha_p_{penalty_params[0]}_alpha_d_{penalty_params[1]}')
os.makedirs(save_path, exist_ok=True)

# -----------------------------------------------------------
# PLOT 1: Map with nothing else
# -----------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))  
cf = ax.contourf(X1g, X2g, y_true_grid, 50, cmap='viridis')
ax.set_xlabel('$K_p$')
ax.set_ylabel('$K_d$')
ax.set_aspect('equal')
# Create a custom colorbar axis manually next to current subplot
cax = fig.add_axes([
    ax.get_position().x1 + 0.015,  # x-position: just right of the subplot
    ax.get_position().y0,        # y-position: same as subplot
    0.021,                       # width of colorbar
    ax.get_position().height     # height of colorbar same as plot
])
fig.colorbar(cf, cax=cax, label="Cost $\\mathcal{J}$")
plt_name = os.path.join(save_path, 'model_cost_map_quad.png')
plt.savefig(plt_name, dpi=400, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------
# PLOT 2: Map with the first point of the opti
# -----------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5)) 
cf = ax.contourf(X1g, X2g, y_true_grid, 50, cmap='viridis')
cf.set_edgecolor("face")  
ax.scatter(1, 1, c='r', edgecolors='k',marker='D', s=80, label='Sampled Points')
ax.set_xlabel('$K_p$')
ax.set_ylabel('$K_d$')
ax.set_aspect('equal')

# Create a custom colorbar axis manually next to current subplot
cax = fig.add_axes([
    ax.get_position().x1 + 0.015,  # x-position: just right of the subplot
    ax.get_position().y0,        # y-position: same as subplot
    0.021,                       # width of colorbar
    ax.get_position().height     # height of colorbar same as plot
])
fig.colorbar(cf, cax=cax, label="Cost $\\mathcal{J}$")

plt_name = os.path.join(save_path, 'model_cost_map_1point_quad.png')
plt.savefig(plt_name, dpi=400, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------
# PLOT 3: Map with all points of the opti
# -----------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))  
cf = ax.contourf(X1g, X2g, y_true_grid, 50, cmap='viridis')
cf.set_edgecolor("face")  
ax.set_xlabel('$K_p$')
ax.set_ylabel('$K_d$')
ax.scatter(1, 1, c='r', edgecolors='k',marker='D', s=80)
ax.scatter(tracking_points[1:, 0], tracking_points[1:, 1], c='r', edgecolors='k', s=80, label='Nelder-Mead')
ax.scatter(tracking_points[-1, 0], tracking_points[-1, 1], c='yellow', edgecolors='k', marker = "*", s=300)
ax.set_aspect('equal')

# Create a custom colorbar axis manually next to current subplot
cax = fig.add_axes([
    ax.get_position().x1 + 0.015,  # x-position: just right of the subplot
    ax.get_position().y0,        # y-position: same as subplot
    0.021,                       # width of colorbar
    ax.get_position().height     # height of colorbar same as plot
])
fig.colorbar(cf, cax=cax, label="Cost $\\mathcal{J}$")
        
# Save and show
# plt.legend(fontsize=14)  # or any size like 16, 18, etc.
plt_name = os.path.join(save_path, 'model_cost_map_wpoints_quad.pdf')
plt.savefig(plt_name, format='pdf', dpi=600, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------
# Save the data
np.save(os.path.join(save_path, 'y_train.npy'), y_true)
np.save(os.path.join(save_path, 'X_train.npy'), X_test)
