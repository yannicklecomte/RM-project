# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:00:21 2025

@author: Y. Lecomte
"""

#%% Import packages ===========================================================

import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.stats import norm
from scipy.linalg import cholesky, cho_solve
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=16)  
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

#%% Define the BO functions

def rbf_kernel(X1, X2, length_scale, variance):
	sqdist = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
	return variance * np.exp(-0.5 * sqdist / length_scale**2)


def mf_gp_predict(X_test, X_L, X_H, rho, theta, alpha, L):
    """
    Predict at new test points using fitted MF-GP (with precomputed alpha).
    """
    length_lf, var_lf, length_d, var_d, sigma_L, sigma_H = theta

    # Cross-covariance terms between test points and training points
    K_s_L = rbf_kernel(X_test, X_L, length_lf, var_lf)
    K_s_H_lf = rbf_kernel(X_test, X_H, length_lf, var_lf)
    K_s_H_d = rbf_kernel(X_test, X_H, length_d, var_d)

    # Combined cross-covariance
    K_s = np.hstack([rho * K_s_L, rho**2 * K_s_H_lf + K_s_H_d])

    # Prior covariance between test points
    K_ss_lf = rbf_kernel(X_test, X_test, length_lf, var_lf)
    K_ss_d = rbf_kernel(X_test, X_test, length_d, var_d)
    K_ss = rho**2 * K_ss_lf + K_ss_d
    
    v = cho_solve((L, True), K_s.T)
    mu = K_s @ alpha
    cov = K_ss - K_s @ v
    return mu , np.clip(np.sqrt(np.diag(cov)), 0, None)


#%% Plot the color maps =======================================================

# -----------------------------------------------------------------------------
# Boundaries of the domain
xmin_kp, xmax_kp = 0.05, 2 
xmin_kd, xmax_kd = 0.05, 1.5 
bounds = ([xmin_kp, xmax_kp], [xmin_kd, xmax_kd])
dx = xmax_kp - xmin_kp
dy = xmax_kd - xmin_kd


# -----------------------------------------------------------------------------
# Load training data
training_path = os.path.join('training_data', 'Case_4')

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

# -----------------------------------------------------------------------------
# Define the initial conditions of the opti

# INIT
Ne_init = 4   # Number of episodes
Ns_init_HF = 5
Ns_init_LF = 200

# ITER
Ne_iter = 4
Ns_iter = 20

xi = 0.01
# Define the path where the plots will be saved
campaign = f"INIT_Ns_L_{Ns_init_LF}_Ns_H_{Ns_init_HF}_Ne_{Ne_init}_ITER_Ns_{Ns_iter}_Ne_{Ne_iter}_xi_{xi}"
data_path = os.path.join("PD_SEQUENCE", f'{campaign}', 'data')
PLT_PATH = os.path.join('PD_SEQUENCE', f'{campaign}', 'plots', 'BO_maps_REPORT')
os.makedirs(PLT_PATH, exist_ok = True)

# -----------------------------------------------------------------------------
# Extract only the defined amoiunt of initial points

# Total number of samples in each data set
Nt_lf = X_L_train.shape[0]; Nt_hf = X_H_train.shape[0]
# Extract the specified amount of info from the training set
y_H_train = np.mean(y_H_train_full[:, :Ne_init], axis=1)
# Select Ns random index for each fidelity system
random_idx_lf = np.random.choice(Nt_lf, size = Ns_init_LF, replace=False)
random_idx_hf = np.random.choice(Nt_hf, size=Ns_init_HF, replace=False)
# Extract these random indexes from the inital data set
X_L = X_L_train[random_idx_lf]
y_L = y_L_train[random_idx_lf]
X_H = X_H_train[random_idx_hf]
y_H = y_H_train[random_idx_hf] 

# -----------------------------------------------------------------------------
# Define various parameters

resolution = 80
alpha_p, alpha_d = 0.05, 0.5

# Generate a dense gid data set for the plot
kp = np.linspace(bounds[0][0], bounds[0][1], resolution)
kd = np.linspace(bounds[1][0], bounds[1][1], resolution)
KP, KD = np.meshgrid(kp, kd)
X_grid = np.column_stack((KP.ravel(), KD.ravel()))


for iter_ in range(Ns_iter):

    # -------------------------------------------------------------------------
    # Load the data set
    BO_input_path = os.path.join(data_path, f'iteration_{iter_}', 'BO_inputs.npz')
    BO_inputs = np.load(BO_input_path)
    
    # Current Kp, Kd of the sequence
    x_next = BO_inputs['x_next']
    Kp, Kd = x_next
    
    # Load of GPR 
    L = BO_inputs['L']
    alpha = BO_inputs['alpha'] 
    X_H =  BO_inputs['X_H']
    X_L = BO_inputs['X_L']
    y_best = BO_inputs['y_best']
    
    
    # -------------------------------------------------------------------------
    # Load the hyper-parameters
    params_path = os.path.join(data_path, 'HPO_data', f'Iteration_{iter_+1:02d}_param_history.npy')
    params_opt = np.load(params_path)[-1, :]
    rho, theta = params_opt[0], params_opt[1:]
    
    # -------------------------------------------------------------------------
    # Posterior prediction on the dense grid
    mu, std = mf_gp_predict(X_grid, X_L, X_H, rho, theta, alpha, L)
    
    # -------------------------------------------------------------------------
    # EI computation over the dense grid
    Z = (y_best - mu - xi) / std
    EI = (y_best - mu - xi) * norm.cdf(Z) + std * norm.pdf(Z)
    EI[std == 0.0] = 0.0
          

    # -------------------------------------------------------------------------
    # Plot setup
    
    # Best point location
    # best_idx = np.where(y_H==y_best)
    best_idx = np.argmin(y_H)
    best_point = X_H[best_idx]
    
    # Reshape for plotting
    mu_grid = mu.reshape(KP.shape)
    std_grid = std.reshape(KP.shape)
    ei_grid = EI.reshape(KP.shape)


    fig, axs = plt.subplots(1, 3, figsize=(20, 3.5))
    titles = ['Posterior mean $\\mu(K_p, K_d)$',
              'Posterior uncertainty $\\sigma(K_p, K_d)$',
              'Expected Improvement (EI)']
    
    color_maps = ['viridis', 'plasma',  'cividis']
    data = [mu_grid, std_grid, ei_grid]
    
    for ax, d, title, col_map in zip(axs, data, titles, color_maps):

        c = ax.contourf(KP, KD, d, levels=100, cmap=col_map)

        c.set_edgecolor("face")  

        # Plot explored points, next, best current (unchanged)
        ax.scatter(X_L[:, 0], X_L[:, 1], c='red', edgecolor='k', s=35, zorder=10, clip_on=False, label='Low-Fidelity Training')
        ax.scatter(X_H[:Ns_init_HF, 0], X_H[:Ns_init_HF, 1], c='green', edgecolor='k', s=50, zorder=10, clip_on=False, label='High-Fidelity Training')
        ax.scatter(X_H[Ns_init_HF:, 0], X_H[Ns_init_HF:, 1], c='blue', edgecolor='k', s=50, zorder=10, clip_on=False, label='High-Fidelity Explored')
        ax.scatter(best_point[0], best_point[1], c='white', edgecolor='k', marker='D', s=50, zorder=10, clip_on=False, label='Best High-Fidelity')
        ax.scatter(x_next[0], x_next[1], c='white', edgecolor='k', marker='*', s=200, zorder=10, clip_on = False, label='Next')

    
        # ax.set_title(title, fontsize=20)
        ax.set_xlabel('$K_p$')
        ax.set_ylabel('$K_d$')
        ax.set_aspect('equal')

        # Create a custom colorbar axis manually next to current subplot
        cax = fig.add_axes([
            ax.get_position().x1 + 0.01,  # x-position: just right of the subplot
            ax.get_position().y0,        # y-position: same as subplot
            0.008,                       # width of colorbar
            ax.get_position().height     # height of colorbar same as plot
        ])
        fig.colorbar(c, cax=cax)
    
        if title == 'Expected Improvement (EI)':
            cax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2e}'))
            
    # Add spacing below the plots
    # fig.subplots_adjust(bottom=0.25) #, wspace=0.3)
    
    # Add a single shared legend below all subplots
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=14, ncol=5, bbox_to_anchor=(0.5, -0.2))

    fig.suptitle("$\\textit{Case 3}$", fontsize=20, y = 1)

    
    # Save plot
    filename = f"BO_maps_iter_{iter_:03d}.pdf"
    full_path = os.path.join(PLT_PATH, filename)
    plt.savefig(full_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.show()  
    
    
    # -------------------------------------------------------------------------
    # Load the time series to compute cost
    
    for eps in range(Ne_iter):
        all_data_list = []
        err_list = []
        file_path = os.path.join(data_path, f'iteration_{iter_}', f"episode_{eps}.csv")
       
        time_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 0]
        theta_list = np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1]
        
        theta_list[0] = theta_list[1]
        all_data_list.append(theta_list)
        Ns = len(theta_list)
        err = np.linalg.norm(theta_list - target_list) / np.sqrt(Ns)
        err_list.append(err + alpha_p*Kp**2 + alpha_d*Kd**2)

    new_evals_mean = np.mean(err_list)
    y_H = np.hstack((y_H, new_evals_mean))   
    
    
