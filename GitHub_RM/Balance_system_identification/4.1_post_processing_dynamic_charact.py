# -*- coding: utf-8 -*-
"""
Created on Wed Mar  28 11:34:19 2025

@author: Y. Lecomte
"""

#%% Packages, initialization and functions

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.signal import butter, filtfilt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import math
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from colorama import Fore, Style, Back, init
init()  

# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=14)   

long_fig_size = (8,3)
square_fig_size = (5,5)
fig_size = (7,3)


auto_fitting = False
cutoff = 1    
  
#%% Define the parameters of the system 

print(Back.GREEN + "Loading the system's coefficients ..." + Style.RESET_ALL)

# =================== System parameters for gamma_ref = 40% ===================
l  = 0.68

print('---> Motor System')
# Define static chracteristics of the motor
Kt = 0.10083599643284652
T0 = -0.5338394640385178

# Define dynamic chracteristics of the motor
tau_UP = 0.1305
tau_d_UP = 0.035

print('---> Balance System')
# Define static chracteristics of the balance
balance_static_campaign = 4 # Without the panels
balance_static_path = os.path.join("Static_characterization", f"campaign_{balance_static_campaign}","data","static_coefs")
C, theta_0 = np.loadtxt(balance_static_path)

# Define dynamic chracteristics of the balance
K_theta = (l*Kt)/C
tau_0 = K_theta * theta_0
known_params = K_theta, tau_0

# First guess of the unknown parameters
b = -3e-1
I = -2.8e-3
params_guess= I, b

# Define the reference power input  and the associated thrust
gamma_ref = 40

# Define the initial position of the balance
theta_init = theta_0

global target
target = 40


#%% Define the functions of the system

def map_pwm(x):
    return ((x-5)/5)*100

def static_thrust(gamma, Kt=Kt, T0=T0):
    return (gamma * Kt) + T0

def gamma_model(y, t, tau_d, gamma_cmd_func):
    gamma_cmd = gamma_cmd_func(t)
    dydt = (gamma_cmd - y) / tau_d
    return dydt

def motor_model(y, t, tau, gamma_func):
    gamma = gamma_func(t)
    T_target = static_thrust(gamma)
    dydt = (T_target - y) / tau
    return dydt

def second_order_system(state, t, params, known_params, T_R_func, T_L_func):
    I, b = params
    K_theta, tau_0 = known_params
    T_R = T_R_func(t)
    T_L = T_L_func(t)
    theta, theta_dot = state

    dtheta_dt = theta_dot
    dtheta_dot_dt = -(b / I) * theta_dot - (K_theta / I) * theta + (l / I) * (T_R - T_L) + (tau_0 / I)
    return [dtheta_dt, dtheta_dot_dt]

def full_system(params, time, known_params, delta_list, theta_init):
    
    delta_list = np.asarray(delta_list)

    # === Step 1: Input gamma_cmd interpolators ===
    gamma_cmd_L_func = interp1d(time, target - 0.5 * delta_list, fill_value="extrapolate")
    gamma_cmd_R_func = interp1d(time, target + 0.5 * delta_list, fill_value="extrapolate")

    # === Step 2: Integrate gamma(t) ===
    gamma_L_init = target - 0.5 * delta_list[0]
    gamma_R_init = target + 0.5 * delta_list[0]

    gamma_L = odeint(gamma_model, gamma_L_init, time, args=(tau_d_UP, gamma_cmd_L_func)).flatten()
    gamma_R = odeint(gamma_model, gamma_R_init, time, args=(tau_d_UP, gamma_cmd_R_func)).flatten()

    # === Step 3: Integrate T(t) ===
    gamma_L_func = interp1d(time, gamma_L, fill_value="extrapolate")
    gamma_R_func = interp1d(time, gamma_R, fill_value="extrapolate")

    T_L_init = static_thrust(gamma_L_init)
    T_R_init = static_thrust(gamma_R_init)

    T_L = odeint(motor_model, T_L_init, time, args=(tau_UP, gamma_L_func)).flatten()
    T_R = odeint(motor_model, T_R_init, time, args=(tau_UP, gamma_R_func)).flatten()

    # === Step 4: Integrate theta ===
    T_L_func = interp1d(time, T_L, fill_value="extrapolate")
    T_R_func = interp1d(time, T_R, fill_value="extrapolate")

    theta_sol = odeint(second_order_system, theta_init, time, args=(params, known_params, T_R_func, T_L_func))
    theta_pred, theta_dot_pred = theta_sol[:,0], theta_sol[:,1]
    return theta_pred, theta_dot_pred, gamma_L, gamma_R, T_R, T_L


def cost_full_system(params, time, known_params, delta_list, theta_real, w_theta=1.0, w_velocity=1.0):
    theta_0 = theta_real[0]
    theta_dot_0 = (theta_real[1] - theta_real[0]) / (time[1] - time[0])
    theta_state_init = [theta_0, theta_dot_0]

    theta_pred, theta_dot_pred, *_ = full_system(params, time, known_params, delta_list, theta_state_init)

    theta_dot_real = np.gradient(theta_real, time)
    theta_dot_pred = np.gradient(theta_pred, time)

    cost_pos = np.mean((theta_pred - theta_real)**2)
    cost_vel = np.mean((theta_dot_pred - theta_dot_real)**2)

    return w_theta * cost_pos + w_velocity * cost_vel



#%% LOAD THE DATA

DATA = "Dynamic_characterization"
campaign = 6
print(Back.GREEN + f"Loading of the data of campaign {campaign:.0f} ..." + Style.RESET_ALL)

# Initialize the dictionnary to store the data
data_dict = {}
delta_dict = {}

# Path to axcess the current campaign data
CASE_PATH = DATA + os.sep + f"campaign_{campaign}"
PWM_NAMES = os.listdir(CASE_PATH)

# Create a path to save the plots
PLT_PATH = CASE_PATH + os.sep + 'plots'
if not os.path.exists(PLT_PATH):
    os.mkdir(PLT_PATH)
    
# Create a path to save the data
DATA_PATH = CASE_PATH + os.sep + 'data'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)


# Extract the PWM and torque ratio values from the folder's names
pwm_list = [re.findall(r'\d+\.\d+|\d+', file) for file in PWM_NAMES]
pwm_list = [float(num) for sublist in pwm_list for num in sublist]

# Add each data case of the campaign in a dictionnary
for pwm in pwm_list:
    data_dict[pwm] = {}
    PWM_PATH = os.path.join(CASE_PATH, str(pwm) + '_PWM')
    DELTA_NAMES = os.listdir(PWM_PATH)
    
    delta_list = [re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', file) for file in DELTA_NAMES]
    delta_list = [(float(sublist[0]), float(sublist[1])) for sublist in delta_list if len(sublist) == 2]
    delta_dict[pwm] = delta_list
    

    for delta_step in delta_list: 
        print(f'---> Case: Delta Step {delta_step}')
        # Construct the file name
        filepath = os.path.join(PWM_PATH, f"{delta_step[0]:.4f}_{delta_step[1]:.4f}.dat")
        print(f'     Path: {filepath}')

        # Load the data, skipping the first row (header)
        data = np.loadtxt(filepath, skiprows=1)  
        
        time = data[:,0] #0
        theta = data[:,1] #1
        mean = np.mean(theta)
        std = np.std(theta, ddof=1)
        z_scores = (theta - mean) / std        
        outlier_indices = np.where(np.abs(z_scores) > 3)[0]
        filt_theta = theta.copy()
        for idx in outlier_indices:
            if idx >= 4:
                filt_theta[idx] = filt_theta[idx - 4]
            else:
                filt_theta[idx] = mean #2
                
        delta = data[:,2] #3
        
        # save all in the dictionnary
        data_dict[pwm][delta_step] = {"data": np.column_stack((time, theta, filt_theta, delta))}


#%% SIMPLE PLOTS

for pwm in pwm_list:
    for delta_step in delta_dict[pwm]:
        time = data_dict[pwm][delta_step]["data"][:,0]
        theta = data_dict[pwm][delta_step]["data"][:,1]       
        filt_theta = data_dict[pwm][delta_step]["data"][:,2]
        delta = data_dict[pwm][delta_step]["data"][:,3]

        plt.figure(figsize=fig_size)
        plt.plot(time, theta, label='Raw signal')
        # plt.plot(time, filt_theta, label='Filtered signal')
        # plt.axvline(time[1600+300])
        plt.xlabel('Time [s]')
        plt.ylabel('Roll angle $\\theta$ [deg]')
        plt.title(f'Reference Power Input: $\\gamma_{{\\mathrm{{ref}}}}$ = {pwm:.1f} \%, Delta Power step: $\\Delta \\gamma = {delta_step[0]:.4f} \\rightarrow {delta_step[1]:.4f}$')
        plt.grid()
        plt.legend()
        fig_name = PLT_PATH + os.sep +  f'plot_{pwm:.1f}_PWM_{delta_step[0]:.4f}_{delta_step[1]:.4f}_r.png'
        plt.savefig(fig_name, dpi=200, bbox_inches='tight')
        plt.show()
        
        
#%% ALL NORMALIZED CURVES PLOT
filt_width = 100

# Static coefficient from campaign 4
theta_0 = 2.38018978
C = -1.54209496

for pwm in pwm_list:
    fig, ax = plt.subplots(figsize=(6, 3))  

    deltas = delta_dict[pwm]  
    cmap = cm.get_cmap("viridis", len(deltas))  
    norm = plt.Normalize(vmin=min(C*delta[1]+theta_0 for delta in deltas), vmax=max(C*delta[1]+theta_0 for delta in deltas))  

    for i, delta_step in enumerate(deltas):
        # ==== Import everything ====
        time_list = data_dict[pwm][delta_step]["data"][:,0]
        theta_list = data_dict[pwm][delta_step]["data"][:,1]       
        filt_theta_list = data_dict[pwm][delta_step]["data"][:,2]
        delta_list = data_dict[pwm][delta_step]["data"][:,3]
        
        # ==== Acquisition frequences ====
        time_diffs = np.diff(time_list)
        fs = 1/np.mean(time_diffs); Ts = 1/fs
        # Locate the index of the step up command
        delta_array = np.array(delta_list)
        step_idx = np.where(delta_array != delta_array[0])[0][0]
        
        # ==== Crop the signal at the step up instant ====
        shift_signal_ixd = 300
        crop_min = step_idx - shift_signal_ixd
        crop_max = crop_min + 2000 + shift_signal_ixd
        time_crop = time_list[crop_min:crop_max] - time_list[crop_min] - shift_signal_ixd*Ts
        filt_theta_list_crop = filt_theta_list[crop_min:crop_max]
    
        # ==== Normalization ====
        theta_offset = filt_theta_list_crop[shift_signal_ixd]  
        theta_final = np.mean(filt_theta_list_crop[-filt_width:])  
        theta_norm = (filt_theta_list_crop - theta_offset) / (theta_final - theta_offset)
        
        # Plot with Viridis colormap
        color = cmap(norm(C*delta_step[1]+theta_0))
        ax.plot(time_crop, theta_norm, color=color)
       
    # Step function
    x = [-1.3, 0, 0, 8.2]
    y = [0, 0, 1, 1]
    ax.plot(x, y, c='r', linestyle='--', zorder=3)

    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"From $\theta_{\mathrm{ref}}$ $\rightarrow$ $\theta_{\mathrm{target}}$")  


    # Labels and ticks
    ax.text(-0.8, 1.1, '$\\Delta \\gamma (t)$', fontsize=14, color='red',
        verticalalignment='bottom', horizontalalignment='left')

    # ax.set_title(f'Step UP from $\\gamma$ = {pwm:.1f} \%')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Normalized Angle $\theta_{\mathrm{norm}}$')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
    ax.set_ylim(-0.2, 2); ax.set_xlim(-1.3, 8)
    
    # Here the plot is sace in .pdf, but it could also be in .png
    fig_name = PLT_PATH + os.sep + f'balance_dynamics_all_steps_{pwm:.1f}_refPWM.pdf'
    plt.savefig(fig_name, format='pdf', dpi = 400, bbox_inches='tight')
    plt.show() 
        
#%% FITTING

print(Back.GREEN + "Fitting of all step responses ..." + Style.RESET_ALL)

filt_width_idx = 1600

total_cases = sum(len(delta_dict[pwm]) for pwm in pwm_list)
# The number of columns might need to be adjusted in function of the campaign
ncols = 5
nrows = math.ceil(total_cases / ncols)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 13))
axs = axs.flatten()  # To index easily even if grid is 1D

for pwm in pwm_list:
    print(f'[INFO] For pwm: {pwm:.2f} %')

    for i, delta_step in enumerate(delta_dict[pwm]):
        ax = axs[i]

        print(f'----> Fitting of delta step: {delta_step}')
        # ==== determine the index of the step up ====
        delta_array = np.array(data_dict[pwm][delta_step]["data"][:,3])
        step_idx = np.where(delta_array != delta_array[0])[0][0]
        crop_min = step_idx
        crop_max= crop_min + filt_width_idx
        
        # ==== Import data and crop ====
        time = data_dict[pwm][delta_step]["data"][crop_min:crop_max,0] -  data_dict[pwm][delta_step]["data"][crop_min,0]
        filt_theta = data_dict[pwm][delta_step]["data"][crop_min:crop_max,2]
        delta_list = data_dict[pwm][delta_step]["data"][crop_min:crop_max,3]

        # ==== Compute initial state of the balance ====
        theta_0 = filt_theta[0]
        theta_dot_0 = (filt_theta[1] - filt_theta[0]) / (time[1] - time[0])
        theta_state_init = [theta_0, theta_dot_0]
        
        # ==== RUn the optimization ====
        bounds = [(-1, 0), (-1, 0)]
        res = minimize(
            cost_full_system,
            params_guess,
            args=(time, known_params, delta_list, filt_theta),
            method='Nelder-Mead',
            bounds=bounds,
            options={
                'disp': False,
                'xtol': 1e-8,   # tighten convergence in parameters
                'fatol': 1e-8,  # tighten convergence in function value (requires recent SciPy versions)
                'maxiter': 1000,
                'maxfev': 5000
            }       
        )

        # ==== Simulate with optimized parameters ==== 
        params_opt = res.x
        print(f'      Solution: I = {params_opt[0]:.4f} and b = {params_opt[1]:.4f}')
        theta_pred, *_ = full_system(params_opt, time, known_params, delta_list, theta_state_init)
        data_dict[pwm][delta_step]["fitting"]  = params_opt
        
        # ==== Check the validity of the fitting ==== 
        # The fitting is valided using an arbitrary threshold, determine empirically
        if res.fun < 100: 
            print(Fore.GREEN + f'      Valid: Final cost {res.fun:.3f} < 100'+ Style.RESET_ALL)
            data_dict[pwm][delta_step]["valid"] = True
        else:
            print(Fore.RED + f'      Not Valid: Final cost {res.fun:.3f} > 100' + Style.RESET_ALL)
            data_dict[pwm][delta_step]["valid"] = False
        
        # ==== plot ====
        ax.plot(time, theta_pred,  label=f'$\\theta(t)$ pred: beta = {params_opt[0]}, I = {params_opt[1]}', linewidth=2)
        ax.plot(time, filt_theta, label='$\\theta(t)$ filt', linewidth=2)
        ax.set_title(f'$\\Delta \\gamma ={delta_step[0]:.3f} \\rightarrow {delta_step[1]:.3f}$')
        ax.set_ylabel('Theta $\\theta$ [deg]')
        ax.grid()

plt.tight_layout()
fig_name = PLT_PATH + os.sep + 'ALL_fitting_grid.png'
plt.savefig(fig_name, dpi=400, bbox_inches='tight')
plt.show()

#%% PLOT THE FITTING COEFS AS FUNCTION OF THE STEPS SIZE

print(Back.GREEN + "Plot all coefficients ..." + Style.RESET_ALL)

# Create a list to store the coefs
coefs_list = [[], []] # I, b

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
for pwm in pwm_list:
    for delta_step in delta_dict[pwm]:        
        fit_coefs = data_dict[pwm][delta_step]["fitting"]
        x_vals = delta_step[1]
        
        coefs_list[0].append(fit_coefs[0])
        coefs_list[1].append(fit_coefs[1])

        
        ax[0].scatter(x_vals, fit_coefs[0], c='b', label=f'PWM {pwm:.2f}', zorder=2)
        ax[0].set_title('Variation of coefficient $I$')
        ax[0].set_xlabel('Target $\\Delta \\gamma$')
        ax[0].set_ylabel('I')
        ax[0].grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)

        ax[1].scatter(x_vals, fit_coefs[1], c='b', label=f'PWM {pwm:.2f}', zorder=2)
        ax[1].set_title('Variation of coefficient $b$')
        ax[1].set_xlabel('Target $\\Delta \\gamma$')
        ax[1].set_ylabel('b')
        ax[1].grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)


# ax[0].legend()
# ax[1].legend()
# fig.suptitle(f"From an initial $\\Delta \\gamma$ = {delta_step[0]:.2f}", fontsize=16)

plt.tight_layout()
fig_name = PLT_PATH + os.sep +  'I_b_vs_step_size.pdf'
plt.savefig(fig_name, format='pdf', dpi=300, bbox_inches='tight')
plt.show()

# Store the mean values of the coefs in a .txt file
I_mean = np.mean(coefs_list[0])
b_mean = np.mean(coefs_list[1])
file_path = os.path.join(DATA_PATH, 'balance_dynamic_coefs')
np.savetxt(file_path, [I_mean, b_mean], fmt="%.6f")
print("---> Mean coefficients")
print(f"     I mean = {I_mean}, b mean = {b_mean}")
print(f"     Saved in {file_path}")

        
        
        