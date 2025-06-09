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
plt.rc('axes', labelsize=14)   # X and Y axis label size

long_fig_size = (8,3)
square_fig_size = (5,5)
fig_size = (6,3.5)

auto_fitting = False
cutoff = 1


# Define the time delay 
tau_d = 0.01

def map_pwm(x):
    return ((x-5)/5)*100

def first_order(y, t, u, tau):
    u_t = np.interp(t, np.arange(len(u)), u)
    dydt = -1/tau * y + 1/tau * u_t
    return dydt

def error(tau, t, u, y_real):
    y0 = y_real[0]
    y_pred = odeint(first_order, y0, t, args=(u, tau)).flatten()
    err = np.mean((y_real - y_pred) ** 2)
    return err

def normalization_down(Signal, filt_width, ref_idx=0):
    signal_offset = Signal[0]  
    signal_final = np.mean(Signal[-filt_width:])  
    return (Signal - signal_offset) / (signal_offset - signal_final)

#%% Load the data + filtering

DATA = "Dynamic_DOWN"
campaign = 9
data_dict = {}
step_dict = {}

CASE_PATH = DATA + os.sep + f"campaign_{campaign}"
PWM_NAMES = os.listdir(CASE_PATH)
PLT_PATH = CASE_PATH + os.sep + 'plots'
if not os.path.exists(PLT_PATH):
    os.mkdir(PLT_PATH)

#extract the values of PWM from the files name in the folder
pwm_list = [re.findall(r"\d+.\d+\d+", file) for file in PWM_NAMES]
pwm_list = [float(num) for sublist in pwm_list for num in sublist]

# Add each data case of the campaign in a dictionnary
for pwm in pwm_list:
    print(pwm)
    data_dict[pwm] = {}
    PWM_PATH = os.path.join(CASE_PATH,  f'{pwm:.2f}_refPWM')
    print(PWM_PATH)
    STEP_NAMES = os.listdir(PWM_PATH)
    
    step_list = [re.findall(r'\d+\.\d+', file) for file in STEP_NAMES]
    step_list = [(float(sublist[0]), float(sublist[1])) for sublist in step_list if len(sublist) == 2]
    step_dict[pwm] = step_list
    
    for step in step_list:
        filepath = os.path.join(PWM_PATH, f"{step[0]:.2f}_{step[1]:.2f}.dat")
        data = np.loadtxt(filepath, skiprows=1) 
        # Extract values
        time = data[:,0] 
        
        thrust = data[:,1] 
        mean_thrust = np.mean(thrust)
        std_thrust = np.std(thrust, ddof=1)
        
        rpm = data[:,2] 
        mean_rpm = np.mean(rpm)
        std_rpm = np.std(rpm, ddof=1)

        gamma = data[:,3]

        # save all in the dictionnary
        data_dict[pwm][step] = {
                    "data": np.column_stack((time, thrust, rpm, gamma))
                    }
        
#%% PLOT ALL RAW CURVE
filt_width=10
for pwm in pwm_list:
    plt.figure(figsize=fig_size)
    ref_pwm_map = ((pwm-5)/5)*100 

    for step in step_dict[pwm]:
        time_list = data_dict[pwm][step]["data"][:,0]
        thrust_list = data_dict[pwm][step]["data"][:,1]
        rpm_list = data_dict[pwm][step]["data"][:,2]
        Ns = len(time_list)
        time_diffs = np.diff(time_list)
        fs = 1/np.mean(time_diffs)
        target_pwm_map = ((step[1]-5)/5)*100 

        plt.plot(time_list, thrust_list, label=f'{target_pwm_map} \%')
        
    plt.title(f'All step UP starting from $\\gamma$ = {map_pwm(step[0]):.1f} \%')
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust $T$ [N]')
    plt.grid()
    fig_name = PLT_PATH + os.sep + f'all_curves_{ref_pwm_map:.1f}_refPWm.png'
    plt.savefig(fig_name, dpi=400, bbox_inches='tight')
    plt.show()


#%% PLOT NORMALIZED ALL STATIC CURVE
filt_width = 10

for pwm in pwm_list:
    fig, ax = plt.subplots(figsize=(6, 3)) 
    ref_pwm_map = ((pwm - 5) / 5) * 100 

    steps = step_dict[pwm]  
    cmap = cm.get_cmap("viridis", len(steps))  
    norm = plt.Normalize(vmin=min(((step[1]-5)/5)*100 for step in steps), vmax=max(((step[1]-5)/5)*100 for step in steps)) 

    for i, step in enumerate(steps):
        time_list = data_dict[pwm][step]["data"][:, 0]
        thrust_list = data_dict[pwm][step]["data"][:, 1]
        rpm_list = data_dict[pwm][step]["data"][:, 2]
        gamma_list = data_dict[pwm][step]["data"][:, 3]

        # Acquisition frequences
        time_diffs = np.diff(time_list)
        fs = 1/np.mean(time_diffs); Ts = 1/fs
        # Locate the index of the step up command
        gamma_array = np.array(gamma_list)
        step_idx = np.where(gamma_array != gamma_array[0])[0][0]
        
        # Cropping of the signal
        Ns = len(time_list)
        shift_signal_idx = 0
        crop_min = step_idx - shift_signal_idx
        crop_max = crop_min + 32
        time_crop = time_list[crop_min:crop_max] - time_list[crop_min] - shift_signal_idx*Ts
        thrust_crop = thrust_list[crop_min:crop_max]
        rpm_crop = rpm_list[crop_min:crop_max]
    
        # RPM - Normalization
        thrust_norm = normalization_down(thrust_crop, filt_width, ref_idx = shift_signal_idx)
        rpm_norm = normalization_down(rpm_crop, filt_width, ref_idx = shift_signal_idx)   
        time_diffs = np.diff(time_list)
        fs = 1/np.mean(time_diffs)
        
        # Plot with Viridis colormap
        color = cmap(norm(((step[1]-5)/5)*100))  # Normalize step value for colormap
        ax.plot(time_crop, thrust_norm, color=color)

    # Step function
    x = [-0.2, 0, 0, 0.8]
    y = [0, 0, -1, -1]
    ax.plot(x, y, c='r', linestyle='--', zorder=3)
    ax.text(0.021, -0.97, '$\\gamma_{cmd}(t)$', fontsize=14, color='red',
        verticalalignment='bottom', horizontalalignment='left')

    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"$\gamma_{\mathrm{target}}$")  

    # ax.set_title(f'Step UP from $\\gamma$ = {ref_pwm_map:.1f} \%')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('$\\hat{T}$')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
    ax.set_xlim(-0.2, 0.8); ax.set_ylim(-1.1, 0.1); ax.set_yticks([-1, -0.5, 0])
    fig_name = PLT_PATH + os.sep + f'ALL_norm_step_DOWN_{ref_pwm_map:.1f}_refPWM.pdf'
    plt.savefig(fig_name, format='pdf', dpi=400, bbox_inches='tight')
    plt.show()


#%% FITTING ALL

# --- Main processing script ---
filt_width = 10  # Number of final points to average for steady-state estimation

for pwm in pwm_list:
    ref_pwm_map = map_pwm(pwm)
    steps = step_dict[pwm]  # All step tests for this PWM

    for i, step in enumerate(steps):
        # ==== Extract time and RPM data ====
        time_list = data_dict[pwm][step]["data"][:, 0]
        thrust_list = data_dict[pwm][step]["data"][:, 1]
        rpm_list = data_dict[pwm][step]["data"][:, 2]
        gamma_list = data_dict[pwm][step]["data"][:, 3]

        # ==== Cropping of the signal ====
        # Locate the index of the step up command
        gamma_array = np.array(gamma_list)
        step_idx = np.where(gamma_array != gamma_array[0])[0][0]
        Ns = len(time_list) # number of sample of the sequence
        # Number of indexes induced by the time delay
        N_shift = int(np.ceil(tau_d / Ts))
        crop_min = step_idx + N_shift
        crop_max = crop_min + 32
        time_crop = time_list[crop_min:crop_max] - time_list[crop_min]
        thrust_crop = thrust_list[crop_min:crop_max]
        rpm_crop = rpm_list[crop_min:crop_max]
    
        # ==== normalize the signal ====
        thrust_norm = -normalization_down(thrust_crop, filt_width)
        rpm_norm = -normalization_down(rpm_crop, filt_width)
        # Normalize target PWM
        target_pwm_map = map_pwm(step[1])

        # Estimate sample frequency (not strictly needed)
        time_diffs = np.diff(time_list)
        fs = 1 / np.mean(time_diffs)

        # Input step signal
        u = np.ones(len(time_crop))
        t_max = time_crop[-1]  # Used in the first_order indexing

        # Optimization setup
        guess = 0.1  # Initial guess for time constant
        bounds_tau = [(1e-3, 1000)]  # Avoid bad values

        # Perform optimization
        p = minimize(
            error,
            guess,
            args=(time_crop, u, thrust_norm),
            bounds=bounds_tau,
            method='Nelder-Mead'
        )

        # Store and print result
        tau_pred = p.x[0]
        print(f"PWM {pwm}, step {map_pwm(step[0]):.1f} % → {map_pwm(step[1]):.1f} %: tau = {tau_pred:.4f} s")

        # ==== Simulate with best-fit tau for the RPM ====
        rpm_pred = odeint(first_order, rpm_norm[0], time_crop, args=(u, tau_pred))[:, 0]
        
        # ==== Simulate with best-fit tau for the Thrust ====
        thrust_pred = odeint(first_order, thrust_norm[0], time_crop, args=(u, tau_pred))[:, 0]
        data_dict[pwm][step]["tau"] = tau_pred

        # Plot comparison
        plt.figure(figsize=(5, 4))
        plt.plot(time_crop, thrust_norm, label="Measured")
        plt.plot(time_crop, thrust_pred, label=f"Model $\\tau$={tau_pred:.2f}s")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Normalized Thrust")
        plt.title(f"Step: {map_pwm(step[0]):.1f} \% → {map_pwm(step[1]):.1f} \%")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


#%% PLOT FITTING COMP


for pwm in pwm_list:
    ref_pwm_map = ((pwm - 5) / 5) * 100 

    steps = step_dict[pwm]
    tau_list = []
    target_step_list = []
    for i, step in enumerate(steps):
        tau =  data_dict[pwm][step]["tau"]
        tau_list.append(tau)

        target_step_list.append(((step[1]-5)/5)*100)
        
    plt.figure(figsize=(5,4))
    plt.scatter(target_step_list, tau_list, zorder=2)
    plt.xlabel('Target $\\gamma$')
    plt.ylabel('Predicted time constant $\\tau$')
    plt.title(f'For steps starting from $\\gamma$ = {ref_pwm_map:.2f} \%')
    plt.grid()
    fig_name = PLT_PATH + os.sep + f'tau_vs_target_{ref_pwm_map:.1f}_refPWM.png'
    plt.savefig(fig_name, dpi=400, bbox_inches='tight')
    plt.show()

        
        