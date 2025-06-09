# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:16:38 2025

@author: Y. Lecomte
"""

#%% PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os 
import all_systems_RK4 as sys
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from colorama import Style, Back, init
init()  

# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=14)  

# Reference power input for the current sequence 
gamma_ref = 40

#%% INITIALIZATION 

# ==== Important parameters to define ====
######################################
# Control action limitations
gamma_clip = 8
# Optimization solver ('Nelder-Mead', 'BFGS', 'BO', ...)
opti_solver = 'Nelder-Mead'
# Cost function type ('classic', 'penalty', ...)
cost_type = 'penalty'
# Penalty parameters (for penalized cost function)
alpha_p, alpha_d = 0.05, 0.5
noise_std = 0
######################################

# === Create output folders ===
OUT_PATH = os.path.join('MB_optimization', f'{opti_solver}', f'{cost_type}_cost', f'g_clip_{gamma_clip}')
os.makedirs(OUT_PATH, exist_ok=True)

PLT_PATH = os.path.join(OUT_PATH, 'plots')
os.makedirs(PLT_PATH, exist_ok=True)
    
DATA_PATH = os.path.join(OUT_PATH, 'data')
os.makedirs(DATA_PATH, exist_ok=True)


#%% CONTROL SEQUENCE
print(Back.GREEN + "> Definition of the sequence to be controlled ..." + Style.RESET_ALL)

# Step sequence definition
target_val = [0, 10, -10, 5, -5]  

# Sequence parameters
fs = 400      # Sampling frequency (Hz)
dt = 1 / fs   # Time step
T_step = 4    # Duration of each step (s)

# Create the full sequence
target_list = sys.control_sequence(target_val, fs, T_step)
print(f'---> Steps sequence created: {target_val}')

#%% OPTIMIZATION
print(Back.GREEN + "> Optimization of the PD control law ..." + Style.RESET_ALL)

# Callbacks to track optimization progress
def callbackF(xk):
    explored_pts.append(np.copy(xk))  
    print(f"     Step: Current params = {xk}")

def callbackF_skopt(result):
    xk = result.x_iters[-1]  # Get the latest tested point
    explored_pts.append(np.copy(xk))
    print(f"     Step: Current params = {xk}")


# ==== Optimization setup ====
initial_guess = [1, 1]                     # Initial guess for (Kp, Kd)
x_min_kp, x_max_kp = 0.05, 2
x_min_kd, x_max_kd = 0.05, 1.5
bounds = [(x_min_kp,  x_max_kp), (x_min_kd, x_max_kd)]                # Bounds for (Kp, Kd)
explored_pts = [initial_guess]             # List to store the explored points

# ==== Choice of optimization method ====
print(f'---> Optimization with {opti_solver}')

# Start the time counting
start_time = time.time()
if opti_solver != 'BO': # Gradient-based / direct search methods
    if cost_type == 'penalty':
        print('  -> Cost function with penalty')
        args = ([alpha_p, alpha_d], target_list, dt, gamma_clip, noise_std)

        result = minimize(sys.cost_function_penalty, initial_guess, bounds=bounds, method=f'{opti_solver}', args=args, callback=callbackF)
        print(f"Optimal parameters found: kp = {result.x[0]}, kd = {result.x[1]}")
        print("Finished in %s s" % (time.time() - start_time))   

    else:
        print('  -> Classic cost function')
        args = (target_list, dt, gamma_clip)
        result = minimize(sys.cost_function, initial_guess, bounds=bounds, method=f'{opti_solver}', args=args, callback=callbackF)
        print(f"Optimal parameters found: kp = {result.x[0]}, kd = {result.x[1]}")
        print("Finished in %s s" % (time.time() - start_time))   

else: # Bayesian Optimization
    search_space = [
    Real(0, 5, name='kp'),
    Real(0, 5, name='kd')
    ]

    if cost_type == 'penalty':
        print('  -> Cost function with penalty')
        args = ([alpha_p, alpha_d], target_list, dt, gamma_clip, noise_std)
    
        @use_named_args(search_space)
        def bayes_cost_function(**params):
            kp = params['kp']
            kd = params['kd']
            return sys.cost_function_penalty([kp, kd], *args)
    else:
        print('  -> Classic cost function')
        args = (target_list, dt, gamma_clip)
    
        @use_named_args(search_space)
        def bayes_cost_function(**params):
            kp = params['kp']
            kd = params['kd']
            return sys.cost_function([kp, kd], *args)

    result = gp_minimize(
        func=bayes_cost_function,
        dimensions=search_space,
        n_calls=40,
        n_initial_points=8,
        acq_func="EI",                # "EI" often works well for deterministic systems
        acq_optimizer="auto",
        xi=0.01,                      # Small exploration factor
        noise=1e-10,                  # Assuming your function is deterministic
        random_state=42,
        callback=[callbackF_skopt] 

    )
    
    explored_pts = result.x_iters  # List of all tested parameter sets
    print(f"  -> Optimal parameters found: kp = {result.x[0]}, kd = {result.x[1]}")
    print("  -> Finished in %s s" % (time.time() - start_time))   


print('---> Optimization DONE!')

#%% PLOTS AND SAVE

print(Back.GREEN + "> Resulting saving ..." + Style.RESET_ALL)

# ==== Final simulation using optimal controller coefficients ====
Kp_opt, Kd_opt = result.x
theta_final, control_final, gamma_L_final, gamma_R_final, error_final, time_list, T_R, T_L = sys.controller_simualtion([Kp_opt, Kd_opt], target_list, dt, gamma_clip)
 
# ==== Save opti coeffs in a .txt file ====
coef_filepath = os.path.join(DATA_PATH, 'opti_coeffs')
np.savetxt(coef_filepath, [Kp_opt, Kd_opt], header="Kp\tKd", comments='', delimiter='\t', fmt='%1.5f')
print(f"---> Optimal PD: Kp = {Kp_opt:.3f}, Kd = {Kd_opt:.3f}")
print(f"     Saved in {coef_filepath}")

# ==== Save the other data in a .txt file ====
data = np.column_stack((time_list, theta_final, error_final, control_final, gamma_L_final, gamma_R_final, target_list))
file_name = os.path.join(OUT_PATH, f'{gamma_ref:.2f}.dat')
header_txt = (
    f"Simulated control sequence with PD control:\n"
    f"Sequence steps: {target_val} \n"
    f"Kp = {Kp_opt}\n"
    f"Kd = {Kd_opt}\n"
    f"Limitations of the gamma control: +/- {gamma_clip}\n"
    f"Optimization solver: {opti_solver}\n"
    f"Cost function type: {cost_type}\n"
    f"Time(s)\tRoll(deg)\tError\tControl_cmd\tgamma_L\tgamma_R\ttarget"
)
np.savetxt(file_name, data, header=header_txt, comments='', delimiter='\t', fmt='%1.5f')
print("---> All simulation data")
print(f"     Saved in {file_name}")


# ==== Save the optimization track in a .txt file ====
data = np.array((explored_pts))
file_name = os.path.join(OUT_PATH, 'opti_tracking.dat')
header_txt = (
    f"History of the explored points by the optimization process:\n"
    f"Sequence steps: target_val\n"
    f"Kp = {Kp_opt}\n"
    f"Kd = {Kd_opt}\n"
    f"Limitations of the gamma control: +/- {gamma_clip}\n"
    f"Optimization solver: {opti_solver}\n"
    f"Cost function type: {cost_type}\n"
    f"Kp\t Kd"
)
np.savetxt(file_name, data, header=header_txt, comments='', delimiter='\t', fmt='%1.5f')
print("---> Explored points of the optimization")
print(f"     Saved in {file_name}")

# ==== Generated and save the plots ====
print("---> All plots")

# Theta
plt.figure(figsize=(7, 4))
plt.plot(time_list, theta_final, 'ko:', label="$\\theta (t)$", markersize=3)
plt.plot(time_list, target_list, color='r', linestyle='--', label="Target")
plt.title("System Response with Optimized PD")
plt.xlabel("Time (s)")
plt.ylabel("$\\theta (t)$")
plt.grid(True)
plt.legend()
figname = os.path.join(PLT_PATH, 'opti_PD_sys_response.png')
plt.savefig(figname, dpi=400, bbox_inches='tight')
plt.show()

# Control
plt.figure(figsize=(7, 4))
plt.plot(time_list, control_final, 'ko:', label="$\\theta (t)$", markersize=3)
plt.title("Control Response with Optimized PD")
plt.xlabel("Time (s)")
plt.ylabel("Control $\\gamma_{\\Delta}$")
plt.grid(True)
plt.legend()
figname = os.path.join(PLT_PATH, 'opti_PD_sys_control.png')
plt.savefig(figname, dpi=400, bbox_inches='tight')
plt.show()

# Thrust
plt.figure(figsize=(7, 4))
plt.plot(time_list, T_R, 'ko:', label="$T_{R}$", markersize=3)
plt.plot(time_list, T_L, 'ro:', label="$T_{L}$", markersize=3)
plt.title("Thrust Response with Optimized PD")
plt.xlabel("Time (s)")
plt.ylabel("Thrust $T$")
plt.grid(True)
plt.legend()
figname = os.path.join(PLT_PATH, 'opti_PD_sys_thrusts.png')
plt.savefig(figname, dpi=400, bbox_inches='tight')
plt.show()

# Input cmd
plt.figure(figsize=(7, 4))
plt.plot(time_list, gamma_L_final, 'ro:', label="$\\gamma_{L}$", markersize=3)
plt.plot(time_list, gamma_R_final, 'ko:', label="$\\gamma_{R}$", markersize=3)
plt.title("System Response with Optimized PID")
plt.xlabel("Time (s)")
plt.ylabel("Input power cmd $\\gamma$")
plt.grid(True)
plt.legend()
figname = os.path.join(PLT_PATH,'opti_PD_sys_gammas.png')
plt.savefig(figname, dpi=400, bbox_inches='tight')
plt.show()

# Error
plt.figure(figsize=(7, 4))
plt.plot(time_list, error_final, 'ko:', label="error", markersize=3)
plt.title("System Response with Optimized PID")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.grid(True)
plt.legend()
figname = os.path.join(PLT_PATH, 'opti_PD_sys_error.png')
plt.savefig(figname, dpi=400, bbox_inches='tight')
plt.show()
print(f"     Saved in {PLT_PATH}")
