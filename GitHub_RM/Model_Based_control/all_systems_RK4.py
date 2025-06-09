# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:15:04 2025

@author: Y. Lecomte
"""

#%% Initialization and packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os 
from colorama import Style, Back, init
init()  

# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=14)  


# Define the main where to load the system coefficients
MAIN_PATH_balance = r"..\Balance_system_identification"
MAIN_PATH_motor = r"..\Motor_system_identification"
    
#%% Define the parameters of the system 
print(Back.GREEN + "> Loading of all the system coefficients" + Style.RESET_ALL)

# =================== System parameters for gamma_ref = 40% ===================

l  = 0.68 # lenght between propeller and center of rotation

print('---> Motor System')
# Load static chracteristics of the motor
motor_static_campaign = 2 
balance_static_path = os.path.join(MAIN_PATH_motor,"Static_characterization",f"campaign_{motor_static_campaign}","data","static_coefs")
Kt, T0 = np.loadtxt(balance_static_path)
print(f'     Kt = {Kt:.3f}, T0 = {T0:.3f}')

# Define dynamic chracteristics of the motor
tau_UP = 0.1305
tau_DOWN = 0.1733
tau_d_UP = 0.035
tau_d_DOWN = 0.01

print('---> Balance System')
# Load static chracteristics of the balance
balance_static_campaign = 4 # Without the panels
balance_static_path = os.path.join(MAIN_PATH_balance,"Static_characterization",f"campaign_{balance_static_campaign}","data","static_coefs")
C, theta_0 = np.loadtxt(balance_static_path)
print(f'     C = {C:.3f}, theta_0 = {theta_0:.3f}')

# Define dynamic chracteristics of the balance
K_theta = (l*Kt)/C
tau_0 = K_theta * theta_0

balance_dyanmic_campaign = 6
balance_dynamic_path = os.path.join(MAIN_PATH_balance,"Dynamic_characterization",f"campaign_{balance_dyanmic_campaign}","data","balance_dynamic_coefs")
I, b = np.loadtxt(balance_dynamic_path)
print(f'     I = {I:.6f}, b = {b:.6f}')

# Define the reference power input  and the associated thrust
gamma_ref = 40
T_ref = Kt*gamma_ref +T0


#%% ALL FUNCTIONS


def pwm_to_RPM(pwm_map, coeffs=(0.01349794364264695, -1.8672816543885395, 151.8771526138152, 746.4757887495163)):
    return np.polyval(coeffs, pwm_map)


# ===== RK4 Integrators =====
def rk4_step(f, y, t, dt, *args):
    k1 = f(y, t, *args)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = f(y + dt * k3, t + dt, *args)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# ===== System Dynamics =====
def gamma_model(y, t, tau_d, gamma_cmd):
    return (gamma_cmd - y) / tau_d

def motor_model(y, t, tau, Kt, T0, gamma):
    return ((Kt * gamma + T0) - y) / tau

def second_order_system(state, t, I, b, K_theta, tau_0, T_R, T_L):
    theta, theta_dot = state
    dtheta_dt = theta_dot
    dtheta_dot_dt = -(b/I) * theta_dot - (K_theta/I) * theta + (l/I) * (T_R - T_L) + (tau_0/I)
    return np.array([dtheta_dt, dtheta_dot_dt])

# ===== Controller Simulation =====
def controller_simualtion(PD_params, target_list, dt, gamma_clip):
    Kp, Kd = PD_params
    n_steps = len(target_list)

    theta = np.zeros(n_steps)
    theta_dot = np.zeros(n_steps)
    gamma_control = np.zeros(n_steps)
    gamma_cmd_L = np.zeros(n_steps)
    gamma_cmd_R = np.zeros(n_steps)
    error = np.zeros(n_steps)
    T_R = np.zeros(n_steps)
    T_L = np.zeros(n_steps)
    time_list = np.zeros(n_steps)

    theta_init = target_list[0]
    theta_state_init = np.array([theta_init, 0.0])
    theta[0] = theta_init

    tau_L, tau_R = tau_UP, tau_UP
    tau_d_L, tau_d_R = tau_d_UP, tau_d_UP
    delta_gamma_init = (theta_init - theta_0) / C
    gamma_L_init = gamma_ref - 0.5 * delta_gamma_init
    gamma_cmd_L[0] = gamma_L_init
    gamma_R_init = gamma_ref + 0.5 * delta_gamma_init
    gamma_cmd_R[0] = gamma_R_init
    T_init_L = Kt * gamma_L_init + T0
    T_L[0] = T_init_L
    T_init_R = Kt * gamma_R_init + T0
    T_R[0] = T_init_R

    derivative_prev = 0.0
    
    for i in range(1, n_steps):
        time_list[i] = time_list[i-1] + dt
        delta_gamma_S = (target_list[i] - theta_0) / C

        error[i] = theta[i-1] - target_list[i-1]
        
        # Evaluate the filtered error value 
        # alpha = tau_d / (tau_d + dt)
        alpha = 0.01
        derivative_filtered = alpha * derivative_prev + (1 - alpha) * ((error[i] - error[i-1]) / dt)
        derivative_prev = derivative_filtered
                

        gamma_control[i] = np.clip(delta_gamma_S + (Kp * error[i] + Kd * derivative_filtered),
                                   delta_gamma_S - gamma_clip, delta_gamma_S + gamma_clip)
        gamma_cmd_L[i] = np.clip((gamma_ref - 0.5 * gamma_control[i]), 20, 60)
        gamma_cmd_R[i] = np.clip((gamma_ref + 0.5 * gamma_control[i]), 20, 60)

        t_now = time_list[i-1]

        gamma_L_init = rk4_step(gamma_model, gamma_L_init, t_now, dt, tau_d_L, gamma_cmd_L[i])
        gamma_R_init = rk4_step(gamma_model, gamma_R_init, t_now, dt, tau_d_R, gamma_cmd_R[i])

        T_init_L = rk4_step(motor_model, T_init_L, t_now, dt, tau_L, Kt, T0, gamma_L_init)
        T_init_R = rk4_step(motor_model, T_init_R, t_now, dt, tau_R, Kt, T0, gamma_R_init)
        T_L[i], T_R[i] = T_init_L, T_init_R

        if (Kt * gamma_L_init + T0) > T_L[i]:
            tau_L, tau_d_L = tau_UP, tau_d_UP
        else:
            tau_L, tau_d_L = tau_DOWN, tau_d_DOWN

        if (Kt * gamma_R_init + T0) > T_R[i]:
            tau_R, tau_d_R = tau_UP, tau_d_UP
        else:
            tau_R, tau_d_R = tau_DOWN, tau_d_DOWN

        theta_state_init = rk4_step(second_order_system, theta_state_init, t_now, dt,
                                    I, b, K_theta, tau_0, float(T_R[i]), float(T_L[i]))
        theta_state_init[0] = np.clip(theta_state_init[0], -20, 20)
        theta[i], theta_dot[i] = theta_state_init

    return theta, gamma_control, gamma_cmd_L, gamma_cmd_R, error, time_list, T_R, T_L

# ===== Cost Function =====
def cost_function(PD_params, target_list, dt, gamma_clip, noise_level):
    theta, *_ = controller_simualtion(PD_params, target_list, dt, gamma_clip)
    # Number of samples
    Nt = len(theta)
    # Compute the noramalized error
    err = (np.linalg.norm(theta - target_list) + np.random.randn() * noise_level) / np.sqrt(Nt)
    return err


# ===== Cost Function with Penalization =====
def cost_function_penalty(PD_params, penalty_params, target_list, dt, gamma_clip, noise_level):
    alpha_p, alpha_d = penalty_params
    kp, kd = PD_params
    theta, *_ = controller_simualtion(PD_params, target_list, dt, gamma_clip)
    # Number of samples
    Nt = len(theta)
    err = (np.linalg.norm(theta - target_list) + np.random.randn() * noise_level) / np.sqrt(Nt)
    err_pen = err + alpha_p*kp**2 + alpha_d*kd**2
    return err_pen


#%% FUNCTION THAT DEFINE THE CONTROL SEQUENCE

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



