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


# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=16)   # X and Y axis label size

long_fig_size = (8,3)
square_fig_size = (5,5)
fig_size = (6,3)

auto_fitting = False
cutoff = 1

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)  # zero-phase distortion -> SP ;)
            
   
def map_pwm(x):
    return ((x-5)/5)*100    
#%% Load the data + filtering

DATA = "Static"
campaign = 2
data_dict = {}
pwm_dict = {}

CASE_PATH = DATA + os.sep + f"campaign_{campaign}"
SAMPLE_NAMES = os.listdir(CASE_PATH)
PLT_PATH = CASE_PATH + os.sep + 'plots'
if not os.path.exists(PLT_PATH):
    os.mkdir(PLT_PATH)

#extract the values of PWM from the files name in the folder
sample_list = [re.findall(r"\d+", file) for file in SAMPLE_NAMES]
sample_list = [float(num) for sublist in sample_list for num in sublist]

# Add each data case of the campaign in a dictionnary
for sample in sample_list:
    print(sample)
    data_dict[sample] = {}
    SAMPLE_PATH = os.path.join(CASE_PATH,  f'{sample:.0f}_samples')
    print(SAMPLE_PATH)
    PWM_NAMES = os.listdir(SAMPLE_PATH)
    pwm_list = [re.findall(r"\d+.\d+\d+", file) for file in PWM_NAMES]
    pwm_list = [float(num) for sublist in pwm_list for num in sublist]

    pwm_dict[sample] = pwm_list
    for pwm in pwm_list:
        filepath = os.path.join(SAMPLE_PATH, f"{pwm:.2f}.dat")
        data = np.loadtxt(filepath, skiprows=1) 
        # Extract theta values + first and second derivatives
        thrust = data[:,0] 
        mean_thrust = np.mean(thrust)
        std_thrust = np.std(thrust, ddof=1)
        
        rpm = data[:,1] 
        mean_rpm = np.mean(rpm)
        std_rpm = np.std(rpm, ddof=1)

        # save all in the dictionnary
        data_dict[sample][pwm] = {
            "data": np.column_stack((thrust, rpm)),
            "mean": [mean_thrust, mean_rpm], 
            "std": [std_thrust, std_rpm]
            }   
        
        
        
#%% THRUST vs RPM

for sample in sample_list:
    plt.figure(figsize=fig_size)
    for pwm in pwm_dict[sample]:
        mean_thrust = data_dict[sample][pwm]["mean"][0]
        std_thrust = data_dict[sample][pwm]["std"][0]
        mean_rpm = data_dict[sample][pwm]["mean"][1]
        std_rpm = data_dict[sample][pwm]["std"][1]

        plt.scatter(mean_rpm, mean_thrust, c='darkblue', zorder=2) 
        plt.errorbar(mean_rpm, mean_thrust, xerr=std_rpm, yerr=std_thrust, fmt='o', color='darkblue', capsize=4, capthick=2)

    plt.title(f'Static curve for {sample} samples')
    plt.xlabel('RPM')
    plt.ylabel('Thrust [N]')
    plt.xlim(1500, 5300); plt.ylim(-1,5)
    plt.grid()
    fig_name = PLT_PATH + os.sep +  f'T_vs_RPM_{sample}_samples.png'
    # plt.savefig(fig_name, dpi=400, bbox_inches='tight')
    plt.show()
        
    
#%% INDIVIDUAL FITTING THRUST vs RPM

for sample in sample_list:
    thrust_list = []
    rpm_list = []
    for pwm in pwm_dict[sample]:    
        thrust_list.append(data_dict[sample][pwm]["mean"][0])
        rpm_list.append(data_dict[sample][pwm]["mean"][1])
        
    coeffs = np.polyfit(rpm_list, thrust_list, deg=2)
    x_fit = np.linspace(min(rpm_list), max(rpm_list), 100)
    y_fit = np.polyval(coeffs, x_fit)

    plt.figure(figsize=fig_size)
    plt.scatter(rpm_list, thrust_list, c='lightblue', edgecolors='black', s=45, marker='o', zorder=2)
    plt.plot(x_fit, y_fit, c='black', label='Fitted function', zorder=1)
    plt.title(f'Static curve for {sample} samples')
    plt.xlabel('RPM')
    plt.ylabel('Thrust [N]')
    # plt.xlim(1500, 5300); plt.ylim(0,5)
    plt.grid()
    fig_name = PLT_PATH + os.sep +  f'FIT_T_vs_RPM_{sample}_samples.png'
    plt.savefig(fig_name, dpi=400, bbox_inches='tight')
    plt.show()
    
#%% ALL FITTING THRUST vs RPM

plt.figure(figsize=fig_size)
for sample in sample_list:
    thrust_list = []
    rpm_list = []
    for pwm in pwm_dict[sample]:    
        thrust_list.append(data_dict[sample][pwm]["mean"][0])
        rpm_list.append(data_dict[sample][pwm]["mean"][1])
        
    coeffs = np.polyfit(rpm_list, thrust_list, deg=2)
    x_fit = np.linspace(min(rpm_list), max(rpm_list), 100)
    y_fit = np.polyval(coeffs, x_fit)

    plt.plot(x_fit, y_fit, label=f'{sample} Samples')
    
    
plt.title('Static curve: all fittings')
plt.xlabel('RPM')
plt.ylabel('Thrust [N]')
plt.xlim(1500, 5300); plt.ylim(0,5)
plt.grid(); plt.legend()
fig_name = PLT_PATH + os.sep +  'All_FIT_T_vs_RPM.png'
# plt.savefig(fig_name, dpi=400, bbox_inches='tight')
plt.show()



#%% INDIVIDUAL FIT PWM VS RPM

for sample in sample_list:
    thrust_list = []
    rpm_list = []
    std_rpm_list = []
    pwm_map_list = []
    for pwm in pwm_dict[sample]:   
        pwm_map_list.append(map_pwm(pwm))
        rpm_list.append(data_dict[sample][pwm]["mean"][1])
        std_rpm_list.append(data_dict[sample][pwm]["std"][1])


    coeffs = np.polyfit(pwm_map_list[6:], rpm_list[6:], deg=3)
    x_fit = np.linspace(min(pwm_map_list[6:]), max(pwm_map_list), 100)
    y_fit = np.polyval(coeffs, x_fit)
    print(f'For {sample} samples, a = {coeffs[0]}, b = {coeffs[1]} and c = {coeffs[2]} and d = {coeffs[3]}')

    plt.figure(figsize=fig_size)
    # plt.scatter(pwm_map_list, rpm_list, c='lightblue', edgecolors='black', s=35, marker='o', zorder=3)
    # plt.plot(pwm_map_list, rpm_list, c='lightblue', zorder=3)

    plt.plot(x_fit, y_fit, c='black', label='Fitted function', zorder=2)
    plt.errorbar(pwm_map_list, rpm_list, yerr=std_rpm_list, fmt='o', color='darkblue', capsize=4, capthick=2)
    plt.axvline(20); plt.axvline(60)
    plt.title(f'Static curve for {sample:.0f} samples')
    plt.xlabel('Power input $\\gamma$ [\%]')
    plt.ylabel('RPM')
    # plt.xlim(1500, 5300); plt.ylim(0,5)
    plt.grid()
    fig_name = PLT_PATH + os.sep +  f'FIT_RPM_vs_PWM_{sample}_samples.png'
    # plt.savefig(fig_name, dpi=400, bbox_inches='tight')
    plt.show()
  
#%% INDIVIDUAL FIT THRUST VS PWM
fig_size = (6,2.8)

for sample in sample_list:
    thrust_list = []
    rpm_list = []
    pwm_map_list = []
    for pwm in pwm_dict[sample]:   
        pwm_map_list.append(map_pwm(pwm))
        thrust_list.append(data_dict[sample][pwm]["mean"][0])
        
    coeffs = np.polyfit(pwm_map_list, thrust_list, deg=1)
    x_fit = np.linspace(min(pwm_map_list), max(pwm_map_list), 100)
    y_fit = np.polyval(coeffs, x_fit)
    print(f'For {sample} samples, K = {coeffs[0]} and p = {coeffs[1]}')

    plt.figure(figsize=fig_size)
    plt.scatter(pwm_map_list, thrust_list, c='forestgreen', edgecolors='black', s=50, marker='o', zorder=3)
    plt.plot(x_fit, y_fit, c='black', linestyle='--',  linewidth=2, label='Fitted function', zorder=2)
    # plt.title(f'Static curve for {sample:.0f} samples')
    plt.xlabel('Power input $\\gamma$ [\%]')
    plt.ylabel('Thrust [N]')
    plt.xlim(20, 60); plt.ylim(0, 6)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
    fig_name = PLT_PATH + os.sep +  f'FIT_T_vs_PWM_{sample}_samples.pdf'
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.show()
    
    
