# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:34:19 2025

@author: Y. Lecomte
"""

#%% Packages, initialization and functions

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from colorama import Fore, Style, Back, init
init()  

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=16)

# fig_size = (7,3)
fig_size = (6,3.5)

def map_pwm(x):
    return ((x-5)/5)*100

#%% Load the data 

CASE = "Static_characterization"
campaign = 5
print(Back.GREEN + f"Loading of the data of campaign {campaign:.0f} ..." + Style.RESET_ALL)

# Initialize the dictionnary to store the data
data_dict = {}
delta_dict = {}

# Path to axcess the current campaign data
CASE_PATH = CASE + os.sep + f"campaign_{campaign}"
PWM_NAMES = os.listdir(CASE_PATH)

# Create a path to save the plots
PLT_PATH = CASE_PATH + os.sep + 'plots'
if not os.path.exists(PLT_PATH):
    os.mkdir(PLT_PATH)
    
# Create a path to save the coefficients
DATA_PATH = CASE_PATH + os.sep + 'data'
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

# Extract the values of PWM from the files name in the folder
pwm_list = [re.findall(r'\d+\.\d+|\d+', file) for file in PWM_NAMES]
pwm_list = [float(num) for sublist in pwm_list for num in sublist]
pwm_list.sort()

# Add each data case of the campaign in a dictionnary
for pwm in pwm_list:
    print(f'---> Case: PWM {pwm} %')
    data_dict[pwm] = {}
    PWM_PATH = os.path.join(CASE_PATH, f'{pwm:.2f}_PWM')
    print(f'     Path: {PWM_PATH}')
    DELTA_NAMES = os.listdir(PWM_PATH)
    
    delta_list = [re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', file) for file in DELTA_NAMES]
    delta_list = [float(num) for sublist in delta_list for num in sublist]
    delta_dict[pwm] = delta_list

    for delta in delta_list: 
        # Construct the file name
        filepath = os.path.join(PWM_PATH, f"{delta:.5f}.dat")
        # Load the data, skipping the first row (header)
        data = np.loadtxt(filepath, skiprows=1)  
        theta = data[:,1]
        # Compute mean and standard deviation
        mean = np.mean(theta)
        std = np.std(theta, ddof=1)
        # Remove outliers values using mean and std
        z_scores = (theta - mean) / std        
        outlier_indices = np.where(np.abs(z_scores) > 3)[0]
        filt_theta = theta.copy()
        for idx in outlier_indices:
            if idx >= 4:
                filt_theta[idx] = filt_theta[idx - 4]
            else:
                filt_theta[idx] = mean  
                
        data_dict[pwm][delta] = {
            "data": data,
            "mean": mean,
            "std": std
            }   
        


#%% PLOTS AND LINEAR REGRESSION + SAVE IN A GIVEN FILE
print(Back.GREEN + "Plotting the static fitting ..." + Style.RESET_ALL)

# Figure size for the report plot
fig_size = (6,2.8)


for pwm in pwm_list:    
    delta_list = delta_dict[pwm]
    mean_theta_list = []
    std_theta_list = []
    
    for delta in delta_list:
        mean_theta = data_dict[pwm][delta]["mean"]
        std = data_dict[pwm][delta]["std"]

        mean_theta_list.append(mean_theta)
        std_theta_list.append(std)

    slope, intercept = np.polyfit(delta_list, mean_theta_list, 1)
    data_dict[pwm]["slope"] =  slope
    data_dict[pwm]["intercept"] =  intercept
    fitted_theta = [slope * g + intercept for g in delta_list]
    print('---> Fitting coefficients:')   
    print(f'     C: {slope}, theta_0: {intercept}')
    
    # Save the coefficients
    coefs_file_name = os.path.join(DATA_PATH, 'static_coefs')
    np.savetxt(coefs_file_name, [slope, intercept], fmt="%.6f")
    print(f'---> Saved in {coefs_file_name}')   


    # Plot + Save 
    plt.figure(figsize=fig_size)
    plt.errorbar(delta_list, mean_theta_list, yerr=std_theta_list, fmt='o', color='forestgreen', capsize=5, capthick=2, zorder=3)
    plt.scatter(delta_list, mean_theta_list, color='forestgreen', edgecolors='black', label="Mean theta", zorder=4)
    plt.plot(delta_list, fitted_theta, c='k', linestyle='--', label='Fitted line', zorder=2)
    plt.xlabel('Power Difference: $\\Delta \\gamma = \\gamma_{R} - \\gamma_{L}$')
    plt.ylabel('Roll angle $\\theta$ [deg]')
    
    # # Remove top and right spines
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)

    fig_name = PLT_PATH + os.sep +  f'static_characterization_{pwm:.1f}_PWM.pdf'
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.show()
    
#%% PLOT all fitted lines


plt.figure(figsize=fig_size)
for pwm in pwm_list:
    delta_list = delta_dict[pwm]
    slope = data_dict[pwm]["slope"]
    intercept = data_dict[pwm]["intercept"]

    fitted_theta = [slope * g + intercept for g in delta_list]
    pwm_map = ((pwm-5)/5)*100 

    plt.plot(delta_list, fitted_theta, label=f'$\\gamma$ = {pwm_map:.1f}\%')

plt.xlabel('Power ratio $r=\\gamma_2 / \\gamma_1$')
plt.ylabel('Roll angle $\\theta$ [deg]')
plt.title('Fitted lines')
plt.grid()
plt.legend()
fig_name = PLT_PATH + os.sep +  'all_fitting.png'
plt.savefig(fig_name, dpi=400, bbox_inches='tight')
plt.show()
