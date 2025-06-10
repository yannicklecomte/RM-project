# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:16:38 2025

@author: Y. Lecomte
"""
#%% Initialization and packages
import numpy as np
import matplotlib.pyplot as plt
import os 
import re
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

#%% Define process function ===================================================

def load_pd_file_numpy(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    kp, kd, gamma_limit = None, None, None

    for line in lines:
        if 'Kp' in line:
            match = re.search(r"Kp\s*=\s*([0-9.eE+-]+)", line)
            if match:
                kp = float(match.group(1))
        elif 'Kd' in line:
            match = re.search(r"Kd\s*=\s*([0-9.eE+-]+)", line)
            if match:
                kd = float(match.group(1))
        elif 'Limitations of the gamma control' in line:
            match = re.search(r"\+/-\s*([0-9.eE+-]+)", line)
            if match:
                gamma_limit = float(match.group(1))

    if kp is None or kd is None:
        raise ValueError("Kp or Kd not found in the file.")
    if gamma_limit is None:
        raise ValueError("Gamma limitation not found in the file.")

    # Keep only lines that start with a number
    data_lines = [line for line in lines if re.match(r'^\s*[-+]?\d', line)]
    data = np.array([
        list(map(float, line.strip().split('\t')))
        for line in data_lines if line.strip()
    ])

    return kp, kd, gamma_limit, data


#%% Load the data =============================================================

print(Back.GREEN + "> Load the system's control data ..." + Style.RESET_ALL)

campaign = 2 # number of the campaign
pwm = 40  # reference pwm    
data_dict = {}
Ne = 10   # Number of episodes

case = "DATA_sequences"

CLIP_NAMES = os.listdir(case)
clip_list = [re.findall(r'\d+', file) for file in CLIP_NAMES]
clip_list = [float(num) for sublist in clip_list for num in sublist]

for clip in clip_list:
    print(f'---> Gamma clip {clip}')
    data_dict[clip] = {}

    all_data_list = []
    time_array = None
    target_array = None

    for i in range(Ne):
        file_path = os.path.join(case, f'gamma_clip_{clip:.0f}', f"episode_{i}", '40.00.dat')
        kp, kd, g_clip, data = load_pd_file_numpy(file_path)

        # Store control coefficients and gamma_clip only once
        if i == 0:
            data_dict[clip] = {"control_coeffs": [kp, kd],
                               "gamma_clip": g_clip
                              }

        if time_array is None and target_array is None:
            time_array = data[:, 0]  # Assuming time is in column 0
            target_array = data[:,6]

        all_data_list.append(data[:, 1])  # Assuming the quantity of interest is in column 1

    # Convert to numpy array of shape (Ne, time_points)
    all_data_array = np.array(all_data_list)

    # Compute ensemble average and standard deviation along axis 0 (episodes)
    ensemble_avg = np.mean(all_data_array, axis=0)
    ensemble_std = np.std(all_data_array, axis=0)

    # Store results in data_dict without overwriting control_coeffs and gamma_clip
    data_dict[clip].update({
        "time": time_array, 
        "data": np.column_stack((ensemble_avg, ensemble_std)), 
        "target": target_array
    })


print(Back.GREEN + "> Control data loading complete." + Style.RESET_ALL)

# Define the path where the plots will be saved
PLT_PATH = os.path.join(case, 'plots')
if not os.path.exists(PLT_PATH):
    os.makedirs(PLT_PATH)


#%% Plots =====================================================================

for idx, clip in enumerate(clip_list):
    time_list = data_dict[clip]['time']
    theta_list = data_dict[clip]['data'][:,0]
    std_list = data_dict[clip]['data'][:,1]
    target_list = data_dict[clip]['target']
    kp, kd = data_dict[clip]["control_coeffs"]
    plt.figure(figsize=(11,1.8))
    
    plt.fill_between(time_list, 
                     theta_list - std_list, 
                     theta_list + std_list, 
                     color='cyan', 
                     alpha=0.5, 
                     label='Standard deviation', zorder=3)

    plt.plot(time_list, theta_list, color='blue', label='Controlled balance', zorder=3)
    plt.plot(time_list, target_list, c='r', linestyle='--', label='Target', zorder=1)
    plt.ylabel('$\\theta$ [deg]')
    if clip == 14.0:
        plt.xlabel('Time [s]')

    plt.title(f"$K_p$ = {kp:.2f}, $K_d$ = {kd:.2f} — "
              f"$\\Delta\\gamma_{{\\mathrm{{clip}}}}$ = ±{clip}")   
    plt.legend(fontsize=7)
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)
    plt.xlim(0, 40)
    filename = os.path.join(PLT_PATH, f'g_clip_{clip}.pdf')
    plt.savefig(filename, format='pdf', dpi=500, bbox_inches='tight')
    plt.show()
