# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:51:27 2025

@author: lyann
"""

#%% Import packages ===========================================================

import numpy as np
import matplotlib.pyplot as plt
import os 

# Figure parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes', labelsize=14)  

data_path = os.path.join('DATA_EI_cost_convergence', 'data')


#%% Load the time serie data of the SINGLE ==================================== 

color_list = ['royalblue', 'firebrick', 'forestgreen']

fig, axs = plt.subplots(1, 2, figsize=(10, 3))  # two side-by-side plots

for i in range(3):
    EI_filename = os.path.join(data_path, f'Case {i+1}', 'EI_list.npy')
    cost_filename = os.path.join(data_path, f'Case {i+1}', 'cost_over_time.npy')

    EI_list = np.load(EI_filename)
    Cost_list = np.load(cost_filename)

    # Plot EI
    axs[0].plot(EI_list, linestyle='-', marker='*', c=color_list[i],  label=f'$\\textit{{Case {i+1}}}$', zorder=1)


    # Plot Cost
    axs[1].plot(Cost_list, linestyle='-', marker='*', c=color_list[i], label=f'Case {i+1}', zorder=1)

# Subplot 1: EI
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Expected Improvement')
axs[0].grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)

# Subplot 2: Cost
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Cost $\\mathcal{J}$')
axs[1].grid(True, linestyle='--', linewidth=0.7, alpha=0.8, zorder=1)

# Create shared legend below both plots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1))

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at the bottom for the legend


filename = os.path.join('DATA_EI_cost_convergence', 'convergence_EI_cost.pdf')
plt.savefig(filename, format='pdf', dpi=600, bbox_inches='tight')
plt.show()


