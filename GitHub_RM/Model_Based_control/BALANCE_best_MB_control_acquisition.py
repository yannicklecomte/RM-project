import time
import smbus2
from rpi_hardware_pwm import HardwarePWM
import math
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os
import struct
from tqdm import tqdm

#_______________________________________________________________________
#INITIALIZATION

# Define control parameters
gamma_ref = 40

# ==== Define system constants =====
# With the panels
#theta_0 = 5.597469
#C = -4.712508

# Without the panels
# from campaign 4
theta_0 = 2.38018978
C = -1.54209496

# Define the optimal coefficient founded based on the model
# ---------------------------------------------
Kp, Kd = 0.892896, 0.520942
# ---------------------------------------------
# Number of episode per sample
Ne = 10

# Folder initialization
Data = "DATA_best_MB_control"

CAMPAIGN_FOLDER = Data + os.sep + f"Kp_{Kp:.2f}_Kd_{Kd:.2f}_Ne_{Ne}"
if not os.path.exists(CAMPAIGN_FOLDER):
    os.makedirs(CAMPAIGN_FOLDER)
    
# Calibration loading
CALIB_PATH_ENC = "data_calib_encoder/calib_value.txt"
theta_calib_enc = np.loadtxt(CALIB_PATH_ENC)

gamma_clip = 8

#_______________________________________________________________________
#TARGET SEQUENCE

# Step parameters
target_val = [0, 10, -10, 5, -5]
fs = 400
T_step = 5
N_steps = len(target_val)
dt = 1/fs

samples_per_step = int(fs * T_step)
total_samples = samples_per_step * N_steps

# Time vector for initial step sequence
t_step = np.linspace(0, T_step * N_steps, total_samples)
target_list = np.zeros_like(t_step)

# Step input
for i in range(N_steps):
    target_list[i*samples_per_step:(i+1)*samples_per_step] = target_val[i]


#_______________________________________________________________________
#FUNCTIONS

class AMT23AngleSensor:
    def __init__(self):
        # I2C configuration
        self.address = 0x04  # Same as set on the Arduino
        self.bus = smbus2.SMBus(1)
        self.rx_bytes = [0, 0, 0, 0]
        self.angle = 0.0

    def read_angle(self):
        try:
            self.rx_bytes = self.bus.read_i2c_block_data(self.address, 0, 4)
            value = struct.unpack('<f', bytes(self.rx_bytes))[0]
        except Exception as e:
            print(f"I2C read error: {e}")
            value = 0

        self.angle = float(value) /4

        if 0.0 <= self.angle <= 360.0:
            return self.angle
        else:
            return 0.0  # Clamp out-of-range values

def setup():    
    # ===== motor setup ===== 
    global pL
    global pR  
    pL = HardwarePWM(pwm_channel=1, hz = 50, chip=2)    #GPIO13
    pR = HardwarePWM(pwm_channel=2, hz = 50, chip=2)    #GPIO18
    pL.start(0)
    pR.start(0)
    print('------Arming ESC------')
    t1 = time.time()
    while (time.time() - t1 <= 3):
        pL.change_duty_cycle(5.5)  
        pR.change_duty_cycle(5.5)
        time.sleep(0.5) 
    
def unmap_pwm(pwm):
    return 0.05*pwm + 5
  
def loop(X, target_list, encoder, Ns):
    # ==== Unmap the controller ====
    Kp, Kd = X

    # ===== Fixed time step =====
    dt = 1/400  
    duration = dt * Ns
    
    # ===== Initialize lists =====
    times = np.zeros(Ns)
    theta = np.zeros(Ns)
    error = np.zeros(Ns)
    gamma_control = np.zeros(Ns)
    gamma_cmd_L = np.zeros(Ns)
    gamma_cmd_R = np.zeros(Ns)

    times[0] = 0
    threshold = 10
    
    # Warm up at the frist value of the target list 
    # before starting a full sequence
    t1 = time.time()
    while (time.time() - t1 <= 8):
        delta_gamma_S_start = (target_list[0] - theta_0)/C
        gamma_cmd_L_start = gamma_ref - 0.5 * delta_gamma_S_start
        gamma_cmd_R_start = gamma_ref + 0.5 * delta_gamma_S_start
        pL.change_duty_cycle(unmap_pwm(gamma_cmd_L_start))
        pR.change_duty_cycle(unmap_pwm(gamma_cmd_R_start))
    
    # Start the sequence
    with tqdm(total=duration, desc="Control", unit="s") as pbar:
        for i in range(1, Ns):
            loop_start = time.time()
                        
            # ===== Measure current position =====
            theta_val = encoder.read_angle() - theta_calib_enc
            if abs(theta_val - theta[i-1]) > threshold:
                theta[i] = theta[i-1]
            else:
                theta[i] = theta_val
            
            # ===== Time update =====
            times[i] = times[i-1] + dt

            # ===== Compute PD error =====
            error[i] = theta[i] - target_list[i]
            derivative = (error[i] - error[i-1]) / dt
            
            # ===== Compute error to apply to each propeller ====
            delta_gamma_S = (target_list[i] - theta_0)/C
            gamma_control[i] = np.clip(delta_gamma_S + (Kp * error[i] + Kd * derivative), delta_gamma_S-gamma_clip, delta_gamma_S+gamma_clip)
            gamma_cmd_L[i] = np.clip((gamma_ref - 0.5 * gamma_control[i]), 20, 60)
            gamma_cmd_R[i] = np.clip((gamma_ref + 0.5 * gamma_control[i]), 20, 60)
            
            # ===== Apply to motors =====
            pL.change_duty_cycle(unmap_pwm(gamma_cmd_L[i]))
            pR.change_duty_cycle(unmap_pwm(gamma_cmd_R[i]))

            # ===== tqdm display =====
            pbar.set_postfix(theta=f"{theta[i]:.2f}°")
            pbar.update(dt)

            # ===== Wait to enforce constant dt =====
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    return times, theta, target_list

#_______________________________________________________________________
#MAIN

if __name__ == '__main__':   # Program entrance
    print ('Program is starting ... ')
    try:
        setup()
        encoder = AMT23AngleSensor()
        for eps in range(Ne):
            time_list, theta_list, target_list = loop([Kp, Kd], target_list, encoder, total_samples)
            file_path = os.path.join(CAMPAIGN_FOLDER, f"episode_{eps}.csv")
            np.savetxt(file_path, np.column_stack((time_list, theta_list, target_list)), delimiter=',', header='time,theta', comments='', fmt='%1.5f')
        pL.stop()
        pR.stop()
    except KeyboardInterrupt: # Press ctrl-c to end the program.
        print('PO')
        pL.stop()
        pR.stop()
