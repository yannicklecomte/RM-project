import time
import smbus2
import busio
import RPi.GPIO as GPIO
import os
import numpy as np
from tqdm import tqdm
from hx711 import HX711
import matplotlib.pyplot as plt
from rpi_hardware_pwm import HardwarePWM
import serial

GPIO.setmode(GPIO.BCM)
#_______________________________________________________________________
#INITIALIZATION

ref_pwm = 7

# Saving folder
Data = "Motor_Model"
campaign = "1"
test_nbre = 1

CAMPAIGN_FOLDER = Data + os.sep + f'campaign_{campaign}'
os.makedirs(CAMPAIGN_FOLDER, exist_ok=True)
    
DATA_FOLDER = f"{CAMPAIGN_FOLDER}{os.sep}test{test_nbre:.0f}"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    

# Global HX711 instance
DT, SCK = 5, 6
hx = HX711(DT, SCK)
hx.reset()
print("Taring of the load cell")
hx.tare(times=15)  # Tare the load cell before use
print(" done!.")

def unmap(pwm):
    return (pwm*0.05) + 5


#_______________________________________________________________________
#CREATION OF THE INPUT SEQUENCE

# Step parameters
g_val = [40, 50, 40, 30, 45, 55, 60]
fs = 600
T_step = 4
N_steps = len(g_val)
dt = 1/fs

samples_per_step = int(fs * T_step)
total_samples = samples_per_step * N_steps

# Time vector for initial step sequence
t_step = np.linspace(0, T_step * N_steps, total_samples)
u_step = np.zeros_like(t_step)

# Step input
for i in range(N_steps):
    u_step[i*samples_per_step:(i+1)*samples_per_step] = g_val[i]

# --- Add linear ramp after last step ---
T_ramp = 4  # seconds
samples_ramp = int(fs * T_ramp)
t_ramp = np.linspace(t_step[-1] + dt, t_step[-1] + T_ramp, samples_ramp)
u_ramp = np.linspace(g_val[-1], g_val[-1] + 20, samples_ramp)  # example: +20 units

# --- Add sinusoidal input after ramp ---
T_sin = 6  # seconds
samples_sin = int(fs * T_sin)
t_sin = np.linspace(t_ramp[-1] + dt, t_ramp[-1] + T_sin, samples_sin)
f_sin = 0.5  # Hz
amplitude = 10
u_sin = g_val[-1] + 20 + amplitude * np.sin(2 * np.pi * f_sin * (t_sin - t_sin[0]))

# --- Combine everything ---
t = np.concatenate([t_step, t_ramp, t_sin])
u = np.concatenate([u_step, u_ramp, u_sin])

#_______________________________________________________________________
#FUNCTIONS

class RPMSensor:
    def __init__(self):
        # Initialize the RPM sensor read by Arduino --- Initialize I2C connection with Arduino
        self.address = 0x04
        self.bus = smbus2.SMBus(1)
        self.rx_bytes = [0, 0, 0]
        self.rpm = 0
        
    # Read rotational speed from Arduino    
    def read_rpm(self):
        try:
            self.rx_bytes = self.bus.read_i2c_block_data(self.address, 0, 4)
            value = self.rx_bytes[0] + (self.rx_bytes[1] << 8) + (self.rx_bytes[2] << 16)
        except:
            value = 0
        self.rpm = float(value) / 1000
        if self.rpm > 10000:
            return 0
        else:
            return self.rpm
    
def motor_arming(ref_pwm):
    print("Connection to motor")
    global p
    p = HardwarePWM(pwm_channel=1, hz=50, chip=2) 
    p.start(0)
    print('------Arming ESC------')
    t1 = time.time()
    while (time.time() - t1 <= 3):
        p.change_duty_cycle(5.5)  
        time.sleep(0.5) 
    while (time.time() - t1 <= 6):
        p.change_duty_cycle(ref_pwm)  
        time.sleep(0.5)  


def measure_thrust_RPM():
    """ Function to measure thrust using HX711. """
    thrust_sensor = hx.get_weight(times=3)
    thrust_g = thrust_sensor / 1.8e3
    thrust_N = (thrust_g / 1000) * 9.81

    rpm_val = rpmsensor.read_rpm()
    return thrust_N, rpm_val
    
    
    
def loop(t, u, DATA_FOLDER):
    thrust_list = np.zeros_like(u)
    rpm_list = np.zeros_like(u)
    
    t0 = time.time()
    for i in range(len(u)):
        elapsed_time = time.time() - t0
        unmap_pwm = unmap(u[i])
        p.change_duty_cycle(unmap_pwm)
        t[i] = (elapsed_time)
        thrust_list[i] = 0
        rpm_list[i] = rpmsensor.read_rpm()
    save_data(t, thrust_list, rpm_list, DATA_FOLDER)
    print("Loop complete.")


def save_data(time_list, thrust_list, rpm_list, DATA_FOLDER):
    data = np.column_stack((time_list, thrust_list, rpm_list))
    file_name = os.path.join(DATA_FOLDER, f'data.dat')
    np.savetxt(file_name, data, header='Time (s)   Thrust (N)   RPM', comments='', delimiter='\t', fmt='%1.5f')
    print(f'Data save in {file_name}')
    
    
    
if __name__ == '__main__':   # Program entrance
    print ('Program is starting ... ')
    try:
        rpmsensor = RPMSensor()
        motor_arming(ref_pwm)
        loop(t, u, DATA_FOLDER)
        p.stop()
    except KeyboardInterrupt: # Press ctrl-c to end the program.
        print("KeyboardInterrupt detected. Stopping motors...")
        p.stop()
        time.sleep(0.5)


