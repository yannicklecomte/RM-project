import time
import math
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import matplotlib.pyplot as plt
import numpy as np
import os
from rpi_hardware_pwm import HardwarePWM
from tqdm import tqdm

# Initialize the I2C interface
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)

# Initialiwe the ADC
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)

x_val = AnalogIn(ads, 2)
z_val = AnalogIn(ads, 3)

# Define accelerameter specs -- GY-61
VCC = 3.3
ZERO_G_VOLTAGE = VCC / 2
SENSITIVITY = 0.8  # V per g

#Define torque ratios
min_torque = 0.955
max_torque = 1.045
torque_steps = 3
r_list = np.linspace(min_torque, max_torque, torque_steps)

# Folder initialization
Data = "Dynamic_response_free"
campaign = "1"
target = 7.5

CAMPAIGN_FOLDER = Data + os.sep + f'campaign_{campaign}'
if not os.path.exists(CAMPAIGN_FOLDER):
    os.mkdir(CAMPAIGN_FOLDER)
    
DATA_FOLDER = f"{CAMPAIGN_FOLDER}{os.sep}{target:.1f}_PWM"
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
    
#Define test duration
duration = 45

#Accel calibration
CALIB_PATH = "data_calib_accel/calib_value.txt"
roll_calib = np.loadtxt(CALIB_PATH)


def setup():    
    global p1
    global p2   
    p1 = HardwarePWM(pwm_channel=1, hz = 50, chip=2)    #GPIO13/19 = channel 1
    p2 = HardwarePWM(pwm_channel=2, hz = 50, chip=2)     #GPIO18/12 = channel 0
    p1.start(0)  #Start with dutycycle 0%
    p2.start(0) #Start with dutycycle 0%
    
    print('------Arming ESC------')
    t1 = time.time()
    while (time.time() - t1 <= 3):
        p1.change_duty_cycle(5.5)  
        p2.change_duty_cycle(5.5)
        time.sleep(0.5)  


def get_tilt_angles(X, roll_calib):
    x,z = X
    roll = math.degrees(math.atan2(-x, z)) - roll_calib
    return roll
    

def loop(duration, target, r, roll_calib, DATA_FOLDER):
    time_list = []
    roll_list = []

    start_time = time.time()
    
    with tqdm(total=duration, desc="Progress", unit="s") as pbar:
        while (time.time() - start_time) < duration:
            elapsed_time = time.time() - start_time

            try:
                p1.change_duty_cycle(target)
                p2.change_duty_cycle(target*r)
            except Exception as e:
                print(f"Error changing duty cycle: {e}")
                break 

            try:
                x_volt, z_volt = x_val.voltage, z_val.voltage
                x_accel = (x_volt - ZERO_G_VOLTAGE) / SENSITIVITY
                z_accel = (z_volt - ZERO_G_VOLTAGE) / SENSITIVITY
                roll = get_tilt_angles([x_accel, z_accel], roll_calib)
            except Exception as e:
                print(f"Error reading sensors: {e}")
                break 

            time_list.append(elapsed_time)
            roll_list.append(roll)
    save_data(time_list, roll_list, target, duration, r, DATA_FOLDER)
    print("Loop complete.")

def save_data(time_list, roll, target, duration, r, DATA_FOLDER):
    data = np.column_stack((time_list, roll))
    file_name = os.path.join(DATA_FOLDER, f'{r:.5f}.dat')
    np.savetxt(file_name, data, header='Time (s)   roll (deg)', comments='', delimiter='\t', fmt='%1.5f')
    print(f'Data save in {file_name}\n')
   


if __name__ == '__main__':   # Program entrance
    print ('Program is starting ... ')
    try:
        setup()
        for i in range(len(r_list)):
            print(f"Torque ratio: {r_list[i]}")
            loop(duration, target, r_list[i], roll_calib, DATA_FOLDER)
        p1.stop()
        p2.stop()
    except KeyboardInterrupt: # Press ctrl-c to end the program.
        print("KeyboardInterrupt detected. Stopping motors...")
        p1.stop()
        p2.stop()
            
    
