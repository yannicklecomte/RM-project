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
import smbus2
import struct

#_______________________________________________________________________
#INITIALIZATION

duration = 30

# Folder initialization
DATA_FOLDER = "data_calib_encoder"
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
    
    
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
            value = struct.unpack('<l', bytes(self.rx_bytes))[0]
        except Exception as e:
            print(f"I2C read error: {e}")
            value = 0

        self.angle = float(value) / 1000.0  # Convert millidegrees to degrees

        if 0.0 <= self.angle <= 360.0:
            return self.angle
        else:
            return 0.0  # Clamp out-of-range values


         
def save_data(time_list, roll_list):
    roll_mean = np.mean(roll_list)
    data = np.column_stack((time_list, roll_list))
    file_name = os.path.join(DATA_FOLDER, 'calib_list.dat')
    calib_name = os.path.join(DATA_FOLDER, 'calib_value.txt')

    np.savetxt(file_name, data, header='Time (s)   roll (deg)', comments='', delimiter='\t', fmt='%1.5f')
    np.savetxt(calib_name, [roll_mean], fmt="%.6f")

    print(f'Data save in {file_name}\n')
   
    
def loop(duration):
    time_list = []
    roll_list = []
    start_time = time.time()
        
    while (time.time() - start_time) < duration:
                
        elapsed_time = time.time() - start_time
        roll = encoder.read_angle()
        print(f"Roll: {roll:.2f}°")
        
        time_list.append(elapsed_time)
        roll_list.append(roll)

    save_data(time_list, roll_list)


#_______________________________________________________________________
#MAIN

if __name__ == '__main__':   # Program entrance
    print ('Program is starting ... ')
    encoder = AMT23AngleSensor()
    try:
        loop(duration)
    except KeyboardInterrupt: # Press ctrl-c to end the program.
        destroy()
        
    
