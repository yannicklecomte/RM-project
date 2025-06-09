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

# Saving folder
Data = "Static"
campaign = "1"
N_sample = 100

CAMPAIGN_FOLDER = Data + os.sep + f'campaign_{campaign}'
os.makedirs(CAMPAIGN_FOLDER, exist_ok=True)

    
DATA_FOLDER = f"{CAMPAIGN_FOLDER}{os.sep}{N_sample}_samples"
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
    
#Define PWM list
min_pwm = 5.5
max_pwm = 8
pwm_steps = 30
pwm_list = np.linspace(min_pwm, max_pwm, pwm_steps)


# Global HX711 instance
DT, SCK = 5, 6
hx = HX711(DT, SCK)
hx.reset()
print("Taring of the load cell")
hx.tare(times=15)  # Tare the load cell before use
print(" done!.")

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
    
def motor_arming():
    print("Connection to motor")
    global p
    p = HardwarePWM(pwm_channel=1, hz=50, chip=2) 
    p.start(0)
    print('------Arming ESC------')
    t1 = time.time()
    while (time.time() - t1 <= 3):
        p.change_duty_cycle(5.5)  
        time.sleep(0.5)  

def measure_thrust_RPM():
    """ Function to measure thrust using HX711. """
    thrust_sensor = hx.get_weight(times=3)
    thrust_g = thrust_sensor / 1.8e3
    thrust_N = (thrust_g / 1000) * 9.81

    rpm_val = rpmsensor.read_rpm()
    return thrust_N, rpm_val
    
    

def loop(N_sample, pwm, DATA_FOLDER):
    thrust_list = []
    rpm_list = []
    step = 0
    p.change_duty_cycle(pwm)
    
    with tqdm(total=N_sample, desc="Running", unit="step") as pbar:
        while step < N_sample:
            thrust_val, rpm_val = measure_thrust_RPM()
            thrust_list.append(thrust_val)
            rpm_list.append(rpm_val)
            #print(f"PWM: {pwm:.2f} %, Thrust: {thrust_val:.2f} N, RPM: {rpm_val:.2f}")
            step += 1
            pbar.update(1)  
    
    save_data(thrust_list, rpm_list, pwm, DATA_FOLDER)
    print("Loop complete.")



def save_data(thrust_list, rpm_list, pwm, DATA_FOLDER):
    data = np.column_stack((thrust_list, rpm_list))
    file_name = os.path.join(DATA_FOLDER, f'{pwm:.2f}.dat')
    np.savetxt(file_name, data, header='Thrust (N)   RPM', comments='', delimiter='\t', fmt='%1.5f')
    print(f'Data save in {file_name}\n')
    
    
    
if __name__ == '__main__':   # Program entrance
    print ('Program is starting ... ')
    try:
        rpmsensor = RPMSensor()
        motor_arming()
        for pwm in pwm_list:
            print(f"PWM: {pwm:.2f}")
            loop(N_sample, pwm, DATA_FOLDER)
        p.stop()
    except KeyboardInterrupt: # Press ctrl-c to end the program.
        print("KeyboardInterrupt detected. Stopping motors...")
        p.stop()
        time.sleep(0.5)


