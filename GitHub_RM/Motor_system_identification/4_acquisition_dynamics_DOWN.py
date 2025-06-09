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

#Define PWM list
ref_pwm = 7
min_pwm = 6.9
max_pwm = 6
pwm_steps = 9
pwm_list = np.linspace(min_pwm, max_pwm, pwm_steps)


# Saving folder
Data = "Dynamic_DOWN"
campaign = "4"
duration = 10

CAMPAIGN_FOLDER = Data + os.sep + f'campaign_{campaign}'
os.makedirs(CAMPAIGN_FOLDER, exist_ok=True)
    
DATA_FOLDER = f"{CAMPAIGN_FOLDER}{os.sep}{ref_pwm:.2f}_refPWM"
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)
    

# Global HX711 instance
DT, SCK = 5, 6
hx = HX711(DT, SCK)
hx.reset()
print("Taring of the load cell")
hx.tare(times=15)  # Tare the load cell before use
print(" done!.")

#_______________________________________________________________________
#FUNCTIONS

def unmap(pwm):
    return 5 + (pwm*0.05)
    
def map(pwm):
    return ((pwm-5)/5)*100
    
    
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
    
    

def loop(duration, ref_pwm, target_pwm, DATA_FOLDER):
    time_list = []
    thrust_list = []
    rpm_list = []
    pre_duration = 0.5*duration
    post_duration = 0.5*duration
    #-------------------- pre step ----------------------------------
    start_time = time.time()
    with tqdm(total=pre_duration, desc="Pre-step", unit="s") as pbar:
        while (time.time() - start_time) < pre_duration:
            elapsed_time = time.time() - start_time
            #Set Motor
            p.change_duty_cycle(ref_pwm)
            #Measure data
            thrust_val, rpm_val = measure_thrust_RPM()
            time_list.append(elapsed_time)
            thrust_list.append(thrust_val)
            rpm_list.append(rpm_val)
            #print(f"PWM: {pwm:.2f} %, Thrust: {thrust_val:.2f} N, RPM: {rpm_val:.2f}")
            
    #-------------------- post step ----------------------------------
    with tqdm(total=post_duration, desc="Post-step", unit="s") as pbar:
        while (time.time() - start_time) < duration:
            elapsed_time = time.time() - start_time
            #Set Motor
            p.change_duty_cycle(target_pwm)
            #Measure data
            thrust_val, rpm_val = measure_thrust_RPM()
            time_list.append(elapsed_time)
            thrust_list.append(thrust_val)
            rpm_list.append(rpm_val)
            #print(f"PWM: {pwm:.2f} %, Thrust: {thrust_val:.2f} N, RPM: {rpm_val:.2f}")
    
    save_data(time_list, thrust_list, rpm_list, ref_pwm, target_pwm, DATA_FOLDER)
    print("Loop complete.")


def save_data(time_list, thrust_list, rpm_list, ref_pwm, target_pwm, DATA_FOLDER):
    data = np.column_stack((time_list, thrust_list, rpm_list))
    file_name = os.path.join(DATA_FOLDER, f'{ref_pwm:.2f}_{target_pwm:.2f}.dat')
    np.savetxt(file_name, data, header='Time (s)   Thrust (N)   RPM', comments='', delimiter='\t', fmt='%1.5f')
    print(f'Data save in {file_name}\n')
    
    
    
if __name__ == '__main__':   # Program entrance
    print ('Program is starting ... ')
    try:
        rpmsensor = RPMSensor()
        motor_arming(ref_pwm)
        for target_pwm in pwm_list:
            print(f"PWM step from: {ref_pwm:.2f} to {target_pwm:.2f}")
            loop(duration, ref_pwm, target_pwm, DATA_FOLDER)
        p.stop()
    except KeyboardInterrupt: # Press ctrl-c to end the program.
        print("KeyboardInterrupt detected. Stopping motors...")
        p.stop()
        time.sleep(0.5)


