import serial, sys, time
import RPi.GPIO as GPIO
from rpi_hardware_pwm import HardwarePWM
from hx711 import HX711
import matplotlib.pyplot as plt
import numpy as np
import os
import busio
import board
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import smbus2


#_______________________________________________________________________
# ACTIVATIONS OF THE COMPONENTS

# Suppress GPIO warnings
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # Use BCM numbering

# Global HX711 instance
DT, SCK = 5, 6
hx = HX711(DT, SCK)
hx.reset()
print("Taring of the load cell")
hx.tare(times=15)  # Tare the load cell before use
print(" done!.")

# Intialize the I2C interface
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)
pot1 = AnalogIn(ads, 0) 

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

def measure(rpmsensor):
    return rpmsensor.read_rpm()
    
    
def motor_arming():
    print("Starting motor.")
    global p
    p = HardwarePWM(pwm_channel=1, hz=50, chip=2) 
    p.start(0)
    print("Motor started.")

def measure_thrust_and_RPM():
    """ Function to measure thrust using HX711. """
    thrust_sensor = hx.get_weight(times=3)
    thrust_g = thrust_sensor / 1.8e3
    thrust_N = (thrust_g / 1000) * 9.81

    return thrust_N
    
    
def loop():
    t0 = time.time() #start timer
    
    counter=0
    steps = 10000
    flag= True
    # while time.time()-t0 <= 3:
        # p.change_duty_cycle(5)
    while flag:
        # Acquiering dutycyle based on potentiometer value
        dutycycle1 = pot1.voltage
            
        #Applying PWM signal to both motors
        dc = max(5, min(10, 5+ (dutycycle1/3.3)*5)) 
        #dc = max(0, min(100, (dutycycle1/3.3)*100)) 
        p.change_duty_cycle(dc)  # set the duty cycle (%) 10% = 10
        
        thrust_val = measure_thrust_and_RPM()

        rpm_val = measure(rpmsensor)

        print(f"PWM: {dc:.2f} %, Thrust: {thrust_val:.2f} N, RPM: {rpm_val:.2f}")

    
def cleanup():
    print("Exit")
    p.stop()  # Engine Stop
    GPIO.cleanup()  # Reset all GPIO
    time.sleep(0.5)
    
    
#_______________________________________________________________________
#MAIN

if __name__ == "__main__":
    try:
       rpmsensor = RPMSensor() 
       motor_arming()
       loop()
    except KeyboardInterrupt:
        cleanup()
