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
# Intialize the I2C interface
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)
pot1 = AnalogIn(ads, 0) 

#_______________________________________________________________________
#FUNCTIONS
def motor_arming():
    print("Starting motor.")
    global p
    p = HardwarePWM(pwm_channel=1, hz=50, chip=2) 
    p.start(0)
    print("Motor started.")

    
    
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
        print(f"PWM: {dc:.2f} %")

    
def cleanup():
    print("Exit")
    p.stop()  # Engine Stop
    GPIO.cleanup()  # Reset all GPIO
    time.sleep(0.5)
    
    
#_______________________________________________________________________
#MAIN

if __name__ == "__main__":
    try:
       motor_arming()
       loop()
    except KeyboardInterrupt:
        cleanup()
