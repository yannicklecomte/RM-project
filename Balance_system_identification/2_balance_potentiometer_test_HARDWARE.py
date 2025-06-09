import time
from rpi_hardware_pwm import HardwarePWM
import math
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import matplotlib.pyplot as plt
import numpy as np

# Initialize the I2C interface
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)

#ads = ADS.ADS1115(i2c , gain = 1 , data_rate = 250) #gain = 0.6666666666666666
pot1 = AnalogIn(ads, 0) 
pot2 = AnalogIn(ads, 1)
        
def setup():    
    global p1
    global p2   
    p1 = HardwarePWM(pwm_channel=1, hz = 50, chip=2)    #GPIO13/19 = channel 1
    p2 = HardwarePWM(pwm_channel=2, hz = 50, chip=2)     #GPIO18/12 = channel 0
    p1.start(0)  #Start with dutycycle 0%
    p2.start(0) #Start with dutycycle 0%
    
def loop():
    t0 = time.time() #start timer
    
    counter=0
    steps = 10000
    flag= True
    while time.time()-t0 <= 3:
        p1.change_duty_cycle(5)
        p2.change_duty_cycle(5)
    while flag:
        # Acquiering dutycyle based on potentiometer value
        dutycycle1 = pot1.voltage
        dutycycle2 = pot2.voltage
            
        #Applying PWM signal to both motors
        dc1 = max(5, min(10, 5 + (dutycycle1/3.3)*5)) 
        dc2 = max(5, min(10, 5 + (dutycycle2/3.3)*5)) 
        #dc1 = max(0, min(100, (dutycycle1/3.3)*100)) 
        #dc2 = max(0, min(100, (dutycycle2/3.3)*100)) 

        p1.change_duty_cycle(dc1)  # set the duty cycle (%) 10% = 10
        p2.change_duty_cycle(dc2)  # set the duty cycle (%) 10% = 10
        
        #print(f"Potentiometer 1: {pot1.voltage:.2f} V, Potentiometer 2: {pot2.voltage:.2f} V")
        print(f"Potentiometer 1: {dc1:.2f} %, Potentiometer 2: {dc2:.2f} %")

def destroy():
    adc.close()    
    p1.close()
    p2.close()
    GPIO.cleanup()  
    
    
if __name__ == '__main__':   # Program entrance
    print ('Program is starting ... ')
    try:
        setup()
        loop()
    except KeyboardInterrupt: # Press ctrl-c to end the program.
        destroy()
        
    
