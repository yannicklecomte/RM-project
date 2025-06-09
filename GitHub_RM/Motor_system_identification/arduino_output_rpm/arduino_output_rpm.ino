#include <Wire.h>
#include <stdio.h>
#define I2CADDR 0x04

volatile long rev=1;
float rpm;
float oldtime=0;        
float time_delta;
int pinA = 3;
volatile long data;//Data to be send// over via I2C

void isr()         
{
    rev++;
}

void setup()
{
Serial.begin(115200);
Wire.begin(I2CADDR); //  I2C bus with address
Wire.onRequest(requestEvent); // register request event
pinMode(pinA, INPUT_PULLUP);
attachInterrupt(digitalPinToInterrupt(pinA),isr,RISING); 
}

void loop()
{
  delayMicroseconds(10000);
  detachInterrupt(digitalPinToInterrupt(pinA));//Disable the interrupt for a short time to perform the calculations etc (Very important!)         
  time_delta=micros()-oldtime;
  rpm=(rev/time_delta)*60000;
  //rpm=(rev/time_delta)*60000000/1;
  data = ((long)(rpm * 1000));//Converting float to long so that it can easily be send to the RPI
  //data = 500;
  Serial.print(data);
  Serial.print("\n");
  rev = 0;  
  attachInterrupt(digitalPinToInterrupt(pinA), isr, RISING);//Restart the interrupt function
  oldtime=micros();
}

//Sending the data to the Raspberry PI
void requestEvent() {
  Wire.write((uint8_t*)&data, sizeof(data));
  //Wire.write((uint8_t*)&rev, sizeof(rev)-1);
}
