#include <Wire.h>

#define I2CADDR 0x04  // Arduino's I2C address

#define DATA_PIN 2
#define CLOCK_PIN 3
#define CS_PIN 4

const int ENCODER_RESOLUTION = 16384; // 14-bit
volatile long dataToSend = 0;

void setup() {
  pinMode(CLOCK_PIN, OUTPUT);
  pinMode(DATA_PIN, INPUT);
  pinMode(CS_PIN, OUTPUT);

  digitalWrite(CLOCK_PIN, LOW);
  digitalWrite(CS_PIN, HIGH);

  Wire.begin(I2CADDR);                // Join I2C bus as slave
  Wire.onRequest(requestEvent);      // Register function to send data

  Serial.begin(115200);
}

void loop() {
  uint16_t position = readEncoderPosition();
  float angle = (position * 360.0) / ENCODER_RESOLUTION;
  dataToSend = (long)(angle * 1000); // Send angle as integer with 3 decimal places

  Serial.print("Angle (°): ");
  Serial.println(angle);

  delay(10); // Optional delay
}

uint16_t readEncoderPosition() {
  uint16_t position = 0;

  digitalWrite(CS_PIN, LOW);
  delayMicroseconds(1);

  for (int i = 0; i < 16; i++) {
    digitalWrite(CLOCK_PIN, HIGH);
    delayMicroseconds(1);

    position <<= 1;
    if (digitalRead(DATA_PIN)) {
      position |= 1;
    }

    digitalWrite(CLOCK_PIN, LOW);
    delayMicroseconds(1);
  }

  digitalWrite(CS_PIN, HIGH);

  position &= 0x3FFF;
  return position;
}

// I2C send handler
void requestEvent() {
  Wire.write((uint8_t*)&dataToSend, sizeof(dataToSend));
}
