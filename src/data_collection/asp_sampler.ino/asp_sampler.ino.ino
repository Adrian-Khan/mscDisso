#include <SPI.h>

const int CS_PIN = 10;

unsigned long last_us = 0;

void setup() {
  Serial.begin(115200);
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  last_us = micros();   // initialise timing
}

int readMCP3208(int channel) {
  digitalWrite(CS_PIN, LOW);
  byte b0 = SPI.transfer(0x06 | (channel >> 2));
  byte b1 = SPI.transfer((channel & 0x03) << 6);
  byte b2 = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  return ((b1 & 0x0F) << 8) | b2;
}

void loop() {
  // timestamp before reading
  unsigned long now_us = micros();
  unsigned long dt_us = now_us - last_us;
  last_us = now_us;

  // read ADC
  int value = readMCP3208(0);

  // print: dt_us, adc_raw
  Serial.print(dt_us);
  Serial.print(",");
  Serial.println(value);

  // no delay so raw ASP sampling rate
}
