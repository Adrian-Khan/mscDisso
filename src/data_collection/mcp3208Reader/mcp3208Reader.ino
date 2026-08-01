#include <SPI.h>

const int CS_PIN = 10;

void setup() {
  Serial.begin(115200);
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
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
  int value = readMCP3208(0);
  Serial.print(millis());
  Serial.print(",");
  Serial.println(value);
  delay(10);
}