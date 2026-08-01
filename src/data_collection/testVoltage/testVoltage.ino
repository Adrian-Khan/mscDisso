void setup() {
  Serial.begin(115200);
}

void loop() {
  int raw = analogRead(A0);
  float voltage = raw * (5.0 / 1023.0);
  float Rs = 3200.0;
  float R_sensor = Rs * voltage / (5.0 - voltage);
  
  Serial.print(millis());
  Serial.print(",");
  Serial.print(raw);
  Serial.print(",");
  Serial.print(voltage, 4);
  Serial.print(",");
  Serial.println(R_sensor, 1);
  
  delay(10);
}
