#include <Servo.h>

Servo fingerServo;
int currentAngle = 0;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(25);
  fingerServo.attach(9);  // change pin if needed
  fingerServo.write(0);
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() > 0) {
      int angle = line.toInt();
      angle = constrain(angle, 0, 180);
      currentAngle = angle;

      fingerServo.write(currentAngle);
      Serial.println(currentAngle);
    }
  }
}
