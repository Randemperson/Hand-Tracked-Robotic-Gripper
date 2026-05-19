# HandTracked3DGripper

A computer-vision-controlled robotic gripper project that tracks hand motion in real time and maps finger bend data to a servo-driven gripper using Python, MediaPipe, OpenCV, and Arduino.

## Overview

This project uses a webcam to detect hand landmarks with MediaPipe, calculates a finger bend angle from the tracked hand, and sends that angle over serial to an Arduino Nano. The Arduino then uses the incoming value to drive a micro servo attached to a 3D-printed gripper.

The goal of the project was to combine computer vision, hardware control, and physical prototyping into a working hand-controlled robotic gripper system.

## Features

- Real-time hand tracking with MediaPipe
- Live OpenCV visualization of the hand skeleton
- Finger bend angle estimation
- Optional serial communication to Arduino
- Servo-driven gripper control
- Test mode for running vision without Arduino connected

## Hardware Used

- Arduino Nano
- Micro servo
- 3D-printed gripper
- Webcam
- USB connection to computer

## Software Used

- Python
- OpenCV
- MediaPipe
- pyserial
- Arduino IDE

## Project Structure

```text
HandTracked3DGripper/
├─ README.md
├─ requirements.txt
├─ main.py
├─ models/
│  └─ hand_landmarker.task
├─ src/
│  └─ finger_tracker/
│     ├─ __init__.py
│     └─ hand_tracker.py
├─ arduino/
│  └─ gripper_servo_control/
│     └─ gripper_servo_control.ino
├─ assets/
│  ├─ images/
│  └─ videos/
└─ docs/
```
Installation
Clone the repository.
Create and activate a virtual environment.
Install dependencies:
pip install -r requirements.txt
Download the MediaPipe hand landmarker model and place it in:
models/hand_landmarker.task
Running the Project
Run the Python program:

python main.py
When prompted:

choose test to run only the hand tracker
choose arduino to also send bend-angle data to the Arduino
Arduino Code
The Arduino sketch is located in:

arduino/gripper_servo_control/gripper_servo_control.ino
It receives a bend angle over serial at 9600 baud and writes that angle to the servo.

Challenges
One of the biggest challenges in this project was working through the physical prototyping side alongside the software. I had to 3D print the gripper and refine its design so it would actually function as intended, which took trial and error. I also had to learn how to properly connect and control a micro servo with an Arduino Nano, which meant figuring out wiring, power, and communication details that were new to me. Another major challenge was experimenting with resin 3D printing silicone components, which took several days of testing and adjustment before I could get usable results. These challenges pushed me to problem-solve across both hardware and software, and they taught me how much persistence and iteration real engineering projects require.

Future Improvements
Improve fist detection and full-hand gesture recognition
Track individual gripper joints
Add more reliable angle smoothing
Calibrate servo motion to match real finger movement more closely
Expand to more advanced robotic hand control
Requirements
opencv-python>=4.10.0
mediapipe>=0.10.14
numpy>=1.26.0
pyserial>=3.5

Author
Your Name Here

