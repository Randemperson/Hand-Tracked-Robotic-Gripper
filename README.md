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

## Hardware Used

- Arduino Nano
- Fixed Rotation Micro servo (external power source of 4 AA batteries or 4.5-6V equivalent needed)
- 3D-printed gripper
- Webcam

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

## Installation

1. Clone the repository
2. Create a venv:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Running the Project

Run the program with:
```powershell
python main.py
```
When prompted:
- Enter test to run only the hand tracking visualization.
- Enter arduino to send bend-angle data to the Arduino.

## Arduino Setup

Upload the arduino sketch from:
```text
arduino/gripper_servo_control/gripper_servo_control.ino
```
Make sure:
- Your Arduino is connected to the correct COM port.
- The baud rate matches 9600.
- Your servo signal wire is connected to pin 9 unless you changed it in the sketch.

## Challenges

One of the biggest challenges in this project was working through the physical prototyping side alongside the software. I had to 3D print the gripper and refine its design so it would actually function as intended, which took trial and error. I also had to learn how to properly connect and control a micro servo with an Arduino Nano, which meant figuring out wiring, power, and communication details that were new to me. Another major challenge was experimenting with resin 3D printing silicone components (40A), which was a surprisingly long process and my PI told me my parts were the first to be successful in a while. These challenges pushed me to problem-solve across both hardware and software, and they taught me how much persistence and iteration real engineering projects require.

## Future Improvements
- Improve fist detection and full-hand gesture recognition
- Track gripper joints for a closed-feedback loop
- Add more reliable angle smoothing
- More advanced robotic hand control
- Robotic arm
- World domination

## Requirements

```txt
opencv-python>=4.10.0
mediapipe>=0.10.14
numpy>=1.26.0
pyserial>=3.5
```

## Author
Alfred Tang
