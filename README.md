# Night-Vision Auto-Tracking Light

This repository contains the code and supporting files for my third-year individual project at the University of Manchester.

The project investigates the feasibility of an autonomous perception-guided lighting system that detects and tracks a human target in real time and directs a motorised light source towards them under day and night conditions. The final prototype combined YOLOv8-based human detection on an NVIDIA Jetson Orin Nano, a Raspberry Pi NoIR camera, a pan-tilt mechanism driven through a PCA9685 servo driver, and GPIO-controlled floodlight switching.

## Project Overview

The system captures live image frames, performs person detection locally on the Jetson Orin Nano, selects a target, calculates image-space tracking error, and updates the pan-tilt mechanism to keep the light directed towards the detected person.

The project was evaluated as a proof of concept in a 7 m × 10 m domestic garden. The main performance areas assessed in the report were:

- detection reliability
- latency
- tracking speed
- field of detection / tracking coverage
- robustness under different operating conditions

## Main Features

- YOLOv8-based real-time person detection
- GPU-accelerated inference on Jetson Orin Nano
- pan-tilt servo control using PCA9685
- GPIO-controlled floodlight switching
- low-light / night-time operation using a NoIR camera and IR illumination
- CSV logging for tracking and performance analysis
- supporting scripts for experimental testing
- exploratory ROS 2 package work for modular architecture development

## Repository Structure

This repository is organised into the following main sections:

### Main implementation
Contains the final standalone runtime script used for the integrated tracking-light system.

### Test scripts
Contains scripts used to support the experiments discussed in the report, including:

- vision accuracy testing
- latency testing
- tracking performance testing
- field-of-detection / coverage testing
- height-dependent tracking testing
- multi-person tracking testing
- zigzag motion tracking and smoothness logging

### Results / logged data
Contains CSV files and other supporting logged outputs used to generate or support the results presented in the report.

### ROS 2 package
Contains exploratory ROS 2 package work used to investigate a more modular software architecture.

The ROS 2 package includes:

- `vision_yolo_node.py` — publishes person-offset data from the camera stream
- `servo_controller_node.py` — subscribes to the offset topic and drives the pan-tilt servos

This ROS 2 package was exploratory. The final prototype described in the report used a standalone Python control script rather than a ROS 2-based final implementation.

## Hardware Used

- NVIDIA Jetson Orin Nano Developer Kit
- Raspberry Pi Camera Module V2 NoIR
- IR illuminator
- PCA9685 PWM driver
- pan servo
- tilt servo
- floodlight
- MOSFET / solid-state relay switching stage

## Software and Libraries

Examples of software and libraries used in this project include:

- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch with CUDA
- Jetson.GPIO
- Adafruit PCA9685 libraries
- ROS 2

## Notes

This repository is provided as supporting material for the final report submission. It is intended to show the final implementation, supporting test scripts, logged data, and selected ROS 2 development work associated with the project.

## Author

Ben Bradwell  
Third Year Individual Project  
University of Manchester  
2026
