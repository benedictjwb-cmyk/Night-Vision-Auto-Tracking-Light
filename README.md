# Night-Vision Auto-Tracking Light

This repository contains the code and supporting files for my third-year individual project at the University of Manchester.

The project investigates the feasibility of an autonomous perception-guided lighting system that detects and tracks a human target in real time and directs a motorised light source towards them under day and night conditions. The final prototype combined YOLOv8-based human detection on an NVIDIA Jetson Orin Nano, a Raspberry Pi Camera Module V2 NoIR, an infrared illuminator, a pan-tilt mechanism driven through a PCA9685 servo driver, and GPIO-controlled floodlight switching.

These two videos demonstrate the light in action: 
Camera POV Demo: https://youtu.be/fggf2fM4tx4
Person Tracking Demo https://youtube.com/shorts/HBrcA1gscnE?feature=share

## Overview

The system captures live image frames, performs person detection locally on the Jetson Orin Nano, selects a target, calculates image-space tracking error, and updates the pan-tilt mechanism so that the light remains directed towards the detected person.

The project was evaluated as a proof of concept in a 7 m × 10 m domestic garden. The main performance areas assessed in the accompanying report were:

- detection reliability
- latency
- tracking speed
- field of detection and tracking coverage
- robustness under different operating conditions

## Repository Structure

This repository is organised into the following main sections.

### `main_tracking_light.py`

The final standalone runtime script used for the integrated tracking-light system.

### `tests/`

Scripts used to support the experiments discussed in the report, including:

- vision accuracy testing
- latency testing
- tracking performance testing
- field-of-detection and coverage testing
- height-dependent tracking testing
- multi-person tracking testing
- zigzag motion tracking and smoothness logging

### `results/`

CSV files and other logged outputs used to support the analysis and figures presented in the report.

### `ros2/auto_tracking_light/`

Exploratory ROS 2 package work used to investigate a more modular software architecture.

This package includes:

- `vision_yolo_node.py` — publishes person-offset data from the camera stream
- `servo_controller_node.py` — subscribes to the offset topic and drives the pan-tilt servos

The ROS 2 package was exploratory. The final prototype described in the report used the standalone Python implementation rather than a ROS 2 final implementation.

## Main Features

- YOLOv8-based real-time person detection
- GPU-accelerated inference on Jetson Orin Nano
- pan-tilt servo control using PCA9685
- GPIO-controlled floodlight switching
- low-light and night-time operation using a NoIR camera and IR illumination
- CSV logging for tracking and performance analysis
- supporting scripts for experimental testing
- exploratory ROS 2 package work for modular software development

## Hardware Used

- NVIDIA Jetson Orin Nano Developer Kit
- Raspberry Pi Camera Module V2 NoIR
- infrared illuminator
- PCA9685 PWM driver
- pan servo
- tilt servo
- floodlight
- MOSFET and solid-state relay switching stage

## Software and Libraries

Examples of software and libraries used in this project include:

- Python 3
- OpenCV
- Ultralytics YOLOv8
- PyTorch with CUDA
- Jetson.GPIO
- Adafruit PCA9685 libraries
- Flask
- ROS 2

## Installation

These instructions assume setup on an NVIDIA Jetson Orin Nano running Python 3 with CUDA-enabled PyTorch already available.

### 1. Clone the repository

```bash
git clone https://github.com/benedictjwb-cmyk/Night-Vision-Auto-Tracking-Light.git
cd Night-Vision-Auto-Tracking-Light
```

### 2. Install Python dependencies

Install the required packages as needed for your setup:

```bash
pip install ultralytics opencv-python flask adafruit-circuitpython-pca9685 adafruit-blinka
```

Depending on your Jetson software image, some hardware libraries may need to be installed or configured separately.

### 3. Connect the hardware

Before running the main script, ensure that:

- the Raspberry Pi NoIR camera is connected to the Jetson CSI port
- the PCA9685 driver is connected over I2C
- the pan and tilt servos are connected to the PCA9685
- the floodlight switching circuit is connected correctly
- the servos are powered from a suitable external supply
- the camera and infrared illuminator are mounted so that sensing and illumination are aligned

### 4. Check platform-specific configuration

Jetson-based camera, GPIO, and I2C access may require platform-specific setup before first use. These steps depend on the software image and connected hardware.

## How to Run

### Main integrated system

Run the final standalone tracking-light script with:

```bash
python3 main_tracking_light.py
```

This script:

- opens the live camera stream
- performs person detection using YOLOv8
- selects and tracks a target in image space
- drives the pan-tilt servos to reduce tracking error
- switches the floodlight when a person is detected
- logs tracking data for later analysis

### Test scripts

Experimental scripts in the `tests/` directory can be run individually, depending on the experiment being repeated. For example:

```bash
python3 tests/<script_name>.py
```

These scripts were used to support the evaluation reported in the final dissertation.

## Outputs

The repository includes logged outputs and result files used to support the final analysis.

Typical outputs include:

- CSV logs of pan and tilt angles
- velocity and acceleration estimates
- target image coordinates
- latency measurements
- other experiment-specific result files

These are primarily stored in the `results/` directory.

## Technical Details

The final prototype used a closed-loop image-based tracking approach.

### Detection

Frames were captured from the camera and processed locally on the Jetson Orin Nano using a YOLOv8 nano model with GPU acceleration. Detection was restricted to the person class to reduce computational load and support real-time operation.

### Target tracking

The detected person’s image position was compared with the image centre to produce horizontal and vertical image-space error. This error was used directly for control rather than reconstructing a full 3D target position.

### Control

The pan-tilt mechanism was driven using proportional image-space control. Deadzone logic was used to reduce unnecessary actuator motion and help limit oscillation when the target was already near the centre of the frame. Command limits were also applied to avoid excessively large step changes.

### Actuation

Servo control was generated through a PCA9685 PWM driver over I2C. This avoided placing timing-critical PWM generation directly on the Jetson while object detection was running.

### Illumination switching

The floodlight was controlled through a GPIO-triggered MOSFET and solid-state relay arrangement so that the embedded platform could safely switch the lighting load.

### Evaluation support

Additional scripts were used to measure latency, assess tracking behaviour, evaluate night-time detection performance, test multi-person behaviour, and log motion smoothness.

## Known Issues

- Performance reduces in adverse weather, especially in rain or damp conditions.
- Initial acquisition coverage is narrower than the achieved post-acquisition tracking range.
- Hardware setup is Jetson-specific and may require additional camera, GPIO, or I2C configuration.
- Servo behaviour depends on mounting rigidity, load distribution, and power supply quality.
- Reproducing the exact hardware setup requires the same or closely matched components.

## Future Improvements

- add a formal setup and calibration mode for installation
- improve environmental robustness in rain and poor outdoor conditions
- extend the modular ROS 2 implementation
- improve initial target acquisition coverage
- support a wider range of lighting units for different deployment scenarios
- improve packaging and enclosure design for long-term outdoor use

## Notes

This repository is provided as supporting material for the final report submission. It is intended to show the final implementation, supporting test scripts, logged data, and exploratory ROS 2 development work associated with the project.

## Author

Ben Bradwell  
Third Year Individual Project  
University of Manchester  
2026
