# Anira Real-time Principle Check
## Introduction
This repository contains code for the real-time evaluation of inference engines: LibTorch, ONNX Runtime, and TensorFlow Lite. Additionally, the Anira library, which ensures real-time safety for these inference engines, is also evaluated using real-time safety checks.

The purpose of this project is to confirm the compatibility of the described inference engines with real-time principles. Real-time principles are essential for applications such as audio processing, where timely execution is crucial. The evaluation is conducted using code sanitizers to detect potential real-time violations during runtime.

Code sanitizers are programming utilities designed to identify anomalies, such as undefined or suspicious behaviors, by introducing instrumentation code during runtime through compilation. In this project, we leverage the "Realtime Sanitizer" (RadSan), a specialized real-time safety testing tool designed for C and C++. RadSan is tailored to detect real-time violations like memory allocation, deallocation, and thread synchronization. It is a customized version of the Clang compiler.

## Available Tests
The repository contains two tests:
- **validate-anira**
  - This test is designed to ascertain the Anira library's adherence to real-time principles, identifying any potential violations.
- **validate-inference-engines**
  - his test aims to quantitatively assess the real-time performance of various inference engines, focusing on identifying the number and nature of real-time violations.

Tests are conducted under diverse conditions to ensure robust analysis: Each engine is evaluated using three models over ten iterations to track performance consistency, alongside varied audio processing configurations for adaptability and real-time compliance assessment.

## How to Use
How to validate real time principles, currently only supported on macOS / Linux / WSL:

Get Radsan:
```bash
docker pull realtimesanitizer/radsan-clang
sudo docker run -v  $(pwd):/anira-rt-principle-check -it realtimesanitizer/radsan-clang /bin/bash
```
Install necessary dependencies:
```bash
apt-get update && apt-get install -y git cmake vim
apt install libasound2-dev libjack-jackd2-dev \
    ladspa-sdk \
    libcurl4-openssl-dev  \
    libfreetype6-dev \
    libx11-dev libxcomposite-dev libxcursor-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev \
    libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev mesa-common-dev
```
Build:
```bash
cd /anira-rt-principle-check/anira-rt-principle-check/
git submodule update --init --recursive
cmake . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release

cmake --build cmake-build-release --config Release --target simple-real-time-violations
cmake --build cmake-build-release --config Release --target validate-anira
cmake --build cmake-build-release --config Release --target validate-inference-engines
```
Execute:
```bash
RADSAN_ERROR_MODE=continue ./simple-real-time-violations 2>&1 | tee simple-real-time-violations.txt
RADSAN_ERROR_MODE=continue ./validate-anira 2>&1 | tee validate-anira.txt
RADSAN_ERROR_MODE=continue ./validate-inference-engines 2>&1 | tee validate-inference-engines.txt
```

## Contributors
- [Valentin Ackva](https://github.com/vackva)
- [Fares Schulz](https://github.com/faressc)

## License
This project is licensed under the [MIT License](LICENSE).

