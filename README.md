# Real-Time Evaluation of Inference Engines

## Introduction
This repository contains code for the real-time evaluation of inference engines, namely LibTorch, ONNX Runtime, and TensorFlow Lite. Additionally, the Anira library, which ensures real-time safety for these inference engines, is also evaluated using real-time safety checks.

## Motivation
The purpose of this project is to confirm the compatibility of the described inference engines with real-time principles. Real-time principles are essential for applications such as audio processing, where timely execution is crucial. The evaluation is conducted using code sanitizers to detect potential real-time violations during runtime.

## Code Sanitizers
Code sanitizers are programming utilities designed to identify anomalies, such as undefined or suspicious behaviors, by introducing instrumentation code during runtime through compilation. In this project, we leverage the "Realtime Sanitizer" (RadSan), a specialized real-time safety testing tool designed for C and C++. RadSan is tailored to detect real-time violations like memory allocation, deallocation, and thread synchronization. It is a customized version of the Clang compiler.

## Test Environment
The test is conducted on a Linux x64 system, emphasizing the evaluation of the inference callback's real-time safety after one initial inference. The initial inference is motivated by Stefani et al. (2022), which asserts that the engines achieve real-time safety following the first run. This real-time evaluation also aims to determine the quantity and types of real-time violations.

## Repository Structure
The repository structure is organized as follows:


## How to Use

## Contributors
- [Valentin Ackva](https://github.com/vackva)
- [Fares Schulz](https://github.com/faressc)
- 
## License
This project is licensed under the [MIT License](LICENSE).

