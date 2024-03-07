# Anira Real-time Principle Check
## Introduction
This repository serves as a supplementary resource for the [anira library](https://github.com/tu-studio/anira), an architecture for real-time inference in real-time audio application.

**This repository contains two validation tests / builds:**
 - **validate-anira:** validate real-time saftey of the proposed architecture of [anira](https://github.com/tu-studio/anira)
 - **validate-inference-engines**: evaluation of real-time violations of the inference engines: [LibTorch](https://github.com/pytorch/pytorch/), [ONNXRuntime](https://github.com/microsoft/onnxruntime/), and [Tensorflow Lite](https://github.com/tensorflow/tensorflow/)
      - evaluate number of violations in multiple inferences
      - evaluate types of violations in multiple inferences 

Tests are conducted under diverse conditions to ensure robust analysis: Each inference engine is evaluated using three models over ten iterations, alongside varied audio processing configurations for adaptability and real-time compliance assessment.

The analysis utilizes code sanitizers specifically designed to detect potential real-time violations during runtime. We leverage [RadSan](https://github.com/realtime-sanitizer/radsan), a real-time safety testing tool optimized for C and C++ environments. RadSan is engineered to identify real-time violations, including but not limited to, memory allocation/deallocation and thread synchronization issues.

## How to Use
The tests are currently only supported on *macOS*, *Linux*, *WSL*:

Get Radsan Docker:
```bash
docker pull realtimesanitizer/radsan-clang
```

Clone repository and get submodules:
```bash
git clone --recursive https://github.com/tu-studio/anira-rt-principle-check/
```

Prepare docker:
```bash
# Start RadSan docker image with mounted directory
sudo docker run -v  $(pwd):/anira-rt-principle-check -it realtimesanitizer/radsan-clang /bin/bash

#Install necessary dependencies:
apt-get update && apt-get install -y git cmake vim
```

Build inside docker:
```
cd anira-rt-principle-check/anira-rt-principle-check/
cmake . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release

cmake --build cmake-build-release --config Release --target simple-real-time-violations
cmake --build cmake-build-release --config Release --target validate-anira
cmake --build cmake-build-release --config Release --target validate-inference-engines
```
Execute:
```bash
RADSAN_ERROR_MODE=continue ./cmake-build-release/simple-real-time-violations 2>&1 | tee cmake-build-release/simple-real-time-violations.txt
RADSAN_ERROR_MODE=continue ./cmake-build-release/validate-anira 2>&1 | tee cmake-build-release/validate-anira.txt
RADSAN_ERROR_MODE=continue ./cmake-build-release/validate-inference-engines 2>&1 | tee cmake-build-release/validate-inference-engines.txt
```

## Contributors
- [Valentin Ackva](https://github.com/vackva)
- [Fares Schulz](https://github.com/faressc)

## License
This project is licensed under [Apache-2.0](LICENSE).

