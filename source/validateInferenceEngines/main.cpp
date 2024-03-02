#include <iostream>

#include "evalModels/CustomCnnInferenceConfig.h"
#include "evalModels/CustomHybridNNConfig.h"
#include "evalModels/CustomStatefulRNNConfig.h"

#include "engines/EngineBaseVal.h"
#include "engines/LibTorchVal.h"
#include "engines/OnnxRuntimeVal.h"
#include "engines/TFLiteVal.h"

enum InferenceBackend {
    LIBTORCH,
    ONNX,
    TFLITE
};

auto modelPath = [](CustomInferenceConfig& config, InferenceBackend backend) -> std::string {
    switch (backend) {
        case InferenceBackend::LIBTORCH:
            return config.m_model_path_torch;
        case InferenceBackend::ONNX:
            return config.m_model_path_onnx;
        case InferenceBackend::TFLITE:
            return config.m_model_path_tflite;
        default:
            return "Unknown";
    }
};


auto engineEnumToString = [](InferenceBackend backend) -> std::string {
    switch (backend) {
        case InferenceBackend::LIBTORCH:
            return "LIBTORCH";
        case InferenceBackend::ONNX:
            return "ONNX";
        case InferenceBackend::TFLITE:
            return "TFLITE";
        default:
            return "Unknown";
    }
};

std::unique_ptr<EngineBaseVal> createInferenceEngine(CustomInferenceConfig currentConfig, InferenceBackend currentEngine) {
    switch (currentEngine) {
        case InferenceBackend::LIBTORCH:
            return std::make_unique<LibTorchVal>(currentConfig);
        case InferenceBackend::ONNX:
            return std::make_unique<OnnxRuntimeVal>(currentConfig);
        case InferenceBackend::TFLITE:
            return std::make_unique<TFLiteVal>(currentConfig);
        default:
            throw std::invalid_argument("Unsupported inference engine");
    }
};

int main() {
    std::vector<InferenceBackend> inferenceEngines = {InferenceBackend::LIBTORCH,
                                                      InferenceBackend::ONNX,
                                                      InferenceBackend::TFLITE};

    std::vector<CustomInferenceConfig> modelsToInference = {hybridNNConfig,
                                                            cnnConfig,
                                                            statefulRNNConfig};

    const size_t numberOfInferences = 10;
    std::unique_ptr<EngineBaseVal> engineToCheck {nullptr};

    for (auto currentEngine : inferenceEngines) {
        for (auto& currentConfig : modelsToInference) {
            engineToCheck = createInferenceEngine(currentConfig, currentEngine);
            for (size_t inferenceCount = 0; inferenceCount < numberOfInferences; ++inferenceCount) {
                std::cout << "-------------------------------------------------------------" << std::endl;
                std::cout << "-------------- Anira-Real-Time-Principle-Check --------------" << std::endl;
                std::cout << "-------------------------------------------------------------" << std::endl;
                std::cout << "-- Inference Engine: " << engineEnumToString(currentEngine) << std::endl;
                std::cout << "-- Model: " << modelPath(currentConfig, currentEngine) << std::endl;
                std::cout << "-- Inference: " << inferenceCount << std::endl;
                std::cout << "-------------------------------------------------------------" << "\n" << std::endl;
                engineToCheck->executeInference();
            }
        }
    }

    return 0;
}
