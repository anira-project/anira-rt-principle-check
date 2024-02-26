#include <iostream>
#include "utils.h"

int main() {
    std::vector<anira::InferenceBackend> inferenceEngines = {anira::InferenceBackend::LIBTORCH,
                                                             anira::InferenceBackend::ONNX,
                                                             anira::InferenceBackend::TFLITE};

    std::vector<anira::InferenceConfig> modelsToInference = {hybridNNConfig,
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
