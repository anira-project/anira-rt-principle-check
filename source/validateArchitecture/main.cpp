#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <algorithm>

#include "../../anira/extras/models/stateful-rnn/StatefulRNNPrePostProcessor.h"
#include "../../anira/extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../anira/extras/models/cnn/CNNPrePostProcessor.h"

struct ModelConfig {
    anira::InferenceConfig& config;
    anira::PrePostProcessor& processor;
    std::string configName;

    ModelConfig(anira::InferenceConfig& c, anira::PrePostProcessor& p, std::string name) : config(c), processor(p), configName(name) {}
};

std::vector<int> generatePowersOfTwo(int min, int max) {
    std::vector<int> powers;

    int startExp = std::ceil(std::log2(min));
#if WIN32
    startExp = max(startExp, 0);
#else
    startExp = std::max(startExp, 0);
#endif

    for (int exp = startExp; ; ++exp) {
        int power = std::pow(2, exp);
        if (power > max) break;
        powers.push_back(power);
    }

    return powers;
}

std::vector<float> generateRandomAudioBuffer(size_t numSamples) {
    std::vector<float> buffer(numSamples);
    srand(static_cast<unsigned int>(time(nullptr)));

    for (size_t i = 0; i < numSamples; ++i) {
        buffer[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    return buffer;
}

[[clang::realtime]] void callProcess(std::unique_ptr<anira::InferenceHandler>& manager, std::vector<float>& bufferToProcess) {
    float* bufferPtr = bufferToProcess.data();
    manager->process(&bufferPtr, bufferToProcess.size());
}

CNNPrePostProcessor cnnPrePostProcessor;
HybridNNPrePostProcessor hybridNnPrePostProcessor;
StatefulRNNPrePostProcessor statefulRnnPrePostProcessor;

int main() {
    std::vector<anira::InferenceConfig> modelsToInference = {hybridNNConfig,
                                                             cnnConfig,
                                                             statefulRNNConfig};

    auto bufferSizesToTest = generatePowersOfTwo(2048, 8192);

    std::vector<ModelConfig> combinedConfigs = {
            ModelConfig(cnnConfig, cnnPrePostProcessor, "CNN"),
            ModelConfig(hybridNNConfig, hybridNnPrePostProcessor, "Hybrid"),
            ModelConfig(statefulRNNConfig, statefulRnnPrePostProcessor, "Stateful RNN")
    };

    const size_t numberOfInferences = 10;
    std::unique_ptr<anira::InferenceHandler> handlerToCheck {nullptr};

    for (auto& modelConfig : combinedConfigs) {
        if (handlerToCheck != nullptr) handlerToCheck.reset();
        handlerToCheck = std::make_unique<anira::InferenceHandler>(modelConfig.processor, modelConfig.config);
        handlerToCheck->setInferenceBackend(anira::InferenceBackend::NONE);
        std::cout << "-------------------------------------------------------------" << std::endl;
        std::cout << "-------------- Anira-Real-Time-Principle-Check --------------" << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
        std::cout << "-- Config: " << modelConfig.configName << std::endl;
        for (auto bufferSize : bufferSizesToTest) {
            std::cout << "-- Buffer Size: " << bufferSize << std::endl;
            anira::HostAudioConfig audioConfig {1, (size_t) bufferSize, 48000};
            handlerToCheck->prepare(audioConfig);
            for (size_t inferenceCount = 0; inferenceCount < numberOfInferences; ++inferenceCount) {
                std::cout << "-- Inference: " << inferenceCount << std::endl;
                auto bufferToProcess = generateRandomAudioBuffer((size_t) bufferSize);
                callProcess(handlerToCheck, bufferToProcess);
            }
        }
        std::cout << "-------------------------------------------------------------" << "\n" << std::endl;
    }

    return 0;
}