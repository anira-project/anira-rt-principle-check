#ifndef ANIRA_RT_PRINCIPLE_CHECK_UTILS_H
#define ANIRA_RT_PRINCIPLE_CHECK_UTILS_H

#include <anira/anira.h>
#include "../../anira/extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../anira/extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../anira/extras/models/cnn/CNNConfig.h"

auto engineEnumToString = [](anira::InferenceBackend backend) -> std::string {
    switch (backend) {
        case anira::InferenceBackend::LIBTORCH:
            return "LIBTORCH";
        case anira::InferenceBackend::ONNX:
            return "ONNX";
        case anira::InferenceBackend::TFLITE:
            return "TFLITE";
        default:
            return "Unknown";
    }
};

auto modelPath = [](anira::InferenceConfig& config, anira::InferenceBackend backend) -> std::string {
    switch (backend) {
        case anira::InferenceBackend::LIBTORCH:
            return config.m_model_path_torch;
        case anira::InferenceBackend::ONNX:
            return config.m_model_path_onnx;
        case anira::InferenceBackend::TFLITE:
            return config.m_model_path_tflite;
        default:
            return "Unknown";
    }
};



#endif //ANIRA_RT_PRINCIPLE_CHECK_UTILS_H
