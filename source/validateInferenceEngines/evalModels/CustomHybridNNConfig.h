#ifndef ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H
#define ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H

#include "CustomInferenceConfig.h"

static CustomInferenceConfig hybridNNConfig(
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/model_0-streaming.pt"),
        {128, 1, 150},
        {128, 1},
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/model_0-tflite-streaming.onnx"),
        {128, 150, 1},
        {128, 1},
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/model_0-streaming.tflite"),
        {128, 150, 1},
        {128, 1},
        128,
        1,
        150,
        1,
        256,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H
