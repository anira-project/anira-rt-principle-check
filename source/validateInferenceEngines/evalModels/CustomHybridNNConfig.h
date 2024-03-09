#ifndef ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H
#define ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H

#include "CustomInferenceConfig.h"

static CustomInferenceConfig hybridNNConfig(
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-dynamic.pt"),
        {2048, 1, 1},
        {2048, 1, 1},
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {2048, 1, 1},
        {2048, 1, 1},
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/GuitarLSTM-2048.tflite"),
        {1, 2048, 1},
        {1, 2048, 1},
        1,
        2048,
        2048,
        2048,
        2048,
        0,
        false,
        0.5f,
        true
);


#endif //ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H
