#ifndef ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H
#define ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H

#include "CustomInferenceConfig.h"

static CustomInferenceConfig hybridNNConfig(
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-dynamic.pt"),
        {2048, 1, 150},
        {2048, 1},
        GUITARLSTM_MODELS_PATH_PYTORCH + std::string("model_0/GuitarLSTM-libtorch-dynamic.onnx"),
        {2048, 1, 150},
        {2048, 1},
        GUITARLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/GuitarLSTM-2048.tflite"),
        {2048, 150, 1},
        {2048, 1},
        2048,
        1,
        150,
        1,
        2048,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_RT_PRINCIPLE_CHECK_CUSTOMHYBRIDNNCONFIG_H
