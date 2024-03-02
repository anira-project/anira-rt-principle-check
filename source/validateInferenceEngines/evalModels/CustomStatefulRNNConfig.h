#ifndef ANIRA_RT_PRINCIPLE_CHECK_CUSTOMSTATEFULRNNCONFIG_H
#define ANIRA_RT_PRINCIPLE_CHECK_CUSTOMSTATEFULRNNCONFIG_H

#include "CustomInferenceConfig.h"

static CustomInferenceConfig statefulRNNConfig(
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm.pt"),
        {2048, 1, 1},
        {2048, 1, 1},
        STATEFULLSTM_MODELS_PATH_PYTORCH + std::string("model_0/stateful-lstm-libtorch.onnx"),
        {2048, 1, 1},
        {2048, 1, 1},
        STATEFULLSTM_MODELS_PATH_TENSORFLOW + std::string("model_0/stateful-lstm.tflite"),
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

#endif //ANIRA_RT_PRINCIPLE_CHECK_CUSTOMSTATEFULRNNCONFIG_H
