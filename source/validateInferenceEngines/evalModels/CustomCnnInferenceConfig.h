#ifndef ANIRA_RT_PRINCIPLE_CHECK_CUSTOMCNNINFERENCECONFIG_H
#define ANIRA_RT_PRINCIPLE_CHECK_CUSTOMCNNINFERENCECONFIG_H

#include "CustomInferenceConfig.h"

static CustomInferenceConfig cnnConfig(
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-2048.pt"),
        {1, 1, 15380},
        {1, 1, 2048},
        STEERABLENAFX_MODELS_PATH_PYTORCH + std::string("model_0/steerable-nafx-libtorch-2048.onnx"),
        {1, 1, 15380},
        {1, 1, 2048},
        STEERABLENAFX_MODELS_PATH_TENSORFLOW + std::string("model_0/steerable-nafx-2048.tflite"),
        {1, 15380, 1},
        {1, 2048, 1},
        1,
        2048,
        15380,
        2048,
        2048,
        0,
        false,
        0.5f,
        false
);


#endif //ANIRA_RT_PRINCIPLE_CHECK_CUSTOMCNNINFERENCECONFIG_H
