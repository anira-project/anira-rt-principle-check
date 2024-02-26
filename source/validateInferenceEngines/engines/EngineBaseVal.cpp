#include "EngineBaseVal.h"

EngineBaseVal::EngineBaseVal(anira::InferenceConfig &conf) : config(conf) {
    inputSize = config.m_batch_size * config.m_model_input_size_backend;
    outputSize = config.m_batch_size * config.m_model_output_size_backend;

    inputData.reserve(inputSize);
    outputData.reserve(outputSize);
}
