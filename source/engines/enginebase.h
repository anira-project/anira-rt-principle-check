#ifndef ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASE_H
#define ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASE_H

#include <anira/anira.h>

class EngineBase {
public:
    EngineBase(anira::InferenceConfig& conf) : config(conf) {
        inputSize = config.m_batch_size * config.m_model_input_size_backend;
        outputSize = config.m_batch_size * config.m_model_output_size_backend;

        inputData.reserve(inputSize);
        outputData.reserve(outputSize);
    }
    virtual void executeInference() = 0;
protected:
    anira::InferenceConfig& config;

    std::vector<float> inputData;
    std::vector<float> outputData;

    size_t inputSize;
    size_t outputSize;
};

#endif //ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASE_H
