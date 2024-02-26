#ifndef ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASEVAL_H
#define ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASEVAL_H

#include "anira/anira.h"

class EngineBaseVal {
public:
    EngineBaseVal(anira::InferenceConfig& conf);
    virtual void executeInference() = 0;

protected:
    anira::InferenceConfig& config;

    std::vector<float> inputData;
    std::vector<float> outputData;

    size_t inputSize;
    size_t outputSize;
};

#endif //ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASEVAL_H
