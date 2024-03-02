#ifndef ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASEVAL_H
#define ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASEVAL_H

#include "../evalModels/CustomInferenceConfig.h"

class EngineBaseVal {
public:
    EngineBaseVal(CustomInferenceConfig& conf);
    virtual void executeInference() = 0;

protected:
    CustomInferenceConfig& config;

    std::vector<float> inputData;
    std::vector<float> outputData;

    size_t inputSize;
    size_t outputSize;
};

#endif //ANIRA_RT_PRINCIPLE_CHECK_ENGINEBASEVAL_H
