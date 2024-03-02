#ifndef ANIRA_RT_PRINCIPLE_CHECK_ONNXRUNTIMEVAL_H
#define ANIRA_RT_PRINCIPLE_CHECK_ONNXRUNTIMEVAL_H

#include "EngineBaseVal.h"
#include <onnxruntime_cxx_api.h>

class OnnxRuntimeVal : public EngineBaseVal {
public:
    OnnxRuntimeVal(CustomInferenceConfig& conf);

    void executeInference() override;
    [[clang::realtime]] void inference();

private:
    Ort::Env env;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    std::vector<Ort::Value> inputTensor;
    std::vector<Ort::Value> outputTensor;

    std::unique_ptr<Ort::AllocatedStringPtr> inputName;
    std::unique_ptr<Ort::AllocatedStringPtr> outputName;

    std::array<const char *, 1> inputNames;
    std::array<const char *, 1> outputNames;
};

#endif //ANIRA_RT_PRINCIPLE_CHECK_ONNXRUNTIME_H>