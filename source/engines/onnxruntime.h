#ifndef ANIRA_RT_PRINCIPLE_CHECK_ONNXRUNTIME_H
#define ANIRA_RT_PRINCIPLE_CHECK_ONNXRUNTIME_H

#include "enginebase.h"

class OnnxRuntime : public EngineBase {
public:
    OnnxRuntime(anira::InferenceConfig& conf) : EngineBase(conf),
                                                memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
    {
        session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
        std::wstring modelWideStr = std::wstring(config.m_model_path_onnx.begin(), config.m_model_path_onnx.end());
        const wchar_t* modelWideCStr = modelWideStr.c_str();
        session = std::make_unique<Ort::Session>(Ort::Session(env, modelWideCStr, session_options));
#else
        session = std::make_unique<Ort::Session>(Ort::Session(env, config.m_model_path_onnx.c_str(), session_options));
#endif

        inputName = std::make_unique<Ort::AllocatedStringPtr>(session->GetInputNameAllocated(0, ort_alloc));
        outputName = std::make_unique<Ort::AllocatedStringPtr>(session->GetOutputNameAllocated(0, ort_alloc));
        inputNames = {(char*) inputName->get()};
        outputNames = {(char*) outputName->get()};

        std::vector<int64_t> inputShape = config.m_model_input_shape_onnx;
        inputData.resize(inputSize, 0.0f);

        inputTensor.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info,
                inputData.data(),
                inputSize,
                inputShape.data(),
                inputShape.size()
        ));
    }

    void executeInference() override {
        auto audioInputPtr = inputTensor[0].GetTensorMutableData<float>();
        for (size_t i = 0; i < inputSize; i++) {
            audioInputPtr[0] = (float) i * 0.000001f;
        }

        inference();

        auto outputPtr = outputTensor[0].GetTensorMutableData<float>();
        for (size_t i = 0; i < outputSize; ++i) {
            outputData[i] = outputPtr[i];
            //std::cout << outputData[i] << std::endl;
        }
    }

    [[clang::realtime]] void inference() {
        outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensor.data(), inputNames.size(), outputNames.data(), outputNames.size());
    }

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