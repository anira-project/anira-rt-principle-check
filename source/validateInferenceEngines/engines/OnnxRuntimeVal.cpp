#include "OnnxRuntimeVal.h"

OnnxRuntimeVal::OnnxRuntimeVal(CustomInferenceConfig &conf) : EngineBaseVal(conf),
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

void OnnxRuntimeVal::executeInference() {
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

void OnnxRuntimeVal::inference() {
    outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensor.data(), inputNames.size(), outputNames.data(), outputNames.size());
}
