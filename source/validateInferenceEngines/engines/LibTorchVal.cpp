#include "LibTorchVal.h"

LibTorchVal::LibTorchVal(CustomInferenceConfig &conf) : EngineBaseVal(conf) {
    torch::set_num_threads(1);

    module = torch::jit::load(config.m_model_path_torch);
    shape = config.m_model_input_shape_torch;
}

void LibTorchVal::executeInference() {
    for (size_t i = 0; i < inputSize; i++) {
        inputData[i] = (float) i * 0.000001f;
    }
    torch::Tensor inputTensor = torch::from_blob(inputData.data(), shape);
    inputs.clear();
    inputs.emplace_back(inputTensor);

    inference();

    outputTensor = outputTensor.view({-1});

    for (size_t i = 0; i < outputSize; i++) {
        outputData[i] = outputTensor[(int64_t) i].item<float>();
        //std::cout << outputData[i] << std::endl;
    }
}

void LibTorchVal::inference() {
    outputTensor = module.forward(inputs).toTensor();
}
