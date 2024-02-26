#ifndef ANIRA_RT_PRINCIPLE_CHECK_LIBTORCH_H
#define ANIRA_RT_PRINCIPLE_CHECK_LIBTORCH_H

#include "enginebase.h"

class LibTorch : public EngineBase {
public:
    LibTorch(anira::InferenceConfig& conf) : EngineBase(conf) {
        std::string omp_num_threads = "OMP_NUM_THREADS=1";
        std::string mkl_num_threads = "MKL_NUM_THREADS=1";
#if WIN32
        _putenv(omp_num_threads.data());
        _putenv(mkl_num_threads.data());
#else
        putenv(omp_num_threads.data());
        putenv(mkl_num_threads.data());
#endif

        module = torch::jit::load(config.m_model_path_torch);
        shape = config.m_model_input_shape_torch;
    }

    void executeInference() override {
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

    [[clang::realtime]] void inference() {
        outputTensor = module.forward(inputs).toTensor();
    }

private:
    torch::Tensor outputTensor;
    std::vector<torch::jit::IValue> inputs;
    torch::jit::Module module;

    std::vector<int64_t> shape;
};

#endif //ANIRA_RT_PRINCIPLE_CHECK_LIBTORCH_H
