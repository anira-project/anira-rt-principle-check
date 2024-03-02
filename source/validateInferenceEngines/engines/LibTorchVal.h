#ifndef ANIRA_RT_PRINCIPLE_CHECK_LIBTORCHVAL_H
#define ANIRA_RT_PRINCIPLE_CHECK_LIBTORCHVAL_H

#include "EngineBaseVal.h"
#include <torch/script.h>

class LibTorchVal : public EngineBaseVal {
public:
    LibTorchVal(CustomInferenceConfig& conf);

    void executeInference() override;
    [[clang::realtime]] void inference();

private:
    torch::Tensor outputTensor;
    std::vector<torch::jit::IValue> inputs;
    torch::jit::Module module;

    std::vector<int64_t> shape;
};

#endif //ANIRA_RT_PRINCIPLE_CHECK_LIBTORCHVAL_H
