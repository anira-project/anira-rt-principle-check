//
// Created by Valentin Ackva on 15/02/2024.
//

#ifndef ANIRA_RT_PRINCIPLE_CHECK_TFLITE_H
#define ANIRA_RT_PRINCIPLE_CHECK_TFLITE_H

#include <comutil.h>
#include "enginebase.h"

class TFLite : public EngineBase {
public:
    TFLite(anira::InferenceConfig& config) : EngineBase(config) {
        model = TfLiteModelCreateFromFile(config.m_model_path_tflite.c_str());
        options = TfLiteInterpreterOptionsCreate();
        interpreter = TfLiteInterpreterCreate(model, options);
        TfLiteInterpreterOptionsSetNumThreads(options, 1);
        TfLiteInterpreterAllocateTensors(interpreter);
        inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    }

    ~TFLite() {
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
    }

    void executeInference() override {
        for (size_t i = 0; i < inputSize; i++) {
            inputData[i] = (float) i * 0.000001f;
        }
        TfLiteTensorCopyFromBuffer(inputTensor, inputData.data(), inputSize * sizeof(float));

        inference();

        outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        TfLiteTensorCopyToBuffer(outputTensor, outputData.data(), outputSize * sizeof(float));

        for (size_t i = 0; i < outputSize; ++i) {
            //std::cout << outputData[i] << std::endl;
        }
    }

    [[clang::realtime]] void inference() {
        TfLiteInterpreterInvoke(interpreter);
    }

private:
    TfLiteModel* model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter* interpreter;

    TfLiteTensor* inputTensor;
    const TfLiteTensor* outputTensor;
};


#endif //ANIRA_RT_PRINCIPLE_CHECK_TFLITE_H
