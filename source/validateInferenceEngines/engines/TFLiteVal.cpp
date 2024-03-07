#include "TFLiteVal.h"

TFLiteVal::TFLiteVal(CustomInferenceConfig &config) : EngineBaseVal(config) {
    model = TfLiteModelCreateFromFile(config.m_model_path_tflite.c_str());
    options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);
    interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);
    inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
}

TFLiteVal::~TFLiteVal() {
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}

void TFLiteVal::executeInference() {
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

void TFLiteVal::inference() {
    TfLiteInterpreterInvoke(interpreter);
}
