#ifndef ANIRA_RT_PRINCIPLE_CHECK_TFLITEVAL_H
#define ANIRA_RT_PRINCIPLE_CHECK_TFLITEVAL_H

#include <comutil.h>
#include "EngineBaseVal.h"

class TFLiteVal : public EngineBaseVal {
public:
    TFLiteVal(anira::InferenceConfig& config);
    ~TFLiteVal();

    void executeInference() override;
    [[clang::realtime]] void inference();

private:
    TfLiteModel* model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter* interpreter;

    TfLiteTensor* inputTensor;
    const TfLiteTensor* outputTensor;
};


#endif //ANIRA_RT_PRINCIPLE_CHECK_TFLITEVAL_H
