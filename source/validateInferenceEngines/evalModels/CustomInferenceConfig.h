#ifndef ANIRA_RT_PRINCIPLE_CHECK_MODELS_H
#define ANIRA_RT_PRINCIPLE_CHECK_MODELS_H

#include <cstdint>
#include <vector>
#include <string>
#include <thread>

struct CustomInferenceConfig {
    CustomInferenceConfig(
            std::string model_path_torch = "",
            std::vector<int64_t> model_input_shape_torch = {},
            std::vector<int64_t> model_output_shape_torch = {},
            std::string model_path_onnx = "",
            std::vector<int64_t> model_input_shape_onnx = {},
            std::vector<int64_t> model_output_shape_onnx = {},
            std::string model_path_tflite = "",
            std::vector<int> model_input_shape_tflite = {},
            std::vector<int> model_output_shape_tflite = {},
            size_t batch_size = 0,
            size_t model_input_size = 0,
            size_t model_input_size_backend = 0,
            size_t model_output_size_backend = 0,
            size_t max_inference_time = 0,
            int model_latency = 0,
            bool warm_up = false,
            float wait_in_process_block = 0.5f,
            bool bind_session_to_thread = false,
            int numberOfThreads = ((int) std::thread::hardware_concurrency() - 1 > 0) ? (int) std::thread::hardware_concurrency() - 1 : 1) :
            m_model_path_torch(model_path_torch),
            m_model_input_shape_torch(model_input_shape_torch),
            m_model_output_shape_torch(model_output_shape_torch),
            m_model_path_onnx(model_path_onnx),
            m_model_input_shape_onnx(model_input_shape_onnx),
            m_model_output_shape_onnx(model_output_shape_onnx),
            m_model_path_tflite(model_path_tflite),
            m_model_input_shape_tflite(model_input_shape_tflite),
            m_model_output_shape_tflite(model_output_shape_tflite),
            m_batch_size(batch_size),
            m_model_input_size(model_input_size),
            m_model_input_size_backend(model_input_size_backend),
            m_model_output_size_backend(model_output_size_backend),
            m_max_inference_time(max_inference_time),
            m_model_latency(model_latency),
            m_warm_up(warm_up),
            m_wait_in_process_block(wait_in_process_block),
            m_bind_session_to_thread(bind_session_to_thread),
            m_number_of_threads(numberOfThreads)
    {}

    std::string m_model_path_torch;
    std::vector<int64_t> m_model_input_shape_torch;
    std::vector<int64_t> m_model_output_shape_torch;

    std::string m_model_path_onnx;
    std::vector<int64_t> m_model_input_shape_onnx;
    std::vector<int64_t> m_model_output_shape_onnx;

    std::string m_model_path_tflite;
    std::vector<int> m_model_input_shape_tflite;
    std::vector<int> m_model_output_shape_tflite;

    size_t m_batch_size;
    size_t m_model_input_size;
    size_t m_model_input_size_backend;
    size_t m_model_output_size_backend;
    size_t m_max_inference_time;
    int m_model_latency;
    bool m_warm_up;

    float m_wait_in_process_block;
    bool m_bind_session_to_thread;
    int m_number_of_threads;
};

#endif //ANIRA_RT_PRINCIPLE_CHECK_MODELS_H
