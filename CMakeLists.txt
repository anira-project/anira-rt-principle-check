cmake_minimum_required(VERSION 3.16)
project(anira-rt-principle-check C CXX)

set(CMAKE_CXX_STANDARD 20)

option(BUILD_SHARED_LIBS "Build the library as a shared library" OFF)

add_subdirectory(anira)

add_executable(validate-inference-engines
        source/validateInferenceEngines/main.cpp
        source/validateInferenceEngines/engines/EngineBaseVal.cpp
        source/validateInferenceEngines/engines/OnnxRuntimeVal.cpp
        source/validateInferenceEngines/engines/LibTorchVal.cpp
        source/validateInferenceEngines/engines/TFLiteVal.cpp
)

add_executable(validate-anira
        source/validateArchitecture/main.cpp
)

set(GUITARLSTM_MODELS_PATH_TENSORFLOW "${CMAKE_CURRENT_SOURCE_DIR}/anira/extras/models/hybrid-nn/GuitarLSTM/tensorflow-version/models/")
set(GUITARLSTM_MODELS_PATH_PYTORCH "${CMAKE_CURRENT_SOURCE_DIR}/anira/extras/models/hybrid-nn/GuitarLSTM/pytorch-version/models/")
set(STEERABLENAFX_MODELS_PATH_TENSORFLOW "${CMAKE_CURRENT_SOURCE_DIR}/anira/extras/models/cnn/steerable-nafx/models/")
set(STEERABLENAFX_MODELS_PATH_PYTORCH "${CMAKE_CURRENT_SOURCE_DIR}/anira/extras/models/cnn/steerable-nafx/models/")
set(STATEFULLSTM_MODELS_PATH_TENSORFLOW "${CMAKE_CURRENT_SOURCE_DIR}/anira/extras/models/stateful-rnn/stateful-lstm/models/")
set(STATEFULLSTM_MODELS_PATH_PYTORCH "${CMAKE_CURRENT_SOURCE_DIR}/anira/extras/models/stateful-rnn/stateful-lstm/models/")

add_compile_definitions(
        GUITARLSTM_MODELS_PATH_TENSORFLOW="${GUITARLSTM_MODELS_PATH_TENSORFLOW}"
        GUITARLSTM_MODELS_PATH_TENSORFLOW="${GUITARLSTM_MODELS_PATH_TENSORFLOW}"
        GUITARLSTM_MODELS_PATH_PYTORCH="${GUITARLSTM_MODELS_PATH_PYTORCH}"
        STEERABLENAFX_MODELS_PATH_TENSORFLOW="${STEERABLENAFX_MODELS_PATH_TENSORFLOW}"
        STEERABLENAFX_MODELS_PATH_PYTORCH="${STEERABLENAFX_MODELS_PATH_PYTORCH}"
        STATEFULLSTM_MODELS_PATH_TENSORFLOW="${STATEFULLSTM_MODELS_PATH_TENSORFLOW}"
        STATEFULLSTM_MODELS_PATH_PYTORCH="${STATEFULLSTM_MODELS_PATH_PYTORCH}"
)

target_link_libraries(validate-inference-engines anira)
target_link_libraries(validate-anira anira)

if (MSVC)
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_ONNX "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/*.dll")
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_TFLITE "${ANIRA_TENSORFLOWLITE_SHARED_LIB_PATH}/*.dll")
    file(GLOB_RECURSE INFERENCE_ENGINE_DLLS_LIBTORCH "${ANIRA_LIBTORCH_SHARED_LIB_PATH}*.dll")

    set(NECESSARY_DLLS
            ${INFERENCE_ENGINE_DLLS_ONNX}
            ${INFERENCE_ENGINE_DLLS_TFLITE}
            ${INFERENCE_ENGINE_DLLS_LIBTORCH}
    )

    set(TARGETS_NEEDING_DLLS validate-inference-engines validate-anira)

    foreach(TARGET IN LISTS TARGETS_NEEDING_DLLS)
        foreach(DLL IN LISTS NECESSARY_DLLS)
            message(STATUS "Copying ${DLL} to target ${TARGET}")
            add_custom_command(TARGET ${TARGET}
                    PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${DLL}
                    $<TARGET_FILE_DIR:${TARGET}>)
        endforeach()
    endforeach()
endif (MSVC)
