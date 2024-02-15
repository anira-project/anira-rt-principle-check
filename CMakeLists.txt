cmake_minimum_required(VERSION 3.16)
project(anira-rt-principle-check C CXX)

set(CMAKE_CXX_STANDARD 20)

option(BUILD_SHARED_LIBS "Build the library as a shared library" OFF)

add_subdirectory(anira)

add_executable(anira-rt-principle-check
        main.cpp
)

target_link_libraries(anira-rt-principle-check anira)

if (MSVC)
    foreach(DLL ${NECESSARY_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME}
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:minimal-libtorch>)
    endforeach()
endif (MSVC)
