#include <iostream>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>

std::mutex audioMutex;

[[clang::realtime]] void processBufferWithMutex(std::vector<float>& buffer) {
    std::lock_guard<std::mutex> lock(audioMutex);
    for (auto& sample : buffer) {
        sample *= 2;
    }
}

[[clang::realtime]] void processBufferWithSleep(std::vector<float>& buffer) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    for (auto& sample : buffer) {
        sample += 1;
    }
}

[[clang::realtime]] void processBufferWithAllocation(std::vector<float>& buffer) {
    std::vector<float> tempBuffer(256);
    for (size_t i = 0; i < buffer.size(); ++i) {
        tempBuffer[i] = buffer[i] * 3;
    }
    buffer = tempBuffer;
}

[[clang::realtime]] void printBuffer(std::vector<float>& buffer) {
    for (const auto& sample : buffer) {
        std::cout << sample << " ";
    }
    std::cout << std::endl;
}

[[clang::realtime]] void invertBuffer(std::vector<float>& buffer) {
    for (auto& sample : buffer) {
        sample = -sample;
    }
}

[[clang::realtime]] void clearBuffer(std::vector<float>& buffer) {
    std::fill(buffer.begin(), buffer.end(), 0.0f);
}

[[clang::realtime]] float sumBuffer(const std::vector<float>& buffer) {
    float sum = 0.0f;
    for (const auto& sample : buffer) {
        sum += sample;
    }
    std::cout << "sum: " << sum << std::endl;
    return sum;
}

[[clang::realtime]] std::vector<float> copyBuffer(std::vector<float> buffer) {
    for (auto& sample : buffer) {
        sample = sample * -1.f;
    }
    return buffer;
}

int main() {
    std::vector<float> audioBuffer(256, 0.5f);

    std::cout << "Processing buffer with mutex..." << std::endl;
    processBufferWithMutex(audioBuffer);

    std::cout << "Processing buffer with sleep..." << std::endl;
    processBufferWithSleep(audioBuffer);

    std::cout << "Processing buffer with dynamic memory allocation..." << std::endl;
    processBufferWithAllocation(audioBuffer);

    std::cout << "Printing buffer..." << std::endl;
    printBuffer(audioBuffer);

    std::cout << "Inverting buffer values..." << std::endl;
    invertBuffer(audioBuffer);

    std::cout << "Clearing buffer..." << std::endl;
    clearBuffer(audioBuffer);

    std::cout << "Printing sum of buffer..." << std::endl;
    float sum = sumBuffer(audioBuffer);

    std::cout << "Copying the whole buffer..." << std::endl;
    std::vector<float> bufferCopy = copyBuffer(audioBuffer);

    return 0;
}


