#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class EmbeddingEngine {
private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    // A ultra-lightweight basic tokenizer stub for demonstration
    std::vector<int64_t> simpleTokenize(const std::string& text, size_t max_length);

public:
    EmbeddingEngine(const std::string& model_path);
    
    // Core function to turn text into a 384-dimensional vector
    std::vector<float> getEmbedding(const std::string& text);
};
