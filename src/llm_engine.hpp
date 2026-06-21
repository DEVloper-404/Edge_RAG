#pragma once
#include <string>
#include <vector>
#include "llama.h"

class LlmEngine {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;

public:
    // Initializes the Qwen3 model with strict memory limits
    LlmEngine(const std::string& model_path);
    ~LlmEngine();

    // Takes the retrieved document context and the user's question, and prints the answer
    void generateAnswer(const std::string& context, const std::string& question);
};
