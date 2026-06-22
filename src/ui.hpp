#pragma once
#include <string>
#include <vector>
#include "rag_engine.hpp"
#include "inference.hpp"
#include "llm_engine.hpp"

class ChatUI {
public:
    // This function will take control of the terminal and run the chat loop
    static void start(EmbeddingEngine& embedder, std::vector<DocumentChunk>& database, LlmEngine& llm);
};
