#include <iostream>
#include "rag_engine.hpp"
#include "inference.hpp"
#include "llm_engine.hpp"
#include "ui.hpp"

int main() {
    std::cout << "=== Loading Local Edge RAG Pipeline (Auto-Scan & Cache) ===" << std::endl;

    // 1. Initialize the Embeddings and the LLM
    EmbeddingEngine embedder("../models/all-minilm-l6-v2.onnx");
    LlmEngine llm("../models/Qwen3-1.7B-Q4_K_M.gguf");

    // 2. Read, chunk and vectorize all Markdown files in the 'Source files' directory using caching
    std::cout << "\nScanning directory for journals..." << std::endl;
    std::vector<DocumentChunk> database = RagEngine::loadOrCreateDatabase("Source files", embedder, "vector_cache.bin");
    std::cout << "Total chunks loaded in vector database: " << database.size() << "\n" << std::endl;

    if (database.empty()) {
        std::cout << "Error: No markdown files found in the 'Source files' directory!" << std::endl;
        return 0;
    }

    // 3. Launch the Terminal UI!
    ChatUI::start(embedder, database, llm);

    return 0;
}
