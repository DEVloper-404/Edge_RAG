#include <iostream>
#include "rag_engine.hpp"
#include "inference.hpp"
#include "llm_engine.hpp"

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
        std::cout << "Error: No markdown files found in the current directory!" << std::endl;
        return 0;
    }

    // 3. User Input
    std::string userQuery = "tell me my project rating and how these good as undergrad project?";
    std::cout << "User: " << userQuery << std::endl;

    // 4. Search for Context
    std::vector<float> queryVector = embedder.getEmbedding(userQuery);
    std::vector<DocumentChunk> topMatches = RagEngine::searchTopK(queryVector, database, 5);

    std::cout << "\n--- TOP MATCHES ---" << std::endl;
    for (size_t i = 0; i < topMatches.size(); ++i) {
        std::cout << "Rank [" << i + 1 << "] Score: " << topMatches[i].score 
                  << " | Context: " << topMatches[i].contextHeader << std::endl;
        std::cout << "Excerpt: " << topMatches[i].text.substr(0, 150) << "...\n" << std::endl;
    }

    if (topMatches.empty()) {
        std::cout << "No relevant documents found to answer the question." << std::endl;
        return 0;
    }

    // 5. Generate the Final Answer
    std::string retrievedContext = topMatches[0].text;
    std::cout << "\n[Retrieved Best Context from " << topMatches[0].contextHeader << "]" << std::endl;
    
    // Pass the retrieved chunk and the user's question to Qwen3
    llm.generateAnswer(retrievedContext, userQuery);

    return 0;
}
