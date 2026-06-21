#pragma once
#include <string>
#include <vector>

struct DocumentChunk {
    std::string text;
    std::string contextHeader;
    std::vector<float> embedding;
    float score = 0.0f; // Tracks how well this chunk matches a user query
};

class EmbeddingEngine; // Forward declaration

class RagEngine {
public:
    // Reads a markdown file and breaks it down into contextual chunks
    static std::vector<DocumentChunk> chunkMarkdownFile(const std::string& filePath);

    // Compares a query vector against all chunks and returns the Top-K closest matches
    static std::vector<DocumentChunk> searchTopK(
        const std::vector<float>& queryEmbedding, 
        std::vector<DocumentChunk>& database, 
        size_t k
    );

    // Loads or creates the database of chunks from all markdown files in a directory, using caching
    static std::vector<DocumentChunk> loadOrCreateDatabase(
        const std::string& directoryPath, 
        EmbeddingEngine& embedder, 
        const std::string& cachePath = "vector_cache.bin"
    );
};
