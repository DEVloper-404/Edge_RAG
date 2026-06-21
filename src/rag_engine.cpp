#include "rag_engine.hpp"
#include "inference.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>   // Required for std::inner_product
#include <algorithm> // Required for std::sort
#include <filesystem>
#include <map>

namespace fs = std::filesystem;

// Helper functions for binary serialization
static void writeString(std::ostream& os, const std::string& str) {
    uint32_t size = str.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(str.data(), size);
}

static std::string readString(std::istream& is) {
    uint32_t size = 0;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::string str(size, '\0');
    is.read(&str[0], size);
    return str;
}

static void serializeCache(
    const std::string& cachePath, 
    const std::map<std::string, std::pair<int64_t, std::vector<DocumentChunk>>>& cache
) {
    std::ofstream os(cachePath, std::ios::binary);
    if (!os.is_open()) return;

    uint32_t num_files = cache.size();
    os.write(reinterpret_cast<const char*>(&num_files), sizeof(num_files));

    for (const auto& [filepath, val] : cache) {
        writeString(os, filepath);
        int64_t last_mod = val.first;
        os.write(reinterpret_cast<const char*>(&last_mod), sizeof(last_mod));

        uint32_t num_chunks = val.second.size();
        os.write(reinterpret_cast<const char*>(&num_chunks), sizeof(num_chunks));

        for (const auto& chunk : val.second) {
            writeString(os, chunk.text);
            writeString(os, chunk.contextHeader);
            
            uint32_t embed_size = chunk.embedding.size();
            os.write(reinterpret_cast<const char*>(&embed_size), sizeof(embed_size));
            os.write(reinterpret_cast<const char*>(chunk.embedding.data()), embed_size * sizeof(float));
        }
    }
}

static std::map<std::string, std::pair<int64_t, std::vector<DocumentChunk>>> deserializeCache(const std::string& cachePath) {
    std::map<std::string, std::pair<int64_t, std::vector<DocumentChunk>>> cache;
    std::ifstream is(cachePath, std::ios::binary);
    if (!is.is_open()) return cache;

    uint32_t num_files = 0;
    is.read(reinterpret_cast<char*>(&num_files), sizeof(num_files));

    for (uint32_t f = 0; f < num_files; ++f) {
        std::string filepath = readString(is);
        int64_t last_mod = 0;
        is.read(reinterpret_cast<char*>(&last_mod), sizeof(last_mod));

        uint32_t num_chunks = 0;
        is.read(reinterpret_cast<char*>(&num_chunks), sizeof(num_chunks));

        std::vector<DocumentChunk> chunks;
        for (uint32_t c = 0; c < num_chunks; ++c) {
            DocumentChunk chunk;
            chunk.text = readString(is);
            chunk.contextHeader = readString(is);

            uint32_t embed_size = 0;
            is.read(reinterpret_cast<char*>(&embed_size), sizeof(embed_size));
            chunk.embedding.resize(embed_size);
            is.read(reinterpret_cast<char*>(chunk.embedding.data()), embed_size * sizeof(float));
            chunk.score = 0.0f;
            chunks.push_back(chunk);
        }
        cache[filepath] = {last_mod, chunks};
    }
    return cache;
}

std::vector<DocumentChunk> RagEngine::chunkMarkdownFile(const std::string& filePath) {
    std::vector<DocumentChunk> chunks;
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return chunks;
    }

    std::string line;
    std::string currentHeaderContext = "Global";
    std::string currentChunkText = "";
    bool inCodeBlock = false;

    // Read the file line-by-line to protect our 4GB RAM from loading massive files all at once
    while (std::getline(file, line)) {
        
        // 1. Check for Code Block boundaries
        if (line.rfind("```", 0) == 0) {
            inCodeBlock = !inCodeBlock;
            currentChunkText += line + "\n";
            if (!inCodeBlock) { // Code block ended, push as a single chunk
                DocumentChunk chunk{currentChunkText, currentHeaderContext, {}, 0.0f};
                chunks.push_back(chunk);
                currentChunkText = "";
            }
            continue;
        }

        if (inCodeBlock) {
            currentChunkText += line + "\n";
            continue;
        }

        // 2. Check for Markdown Headings
        if (line.rfind("#", 0) == 0) {
            // If we have an existing accumulated chunk, save it before shifting topics
            if (!currentChunkText.empty()) {
                DocumentChunk chunk{currentChunkText, currentHeaderContext, {}, 0.0f};
                chunks.push_back(chunk);
                currentChunkText = "";
            }
            currentHeaderContext = line; // Update the context tracking
            continue;
        }

        // 3. Accumulate regular text paragraphs
        if (line.empty()) {
            if (!currentChunkText.empty()) {
                DocumentChunk chunk{currentChunkText, currentHeaderContext, {}, 0.0f};
                chunks.push_back(chunk);
                currentChunkText = "";
            }
        } else {
            currentChunkText += line + "\n";
        }
    }

    // Grab any remaining text at the very end of the file
    if (!currentChunkText.empty()) {
        DocumentChunk chunk{currentChunkText, currentHeaderContext, {}, 0.0f};
        chunks.push_back(chunk);
    }

    return chunks;
}

std::vector<DocumentChunk> RagEngine::searchTopK(
    const std::vector<float>& queryEmbedding, 
    std::vector<DocumentChunk>& database, 
    size_t k
) {
    // 1. Calculate similarity score for every chunk in our local database
    for (auto& chunk : database) {
        if (chunk.embedding.size() != queryEmbedding.size()) {
            chunk.score = 0.0f; // Safety check in case sizes mismatch
            continue;
        }

        // The Pentium Speed Hack: Since both vectors are normalized, 
        // the inner_product (dot product) IS the cosine similarity score.
        chunk.score = std::inner_product(
            queryEmbedding.begin(), queryEmbedding.end(), // Vector A
            chunk.embedding.begin(),                      // Vector B
            0.0f                                          // Starting sum value
        );
    }

    // 2. Sort the database so the chunks with the HIGHEST scores come first
    std::sort(database.begin(), database.end(), [](const DocumentChunk& a, const DocumentChunk& b) {
        return a.score > b.score; // Descending order
    });

    // 3. Extract the top K highest-ranking chunks
    std::vector<DocumentChunk> results;
    size_t limit = std::min(k, database.size());
    for (size_t i = 0; i < limit; ++i) {
        results.push_back(database[i]);
    }

    return results;
}

std::vector<DocumentChunk> RagEngine::loadOrCreateDatabase(
    const std::string& directoryPath, 
    EmbeddingEngine& embedder, 
    const std::string& cachePath
) {
    // Deserialize the existing cache
    auto cache = deserializeCache(cachePath);
    bool cache_dirty = false;

    // Track active files to clean up cache from deleted files
    std::map<std::string, std::pair<int64_t, std::vector<DocumentChunk>>> new_cache;
    std::vector<DocumentChunk> master_database;

    // Scan directory for .md files
    if (fs::exists(directoryPath) && fs::is_directory(directoryPath)) {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".md") {
                std::string filepath = entry.path().string();
                
                // Get last write time as timestamp
                int64_t last_mod = fs::last_write_time(entry).time_since_epoch().count();

                // Check cache hit
                auto cache_it = cache.find(filepath);
                if (cache_it != cache.end() && cache_it->second.first == last_mod) {
                    std::cout << "[Cache Hit] Loaded " << cache_it->second.second.size() 
                              << " chunks for: " << filepath << std::endl;
                    new_cache[filepath] = cache_it->second;
                } else {
                    // Cache miss: process file
                    std::cout << "[Cache Miss] Chunking and Vectorizing: " << filepath << "..." << std::endl;
                    auto chunks = chunkMarkdownFile(filepath);

                    // Generate embeddings
                    for (auto& chunk : chunks) {
                        chunk.embedding = embedder.getEmbedding(chunk.text);
                    }

                    new_cache[filepath] = {last_mod, chunks};
                    cache_dirty = true;
                }

                // Add to master database
                master_database.insert(
                    master_database.end(), 
                    new_cache[filepath].second.begin(), 
                    new_cache[filepath].second.end()
                );
            }
        }
    }

    // Check if any files were deleted from the directory (cache cleaning)
    if (cache.size() != new_cache.size()) {
        cache_dirty = true;
    }

    // Serialize database if cache was dirty
    if (cache_dirty) {
        std::cout << "Saving updated vector database cache to: " << cachePath << std::endl;
        serializeCache(cachePath, new_cache);
    }

    return master_database;
}
