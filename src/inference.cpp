#include "inference.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <sstream>

EmbeddingEngine::EmbeddingEngine(const std::string& model_path) 
    : env(ORT_LOGGING_LEVEL_WARNING, "EmbeddingEngine"),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {
    
    Ort::SessionOptions session_options;
    
    // CRITICAL PENTIUM OPTIMIZATIONS: Force 1 thread to avoid thrashing your 4GB RAM
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session = Ort::Session(env, model_path.c_str(), session_options);
        std::cout << "ONNX Embedding Model loaded successfully from: " << model_path << std::endl;
        
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "Number of inputs: " << num_input_nodes << std::endl;
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            std::cout << "Input " << i << " : name=" << input_name.get();
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::cout << ", type=" << tensor_info.GetElementType() << std::endl;
        }

        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "Number of outputs: " << num_output_nodes << std::endl;
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            std::cout << "Output " << i << " : name=" << output_name.get();
            auto type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::cout << ", type=" << tensor_info.GetElementType() << std::endl;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
    }
}

// A crude space-based fallback tokenizer for testing purposes
std::vector<int64_t> EmbeddingEngine::simpleTokenize(const std::string& text, size_t max_length) {
    std::vector<int64_t> tokens;
    tokens.push_back(101); // BERT [CLS] start token
    
    std::string word;
    std::stringstream ss(text);
    while (ss >> word && tokens.size() < max_length - 1) {
        // Basic deterministic hashing to simulate token IDs for testing without a massive vocab file
        int64_t hash_id = 0;
        for (char c : word) hash_id = (hash_id * 31) + c;
        tokens.push_back(std::abs(hash_id % 30000)); 
    }
    
    while (tokens.size() < max_length) {
        tokens.push_back(0); // Pad remaining space with [PAD] token
    }
    return tokens;
}

std::vector<float> EmbeddingEngine::getEmbedding(const std::string& text) {
    const size_t max_length = 128; // Keep context window tight to save memory
    
    // Generate inputs
    std::vector<int64_t> input_ids = simpleTokenize(text, max_length);
    // Commented out temporary bounds test to restore original tokenization mapping:
    // for (size_t i = 1; i < input_ids.size(); ++i) {
    //     if (input_ids[i] != 0) input_ids[i] = 102;
    // }
    std::vector<int64_t> attention_mask(max_length, 0);
    std::vector<int64_t> token_type_ids(max_length, 0);
    
    for(size_t i=0; i<max_length; ++i) {
        if (input_ids[i] != 0) attention_mask[i] = 1;
    }

    // Define dimensions for the ONNX tensors [Batch Size (1) x Sequence Length (128)]
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(max_length)};

    // Map raw C++ vectors directly into ONNX Runtime tensors (Zero-copy optimization)
    // Grouped into a contiguous array of Ort::Value to prevent stack out-of-bounds access in session.Run
    Ort::Value input_tensors[] = {
        Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()),
        Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()),
        Ort::Value::CreateTensor<int64_t>(memory_info, token_type_ids.data(), token_type_ids.size(), input_shape.data(), input_shape.size())
    };

    // Define input and output names as expected by the all-MiniLM model export
    const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
    const char* output_names[] = {"last_hidden_state"}; // Or "sentence_embedding" depending on export

    try {
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors, 3,
            output_names, 1
        );

        // Get access to the raw float data containing our vector coordinates
        float* float_array = output_tensors[0].GetTensorMutableData<float>();
        
        // Extract the first 384 dimensions (The hidden size dimensions of miniLM)
        std::vector<float> embedding(float_array, float_array + 384);

        // Perform L2 Normalization (Essential step so Cosine Similarity is accurate later)
        float norm = 0.0f;
        for (float val : embedding) norm += val * val;
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& val : embedding) val /= norm;
        }

        return embedding;

    } catch (const Ort::Exception& e) {
        std::cerr << "Inference Execution Failed: " << e.what() << std::endl;
        return std::vector<float>(384, 0.0f);
    }
}
