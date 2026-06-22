#include "llm_engine.hpp"
#include <iostream>

LlmEngine::LlmEngine(const std::string& model_path) {
    llama_backend_init();

    // 1. Model Parameters (Optimized for Low RAM)
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true; // Reads weights from disk instead of filling up RAM!
    
    // Using llama_model_load_from_file to avoid deprecation warnings
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load LLM from " << model_path << std::endl;
        return;
    }

    // 2. Context Parameters (Optimized for Intel Pentium)
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;      // Increased context size to prevent KV cache exhaustion
    ctx_params.n_threads = 1;     // STRICT: 1 thread so your OS doesn't freeze
    ctx_params.n_threads_batch = 1;

    // Using llama_init_from_model to avoid deprecation warnings
    ctx = llama_init_from_model(model, ctx_params);
    std::cout << "Qwen3 LLM loaded successfully!" << std::endl;
}

LlmEngine::~LlmEngine() {
    if (ctx) llama_free(ctx);
    // Using llama_model_free to avoid deprecation warnings
    if (model) llama_model_free(model);
    llama_backend_free();
}

std::string LlmEngine::generateAnswer(const std::string& context, const std::string& question) {
    // Get the vocabulary from the model (required by updated llama.cpp API)
    const struct llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        return "Error: Could not retrieve vocabulary from model.";
    }

    // Clear the KV cache/memory to start fresh for each new prompt
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // 1. Format the prompt specifically for Qwen Instruct models.
    // Suppress thinking blocks at prompt level by instructing the system.
    std::string prompt = 
        "<|im_start|>system\nYou are a helpful technical assistant. "
        "Answer the question using ONLY the provided context. Do NOT think. Do NOT output a <think> tag or internal thinking process. Go straight to the answer.\n\n"
        "Context:\n" + context + "<|im_end|>\n"
        "<|im_start|>user\n" + question + "<|im_end|>\n"
        "<|im_start|>assistant\n";

    // 2. Convert the text prompt into model tokens
    std::vector<llama_token> tokens_list(prompt.size() + 2); // Buffer size
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
        return "Error: Prompt is too long for the tokenizer buffer.";
    }
    tokens_list.resize(n_tokens);

    // 3. Feed the prompt tokens into the context
    llama_batch prompt_batch = llama_batch_init(n_tokens, 0, 1);
    prompt_batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        prompt_batch.token[i] = tokens_list[i];
        prompt_batch.pos[i] = i;
        prompt_batch.n_seq_id[i] = 1;
        prompt_batch.seq_id[i][0] = 0;
        prompt_batch.logits[i] = (i == n_tokens - 1) ? 1 : 0; // Only get logits for the last prompt token
    }

    if (llama_decode(ctx, prompt_batch) != 0) {
        llama_batch_free(prompt_batch);
        return "Error: Failed to decode prompt.";
    }

    // 4. The Generation Loop (Predicting the next word)
    int n_cur = n_tokens;
    int n_max = 1024; // Max words the AI is allowed to write

    // Allocate a batch of size 1 for single-token generation
    llama_batch loop_batch = llama_batch_init(1, 0, 1);
    loop_batch.n_tokens = 1;
    loop_batch.n_seq_id[0] = 1;
    loop_batch.seq_id[0][0] = 0;
    loop_batch.logits[0] = 1; // Always get logits to predict next token

    bool is_first_iteration = true;
    bool in_thinking = false;
    std::string token_buffer = "";
    std::string final_answer = "";

    while (n_cur < n_tokens + n_max) {
        // Get the probabilities for the next token.
        // In the first iteration, retrieve from the prompt batch (index n_tokens - 1).
        // In subsequent iterations, retrieve from the loop batch (index 0).
        auto* logits = llama_get_logits_ith(ctx, is_first_iteration ? (n_tokens - 1) : 0);
        int n_vocab = llama_vocab_n_tokens(vocab);

        // Simple Greedy Decoding: Pick the token with the highest probability
        llama_token new_token_id = 0;
        float max_prob = -1e9;
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] > max_prob) {
                max_prob = logits[i];
                new_token_id = i;
            }
        }

        // Check if the AI wants to stop writing (End of Text token)
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break; 
        }

        // Convert the token ID back into readable English text.
        // Also includes a fallback streaming filter to skip saving any <think>...</think> block.
        char buf[128];
        int n_chars = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, false);
        if (n_chars > 0) {
            std::string piece(buf, n_chars);
            token_buffer += piece;
            
            // Check for think tag entry
            size_t think_start = token_buffer.find("<think>");
            if (think_start != std::string::npos) {
                in_thinking = true;
                token_buffer = token_buffer.substr(think_start + 7);
            }
            
            // Check for think tag exit
            size_t think_end = token_buffer.find("</think>");
            if (think_end != std::string::npos) {
                in_thinking = false;
                token_buffer = token_buffer.substr(think_end + 8);
            }
            
            // Flush safe buffer content into final_answer
            if (!in_thinking && !token_buffer.empty()) {
                size_t pos_lt = token_buffer.find('<');
                if (pos_lt == std::string::npos) {
                    final_answer += token_buffer;
                    token_buffer.clear();
                } else if (pos_lt > 0) {
                    final_answer += token_buffer.substr(0, pos_lt);
                    token_buffer = token_buffer.substr(pos_lt);
                }
                if (token_buffer.size() > 15) { // Unmatched '<' fallback
                    final_answer += token_buffer;
                    token_buffer.clear();
                }
            }
        }

        // Prepare the new token to be fed back into the loop
        loop_batch.token[0] = new_token_id;
        loop_batch.pos[0] = n_cur; // Explicitly set position to n_cur for sequential decoding

        if (llama_decode(ctx, loop_batch) != 0) {
            std::cerr << "\nError: Failed to decode generated token." << std::endl;
            break;
        }
        
        is_first_iteration = false;
        n_cur++;
    }
    
    // Free the allocated batches memory
    llama_batch_free(prompt_batch);
    llama_batch_free(loop_batch);

    // Make sure to append any residual unprinted buffer characters if we didn't end in thinking mode
    if (!in_thinking && !token_buffer.empty()) {
        final_answer += token_buffer;
    }

    return final_answer;
}
