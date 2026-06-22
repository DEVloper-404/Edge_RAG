# 🧠 Edge RAG Pipeline: Error & Resolution Guide

Welcome! This guide explains all the core technical challenges, errors, and crashes we faced while building our local C++ RAG pipeline, the solutions we implemented, and the computer science topics you should study to master these concepts.

---

## 💾 1. ONNX Runtime Memory Alignment (NaN Embeddings)
### ❌ The Error
The model either generated `NaN` (Not a Number) scores or crashed when calculating document embeddings.
### 🔍 Cause (The "Why")
ONNX Runtime requires its input tensors to be laid out in a **single contiguous block** of memory. When we passed separate variables allocated on the stack (which are stored at arbitrary positions in RAM), ONNX read junk memory, causing corruption.
### 🛠️ Solution
We stored all input values into a single `std::vector<Ort::Value>` array. This guarantees that all data elements are perfectly contiguous in RAM, ensuring ONNX reads them correctly.
### 📚 Topics & Tools to Learn
* **C++ Memory Management**: Study the difference between the **Stack** and the **Heap**, and how `std::vector` stores elements contiguously.
* **ONNX Runtime C++ API**: Learn how tensors are shaped, initialized, and loaded into inference sessions.
* **Valgrind**: A memory debugging tool that helps track memory leaks and alignment corruptions.

---

## 🧠 2. llama.cpp Stack Crashes (`munmap_chunk(): invalid pointer`)
### ❌ The Error
The program crashed with memory segmentation faults or invalid pointer errors during text generation.
### 🔍 Cause (The "Why")
We used `llama_batch_get_one`—a legacy helper function in `llama.cpp` that references variables on the stack. When we subsequently called `llama_batch_free`, the OS tried to free memory on the stack (which is illegal), causing a crash.
### 🛠️ Solution
We replaced it with manual heap allocation using `llama_batch_init` and `llama_batch_free`. We manually mapped token sequences and sequence positions (`pos[0] = n_cur`) to let the decoder know exactly where the next word should go in memory.
### 📚 Topics & Tools to Learn
* **Manual Memory Management**: Learn how C++ heap buffers work, and rules surrounding `malloc/free` or `new/delete`.
* **llama.cpp APIs**: Study how `llama_batch` constructs tokens, and how the KV cache maps token positions (`pos`) to sequences.

---

## 💬 3. LLM Internal Thinking Leakage (The `<think>` Tag)
### ❌ The Error
Qwen3 printed reasoning thoughts (like `<think> analyzing context... </think>`) inside the final answer.
### 🔍 Cause (The "Why")
Modern Reasoning LLMs are trained to "think" step-by-step using internal tags before giving the final answer. In a chat box, showing these thoughts ruins the formatting.
### 🛠️ Solution
1. **Prompt Tuning**: Instructed the system prompt to avoid outputting `<think>` tags.
2. **Streaming Character Filter**: Created a parser state-machine inside `generateAnswer` that reads tokens as they are produced, detects `<think>` and `</think>` tags, and silently discards everything in between before returning the text.
### 📚 Topics & Tools to Learn
* **State Machines / Parsers**: Learn how to design a simple state machine that shifts state depending on the text stream.
* **LLM Prompts**: Study Prompt Engineering, System Messages, and Instruct Templates.

---

## 🎨 4. UI Layout Corruption (Stdout Collisions)
### ❌ The Error
The text printed by the LLM broke the borders and corrupted the console UI layout.
### 🔍 Cause (The "Why")
We were printing output tokens to `std::cout` word-by-word at the same time FTXUI was trying to draw its bordered layout.
### 🛠️ Solution
Changed the return type of `LlmEngine::generateAnswer` to `std::string` so that the LLM silently builds the complete answer and returns it, allowing FTXUI to render it cleanly at once.
### 📚 Topics & Tools to Learn
* **Model-View-Controller (MVC)**: Study the architecture pattern of decoupling logic/data generation (LLM) from visual display (TUI).
* **Terminal User Interfaces (TUI)**: Learn how consoles draw frames using ANSI escape codes.

---

## 🔄 5. Prompt Decoding Failure on Repeated Queries
### ❌ The Error
On the second query, the UI printed `Error: Failed to decode prompt`.
### 🔍 Cause (The "Why")
When we ask a second question, the model tries to load the new prompt into the KV (Key-Value) cache context. Because we never cleared the cache from the first question, the context overflowed the 2048 token limit.
### 🛠️ Solution
Called `llama_memory_clear` at the start of each `generateAnswer` call. This completely flushes the model's memory buffer, preparing it fresh for the next query.
### 📚 Topics & Tools to Learn
* **KV Cache Mechanics**: Read how Transformers store past keys and values to save computing power, and why context limits exist.

---

## ⚡ 6. UI Freeze/Hanging during Computation
### ❌ The Error
The UI froze, became unresponsive, and couldn't show any loading status or animation while searching and generating answers.
### 🔍 Cause (The "Why")
All calculations (embedding, search, and LLM text generation) were running on the main UI thread, stopping the interface from updating.
### 🛠️ Solution
Moved the calculations to a background worker thread (`std::thread`). We added a timer thread that updates a spinner animation (`⠋`, `⠙`, `⠹`) and wakes up the UI loop using thread-safe events (`screen.PostEvent`), while posting results via `screen.Post`.
### 📚 Topics & Tools to Learn
* **Multithreading**: Study threads (`std::thread`), thread safety, and race conditions.
* **Atomic Variables (`std::atomic`)**: Learn how threads communicate safely without locking.
* **Event Loop Architecture**: Learn how event-driven frameworks (like FTXUI, Node.js, or Android) process UI tasks.

---

## 🚪 7. Terminal Corruption/Freeze on ESC Exit
### ❌ The Error
Pressing the `ESC` key to exit the RAG interface resulted in the terminal freezing, hiding the cursor, and not echoing keyboard inputs.
### 🔍 Cause (The "Why")
We used `std::exit(0)` inside the `CatchEvent` event handler. While this stopped the process immediately, it bypassed C++ object destructors. Specifically, it prevented the `ScreenInteractive` destructor from running, which sends ANSI escape sequences to restore normal terminal mode (re-enabling echo, showing the cursor, and releasing raw input mode).
### 🛠️ Solution
We replaced `std::exit(0)` with `screen.ExitLoopClosure()()` to cleanly break the FTXUI rendering loop. We then handled background thread cleanup (using `.join()` on worker and spinner threads after the loop exits) so that `ChatUI::start` returns normally to `main()`, letting the terminal restore itself cleanly.
### 📚 Topics & Tools to Learn
* **Destructors & RAII**: Learn how Resource Acquisition Is Initialization (RAII) ensures clean resource release in C++.
* **ANSI Escape Sequences**: Study how terminal programs communicate with console buffers to control formatting, cursor position, and raw mode.

---

## 📄 8. Truncated Output (Only Showing 1 Line)
### ❌ The Error
Multi-line answers generated by Qwen3 (containing newlines `\n`) only rendered on a single line, causing paragraphs to be cut off.
### 🔍 Cause (The "Why")
FTXUI's `text(...)` element is designed to display string data as a single continuous line. It does not automatically split text on `\n` characters or wrap word boundaries for narrow screens.
### 🛠️ Solution
We updated the chat history renderer to:
1. Split each message string into separate lines using `std::stringstream` and `std::getline(ss, line)`.
2. Wrap each line inside the FTXUI `paragraph(...)` element instead of `text(...)`, which automatically calculates and wraps lines according to the current width of the terminal.
### 📚 Topics & Tools to Learn
* **String Parsing in C++**: Learn how to use string streams (`std::stringstream`) and standard IO helpers to split string data.
* **Layout Constraints**: Study how word wrapping, alignment, and container sizes are processed in UI frameworks.
