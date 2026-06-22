# 🧠 Local Edge RAG Pipeline (Terminal UI)

A lightweight, high-performance, and completely local Retrieval-Augmented Generation (RAG) pipeline designed for resource-constrained systems (like Intel Pentium or systems with 4GB RAM). It features an interactive Terminal User Interface (TUI) powered by **FTXUI**, local embeddings from **all-MiniLM-L6-v2 (ONNX)**, and text generation via **Qwen3 (GGUF)** running on a CPU-optimized **llama.cpp** engine.

---

## ✨ Features

- **🚀 100% Local & Private**: No API keys, no internet connection required after initial model download, and zero data leakage.
- **🎨 Interactive Terminal UI (TUI)**: Beautiful, responsive console interface with a loading spinner, scrolling conversation history, and real-time word-wrapped outputs.
- **💾 Low-Memory Optimization**: 
  - Model weights read directly from disk via `mmap` to keep RAM usage under 1.5GB.
  - Multi-threaded rendering using a dedicated background thread for calculations, keeping the UI responsive.
  - Periodic KV cache management to prevent memory overflows.
- **⚡ Hybrid Architecture**: Combines C++ efficiency with lightweight Python helper scripts for automated model management.

---

## 🛠️ Tech Stack & Dependencies

The project is built on the following technologies:
1. **Language**: C++20 (using compiler optimizations like `-O3` and `-march=native`).
2. **Build System**: CMake 3.24+.
3. **UI Engine**: Arthur Sonzogni's [FTXUI](https://github.com/ArthurSonzogni/FTXUI) (automatically fetched and built).
4. **LLM Inference**: [llama.cpp](https://github.com/ggerganov/llama.cpp) (statically compiled inside `third_party/llama.cpp`).
5. **Embedding Inference**: [ONNX Runtime](https://onnxruntime.ai/) (local C++ shared library inside `third_party/onnxruntime`).
6. **Model Downloader**: Python 3 + `huggingface_hub` for simple, automated dependency retrieval.

---

## ⚙️ Setup & Installation

Follow these steps to build and run the RAG pipeline on your machine:

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd my_edge_rag
```

### 2. Download the Models (Python)
Since the model binary files are large (~1.2GB total) and ignored by Git, use the included Python helper script to download them directly from the Hugging Face Hub:

```bash
# 1. Install the download dependencies
pip install -r requirements.txt

# 2. Run the model setup script
python download_models.py
```
This script will automatically download and place:
* `all-minilm-l6-v2.onnx` into `models/`
* `Qwen3-1.7B-Q4_K_M.gguf` into `models/`

### 3. Add Source Documents
Place the Markdown (.md) documents you want the pipeline to search and learn from into the `Source files/` directory.

### 4. Build the C++ Application
Compile the executable using CMake:

```bash
# Create build directory
mkdir -p build && cd build

# Configure and generate build files
cmake ..

# Compile the project
make -j$(nproc)
```

---

## 🚀 Running the RAG Pipeline

Once built, you can run the pipeline directly:

```bash
./my_edge_rag
```

### 🎮 Interface Controls
* **Type your message** in the input field at the bottom.
* **Press Enter** to send your query and retrieve relevant context.
* **Press Esc** to clean up threads and exit the program safely.

---

## 📁 Repository Structure

```
my_edge_rag/
├── CMakeLists.txt              # CMake build configuration
├── requirements.txt            # Python dependencies for model setup
├── download_models.py          # Python downloader script for GGUF & ONNX models
├── error&resolve.md            # Troubleshooting guide & computer science concepts
├── Source files/               # Directory containing source markdown files to index
├── models/                     # Directory where model weights are stored (Git ignored)
├── src/                        # C++ Source Code
│   ├── main.cpp                # App entry point
│   ├── ui.cpp / ui.hpp         # FTXUI Terminal UI Loop & Threads
│   ├── llm_engine.cpp / .hpp   # llama.cpp Wrapper
│   ├── inference.cpp / .hpp    # ONNX Runtime Embeddings Wrapper
│   └── rag_engine.cpp / .hpp   # Chunking, Vector Store, & Cosine Similarity
└── third_party/                # Vendor-managed local dependencies (llama.cpp, ONNX Runtime)
```

---

## 📦 How to Deploy/Push to GitHub

If you want to push this repository to your own GitHub profile, run the following commands:

```bash
# 1. Stage the files (including forcing the ONNX shared library)
git add .
git add -f third_party/onnxruntime/lib/libonnxruntime.so
git add -f third_party/onnxruntime/lib/libonnxruntime.so.1.17.1

# 2. Commit the changes
git commit -m "feat: Add interactive FTXUI terminal interface, requirements.txt, and download_models.py"

# 3. Rename the branch to main (if not done already)
git branch -M main

# 4. Link your remote GitHub repository
git remote add origin <YOUR_GITHUB_REPO_URL>

# 5. Push to GitHub
git push -u origin main
```

---

## ⚠️ Troubleshooting

For details on memory alignment issues, threading freezes, terminal layout corruptions, and `<think>` tag leakage, refer to the [error&resolve.md](file:///home/db/Desktop/my_edge_rag/error&resolve.md) guide.
