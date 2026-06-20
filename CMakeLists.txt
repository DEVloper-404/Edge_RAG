cmake_minimum_required(VERSION 3.24)
project(my_edge_rag LANGUAGES CXX)

# Force C++20 standard for modern features and compatibility
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# ⚡ COMPILER FLAGS (PENTIUM OPTIMIZED) ⚡
# ==========================================
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -O3             # Aggressive optimizations for maximum speed
        -march=native   # Optimizes for your specific Pentium instruction set (SSE)
        -ffast-math     # Drastically speeds up vector math for embeddings
	-fno-finite-math-only # 👈 ADD THIS LINE TO FIX THE LLAMA.CPP ERROR
	-Wall           # Show all standard warnings
        -Wextra         # Show extra warnings
    )
endif()

# ==========================================
# 🎨 DEPENDENCY 1: FTXUI (Terminal UI)
# ==========================================
# Downloads and compiles FTXUI directly into the build folder (no global install)
include(FetchContent)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG v5.0.0 
)
FetchContent_MakeAvailable(ftxui)

# ==========================================
# 🧠 DEPENDENCY 2: llama.cpp (LLM Engine)
# ==========================================
# Protects your 4GB RAM by NOT compiling heavy tests/examples
set(LLAMA_BUILD_SERVER OFF CACHE INTERNAL "Disable server build to save RAM")
set(LLAMA_BUILD_EXAMPLES OFF CACHE INTERNAL "Disable examples build to save RAM")
set(LLAMA_BUILD_TESTS OFF CACHE INTERNAL "Disable tests build to save RAM")

# Link the local third_party folder you cloned
add_subdirectory(third_party/llama.cpp)

# ==========================================
# 🗺️ DEPENDENCY 3: ONNX Runtime (Embeddings)
# ==========================================
# Point CMake to your downloaded local ONNX runtime folder
set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT_DIR}/include)

# Locate the ONNX shared library file (.so)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
    NO_DEFAULT_PATH # Forces it to use your local folder, not system libraries
    REQUIRED
)

# ==========================================
# 🚀 TARGET: Main Executable
# ==========================================
# Add your source files here. (If you only have main.cpp right now, remove the others)
add_executable(${PROJECT_NAME}
    src/main.cpp
    # src/ui.cpp         # Uncomment when you create these files
    # src/rag_engine.cpp # Uncomment when you create these files
    # src/inference.cpp  # Uncomment when you create these files
)

# Tell the compiler where your header files are
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link all the libraries together into your final program
target_link_libraries(${PROJECT_NAME} PRIVATE
    ftxui::screen
    ftxui::dom
    ftxui::component
    llama          # Core LLM engine
    ${ONNXRUNTIME_LIB}
)

# ==========================================
# 📦 POST-BUILD COPY (PREVENTS LINUX CRASHES)
# ==========================================
# Automatically copies libonnxruntime.so next to your program so Ubuntu finds it instantly
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)
