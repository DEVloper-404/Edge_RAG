#include "ui.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <thread>
#include <chrono>
#include <atomic>
#include <memory>
#include <sstream>

using namespace ftxui;

void ChatUI::start(EmbeddingEngine& embedder, std::vector<DocumentChunk>& database, LlmEngine& llm) {
    // This takes over your terminal window
    auto screen = ScreenInteractive::TerminalOutput();
    
    std::string input_text;
    auto chat_history = std::make_shared<std::vector<std::string>>();

    // Loading states for background processing
    auto is_loading = std::make_shared<std::atomic<bool>>(false);
    auto spinner_index = std::make_shared<std::atomic<int>>(0);
    const std::vector<std::string> spinner_chars = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};

    // ASCII Art Header
    const std::vector<std::string> ascii_header = {
        "  ███████╗██████╗  ██████╗ ███████╗    ██████╗  █████╗  ██████╗ ",
        "  ██╔════╝██╔══██╗██╔════╝ ██╔════╝    ██╔══██╗██╔══██╗██╔════╝ ",
        "  █████╗  ██║  ██║██║  ███╗█████╗      ██████╔╝███████║██║  ███╗",
        "  ██╔══╝  ██║  ██║██║   ██║██╔══╝      ██╔══██╗██╔══██║██║   ██║",
        "  ███████╗██████╔╝╚██████╔╝███████╗    ██║  ██║██║  ██║╚██████╔╝",
        "  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ "
    };

    // Thread handles to join on exit
    std::thread worker_thread;
    std::thread spinner_thread;

    // What happens when the user presses Enter?
    InputOption option;
    option.transform = [](InputState state) {
        if (state.is_placeholder) {
            state.element |= color(Color::GrayDark);
        } else {
            state.element |= color(Color::White);
        }
        if (state.focused) {
            state.element |= color(Color::Cyan) | bold;
        }
        return state.element;
    };

    option.on_enter = [&] {
        if (input_text.empty() || *is_loading) return;
        
        // Join previous threads if they are finished and joinable
        if (worker_thread.joinable()) worker_thread.join();
        if (spinner_thread.joinable()) spinner_thread.join();

        // 1. Add user question to screen and clear the box
        std::string query = input_text;
        chat_history->push_back("👤 You: " + query);
        input_text.clear();
        
        chat_history->push_back("🔍 [Searching & Generating answer...]");
        size_t loading_msg_index = chat_history->size() - 1;
        
        *is_loading = true;

        // Start spinner animation thread
        spinner_thread = std::thread([&screen, is_loading, spinner_index]() {
            while (*is_loading) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                (*spinner_index)++;
                screen.PostEvent(Event::Custom);
            }
        });

        // Start background worker thread for embedding + search + LLM
        worker_thread = std::thread([&embedder, &database, &llm, &screen, chat_history, is_loading, query, loading_msg_index]() {
            auto queryVector = embedder.getEmbedding(query);
            auto matches = RagEngine::searchTopK(queryVector, database, 1);
            
            std::string search_header = "";
            std::string answer = "";
            
            if (matches.empty()) {
                search_header = "🤖 Qwen: I couldn't find any relevant context.";
            } else {
                search_header = "🔍 [Searching in: " + matches[0].contextHeader + "]";
                answer = llm.generateAnswer(matches[0].text, query);
            }
            
            // Post update back to the UI thread
            screen.Post([chat_history, is_loading, search_header, answer, loading_msg_index]() {
                if (!answer.empty()) {
                    (*chat_history)[loading_msg_index] = search_header;
                    chat_history->push_back("🤖 Qwen: " + answer);
                } else {
                    (*chat_history)[loading_msg_index] = search_header;
                }
                *is_loading = false;
            });
            screen.PostEvent(Event::Custom);
        });
    };

    // The actual text input component
    Component input = Input(&input_text, "Ask a question... (Press Enter to send, ESC to quit)", option);
    
    // The renderer draws the layout every time the screen updates
    auto renderer = Renderer(input, [&] {
        Elements history_elements;
        for (const auto& message : *chat_history) {
            std::stringstream ss(message);
            std::string line;
            while (std::getline(ss, line)) {
                history_elements.push_back(paragraph(line));
            }
        }
        
        // Show spinner/status if loading
        Element status_bar;
        if (*is_loading) {
            status_bar = hbox({
                text(" ⚙️  Thinking " + spinner_chars[*spinner_index % spinner_chars.size()]) | bold | color(Color::Yellow),
                text(" (Qwen3 is processing locally)...") | dim
            });
        } else {
            status_bar = text(" Ready ") | bold | color(Color::Green);
        }

        Elements header_elements;
        for (const auto& line : ascii_header) {
            header_elements.push_back(text(line) | color(Color::Cyan) | bold | center);
        }
        header_elements.push_back(text("─── LOCAL EDGE RAG TUI ───") | color(Color::BlueLight) | center);
        
        return vbox({
            vbox(header_elements),
            separator(),
            vbox(history_elements) | flex | frame, // Chat history (flex means it grows to fill space)
            separator(),
            status_bar,
            separator(),
            hbox({
                text(" ➔ ") | color(Color::Cyan) | bold,
                input->Render() | flex
            })   // Input row at the bottom with transparent styling
        }) | border;
    });

    // Capture the Escape key to quit the program safely
    auto final_component = CatchEvent(renderer, [&](Event event) {
        if (event == Event::Escape) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    // Start the endless loop
    screen.Loop(final_component);

    // Clean shutdown: Signal threads to stop and join them to prevent crashes on stack unwind
    *is_loading = false;
    if (worker_thread.joinable()) worker_thread.join();
    if (spinner_thread.joinable()) spinner_thread.join();
}
