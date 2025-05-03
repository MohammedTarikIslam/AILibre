#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <algorithm>
#include <httplib.h>
#include <nlohmann/json.hpp>

//Global variables
const std::string MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-r1.gguf";
const int MAX_TOKENS = 128;
const int MAX_CTX = 8192;  //match server --ctx-size
const int NUM_PROCESSES = 5;

int mode = 1;  // 1 = summariser, 2 = editing

//input text
const std::vector<std::string> highlighted =  ["Put 100g plain flour, 2 large eggs, 300ml milk, 1 tbsp sunflower or vegetable oil and a pinch of salt into a bowl or large jug, then whisk to a smooth batter. This should be similar in consistency to single cream.", 
    "Set aside for 30 mins to rest if you have time, or start cooking straight away.", 
    "Set a medium frying pan or crêpe pan over a medium heat and carefully wipe it with some oiled kitchen paper.",
    "When hot, cook your pancakes for 1 min on each side until golden, using around half a ladleful of batter per pancake. Keep them warm in a low oven as you make the rest.",
    "Serve with lemon wedges and caster sugar, or your favourite filling. Once cold, you can layer the pancakes between baking parchment, then wrap in cling film and freeze for up to two months."]
    
//tokeniser logic
static std::unique_ptr<Llama> _tokeniser = nullptr;
Llama& get_tokeniser() {
    if (!_tokeniser) {
        _tokeniser = std::make_unique<Llama>(MODEL_PATH, MAX_CTX, /*vocab_only=*/true);
    }
    return *_tokeniser;
}

int count_tokens(Llama& tokeniser, const std::string& text) {
    std::vector<int> tokens = tokeniser.tokenize(text);
    return static_cast<int>(tokens.size());
}
    
//splits text into chunks when too large
//future implementation should split by paragraph or pages
std::vector<std::string> split_by_tokens(Llama& tokenizer, const std::string& text, size_t max_tokens) {
        //pass by reference encoding not needed
        auto tokens = tokenizer.tokenize(text);
        std::vector<std::string> chunks;
        const int NEWLINE_TOKEN_ID = 13;
        
        size_t token_count = count_tokens(tokenizer, text);
        
        if (token_count <= static_cast<int>(max_tokens)) {
            chunks.push_back(trim(tokenizer.detokenize(tokens)));
            return chunks;
        } 
        
        for (size_t start = 0; start < tokens.size(); ) {
            size_t end = std::min(start + max_tokens, token_count);
            
            size_t split = end;
            for (size_t i = end; i > start; --i) {
                if (tokens[i - 1] == NEWLINE_TOKEN_ID) {
                    if ((i - start) > 300) 
                    { 
                        split = i;
                        break;
                    }
                }
        }

        //takes the chunk and detokenises it
        std::vector<int> chunk_tokens(tokens.begin() + start, tokens.begin() + split);
        std::string chunk_text = tokenizer.detokenize(chunk_tokens);


        chunks.push_back(chunk_text);

        start = split;
    }

    return chunks;
}

//takes input and forms a full prompt for the model
std::string make_prompt(const std::string& text, const int mode) {
    if (mode ==1){
        return f"### Instruction: \n" 
                " Read the text carefully. Provide a concise but detailed summary that includes: \n" 
                " - Any important assertion, directive, commitment, emotion and declaration where they are applicable in bullet point format \n" 
                " - If the text is only instructions, provide simpler short instructions in bullet point format with all details included \n" 
                "### Text:\n" + text + "\n\n### Response:";
    }
    
    else if (mode == 2){
        return "### Instruction:\n"
                "Read the text carefully and improve clarity, grammar, and style without changing the factual content.\n"
                "Preserve formatting where possible.\n"
                "Correct any spelling mistakes and grammatical errors.\n"
                "Ensure the tense is consistent throughout.\n"
                "### Text:\n" + text + "\n\n### Response:";
    } else {
        return "invalid mode";
    }
}


//server query function
std::string query_llama(httplib::Client &client, const std::string prompt, int max_tokens, int retries=3){
    std::cout << "starting summariser '" << prompt.substr(0, 40) << "'\n";

    //default server location
    const std::string url = "http://127.0.0.1:8080/completion";
    client.set_connection_timeout(15);
    client.set_read_timeout(60);
    int delay = 1;
    std::string last_error;

    nlohmann::json payload = {
        {"prompt", prompt},
        {"stop", {"###"}},
        {"n_predict", max_tokens},
        {"temperature", 0.2},
        {"top_k", 40},
        {"top_p", 0.95}
    };

    //attempts to contact the server
    for (int attempt = 0; attempt < retries; attempt++) {
        auto resp = client.Post(url, payload.dump(), "application/json");
        if (resp) {
            if (resp->status == 200) {
                auto data = nlohmann::json::parse(resp->body);
                return data["content"].get<std::string>();
            } else {
                std::cerr << "Request failed with status " << resp->status
                            << " (attempt " << attempt << "/" << retries << ")\n";
                last_error = "[HTTP Status " + std::to_string(resp->status) + "]";
            }
        } else {
            last_error = httplib::to_string(resp.error());
            std::cerr << "Request error: " << last_error
            << " (attempt " << attempt << "/" << retries << ")\n";
        }
        if (attempt < retries) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }    
    }
    return "[Error: " + last_error + "]";
}


// Main function
std::string summarise(const std::string &text, Tokenizer &tokenizer, httplib::Client &client ) {
    //handles oversized text
    std::string prompt = make_prompt(text, mode);
    int tokens = count_tokens(tokenizer, prompt);

    if (tokens < MAX_CTX - MAX_TOKENS - 256) {
        return query_llama(client, prompt, MAX_TOKENS);
    } 
    else {
        std::cout << "Oversized text detected. Splitting...\n";
        auto subchunks = split_by_tokens(tokenizer, text, 4096);
        std::vector<std::string> results;
        std::string merged;
        for (size_t i = 0; i < subchunks.size(); ++i) {
            std::string sub_prompt = make_prompt(subchunks[i], 2);
            std::string res = query_llama(client, sub_prompt, MAX_TOKENS);
            results.push_back("[Part " + std::to_string(i) + "]" + res);
        }
    for (const auto &part : results) {
        merged += part + "\n";
    }
    std::string final_prompt = "### Instruction:\n"
    "Summarize the following parts of a large text as a single cohesive summary.\n\n"
    "### Parts:\n" + merged + "\n\n### Response:\n";
    return query_llama(client, final_prompt, MAX_TOKENS);
    }
}

std::unordered_map<std::string, std::string> summarise_all(httplib::Client &client, const std::vector<std::string> &highlighted) {
    Llama& tokeniser = get_tokeniser();
    //changed into a map which can help add features later 
    std::unordered_map<std::string, std::string> summaries;
    //progress bar not a c++ feature
    for (size_t i = 0; i < highlighted.size(); ++i) {
        std::string summary = summarise(highlighted[i], tokenizer, client);
        summaries["chunk_" + std::to_string(i)] = summary;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return summaries;
}

int main() {
    std::cout << "Script started" << std::endl;
    httplib::Client client("http://127.0.0.1:8080");
    auto summaries = summarise_all(client, highlighted);

    for (const auto& pair : summaries) {
        std::cout << pair.first << ": " << pair.second << "\n";
    }
    return 0;
}