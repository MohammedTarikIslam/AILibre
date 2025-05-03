#include "aihelper.hxx"
#include <string>
#include <sstream>
#include <httplib.h>
#include <nlohmann/json.hpp>

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
std::vector<std::string> split_by_tokens(Llama& tokenizer, const std::string& text, size_t max_tokens) {
    //pass by reference encoding not needed
    auto tokens = tokenizer.tokenize(text);
    std::vector<std::string> chunks;
    const int NEWLINE_TOKEN_ID = 0x0A;
    
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
std::string make_prompt(const std::string& text, const std::string& command) {
    
    //converts to UTF8 for the ai to process it
    if (command == "summarise"){
        return "### Instruction: \n" 
                " Read the text carefully. Provide a concise but detailed summary that includes: \n" 
                " - Any important assertion, directive, commitment, emotion and declaration where they are applicable in bullet point format \n" 
                " - If the text is only instructions, provide simpler short instructions in bullet point format with all details included \n" 
                "### Text:\n" + text + "\n\n### Response:";
    } else if (command == "edit"){
        return "### Instruction:\n"
                "Read the text carefully and improve clarity, grammar, and style without changing the factual content.\n"
                "Preserve formatting where possible.\n"
                "Correct any spelling mistakes and grammatical errors.\n"
                "Ensure the tense is consistent throughout.\n"
                "### Text:\n" + text + "\n\n### Response:";
    } else if (command == "extend") {
        return "### Instruction: Continue writing from the following text.\n"
                "### Text:\n" + text + "\n\n### Response:\n";
    } else {
        return "invalid mode";
    }
}


OUString query_llama(httplib::Client &client, const std::string prompt)
{
    nlohmann::json payload = {
        {"prompt", prompt},
        {"stop", {"###"}},
        {"n_predict", 128},
        {"temperature", 0.2},
        {"top_k", 40},
        {"top_p", 0.95}
    };
    const int retries = 3;

    const std::string url = "http://127.0.0.1:8080/completion";
    client.set_connection_timeout(15);
    client.set_read_timeout(60);
    int delay = 1;
    std::string last_error;

    //attempts to contact the server
    for (int attempt = 0; attempt < retries; attempt++) {
        auto resp = client.Post(url, payload.dump(), "application/json");
        if (resp) {
            if (resp->status == 200) {
                auto data = nlohmann::json::parse(resp->body);
                std::string result = data["content"];
                return OUString::fromUtf8(result.c_str());                
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
    return OUString::fromUtf8(("[Error: " + last_error + "]";).c_str());

}

OUString chunker(const std::string &text, Tokenizer &tokenizer, httplib::Client &client , const std::string command) {
    const int MAX_TOKENS = 128;
    const int MAX_CTX = 8192;  //match server --ctx-size

    // Build prompt
    std::string prompt = make_prompt(text, command);
    if (prompt == "invalid mode") {
        return OUString::fromUtf8(text.c_str()); //fallback
    } 
    
    //handles oversized text
    int tokens = count_tokens(tokenizer, text);
    if (tokens < MAX_CTX - MAX_TOKENS - 256) {
        return query_llama(client, prompt);
    } 
    else {
        std::cout << "Oversized text detected. Splitting...\n";
        auto subchunks = split_by_tokens(tokenizer, text, 4096);
        std::vector<std::string> results;
        std::string merged;
        for (size_t i = 0; i < subchunks.size(); ++i) {
            std::string sub_prompt = make_prompt(subchunks[i], command);
            std::string res = query_llama(client, sub_prompt);
            results.push_back("[Part " + std::to_string(i) + "]" + res);
        }
    for (const auto &part : results) {
        merged += part + "\n";
    }

    std::string final_prompt = "### Instruction:\n"
    "Combine the set of small summaries which are parts of a large text into a single cohesive summary.\n\n"
    "### Parts:\n" + merged + "\n\n### Response:\n";

    return query_llama(client, final_prompt);

    }
}

OUString ProcessAICommand(const OUString& rSelectedText, const OUString& rCommandType){

    httplib::Client client("http://127.0.0.1:8080");
    
    std::string command = std::string(rCommandType.toUtf8());
    std::string text = std::string(rSelectedText.toUtf8());
    auto& tokenizer = get_tokeniser();

    return chunker(text, tokenizer, client, command);
}