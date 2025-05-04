#include <string>
#include <sstream>
#include <httplib.h>
#include <nlohmann/json.hpp>

//global variables
const std::string MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-r1.gguf";

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
std::vector<std::string> split_text(const std::string& text, size_t max_chars) {
    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = std::min(start + max_chars, text.size());
        size_t split = text.rfind("\n\n", end);
        size_t mid = start + (max_chars / 2);
        if (split == std::string::npos || split <= mid) {
            split = text.rfind("\n", end);
            //if good new line not found find scentence end
            if (split == std::string::npos || split <= mid) {
                size_t p1 = text.rfind(".", end);
                size_t p2 = text.rfind("!", end);
                size_t p3 = text.rfind("?", end);
                split = std::max({p1, p2, p3});

                // No good sentence end found or too early = find word end
                if (split == std::string::npos || split <= mid) {
                    split = text.rfind(" ", end);

                    // No good newline found or too early = hard split
                    if (split == std::string::npos || split <= mid) {
                        //cancels out split+1 later
                        split = end - 1;
                    }
                }
            }
        } else {
            //if new paragraph found add the 2 new lines to chunk to give better next split
            split = split + 1;
        }

        //include punctuation in split
        split = split + 1;
        std::string chunk = text.substr(start, split - start);
        chunks.push_back(chunk);
        start = split;
            }

    return chunks;
}

//takes input and forms a full prompt for the model
std::string make_prompt(const std::string& text, const std::string& command) {
        //converts to UTF8 for the ai to process it
    if (command == "summarise"){
        return "### Instruction: \n" 
                " Read the text carefully. Provide a concise summary that includes: \n"
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


OUString query_llama(httplib::Client &client, const std::string &prompt)
{
    const int retries = 3;

    client.set_connection_timeout(15);
    client.set_read_timeout(60);
    //delay removed due to libreoffice requirement
    std::string last_error;
    std::string bug;

                                        // create code that changes n_predict based on the command
                                        //n predict is the length of the output

    nlohmann::json payload = {
        {"prompt", prompt},
        {"stop", {"###"}},
        {"n_predict", 128},
        {"temperature", 0.2},
        {"top_k", 40},
        {"top_p", 0.95}
    };

    //attempts to contact the server
    for (int attempt = 0; attempt < retries; attempt++) {
        auto resp = client.Post("/completion", payload.dump(), "application/json");
        if (resp) {
            bug = resp->body;
            if (resp->status == 200) {
                try {
                    auto data = nlohmann::json::parse(resp->body);
                    if (!data.contains("content") || !data["content"].is_string()) {
                        //will change to libre accepted error message
                        //throw std::runtime_error("Missing or bad ‘content’ field");
                        continue;
                    }
                    std::string result = data["content"].get<std::string>();
                    return OUString::fromUtf8(result.c_str());
                }
                catch (const std::exception &e) {
                    last_error = std::string("JSON error: ") + e.what();
                    continue;
                }
            } else {
                //std::cerr << "Request failed with status " << resp->status << " (attempt " << attempt << "/" << retries << ")\n";
                last_error = "[HTTP Status " + std::to_string(resp->status) + "]";
            }
            bug = resp->body;
        } else {
            last_error = httplib::to_string(resp.error());
            //std::cerr << "Request error: " << last_error
            //<< " (attempt " << attempt << "/" << retries << ")\n";
        }
        //delay removed. not allowed by libreoffice
    }
    return OUString::fromUtf8(("[Error: " + last_error + "]" + bug).c_str());
}

OUString chunker(const std::string &text, httplib::Client &client , const std::string command) {
    //using estimation of 4 tokens per char
    const int MAX_CTX_BYTES = 8192;  //leaves space on server for the prompt and response

    //handles oversized text
    if (text.size() < MAX_CTX_BYTES) {
        // Build prompt and check for valid output
        std::string prompt = make_prompt(text, command);
        if (prompt == "invalid mode") {
            return OUString::fromUtf8(text.c_str()); //fallback
        }
        return query_llama(client, prompt);
    } 
    else {
        // std::cout << "Oversized text detected. Splitting...\n";
        auto subchunks = split_text(text, MAX_CTX_BYTES);
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
    httplib::Client client("127.0.0.1", 8080);
        std::string command = std::string(rCommandType.toUtf8());
    std::string text = std::string(rSelectedText.toUtf8());
    std::string result = chunker(text, client, command);
    return OUString::fromUtf8(result.c_str());
}

int main() {
    httplib::Client client("127.0.0.1", 8080);
    std::string INPUT = "Here is the single block of text you want summarised.";
    std::string summary = summarise(INPUT, client);
    std::cout << summary << std::endl;
    return 0;
}


