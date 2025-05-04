#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <algorithm>
#include "httplib.h"
#include <nlohmann/json.hpp>

//Global variables
const int MAX_TOKENS = 128;
const int MAX_CHARS = 2048; //match server --ctx-size
int mode = 1; // 1 = summariser, 2 = editing

//input text
const std::vector<std::string> highlightedbackup
    = { "Put 100g plain flour, 2 large eggs, 300ml milk, 1 tbsp sunflower or vegetable oil and a "
        "pinch of salt into a bowl or large jug, then whisk to a smooth batter.",
        "This should be similar in consistency to single cream.",
        "Set aside for 30 mins to rest if you have time, or start cooking straight away.",
        "Set a medium frying pan or crêpe pan over a medium heat and carefully wipe it with some "
        "oiled kitchen paper.",
        "When hot, cook your pancakes for 1 min on each side until golden, using around half a "
        "ladleful of batter per pancake.",
        " Keep them warm in a low oven as you make the rest.",
        "Serve with lemon wedges nd caster sugar, or your favourite filling. Once cold, you can "
        "layer the pancakes between baking parchment, then wrap in cling film and freeze for up to "
        "two months." };

std::string highlighted
    = "Put 100g plain flour, 2 large eggs, 300ml milk, 1 tbsp sunflower or vegetable oil and a "
      "pinch of salt into a bowl or large jug, then whisk to a smooth batter. This should be "
      "similar in consistency to single cream. Set aside for 30 mins to rest if you have time, or "
      "start cooking straight away. Set a medium frying pan or crêpe pan over a medium heat and "
      "carefully wipe it with some oiled kitchen paper.When hot, cook your pancakes for 1 min on "
      "each side until golden, using around half a ladleful of batter per pancake. Keep them warm "
      "in a low oven as you make the rest. Serve with lemon wedges nd caster sugar, or your "
      "favourite filling. Once cold, you can layer the pancakes between baking parchment, then "
      "wrap in cling film and freeze for up to two months.";

//splits text into chunks when too large
//future implementation should split by paragraph or pages
std::vector<std::string> split_text(const std::string& text, size_t max_chars)
{
    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < text.size())
    {
        size_t end = std::min(start + max_chars, text.size());
        size_t split = text.rfind("\n\n", end);
        size_t mid = start + (max_chars / 2);
        if (split == std::string::npos || split <= mid)
        {
            split = text.rfind("\n", end);
            //if good new line not found find scentence end
            if (split == std::string::npos || split <= mid)
            {
                size_t p1 = text.rfind(".", end);
                size_t p2 = text.rfind("!", end);
                size_t p3 = text.rfind("?", end);
                split = std::max({ p1, p2, p3 });
                // No good sentence end found or too early = find word end
                if (split == std::string::npos || split <= mid)
                {
                    split = text.rfind(" ", end);
                    // No good newline found or too early = hard split
                    if (split == std::string::npos || split <= mid)
                    {
                        //cancels out split+1 later
                        split = end - 1;
                    }
                }
            }
        }
        else
        {
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
std::string make_prompt(const std::string& text, const int mode)
{
    // 1 = summariser, 2 = editing
    if (mode == 1)
    {
        return "### Instruction:\n",
               " Read the text carefully. Provide a concise but detailed summary that includes: \n",
               " - Any important assertion, directive, commitment, emotion and declaration where "
               "they are applicable in bullet point format \n",
               " - If the text is only instructions, provide simpler short instructions in bullet "
               "point format with all details included \n",
               "### Text:\n" + text + "\n\n### Response:";
    }
    else if (mode == 2)
    {
        return "### Instruction:\n",
               "Read the text carefully and improve clarity, grammar, and style without changing "
               "the factual content.\n",
               "Preserve formatting where possible.\n",
               "Correct any spelling mistakes and grammatical errors.\n",
               "Ensure the tense is consistent throughout.\n",
               "### Text:\n" + text + "\n\n### Response:";
    }
    else
    {
        return "invalid mode";
    }
}

//server query function
std::string query_llama(httplib::Client& client, const std::string& prompt, int max_tokens)
{
    std::cout << "starting summariser '" << prompt << "'\n";
    int retries = 3;
    //default server location
    client.set_connection_timeout(15);
    client.set_read_timeout(60);
    //int delay = 1;
    std::string last_error;
    nlohmann::json payload
        = { { "prompt", prompt },   { "stop", { "###" } }, { "n_predict", max_tokens },
            { "temperature", 0.2 }, { "top_k", 40 },       { "top_p", 0.95 } };

    //attempts to contact the server
    for (int attempt = 0; attempt < retries; attempt++)
    {
        auto resp = client.Post("/completion", payload.dump(), "application/json");
        if (resp)
        {
            if (resp->status == 200)
            {
                auto data = nlohmann::json::parse(resp->body);
                std::cerr << "Successfully received model response. \n";
                return data["content"].get<std::string>();
            }
            else
            {
                std::cerr << "Request failed with status " << resp->status << " (attempt "
                          << attempt << "/" << retries << ")\n";
                last_error = "[HTTP Status " + std::to_string(resp->status) + "]";
            }
        }
        else
        {
            last_error = httplib::to_string(resp.error());
            std::cerr << "Request error: " << last_error << " (attempt " << attempt << "/"
                      << retries << ")\n";
        }
        if (attempt < retries)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    return "[Error: " + last_error + "]";
}

// Main function
std::string summarise(const std::string& text, httplib::Client& client)
{
    //handles oversized text
    std::string prompt = make_prompt(text, mode);
    if (prompt.size() < MAX_CHARS - (MAX_TOKENS * 4) - 256)
    {
        return query_llama(client, prompt, MAX_TOKENS);
    }
    else
    {
        std::cout << "Oversized text detected. Splitting...\n";
        auto subchunks = split_text(text, MAX_CHARS);
        std::vector<std::string> results;
        std::string merged;
        for (size_t i = 0; i < subchunks.size(); ++i)
        {
            std::string sub_prompt = make_prompt(subchunks[i], 2);
            std::string res = query_llama(client, sub_prompt, MAX_TOKENS);
            results.push_back("[Part " + std::to_string(i) + "]" + res);
        }
        for (const auto& part : results)
        {
            merged += part + "\n";
        }
        std::string final_prompt
            = "### Instruction:\n"
              "Summarize the following parts of a large text as a single cohesive summary.\n\n"
              "### Parts:\n"
              + merged + "\n\n### Response:\n";
        return query_llama(client, final_prompt, MAX_TOKENS);
    }
}

std::map<std::string, std::string> summarise_all(httplib::Client& client,
                                                 const std::vector<std::string>& highlighted)
{
    //changed into a map which can help add features later
    std::map<std::string, std::string> summaries;
    //progress bar not a c++ feature
    for (size_t i = 0; i < highlighted.size(); ++i)
    {
        std::string summary = summarise(highlighted[i], client);
        summaries["chunk_" + std::to_string(i)] = summary;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return summaries;
}

int main()
{
    std::cout << "Script started" << std::endl;
    httplib::Client client("127.0.0.1", 8080);
    auto summaries = summarise_all(client, highlightedbackup);
    for (const auto& pair : summaries)
    {
        std::cout << pair.first << ": " << pair.second << "\n";
    }
    return 0;
}