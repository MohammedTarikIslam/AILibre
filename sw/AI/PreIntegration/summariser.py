import asyncio
import aiohttp
import orjson
from pathlib import Path

#Global variables
MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-r1.gguf"
MAX_TOKENS = 256
MAX_CTX = 8192  #match server --ctx-size

#takes input and forms a full prompt for the model
def make_summary_prompt(text):
    return f"""### Instruction:
Read the text carefully. Provide a concise but detailed summary that includes:
- Explanation of each assertion, directive, commitment, emotion and declaration where applicable  
- Any important tasks or information in bullet point format

### Text:
{text}

### Response:
"""

#server query function
async def query_llama(session, prompt, max_tokens):
    #default server location
    url = "http://127.0.0.1:8080/completion" 
    payload = {
        "prompt": prompt,
        "stop": ["###"],
        "n_predict": max_tokens,
        "temperature": 0.2,
        "top_k": 40,
        "top_p": 0.95,
    }
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        return data["content"].strip()
    
#Main function 
def summarise(input_text):
    async def short_summary():
        async with aiohttp.ClientSession() as session:
            prompt = make_summary_prompt(input_text)
            summary = await query_llama(session, prompt, MAX_TOKENS)
            print("Summary:\n", summary)
    asyncio.run(short_summary())


if __name__ == "__main__":
    highlighted= "lorem"
    summarise(highlighted)