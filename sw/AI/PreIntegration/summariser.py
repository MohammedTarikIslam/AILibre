import asyncio
import logging
import random
import time
from tqdm import tqdm
import json
import aiohttp
import orjson
from pathlib import Path
from llama_cpp import Llama


#configures logging. output only to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

#Global variables
MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-r1.gguf"
FAILED_LOG = Path("failed.log")
MAX_TOKENS = 256
MAX_CTX = 8192  #match server --ctx-size

#tokeniser logic
_tokeniser = None
def get_tokeniser():
    global _tokeniser
    if _tokeniser is None:
        _tokeniser = Llama(model_path=MODEL_PATH, n_ctx=MAX_CTX, vocab_only=True)
    return _tokeniser

def count_tokens(tokeniser, text):
    return len(tokeniser.tokenize(text.encode("utf-8")))

#splits text into chunks when too large
#future implementation should split by paragraph or pages
def split_by_tokens(tokeniser, text, max_tokens):
    tokens = tokeniser.tokenize(text.encode("utf-8"), add_bos=False)
    chunks = []
    for start in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[start:start + max_tokens]
        chunk_text = tokeniser.detokenize(chunk_tokens).decode("utf-8", errors="ignore")
        chunks.append(chunk_text.strip())
    return chunks

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
async def query_llama(session, prompt, max_tokens, retries=3 ):
    #default server location
    url = "http://127.0.0.1:8080/completion" 
    timeout=aiohttp.ClientTimeout(total=60)
    delay = 1    
    payload = {
        "prompt": prompt,
        "stop": ["###"],
        "n_predict": max_tokens,
        "temperature": 0.2,
        "top_k": 40,
        "top_p": 0.95,
    }
    #attempts to contact the server
    for attempt in range(retries):
        try: 
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.error(f"Request failed with status {resp.status}")
                    return f"[Error {resp.status}]"
                else:
                    print(f"Connection successful. Response received for prompt starting with: {prompt[:40]!r}")
                    data = await resp.json()
                    logger.info("Successfully received model response.") 
                    return data["content"].strip()
        except Exception as e:
            logger.warning(f"An error occurred during query: {e}. retry {attempt+1} of {retries}")
            await asyncio.sleep(delay + random.uniform(0, 0.1))            
            return f"[Error: {e}]"
        
    logger.error(f"Failed after {retries} retries for prompt start: {prompt[:40]!r}")
    FAILED_LOG.write_text(f"{prompt[:60]}\nError: {e}\n", encoding="utf-8")
    return "[Error]"
    
#Main function
async def summarise(text):
    tokeniser = get_tokeniser()
    prompt = make_summary_prompt(text)
    tokens = count_tokens(tokeniser, prompt)
    
    async with aiohttp.ClientSession() as session:
        #handles oversied text
        if tokens < MAX_CTX - MAX_TOKENS - 256:
            return await query_llama(session, prompt, MAX_TOKENS)
        else:
            logger.info("Oversized text detected. Splitting...")
            subchunks = split_by_tokens(tokeniser, text, 4096)
            results = []
            for i, sub in enumerate(subchunks):
                sub_prompt = make_summary_prompt(sub)
                res = await query_llama(session, sub_prompt, MAX_TOKENS)
                results.append(f"[Part {i}]{res}")
            merged = "\n".join(results)
            final_prompt = f"""### Instruction:
Summarize the following parts of a large text as a single cohesive summary.

### Parts:
{merged}

### Response:
"""
            return await query_llama(session, final_prompt, MAX_TOKENS)

#summarises all text in the list
async def summarise_all(highlighted):
    #adds progress bar
    pbar = tqdm(total=len(highlighted), desc="Summarizing")
    for i, text in enumerate(highlighted):
        summary = await summarise(text)
        logger.info(f"Summary {i+1}:{summary}\n")
        time.sleep(0.1)
        pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    highlighted= ["lorem", "ipsum", "dolor", "sit", "amet"]
    print("Script started", flush=True)
    asyncio.run(summarise_all(highlighted))