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

#Global variables
MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-r1.gguf"
MAX_TOKENS = 256
MAX_CTX = 8192  #match server --ctx-size
NUM_PROCESSES = 5

#configures logging
FAILED_LOG = Path("failed.log")
file_handler = logging.FileHandler(FAILED_LOG, encoding="utf-8")
file_handler.setLevel(logging.ERROR)

#configures logging. output only to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        file_handler
    ]
)
logger = logging.getLogger(__name__)

#input text
highlighted= ["Put 100g plain flour, 2 large eggs, 300ml milk, 1 tbsp sunflower or vegetable oil and a pinch of salt into a bowl or large jug, then whisk to a smooth batter. This should be similar in consistency to single cream.", 
              "Set aside for 30 mins to rest if you have time, or start cooking straight away.", 
              "Set a medium frying pan or crêpe pan over a medium heat and carefully wipe it with some oiled kitchen paper.",
              "When hot, cook your pancakes for 1 min on each side until golden, using around half a ladleful of batter per pancake. Keep them warm in a low oven as you make the rest.",
              "Serve with lemon wedges and caster sugar, or your favourite filling. Once cold, you can layer the pancakes between baking parchment, then wrap in cling film and freeze for up to two months."]

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
    print(f"starting summariser {prompt[:40]!r}")

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
                    logger.error(f"Request failed with status {resp.status} — Prompt: {prompt[:60]!r}")
                    return f"[Error {resp.status}]"
                else:
                    logger.info(f"Connection successful. Response received for prompt starting with: {prompt[:40]!r}")
                    data = await resp.json()
                    logger.info("Successfully received model response.") 
                    return data["content"].strip()
        except Exception as e:
            last_error = e
            logger.warning(f"An error occurred during query: {e}. retry {attempt+1} of {retries}")
            await asyncio.sleep(delay + random.uniform(0, 0.1))            
    
    logger.error(f"Failed after {retries} retries for prompt start: {prompt[:40]!r}")
    FAILED_LOG.write_text(f"{prompt[:60]}\nError: {last_error}\n", encoding="utf-8")
    return f"[Error: {last_error}]"
    
#Main function
async def summarise(text, tokeniser, session):
    prompt = make_summary_prompt(text)
    tokens = count_tokens(tokeniser, prompt)

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
    tokeniser = get_tokeniser()
    summaries = {}
    #adds progress bar
    pbar = tqdm(total=len(highlighted), desc="Summarizing")
    #switches to async for multiple requests
    async with aiohttp.ClientSession() as session:
        async def summarise_wrapper(i, text):
            summary = await summarise(text, tokeniser, session)
            logger.info(f"Summary {i+1}: {summary}\n")
            summaries[f"chunk_{i}"] = summary
            await asyncio.sleep(0.1)  # was time.sleep, now non-blocking!
            pbar.update(1)

        tasks = [summarise_wrapper(i, text) for i, text in enumerate(highlighted)]
        await asyncio.gather(*tasks)
        pbar.close()

    return summaries


if __name__ == "__main__":
    print("Script started", flush=True)
    asyncio.run(summarise_all(highlighted))