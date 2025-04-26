import asyncio
import logging
import random
import gc
import time
from tqdm import tqdm
import json
import aiohttp
import orjson
from pathlib import Path
from llama_cpp import Llama
import multiprocessing
import threading


#Global variables
MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-coder-33b-instruct.Q4_K_M.gguf"
MAX_TOKENS = 128
MAX_CTX = 8192  #match server --ctx-size
NUM_PROCESSES = 5
BATCH_SIZE = 5  #num of paragraphs or pages
global_semaphore = asyncio.Semaphore(5)

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
- Any important assertion, directive, commitment, emotion and declaration where they are applicable in bullet point format
### Text:
{text}

### Response:
"""

#server query function
async def query_llama(session, prompt, max_tokens, retries=3, use_semaphore=True):
    print(f"starting summariser {prompt[:40]!r}")

    #default server location
    url = "http://127.0.0.1:8080/completion" 
    timeout=aiohttp.ClientTimeout(total=60)
    delay = 1    
    payload = {
        "prompt": prompt,
        "stop": ["###"],
        "n_predict": max_tokens,
        "temperature": 0.15,
        "top_k": 35,
        "top_p": 0.95,
    }

    #attempts to contact the server
    ctx_manager = global_semaphore if use_semaphore else asyncio.Lock()
    async with ctx_manager:
        for attempt in range(retries):
            try: 
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.error(f"Request failed with status {resp.status} — Prompt: {prompt[:60]!r}")
                        return f"[Error {resp.status}]"
                    else:
                        #logger.info(f"Connection successful. Response received for prompt starting with: {prompt[:40]!r}")
                        data = await resp.json()
                        logger.info("Successfully received model response.") 
                        return data["content"].strip()
            except Exception as e:
                last_error = e
                logger.warning(f"An error occurred during query: {e}. retry {attempt+1} of {retries}")
                await asyncio.sleep(delay + random.uniform(0, 0.1))            
        
    logger.error(f"Failed after {retries} retries for prompt start: {prompt[:40]!r}")
    logger.error(f"{prompt[:60]}\nError: {last_error}\n", encoding="utf-8")
    return f"[Error: {last_error}]"
    
#starmap doesnt accept async
def summariser_entry(all_text, summaries_dict, progress_counter, progress_lock, worker_id):
    asyncio.run(summariser(all_text, summaries_dict, progress_counter, progress_lock, worker_id))

#Main function
async def summariser(all_text, summaries_dict, progress_counter, progress_lock, worker_id):
    logger.info(f"Worker {worker_id} started")
    #initial sleep to stagger start
    time.sleep(worker_id * 0.1)

    tokeniser = get_tokeniser()

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                texts = all_text.get_nowait()
            except Exception:
                #empty queue
                break

            i, text = texts
            logger.info(f"Worker {worker_id} processing chunk {i}")

            tokens = count_tokens(tokeniser,text)
            if tokens > MAX_CTX - MAX_TOKENS - 256:
                try:
                    logger.info("Oversized text detected. Splitting...")
                    subchunks = split_by_tokens(tokeniser, text, 4096)
                    #summarises each subchunk
                    async def summarise_subchunk(index, sub):
                        sub_prompt = make_summary_prompt(sub)
                        res = await query_llama(session, sub_prompt, MAX_TOKENS)
                        return f"[Part {index}]{res}"
            
                    summarise_each = [summarise_subchunk(idx, sub) for idx, sub in enumerate(subchunks)]
                    results = await asyncio.gather(*summarise_each)
                    full = "\n".join(f"[Part {i}]{res}" for i, res in enumerate(results))
                    
                    final_prompt = f"""### Instruction:
            Summarize the following parts of a large text as a single cohesive summary.

            ### Parts:
            {full}

            ### Response:
            """
                    full_summary = await query_llama(session, final_prompt, MAX_TOKENS)
                    summaries_dict[i] = full_summary
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed on oversized text {i}: {e}")
                    summaries_dict[i] = "[ERROR]"
            else:
                try:
                    prompt = make_summary_prompt(text)
                    summary = await query_llama(session, prompt, MAX_TOKENS)
                    summaries_dict[i] = summary
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed on text {i}: {e}")
                    summaries_dict[i] = "[ERROR]"
            with progress_lock:
                progress_counter.value += 1
            gc.collect()
    
    logger.info(f"Worker {worker_id} finished processing")

#adds progress bar
def progress_monitor(num_tasks, counter):
    pbar = tqdm(num_tasks, desc="Summarizing")
    last_task = 0
    while last_task < num_tasks:
        time.sleep(0.5)
        with counter.get_lock():
            current = counter.value
        diff = current - last_task
        if diff > 0:
            pbar.update(diff)
            last_task = current
    pbar.close()


#summarises all text in the list
async def summarise_all(highlighted):
    manager = multiprocessing.Manager()
    all_text = manager.Queue()
    num_tasks = len(highlighted)
    summaries_dict = manager.dict()
    #creates tuples (index, text)
    for i, text in enumerate(highlighted):
        all_text.put((i, text))

    #creates progress bar
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()
    progress_thread = threading.Thread(target=progress_monitor, args=(num_tasks, progress_counter))
    progress_thread.start()
 

    #sets up multiprocessing
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=NUM_PROCESSES)
    args = [(all_text, summaries_dict, progress_counter, progress_lock, wid) for wid in range(NUM_PROCESSES)]

    pool.starmap(summariser_entry, args)

    progress_thread.join()

    summaries = dict(summaries_dict)
    sorted_summaries = [summaries[i] for i in sorted(summaries)]
    
    #output results
    logger.info(f"Summaries:")
    for i, summary in enumerate(sorted_summaries):
        logger.info(f"Summary {i+1}: {summary}\n")

    return dict(summaries_dict), []



if __name__ == "__main__":
    print("Script started", flush=True)
    summaries, failed = asyncio.run(summarise_all(highlighted))

    print("Summaries:", summaries)

    if failed != []:
        print("Failed Summaries:", failed)
    else:
        print("No failed summaries.")
