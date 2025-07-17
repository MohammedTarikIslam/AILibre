import logging
import asyncio
import aiohttp
import orjson
import time
import gc
import random
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama
import json

# Configure logging to output only to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

CACHE_FILE = Path("summaries/skipped_and_completed_files.json")

# ---- CONFIGURE THESE ----
MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-coder-33b-instruct.Q4_K_M.gguf"
CHUNK_DIR = Path("/home/tarik8422/chunks/dynamic/sw")
SUMMARY_DIR = Path("summaries/dynamic/sw")
MAX_TOKENS = 256
MAX_CTX = 8192  # match server --ctx-size
MAX_FILE_SIZE_BYTES = 12000000  # Only process files smaller than this
CHUNKS_PER_WORKER = 1
NUM_PROCESSES = 2
FAILED_LOG = Path("summaries/failed_chunks.log")
MAX_FILES = 1000  # ✅ Stop after summarizing this many files
SMALL_FILE_BYTES = 50
global_semaphore = asyncio.Semaphore(1)

# Glob and preload all chunks into memory
PRELOADED_CHUNKS = {}
for path in CHUNK_DIR.rglob("*.jsonl"):
    rel_path = path.relative_to(CHUNK_DIR)
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        PRELOADED_CHUNKS[str(rel_path)] = lines
    except Exception as e:
        print(f"⚠️ Could not load {rel_path}: {e}")

RESET_SKIPPED = True  # or use a command-line flag

if RESET_SKIPPED and CACHE_FILE.exists():
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
    cache["skipped"] = []
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def load_tokenizer():
    return Llama(model_path=MODEL_PATH, n_ctx=MAX_CTX, vocab_only=True)

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = load_tokenizer()
    return _tokenizer

def count_tokens(llm, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8")))

def split_by_tokens(llm, text: str, max_tokens: int):
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = llm.detokenize(chunk_tokens).decode("utf-8", errors="ignore")
        chunks.append(chunk_text.strip())
        start = end
    return chunks

def make_summary_prompt(code_chunk):
    return f"""### Instruction:
Read the following code carefully. Provide a concise but detailed summary that includes:
- The main purpose of the code
- Explanations of each function or class
- Any important design patterns, algorithms, or side effects

### Code:
{code_chunk}

### Response:
"""

# --- Asynchronous query function using aiohttp ---
async def async_query_llama_server(session, prompt, max_tokens=512, retries=1, use_semaphore=True):
    delay = 1
    url = "http://127.0.0.1:8080/completion"  # completion was removed
    timeout = aiohttp.ClientTimeout(total=5000)
    payload = {
        "prompt": prompt,
        "stop": ["###"],
        "n_predict": max_tokens,
        "temperature": 0.2,
        "top_k": 40,
        "top_p": 0.95,
        "stream": False
    }
    # Use the global semaphore if requested; otherwise use a dummy lock.
    ctx_manager = global_semaphore if use_semaphore else asyncio.Lock()
    async with ctx_manager:
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload, timeout=timeout) as response:
                    response.raise_for_status()
                    resp = await response.json()
                    return resp["content"].strip()
            except Exception as e:
                logger.error(f"[{type(e).__name__}] Prompt start: {prompt[:60]!r} — Error: {e}")
                print(f"Retry {attempt+1} failed: {type(e).__name__}: {e}")
                await asyncio.sleep(delay + random.uniform(0, 0.1))
                delay *= 2
        logger.error(f"[Worker] Failed to query llama-server after {retries} retries. Prompt start: {prompt[:60]!r}")

async def process_tasks(task_tuples, session, timeout=6000):
    """ async with global_semaphore:
    task_tuples: list of (chunk_index, prompt, max_tokens)
    Returns a list of (chunk_index, result)
    """
    coros = [async_query_llama_server(session, prompt, max_tokens) for (_, prompt, max_tokens) in task_tuples]
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [(index, res) for ((index, _, _), res) in zip(task_tuples, results)]

# --- Main Worker Function (runs in each process) ---
def summarize_worker(file_list, worker_id, progress_counter, progress_lock):
    import asyncio
    time.sleep(worker_id * 0.1)  # Stagger start by worker id
    llm = get_tokenizer()  # Load tokenizer (cached per process)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # failed = []
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    def log_failure(reason: str):
        try:
            with open(FAILED_LOG, 'ab') as flog:
                flog.write(orjson.dumps([reason], option=orjson.OPT_APPEND_NEWLINE))
        except Exception as e:
            print(f"⚠️ Failed to log failure: {e}")    

    async def worker_session():
        async with aiohttp.ClientSession() as session:
            for rel_path in file_list:
                # ✅ Stop globally if MAX_FILES is reached
                with progress_lock:
                    if progress_counter.value >= MAX_FILES:
                        return []
                if str(rel_path) not in PRELOADED_CHUNKS:
                    print(f"⚠️ [Worker {worker_id}] Skipping missing preload: {rel_path}")
                    continue
                # Use preloaded lines, but re-read file for size check and actual content.
                input_file = CHUNK_DIR / rel_path
                file_size = input_file.stat().st_size
                output_file = SUMMARY_DIR / rel_path.with_suffix(".summary.json")

                # Skip if summary exists
                if output_file.exists():
                    with progress_lock:
                        progress_counter.value += 1
                        gc.collect()
                    continue

                # Handle small file: simply skip (or add your log here)
                if file_size < SMALL_FILE_BYTES:
                    raw_code = "\n".join(PRELOADED_CHUNKS[str(rel_path)]).strip()
                    combined_summary = f"[Chunk 0]\n{raw_code}"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'wb') as f_out:
                        f_out.write(orjson.dumps({"file": str(rel_path), "summary": combined_summary}, option=orjson.OPT_INDENT_2))
                    print(f"✅ [Worker {worker_id}] {rel_path}: small file formatted as AI-style chunk")
                    with progress_lock:
                        progress_counter.value += 1
                        gc.collect()
                    continue

                # Don't process files that are too large.
                #     print(f"⏩ [Worker {worker_id}] Skipping large file: {rel_path}")
                #     with progress_lock:
                #         progress_counter.value += 1
                #         gc.collect()
                #     continue        
                # combined_summary = ""
                if file_size > MAX_FILE_SIZE_BYTES:
                    try:
                        combined_summary = ""
                        lines = PRELOADED_CHUNKS[str(rel_path)]
                        normal_tasks = []
                        for i, line in enumerate(lines):
                            try:
                                data = orjson.loads(line.encode("utf-8"))
                                chunk = data["code"]
                                prompt = make_summary_prompt(chunk)
                                prompt_tokens = count_tokens(llm, prompt)
                                safe_n_predict = max(256, MAX_CTX - prompt_tokens - 256)
                                
                                if safe_n_predict <= 0:
                                    print(f"⛔ [Worker {worker_id}] Skipping chunk {i} in {rel_path} — exceeds context window")
                                    log_failure(f"{rel_path} [chunk {i} skipped: too large]")
                                    continue
                                
                                # Oversized chunk: process subchunks concurrently.
                                if prompt_tokens > (4*MAX_CTX) - MAX_TOKENS - 256:
                                    print(f"⛔ [Worker {worker_id}] Skipping chunk {i} in {rel_path} — fucked up")
                                    log_failure(f"{rel_path} [chunk {i} skipped: fucked up]")
                                    continue


                                # Oversized chunk: process subchunks concurrently.
                                if prompt_tokens > MAX_CTX - MAX_TOKENS - 256:
                                    print(f"✅ [Worker {worker_id}] {rel_path} summarized")
                                    subchunk_semaphore = asyncio.Semaphore(3)
                                    print(f"⚠️ [Worker {worker_id}] Oversized chunk {i} in {rel_path}, splitting by tokens")
                                    subchunks = split_by_tokens(llm, chunk, 4096)
                                    sub_tasks = []
                                    for j, sub in enumerate(subchunks):
                                        sub_prompt = make_summary_prompt(sub)
                                        safe_sub_tokens = max(256, MAX_CTX - count_tokens(llm, sub_prompt) - 256)
                                        sub_tasks.append((j, sub_prompt, min(MAX_TOKENS, safe_sub_tokens)))
                                    
                                    queue = asyncio.Queue()
                                    results = {}

                                    async def producer():
                                        for j, (sub_prompt, tokens) in enumerate([(p, t) for (_, p, t) in sub_tasks]):
                                            await queue.put((j, sub_prompt, tokens))
                                        for _ in range(3):  # stop signals for workers
                                            await queue.put(None)

                                    async def worker():
                                        while True:
                                            item = await queue.get()
                                            if item is None:
                                                break
                                            j, sub_prompt, tokens = item
                                            try:
                                                res = await async_query_llama_server(session, sub_prompt, tokens, use_semaphore=False)
                                            except Exception as e:
                                                res = f"[Error] {e}"
                                                log_failure(f"{rel_path} [chunk {i}, subchunk {j} error: {e}]")
                                            results[j] = res

                                    await asyncio.gather(
                                        producer(),
                                        *[worker() for _ in range(3)]
                                    )
                                    parts_text = "\n".join([f"[Part {j}]\n{results[j]}" for j in sorted(results)])

                                    # async def sem_query(prompt, tokens):
                                    #     async with subchunk_semaphore:
                                    #         return await async_query_llama_server(session, prompt, tokens, use_semaphore=False)
                                    
                                    # sub_results = await asyncio.gather(*[sem_query(p, t) for (_, p, t) in sub_tasks], return_exceptions=True)
                                    # partial_summaries = [
                                    #     res if not isinstance(res, Exception) else f"[Error] {res}"
                                    #     for res in sub_results
                                    # ]
                                    # if partial_summaries:
                                    #     parts_text = "\n".join([f"[Part {j}]\n{res}" for j, res in enumerate(partial_summaries)])
                                    summary_prompt = f"""### Instruction:
    This is a summary of a long code file broken into parts. Please summarize it as a whole.

    ### Parts:
    {parts_text}

    ### Response:"""
                                    final_result = await sem_query(summary_prompt, MAX_TOKENS)
                                    combined_summary += f"\n[Chunk {i}]\n{final_result}\n"
                                    continue
                                # Normal processing: enqueue task
                                normal_tasks.append((i, prompt, min(MAX_TOKENS, safe_n_predict)))
                            except Exception as e:
                                logger.error(f"[{type(e).__name__}] Prompt start: {prompt[:60]!r} — Error: {e}")
                                print(f"⚠️ [Worker {worker_id}] Error processing chunk {i} of {rel_path}: {e}")
                                log_failure(f"{rel_path} [chunk {i} error: {e}]")
                        if normal_tasks:
                            normal_results = await process_tasks(normal_tasks, session)
                            normal_results.sort(key=lambda x: x[0])
                            for i, res in normal_results:
                                if isinstance(res, Exception):
                                    combined_summary += f"\n[Chunk {i}]\n[Error] {res}\n"
                                else:
                                    combined_summary += f"\n[Chunk {i}]\n{res}\n"
                        
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        if combined_summary.strip():
                            with open(output_file, 'wb') as f_out:
                                f_out.write(orjson.dumps({"file": str(rel_path), "summary": combined_summary.strip()}, option=orjson.OPT_INDENT_2))
                            print(f"✅ [Worker {worker_id}] {rel_path} summarized")
                        else:
                            print(f"⚠️ [Worker {worker_id}] Skipped writing empty summary for {rel_path}")
                            log_failure(f"{rel_path} [empty summary]")
                    except Exception as e:
                        print(f"⚠️ [Worker {worker_id}] Failed to read {rel_path}: {e}")
                        log_failure(f"{rel_path} [file error: {e}]")
                    with progress_lock:
                        progress_counter.value += 1
                        gc.collect()
                # return failed
    
    failure_list = loop.run_until_complete(worker_session())
    loop.close()
    if failure_list:
        with open(FAILED_LOG, 'ab') as flog:
            flog.write(orjson.dumps(failure_list, option=orjson.OPT_INDENT_2))
    return

def split_workload():
    all_jsonl = sorted(list(CHUNK_DIR.rglob("*.jsonl")), key=lambda f: f.stat().st_size)
    # , reverse=True
    skipped_cache = set()
    completed_cache = set()

    # Load cache if it exists, converting cached paths to strings.
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
                skipped_cache = set(cache.get("skipped", []))
                completed_cache = set(cache.get("completed", []))
        except Exception:
            print("⚠️ Failed to read cache file. Rebuilding cache.")

    new_skipped = set()
    new_completed = set()
    eligible = []

    for f in all_jsonl:
        rel_str = str(f.relative_to(CHUNK_DIR))
        # Compare using strings since caches store string paths.
        if rel_str in skipped_cache or rel_str in completed_cache:
            print(f"⚠️ Skipping cached file: {rel_str}")
            continue
        # Use consistent key type for PRELOADED_CHUNKS (strings).
        if rel_str not in PRELOADED_CHUNKS:
            new_skipped.add(rel_str)
            print(f"⚠️ Skipping missing preload: {rel_str}")
            continue
        f_size = f.stat().st_size
        if f_size > MAX_FILE_SIZE_BYTES:
            new_skipped.add(rel_str)
            continue
        output_path = SUMMARY_DIR / Path(rel_str).with_suffix(".summary.json")
        if output_path.exists():
            new_completed.add(rel_str)
            continue
        eligible.append(Path(rel_str))

    # Update and save new cache
    full_skipped = skipped_cache.union(new_skipped)
    full_completed = completed_cache.union(new_completed)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump({
            "skipped": list(full_skipped),
            "completed": list(full_completed)
        }, f, indent=2)
    if new_skipped:
        with open("skipped_files.log", "a") as log:
            for s in sorted(new_skipped):
                log.write(s + "\n")

    # Chunk work for workers.
    chunks = [eligible[i:i + CHUNKS_PER_WORKER] for i in range(0, len(eligible), CHUNKS_PER_WORKER)]
    return chunks

def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    workloads = split_workload()
    total_files = sum(len(chunk) for chunk in workloads)
    print(f"🧪 Number of eligible files to summarize: {total_files}")
    from multiprocessing import Manager
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()  # Create a lock using the manager
    
    # Create a progress bar in a separate thread to track total progress.
    from tqdm import tqdm
    pbar = tqdm(total=total_files, desc="Overall Progress")
    
    import threading
    def progress_monitor():
        last = 0
        while progress_counter.value < total_files:
            diff = progress_counter.value - last
            if diff:
                pbar.update(diff)
                last = progress_counter.value
            time.sleep(0.5)
        diff = total_files - last
        if diff:
            pbar.update(diff)
        pbar.close()
    
    progress_thread = threading.Thread(target=progress_monitor)
    progress_thread.start()
    
    try:
        with multiprocessing.get_context("spawn").Pool(processes=NUM_PROCESSES) as pool:
            args = [(wl, wid, progress_counter, progress_lock) for wid, wl in enumerate(workloads)]
            pool.starmap(summarize_worker, args)
    except Exception as e:
        print(f"❌ Error during multiprocessing: {e}")
    
    progress_thread.join()
    print("\n✅ All summaries complete!")

if __name__ == "__main__":
    main()
