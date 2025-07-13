import uno
from com.sun.star.awt import MessageBoxButtons as MSG_BUTTONS
from com.sun.star.awt.MessageBoxType import MESSAGEBOX
import requests
import json
import asyncio
import logging
import random
import gc
import time
from tqdm import tqdm
import aiohttp
import orjson
from pathlib import Path
from llama_cpp import Llama
import multiprocessing
import threading
import queue

# Global variables
MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-coder-33b-instruct.Q4_K_M.gguf"
MAX_TOKENS = 128
# match server --ctx-size
MAX_CTX = 8192
NUM_PROCESSES = 5
# num of paragraphs or pages
BATCH_SIZE = 5
global_semaphore = asyncio.Semaphore(5)
mode = "edit"


# region for logging
# configures logging
FAILED_LOG = Path("failed.log")
file_handler = logging.FileHandler(FAILED_LOG, encoding="utf-8")
file_handler.setLevel(logging.ERROR)

# configures logging. output only to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(), file_handler],
)
logger = logging.getLogger(__name__)
# endregion


def insert_debug_line(line):
    try:
        doc = XSCRIPTCONTEXT.getDocument()
        text = doc.Text
        cursor = text.createTextCursor()
        text.insertString(cursor, "[DEBUG] " + line + "\n", 0)
    except:
        pass


# region for tokeniser logic


# _tokeniser = None

# def get_tokeniser():
#     global _tokeniser
#     if _tokeniser is None:
#         _tokeniser = Llama(model_path=MODEL_PATH, n_ctx=MAX_CTX, vocab_only=True)
#     return _tokeniser


# def count_tokens(tokeniser, text):
#     return len(tokeniser.tokenize(text.encode("utf-8")))

# endregion


# splits text into chunks when too large
def split_text(text, max_chars):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # ensures split is
        mid = start + (max_chars // 2)
        split = text.rfind("\n\n", start, end)
        if split == -1 or split < mid:
            split = text.rfind("\n", start, end)
            if split == -1 or split < mid:
                p1 = text.rfind(".", start, end)
                p2 = text.rfind("?", start, end)
                p3 = text.rfind("!", start, end)
                split = max(p1, p2, p3)
                if split == -1 or split < mid:
                    p1 = text.rfind(",", start, end)
                    p2 = text.rfind(":", start, end)
                    p3 = text.rfind(";", start, end)
                    split = max(p1, p2, p3)
                    if split == -1 or split < mid:
                        split = text.rfind(" ", start, end)
                        if split == -1 or split < mid:
                            split = end - 1
        else:
            # skips one newline token if new paragraph /n/n
            split += 1
        # skips punctuation at the end of the chunk
        split += 1

        chunk = text[start:split]
        chunks.append(chunk)
        # chunks.append(chunk.strip())
        start = split
    return chunks


# region for prompts
# takes input and forms a full prompt for the model
def make_summary_prompt(text):
    return f"""### Instruction:
 Read the text carefully. Provide a concise but detailed summary that includes:
- Any important assertion, directive, commitment, emotion and declaration where they are applicable in bullet point format
- If the text is only instructions, provide simpler short instructions in bullet point format with all details included
### Text:
{text}

### Response:
"""


def make_edit_prompt(text):
    return f"""### Instruction:
Read the text carefully and improve clarity, grammar, and style without changing the factual content.
Preserve formatting where possible.
correct any spelling mistakes and grammatical errors.
ensure the tense is consistent throughout.
### Text:
{text}

### Response:
"""


# endregion


# server query function
async def query_llama(session, prompt, max_tokens, retries=3, use_semaphore=True):
    ctx = XSCRIPTCONTEXT.getComponentContext()
    smgr = ctx.ServiceManager
    toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
    parent = toolkit.getDesktopWindow()
    box = toolkit.createMessageBox(
        parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "Notice", " starting summariser "
    )
    box.execute()

    # logger.info(f"starting summariser {prompt[:40]!r}")

    # default server location
    url = "http://127.0.0.1:8080/completion"
    timeout = aiohttp.ClientTimeout(total=60)
    delay = 1
    payload = {
        "prompt": prompt,
        "stop": ["###"],
        "n_predict": max_tokens,
        "temperature": 0.15,
        "top_k": 35,
        "top_p": 0.95,
    }

    # attempts to contact the server
    ctx_manager = global_semaphore if use_semaphore else asyncio.Lock()
    async with ctx_manager:
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.error(
                            f"Request failed with status {resp.status} — Prompt: {prompt[:60]!r}"
                        )
                        return f"[Error {resp.status}]"
                    else:
                        # logger.info(f"Connection successful. Response received for prompt starting with: {prompt[:40]!r}")
                        data = await resp.json(loads=orjson.loads)
                        logger.info("Successfully received model response.")
                        return data["content"].strip()
            except Exception as e:
                last_error = e
                logger.warning(
                    f"An error occurred during query: {e}. retry {attempt+1} of {retries}"
                )
                await asyncio.sleep(delay + random.uniform(0, 0.1))

    logger.error(f"Failed after {retries} retries for prompt start: {prompt[:40]!r}")
    logger.error(f"{prompt}\n Error: {last_error}\n")
    return f"[Error: {last_error}]"


# high chance this is not needed anymore


# # starmap doesnt accept async
# def text_proc_entry(all_text, results_dict, progress_counter, progress_lock, worker_id):
#     try:
#         with open("/tmp/debug_ai_macro.log", "a") as f:
#             f.write(f"[{worker_id}] Reached checkpoint\n")

#         # insert_debug_line("Reached textprocentry")
#         #
#         # ctx = XSCRIPTCONTEXT.getComponentContext()
#         # smgr = ctx.ServiceManager
#         # toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
#         # parent = toolkit.getDesktopWindow()

#         # box = toolkit.createMessageBox(
#         #     parent,
#         #     MESSAGEBOX,
#         #     MSG_BUTTONS.BUTTONS_OK,
#         #     "alltext_proc",
#         #     "rsoitebsobeinbioesrbreinnbsirobne",
#         # )
#         # box.execute()

#         # asyncio.run(
#         #     text_proc(
#         #         all_text, results_dict, progress_counter, progress_lock, worker_id
#         #     )
#         # )
#     except Exception as e:
#         with open("/tmp/ai_macro_debug.log", "a") as f:
#             f.write(f"[{worker_id}] Exception in text_proc_entry: {str(e)}\n")
#         #
#         # insert_debug_line("textproc entry error")
#         #
#         # ctx = XSCRIPTCONTEXT.getComponentContext()
#         # smgr = ctx.ServiceManager
#         # toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
#         # parent = toolkit.getDesktopWindow()
#         # box = toolkit.createMessageBox(
#         #     parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "textproc entry error", e
#         # )
#         # box.execute()


# # Main function
# async def text_proc(all_text, results_dict, progress_counter, progress_lock, worker_id):
#     try:
#         # initial sleep to stagger start
#         #         potentially an issue with pausing
#         # await asyncio.sleep(worker_id * 0.1)

#         async with aiohttp.ClientSession() as session:
#             while True:
#                 try:
#                     texts = all_text.get_nowait()
#                 except Exception:
#                     # empty queue
#                     break

#                 i, text = texts
#                 # logger.info(f"Worker {worker_id} processing chunk {i}")

#                 tokens = count_tokens(tokeniser, text)
#                 if tokens > MAX_CTX - MAX_TOKENS - 256:
#                     try:
#                         logger.info("Oversized text detected. Splitting...")
#                         subchunks = split_text(tokeniser, text, 4096)

#                         # summarises each subchunk
#                         async def process_subchunk(index, sub):
#                             if mode == "edit":
#                                 sub_prompt = make_edit_prompt(sub)
#                             elif mode == "summary":
#                                 sub_prompt = make_summary_prompt(sub)

#                             res = await query_llama(session, sub_prompt, MAX_TOKENS)
#                             return f"[Part {index}]{res}"

#                         process_each = [
#                             process_subchunk(idx, sub)
#                             for idx, sub in enumerate(subchunks)
#                         ]
#                         results = await asyncio.gather(*process_each)
#                         full = "\n".join(
#                             f"[Part {i}]{res}" for i, res in enumerate(results)
#                         )

#                         final_prompt = f"""### Instruction:
#                 Summarize the following parts of a large text as a single cohesive summary.

#                 ### Parts:
#                 {full}

#                 ### Response:
#                 """
#                         full_result = await query_llama(
#                             session, final_prompt, MAX_TOKENS
#                         )
#                         results_dict[i] = full_result
#                     except Exception as e:
#                         logger.error(
#                             f"Worker {worker_id} failed on oversized text {i}: {e}"
#                         )
#                         results_dict[i] = "[ERROR]"
#                 else:
#                     try:
#                         if mode == "edit":
#                             prompt = make_edit_prompt(text)
#                         elif mode == "summary":
#                             prompt = make_summary_prompt(text)
#                         summary = await query_llama(session, prompt, MAX_TOKENS)
#                         results_dict[i] = summary
#                     except Exception as e:
#                         logger.error(f"Worker {worker_id} failed on text {i}: {e}")
#                         results_dict[i] = "[ERROR]"
#                 with progress_lock:
#                     progress_counter.value += 1
#                 gc.collect()
#         logger.info(f"Worker {worker_id} finished processing")
#     except Exception as e:
#         ctx = XSCRIPTCONTEXT.getComponentContext()
#         smgr = ctx.ServiceManager
#         toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
#         parent = toolkit.getDesktopWindow()
#         box = toolkit.createMessageBox(
#             parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "textproc error", e
#         )
#         box.execute()
#     placeholder = "This is a placeholder for the text processing function."
#     return placeholder
#     # insert_debug_line(placeholder)


# adds progress bar
def progress_monitor(num_tasks, counter):
    pbar = tqdm(total=num_tasks, desc="Summarizing")
    last_task = 0
    while last_task < num_tasks:
        time.sleep(0.5)
        current = counter.value
        diff = current - last_task
        if diff > 0:
            pbar.update(diff)
            last_task = current
    pbar.close()


def worker_thread(
    all_text_queue, results_dict, progress_counter, progress_lock, thread_id
):
    while True:
        try:
            i, text = all_text_queue.get_nowait()
        except queue.Empty:
            break
        result = asyncio.run(
            text_proc(text, results_dict, progress_counter, progress_lock, thread_id)
        )  # or whatever async method you call
        with progress_lock:
            results_dict[i] = result
            progress_counter[0] += 1
        all_text_queue.task_done()


# summarises all text in the list
# async
async def alltext_proc():
    try:
        insert_debug_line("entering alltext_proc")
        # check to see if alltextproc starts
        ctx = XSCRIPTCONTEXT.getComponentContext()
        smgr = ctx.ServiceManager
        toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
        parent = toolkit.getDesktopWindow()

        # await asyncio.sleep(5)
        highlighted = ["abc"]

        # manager = multiprocessing.Manager()
        alltext_queue = queue.Queue()

        # creates tuples (index, text)
        for i, text in enumerate(highlighted):
            alltext_queue.put((i, text))

        num_tasks = len(highlighted)
        results_dict = {}
        # creates progress bar
        progress_counter = [0]
        progress_lock = threading.Lock()
        progress_thread = threading.Thread(
            target=progress_monitor, args=(num_tasks, progress_counter)
        )
        progress_thread.start()
        # sets up multiprocessing

        threads = []
        for i in range(NUM_PROCESSES):
            t = threading.Thread(
                target=worker_thread,
                args=(alltext_queue, results_dict, progress_counter, progress_lock, i),
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        progress_thread.join()

        sorted_results = [results_dict[i] for i in sorted(results_dict)]

        # context = multiprocessing.get_context("spawn")
        # pool = context.Pool(processes=NUM_PROCESSES)
        # args = [
        #     (all_text, results_dict, progress_counter, progress_lock, wid)
        #     for wid in range(NUM_PROCESSES)
        # ]

        # # starts the worker pool
        # pool.starmap(text_proc_entry, args)
        # pool.close()
        # pool.join()
        # progress_thread.join()
        result = dict(results_dict)

        # output results
        # logger.info(f"Results:")
        for i, result in enumerate(sorted_results):
            placeholder = f"arst {i+1}: {result}"
            insert_debug_line(placeholder)
            # logger.info(f"Summary {i+1}: {results}\n")
            insert_debug_line(f"Placeholder {i+1}: {sorted_results[i]}")
            box = toolkit.createMessageBox(
                parent,
                MESSAGEBOX,
                MSG_BUTTONS.BUTTONS_OK,
                "alltext_proc",
                "for loop",
            )
            box.execute()

        box = toolkit.createMessageBox(
            parent,
            MESSAGEBOX,
            MSG_BUTTONS.BUTTONS_OK,
            "alltext_proc",
            "part 3 post for loop",
        )
        box.execute()

        return dict(results_dict), []

    except Exception as e:
        ctx = XSCRIPTCONTEXT.getComponentContext()
        smgr = ctx.ServiceManager
        toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
        parent = toolkit.getDesktopWindow()
        box = toolkit.createMessageBox(
            parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "all text proc Error", str(e)
        )
        box.execute()
        insert_debug_line("alltext proc error")
        logger.error(f"Error in alltext_proc: {e}")
        return {}, [str(e)]


def send_to_ai(highlighted):
    ctx = XSCRIPTCONTEXT.getComponentContext()
    smgr = ctx.ServiceManager
    toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
    parent = toolkit.getDesktopWindow()

    fulltext = " ".join(highlighted)

    if mode == "edit":

        # asyncio.run(alltext_proc())

        results, failed = asyncio.run(alltext_proc())
        # failed = []
        # results = []

        if failed != []:
            # error printing an array
            insert_debug_line("Reached posrt alltext_proc")
            box = toolkit.createMessageBox(
                parent,
                MESSAGEBOX,
                MSG_BUTTONS.BUTTONS_OK,
                "Notice",
                "failed to procsess",
            )
            box.execute()
        else:
            box = toolkit.createMessageBox(
                parent,
                MESSAGEBOX,
                MSG_BUTTONS.BUTTONS_OK,
                "Notice",
                "All tasks completed successfully",
            )
            box.execute()
    elif mode == "summary":
        summaries, failed = asyncio.run(alltext_proc(highlighted))
        if failed != []:
            failedstr = ", ".join(failed)
            box = toolkit.createMessageBox(
                parent,
                MESSAGEBOX,
                MSG_BUTTONS.BUTTONS_OK,
                "Notice",
                failedstr + "failed to procsess",
            )
            box.execute()
        else:
            box = toolkit.createMessageBox(
                parent,
                MESSAGEBOX,
                MSG_BUTTONS.BUTTONS_OK,
                "Notice",
                "Summaries completed successfully",
            )
            box.execute()

        # summaries, failed = asyncio.run(alltext_proc(highlighted))
        # # print("\n Summaries:", summaries)
        # if failed != []:
        #     print("Failed to process:", failed)
        # else:
        #     print("All tasks completed successfully")
    else:
        box = toolkit.createMessageBox(
            parent,
            MESSAGEBOX,
            MSG_BUTTONS.BUTTONS_OK,
            "Notice",
            highlighted + "no mode",
        )
        box.execute()


# entry point of code
# sends selected text to processing
def send_selected_text_to_ai():
    # try:
    # Get the current document and view cursor
    xDoc = XSCRIPTCONTEXT.getDocument()
    xController = xDoc.getCurrentController()
    view_cursor = xController.getSelection()

    ctx = XSCRIPTCONTEXT.getComponentContext()
    smgr = ctx.ServiceManager
    toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
    parent = toolkit.getDesktopWindow()
    insert_debug_line("sending selected text to ai")

    selected_text = []
    if hasattr(view_cursor, "getCount") and view_cursor.getCount() > 0:
        insert_debug_line("get count")
        for i in range(1, view_cursor.getCount()):
            # starts at 1. quirk of libreoffice
            text = view_cursor.getByIndex(i).getString()
            selected_text.append(text)
            insert_debug_line(text + " " + str(i))
    elif hasattr(view_cursor, "getString"):
        insert_debug_line("get string")
        selected_text = view_cursor.getString()
    else:
        selected_text = ""

    # checks if the selection is valid
    if selected_text == [] or selected_text == "":
        box = toolkit.createMessageBox(
            parent,
            MESSAGEBOX,
            MSG_BUTTONS.BUTTONS_OK,
            "Notice",
            "No valid text selected.",
        )
        box.execute()
        return
    else:
        for i in range(1, len(selected_text)):
            # starts at 1. quirk of libreoffice
            box = toolkit.createMessageBox(
                parent,
                MESSAGEBOX,
                MSG_BUTTONS.BUTTONS_OK,
                "sending selected text to AI",
                selected_text[i],
            )
            box.execute()

    # cut_text = split_text(selected_text, 2048)
    cut_text = ["abc"]

    send_to_ai(cut_text)

    # selected = [selection.getByIndex(i).getString() for i in range(selection.getCount())]

    # splits the chunks into smaller parts if they are too large
    # highlighted = []
    # highlighted = cut_text

    # except Exception as e:
    #     box = toolkit.createMessageBox(
    #         parent,
    #         MESSAGEBOX,
    #         MSG_BUTTONS.BUTTONS_OK,
    #         "Error",
    #         e,
    #     )
    #     box.execute()
    #     print("Error during macro execution:", e)

    # region DELETE MAYBE IESTNATRIENTOKYUWNPIUNEITNRIETNOIETNISTENSIONAIERTNOENTOEANRIETSRNIOTEN

    #     # Prepare AI request
    #     url = "http://localhost:5000/api"
    #     headers = {"Content-Type": "application/json"}
    #     payload = {"input": selected_text}

    #     # Send request to AI backend
    #     response = requests.post(url, data=json.dumps(payload), headers=headers)

    #     if response.status_code == 200:
    #         result = response.json().get("output", "")
    #         if result:
    #             # Replace selected text with AI output
    #             view_cursor.setString(result)
    #             print("Text replaced with AI response.")
    #         else:
    #             print("AI returned no output.")
    #     else:
    #         print(f"AI request failed: {response.status_code}")

    # endregion


g_exportedScripts = (send_selected_text_to_ai,)
