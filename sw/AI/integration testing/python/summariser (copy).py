import uno
from com.sun.star.awt import MessageBoxButtons as MSG_BUTTONS
from com.sun.star.awt.MessageBoxType import MESSAGEBOX
from com.sun.star.awt.MessageBoxButtons import BUTTONS_YES_NO
from com.sun.star.awt.MessageBoxResults import YES, NO
import requests
import json
import asyncio
import logging
import random
import gc
import socket
import time
from tqdm import tqdm
import aiohttp
import orjson
from pathlib import Path
from llama_cpp import Llama
import threading
import queue
import subprocess
import time

# Global variables
MODEL_PATH = "/home/tarik8422/llama.cpp/models/deepseek-coder-33b-instruct.Q4_K_M.gguf"
MAX_TOKENS = 128
# match server --ctx-size
MAX_CTX = 8192
# num of paragraphs or pages
BATCH_SIZE = 5
global_semaphore = asyncio.Semaphore(5)
ai_process = None


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


# runs the AI model in a separate process
def run_ai():
    try:
        global ai_process
        command = "source ~/llama_env/bin/activate && ./run_llama.sh"
        ai_process = subprocess.Popen(["bash", "-c", command])
        if ai_process is None:
            return "Failed to start AI process: got None"
        text_box("AI Status", f"AI started with PID {ai_process.pid}")
        # wait for the ai to start up
        time.sleep(3)
        return f"AI started with PID {ai_process.pid}"
    except Exception as e:
        return f"Failed to start AI process: {str(e)}"


def stop_ai():
    global ai_process
    if ai_process and ai_process.poll() is None:
        ai_process.kill()
        ai_process.wait()
        return "AI process terminated"
    else:
        return "No running AI process to kill"


def insert_debug_line(line):
    try:
        doc = XSCRIPTCONTEXT.getDocument()
        text = doc.Text
        controller = doc.getCurrentController()
        view_cursor = controller.getViewCursor()
        text.insertString(view_cursor, "[DEBUG] " + line + "\n\n", 0)
    except:
        pass


def text_box(title, text):
    ctx = XSCRIPTCONTEXT.getComponentContext()
    smgr = ctx.ServiceManager
    toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
    parent = toolkit.getDesktopWindow()

    box = toolkit.createMessageBox(
        parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, title, text
    )
    box.execute()


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
    test = "query worked"
    ctx_manager = global_semaphore if use_semaphore else asyncio.Lock()
    async with ctx_manager:
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.error(
                            f"Request failed with status {resp.status} — Prompt: {prompt[:60]!r}"
                        )
                        # test = "failed response"
                        # return test
                        return f"[Error {resp.status}]"
                    else:
                        # logger.info(f"Connection successful. Response received for prompt starting with: {prompt[:40]!r}")
                        data = await resp.json(loads=orjson.loads)
                        logger.info("Successfully received model response.")
                        # test = "query worked"
                        # return test
                        return data["content"].strip()
            except Exception as e:
                last_error = e
                # test = "exception occurred"
                await asyncio.sleep(delay + random.uniform(0, 0.1))
                # return test
                logger.warning(
                    f"An error occurred during query: {e}. retry {attempt+1} of {retries}"
                )

    last_error(f"Failed after {retries} retries for prompt start: {prompt[:40]!r}")
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


async def text_proc(
    index, i, text, results_dict, progress_counter, progress_lock, worker_id
):
    mode = "summary"
    try:
        # initial sleep to stagger start
        #         potentially an issue with pausing
        placeholder = [1, 1, "1xxxxxxxxxxxxx"]

        await asyncio.sleep(worker_id * 0.1)

        async with aiohttp.ClientSession() as session:
            placeholder = [1, 1, "133333333333"]
            try:
                placeholder = [1, 1, "try 12222222222222"]
                try:
                    placeholder = [1, 1, "success tryyyyyyyyyyyyy "]
                    if mode == "edit":
                        prompt = make_edit_prompt(text)
                    elif mode == "summary":
                        prompt = make_summary_prompt(text)
                    placeholder = [index, i, prompt]
                    ai_response = await query_llama(session, prompt, MAX_TOKENS)
                    placeholder = [index, i, ai_response]
                    # results_dict[i] = summary
                except Exception as e:
                    placeholder = [1, 1, "tryyyyyyyyyyyyy "]
                    placeholder = [index, i, ai_response]
                    return placeholder
                    # results_dict[i] = "[ERROR]"  88888888888888888888888
                with progress_lock:
                    progress_counter[0] += 1
                # gc.collect()
                # placeholder = ["finished tryyyyyyyyyyyyy ", 1, 1]
            except Exception as e:
                # placeholder = [1, 1, "failed 12222222222222"]
                return placeholder
            # insert_debug_line(f"Worker {worker_id} error: {e}")
        return placeholder
    except Exception as e:
        # placeholder = ["failed 1xxxxxxxxxxxxx", 1, 1]
        return placeholder
        # insert_debug_line(f"Worker {worker_id} error: {e}")


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
            index, i, text = all_text_queue.get(timeout=1)
            # result = ["failed 6666666666666666", 1, 1]
            result = [index, i, text]
            # text_new
            result = asyncio.run(
                text_proc(
                    index,
                    i,
                    text,
                    results_dict,
                    progress_counter,
                    progress_lock,
                    thread_id,
                )
            )
            # result = [index, i, text_new]
            with progress_lock:
                results_dict[index] = result
                progress_counter[0] += 1
            all_text_queue.task_done()
        except queue.Empty:
            # empty queue
            break
        except Exception as e:
            # result = [index, i, text]
            result = ["failed 6666666666666666", e, 1]
            with progress_lock:
                results_dict[index] = result
                progress_counter[0] += 1
            all_text_queue.task_done()


####################################### redo comment ########################################
# summarises all text in the list
async def alltext_proc(highlighted):
    NUM_PROCESSES = 5
    try:
        result = ["failed 6666666666666666", 1, 1]
        # insert_debug_line("entering alltext_proc")
        # check to see if alltextproc starts
        ctx = XSCRIPTCONTEXT.getComponentContext()
        smgr = ctx.ServiceManager
        toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
        parent = toolkit.getDesktopWindow()

        alltext_queue = queue.Queue()

        # creates tuples (index, i, text)
        for index, (i, text) in enumerate(highlighted):
            alltext_queue.put((index, i, text))

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

        result = results_dict.copy()
        box = toolkit.createMessageBox(
            parent, MESSAGEBOX, MSG_BUTTONS.BUTTONS_OK, "processing text", "1111"
        )
        box.execute()
        # output results
        # insert_debug_line("Results:")
        # insert_debug_line(str(sorted_results))
        for index, i, text in sorted_results:
            insert_debug_line(f"Result {index} {i}: {text}")

        placeholder = result

        stop_ai()

        # insert_debug_line("alltext_proc completed successfully")

        return result, []

    except Exception as e:
        ctx = XSCRIPTCONTEXT.getComponentContext()
        smgr = ctx.ServiceManager
        toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
        parent = toolkit.getDesktopWindow()
        text_box("all text proc Error", str(e))
        # insert_debug_line("alltext proc error")
        # logger.error(f"Error in alltext_proc: {e}")
        return {}, [str(e)]


def send_to_ai(highlighted):
    ctx = XSCRIPTCONTEXT.getComponentContext()
    smgr = ctx.ServiceManager
    toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
    parent = toolkit.getDesktopWindow()

    box = toolkit.createMessageBox(
        parent,
        # uno.createUnoStruct("com.sun.star.awt.Rectangle"),
        "QUERYBOX",
        3,  # BUTTONS_YES_NO
        "Choose AI Mode",
        "Click YES Summarise mode [default] or NO Edit mode",
    )

    mode = box.execute()

    if mode == 2:
        text_box("AI Mode", "Edit mode selected. Click OK to continue.")

        results, failed = asyncio.run(alltext_proc(highlighted))
        failedstr = ", ".join(failed)

        ai = stop_ai()
        text_box("AI Status", f"AI stopped: {ai}")

        if failed != []:
            # error printing an array
            text_box("Error", f"{failedstr} failed to process")
        else:
            text_box("Processing completed", "All tasks completed successfully")
    elif mode == 3:
        text_box("AI Mode", "Summarise mode selected. Click OK to continue.")
        summaries, failed = asyncio.run(alltext_proc(highlighted))
        if failed != []:
            text_box("Error", f"{failedstr} failed to process")
        else:
            text_box("Processing completed", "Summaries completed successfully")

    else:
        text_box("Error", f"{highlighted} + no mode")


# entry point of code
# sends selected text to processing
def send_selected_text_to_ai():
    MAX_CHARS = 4000

    ai = run_ai()

    # insert_debug_line(f"AI started: {ai}")

    text_box("AI Status", f"AI started: {ai}")

    # Get the current document and view cursor
    xDoc = XSCRIPTCONTEXT.getDocument()
    xController = xDoc.getCurrentController()
    view_cursor = xController.getSelection()

    ctx = XSCRIPTCONTEXT.getComponentContext()
    smgr = ctx.ServiceManager
    toolkit = smgr.createInstanceWithContext("com.sun.star.awt.Toolkit", ctx)
    parent = toolkit.getDesktopWindow()

    selected_text = []
    if hasattr(view_cursor, "getCount") and view_cursor.getCount() > 1:
        # insert_debug_line("get count")
        for i in range(1, view_cursor.getCount()):
            # starts at 1. quirk of libreoffice
            text = view_cursor.getByIndex(i).getString()
            selected_text.append(text)
    elif hasattr(view_cursor, "getCount") and view_cursor.getCount() == 1:
        # insert_debug_line("get string")
        text = view_cursor.getByIndex(0).getString()
        selected_text.append(text)
    else:
        selected_text = ""

    # checks if the selection is valid
    if not selected_text:
        text_box("Error", "No valid text selected.")
        return

    # insert_debug_line("sending selected text to ai")

    selected_cut = []
    for i in range(len(selected_text)):
        cut_text = split_text(selected_text[i], MAX_CHARS)
        for text in cut_text:
            selected_cut.append((i, text))

    send_to_ai(selected_cut)


g_exportedScripts = (send_selected_text_to_ai,)
