# annotator.py
import random, json, datetime, os, pathlib
import gradio as gr
from datasets import load_dataset

# ───────────────────────────────────────────────────────────────────
# 1. CONFIG
# ───────────────────────────────────────────────────────────────────
ANNOT_FILE = "annotations.jsonl"         # cache
DATASET_ID = "potsawee/speakbench-v1-all-outputs"
SPLIT      = "train"

models = [
    "gpt4o-audio",
    "gpt4o-audio+asr+tts",
    "gemini2-flash-exp",
    "gemini2-flash-exp+asr+tts",
    # ------------------- #
    "typhoon2-audio",
    "llama-omni",
    "moshi",
    # ------------------- #
    "gpt4o-text+tts",
    "gemini2-flash-text+tts",
    "diva+tts",
    "typhoon2-audio+tts",
    "qwen2-audio+tts",
    "asr+llama3+tts",
]

# ───────────────────────────────────────────────────────────────────
# 2. LOAD DATASET  (≈ 3 s on first run, then cached by 🤗 datasets)
# ───────────────────────────────────────────────────────────────────
print("Loading dataset …")
ds = load_dataset(DATASET_ID, split=SPLIT)        #   <─ HF cache ~/.cache
N  = len(ds) - 1
print(f"Loaded {N+1} items.")

# ───────────────────────────────────────────────────────────────────
# 3. HELPERS
# ───────────────────────────────────────────────────────────────────
def pick_two():
    """Return two distinct model names."""
    return random.sample(models, 2)

def fetch_example(idx: int):
    """Return instruction, two audio tuples, radio choices, hidden names."""
    if idx < 0 or idx > N:
        raise gr.Error(f"data_ID must be between 0 and {N}")

    ex  = ds[int(idx)]
    m1, m2 = pick_two()

    audio1 = (ex[m1]["sampling_rate"], ex[m1]["array"])  #  SR first!
    audio2 = (ex[m2]["sampling_rate"], ex[m2]["array"])

    return (
        f"### Instruction\n\n{ex['instruction']}",       # Markdown
        audio1, audio2,
        gr.update(choices=[m1, m2, "tie"], value=None),
        m1, m2
    )

def append_jsonl(obj: dict, path: str):
    """Write dict as one JSON line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def save_pref(idx, m1, m2, choice, annotator):
    if choice is None:
        raise gr.Error("☝️ Pick an option before submitting!")

    pref_token = "model1" if choice == m1 else "model2" if choice == m2 else "tie"

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "annotator": annotator or "anon",
        "data_ID": int(idx),
        "model1": m1,
        "model2": m2,
        "preference": pref_token
    }
    append_jsonl(record, ANNOT_FILE)

    # ── count lines AFTER writing ─────────────────────────────────
    with open(ANNOT_FILE, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    status_msg = f"✅ Saved! Annotations in file: {total}"
    return status_msg, progress_table_data()   # ← two outputs now

def progress_table_data():
    """Return [[data_ID, count], …] for the whole split."""
    counts = {}
    if os.path.getsize(ANNOT_FILE):
        with open(ANNOT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    did = int(rec["data_ID"])
                    counts[did] = counts.get(did, 0) + 1
                except Exception:
                    pass                       # skip malformed lines
    return [[i, counts.get(i, 0)] for i in range(N + 1)]

# make sure the file exists
pathlib.Path(ANNOT_FILE).touch(exist_ok=True)

# ───────────────────────────────────────────────────────────────────
# 4. UI
# ───────────────────────────────────────────────────────────────────
with gr.Blocks(title="SpeakBench Audio Preference Annotator") as demo:
    gr.Markdown(
        f"""
        ## 🔊 SpeakBench — Audio Preference Annotator
        * Dataset: **{DATASET_ID}**  (split “{SPLIT}”)
        * Total items: **{N+1}**
        """
    )

    with gr.Row():
        idx_in      = gr.Number(label="data_ID", value=0, minimum=0,
                                maximum=N, step=1)
        annot_in    = gr.Textbox(label="Your name (optional)", value="pm574")
        load_btn    = gr.Button("🔄 Load sample")

    instr_md = gr.Markdown()
    aud1 = gr.Audio(label="Response 1", interactive=False, type="numpy")
    aud2 = gr.Audio(label="Response 2", interactive=False, type="numpy")

    radio    = gr.Radio(label="Which response is better?")
    submit   = gr.Button("✅ Submit")
    status   = gr.Textbox(interactive=False)

    hidden1  = gr.State("")   # model1
    hidden2  = gr.State("")   # model2

    load_btn.click(
        fn=fetch_example,
        inputs=[idx_in],
        outputs=[instr_md, aud1, aud2, radio, hidden1, hidden2],
        api_name="load",
    )

    gr.Markdown("### 📊 Progress")
    progress_df = gr.Dataframe(
        headers=["data_ID", "count"],
        datatype=["number", "number"],
        interactive=False,
    )
    # call once at *every* page-load
    demo.load(
        fn=progress_table_data,   # returns the up-to-date [[id, count], …]
        inputs=None,
        outputs=progress_df,
        queue=False               # send instantly, no queuing needed
    )

    submit.click(
        fn=lambda i, m1, m2, pref, user:
                save_pref(i, m1, m2, pref, user.strip()),
        inputs=[idx_in, hidden1, hidden2, radio, annot_in],
        outputs=[status, progress_df],             # ← matches 2-tuple return
        api_name="submit",
    )

if __name__ == "__main__":
    demo.launch(share=True)