import os
import json
import pandas as pd
import argparse
import random
import base64
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

audio_generation_template = "Say the following text (between <text> </text>) using the '###style###' speaking style. You must not add or remove any word, just say these words exactly as written using the '###style###' style: <text> ###text### </text>"

def main(
    output_dir: str,
    split: str,
):
    print("Output directory:", output_dir)
    print("Split ID:", split)

    # Load the data
    path = "/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english.json"
    with open(path) as f:
        data = json.load(f)

    # the conversation
    conversation_style = pd.read_csv("batch1.1.tsv", sep="\t")

    for i in tqdm(range(len(conversation_style))):
        x = conversation_style.iloc[i]
        style = x["style"]
        conversation_id = x[split]
        data_i = data[conversation_id]

        # check if the wav_a file already exists
        if os.path.exists(f"{output_dir}/{conversation_id}_question.{style}.wav") and os.path.exists(f"{output_dir}/{conversation_id}_question.natural.wav"):
            print(f"Skipping {i}")
            continue

        tts_prompt_style = audio_generation_template.replace("###style###", style).replace("###text###", data_i["conversation_a"][0]["content"])
        tts_prompt_natural = audio_generation_template.replace("###style###", "natural").replace("###text###", data_i["conversation_a"][0]["content"])
        
        completion_style = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": tts_prompt_style
                }
            ]
        )
        wav_bytes_style = base64.b64decode(completion_style.choices[0].message.audio.data)
        wav_file_style = f"{output_dir}/{conversation_id}_question.{style}.wav"
        with open(wav_file_style, "wb") as f:
            f.write(wav_bytes_style)
        print("Generated audio for question with style:", wav_file_style)

        completion_natural = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": tts_prompt_natural
                }
            ]
        )
        wav_bytes_natural = base64.b64decode(completion_natural.choices[0].message.audio.data)
        wav_file_natural = f"{output_dir}/{conversation_id}_question.natural.wav"
        with open(wav_file_natural, "wb") as f:
            f.write(wav_bytes_natural)
        print("Generated audio for question (natural):", wav_file_natural)
        print("----------------------------------------------")

if __name__ == "__main__":
    # argument parser setup
    parser = argparse.ArgumentParser(description="Generate audio using GPT4o.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--split", default="ID1", type=str, help="Split ID")
    args = parser.parse_args()
    main(args.output_dir, args.split)

    # usage: python gpt4o_audio_generation_question.py --output_dir batch1_generated_gpt_audio_question --split ID1