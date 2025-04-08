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

style_winners = ['style_winner_a', 'style_winner_b']

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
        if os.path.exists(f"{output_dir}/{conversation_id}_model_a.{style}.wav") or os.path.exists(f"{output_dir}/{conversation_id}_model_a.natural.wav") or os.path.exists(f"{output_dir}/{conversation_id}_model_b.{style}.wav") or os.path.exists(f"{output_dir}/{conversation_id}_model_b.natural.wav"):
            print(f"Skipping {i}")
            continue



        style_winner = random.choice(style_winners)
        print("Style winner:", style_winner)

        if style_winner == "style_winner_a":
            model_a_style = style
            model_b_style = "natural"
        elif style_winner == "style_winner_b":
            model_a_style = "natural"
            model_b_style = style
        else:
            raise ValueError(f"Invalid style_winner: {style_winner}")
        
        tts_prompt_model_a = audio_generation_template.replace("###style###", model_a_style).replace("###text###", data_i["conversation_a"][1]["content"])
        tts_prompt_model_b = audio_generation_template.replace("###style###", model_b_style).replace("###text###", data_i["conversation_b"][1]["content"])
        
        # Generate audio for model A
        completion_model_a = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": tts_prompt_model_a
                }
            ]
        )
        wav_bytes_a = base64.b64decode(completion_model_a.choices[0].message.audio.data)
        wav_file_a = f"{output_dir}/{conversation_id}_model_a.{model_a_style}.wav"
        with open(wav_file_a, "wb") as f:
            f.write(wav_bytes_a)
        print("Generated audio for model A:", wav_file_a)

        # Generate audio for model B
        completion_model_b = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": tts_prompt_model_b
                }
            ]
        )
        wav_bytes_b = base64.b64decode(completion_model_b.choices[0].message.audio.data)
        wav_file_b = f"{output_dir}/{conversation_id}_model_b.{model_b_style}.wav"
        with open(wav_file_b, "wb") as f:
            f.write(wav_bytes_b)
        print("Generated audio for model B:", wav_file_b)
        print("----------------------------------------------")

if __name__ == "__main__":
    # argument parser setup
    parser = argparse.ArgumentParser(description="Generate audio using GPT4o.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--split", default="ID1", type=str, help="Split ID")
    args = parser.parse_args()
    main(args.output_dir, args.split)

    # usage: python gpt4o_audio_generation.py --output_dir batch1_generated_gpt_audio --split ID1