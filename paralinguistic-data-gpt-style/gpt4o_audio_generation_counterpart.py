import os
import json
import pandas as pd
import argparse
import random
import base64
from openai import OpenAI
from tqdm import tqdm
from glob import glob

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

audio_generation_template = "Say the following text (between <text> </text>) using the 'natural/normal' speaking style. You must not add or remove any word, just say these words exactly as written in a norma speaking style: <text> ###text### </text>"

def main(
    output_dir: str,
):
    print("Output directory:", output_dir)

    # Load the data
    path = "/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english.json"
    with open(path) as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        conversation_id = i
        data_i = data[i]
        wav_paths = glob(f"batch1_generated_gpt_audio/{conversation_id}_model_*.*.wav")
        if len(wav_paths) == 0:
            continue
        elif len(wav_paths) == 2:
            if "natural" in wav_paths[0]:
                wav_path = wav_paths[1]
            else:
                wav_path = wav_paths[0]
        else:
            raise ValueError(f"Invalid number of wav files: {len(wav_paths)}, {wav_paths}")
        
        y = wav_path.split(".") # ['batch1_generated_gpt_audio/202_model_a', 'disgusted', 'wav']
        file_name = y[0].split("/")[-1]
        natural_wav_path = f"{output_dir}/{file_name}.natural.wav"
        # check if the natural wav file already exists
        if os.path.exists(natural_wav_path):
            print(f"Skipping {i}")
            continue

        if "model_a" in file_name:
            conversation = "conversation_a"
        elif "model_b" in file_name:
            conversation = "conversation_b"
        else:
            raise ValueError(f"Invalid file name: {file_name}")
        
        response = data_i[conversation][1]['content']
        tts_prompt = audio_generation_template.replace("###text###", response)


        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": tts_prompt
                }
            ]
        )
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        with open(natural_wav_path, "wb") as f:
            f.write(wav_bytes)
        print("Generated audio for natural tone:", natural_wav_path)

if __name__ == "__main__":
    # argument parser setup
    parser = argparse.ArgumentParser(description="Generate audio using GPT4o.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()
    main(args.output_dir)

    # usage: python gpt4o_audio_generation_counterpart.py --output_dir batch1_generated_gpt_audio/counterpart_natural