import os
import argparse
import json
from tqdm import tqdm
from glob import glob
import numpy as np
import soundfile as sf
from google import genai


def experiment(
    input_dir: str,
    text_dir: str,
):
    print("-------------------------------------------")
    print("input_dir:", input_dir)
    print("text_dir:", text_dir)
    print("-------------------------------------------")
    
    client = genai.Client()
    wav_paths = sorted(glob(f"{input_dir}/*.wav"))
    for wav_path in wav_paths:
        id = int(wav_path.replace(input_dir, "").split(".")[0].strip("/"))
        txt_path = f"{text_dir}/{id}.txt"
        # check if the transcript file already exists
        if os.path.exists(txt_path):
            print(f"Skipping {id}")
            continue

        myfile = client.files.upload(file=wav_path)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                'Generate a transcript of the speech. Provide only the text transcript and no other information. Please transcribe the speech exactly as spoken. Do not attempt to answer the question or provide any additional information.',
                myfile,
            ]
        )
        print(id, response.text.strip())
        with open(txt_path, "w") as f:
            f.write(response.text.strip())
    print("Done.")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", 
        type=str, 
        help="Path to the input dir containing files to transcribe. (*.wav files)"
    )
    parser.add_argument("--text_dir", 
        type=str, 
        help="Path to the output file to save the ASR results. (*.txt files)"
    )
    args = parser.parse_args()

    num_try = 10
    for n in range(num_try):
        try:
            experiment(args.input_dir, args.text_dir)
            break
        except genai.errors.ClientError as e:
            print(f"Error: {e}")
            print(f"Retrying... {n+1}/{num_try}")

    # usage: python asr_gemini_generation.py --input_dir experiments/advvoiceq1/gemini2flash-exp/audio --text_dir experiments/advvoiceq1/gemini2flash-exp/audio_transcript

    # usage: python asr_gemini_generation.py --input_dir ../advanced-voice-gen-task-v1/questions1_kokoro_wav --text_dir  experiments/advvoiceq1/asr_gemini/transcript
