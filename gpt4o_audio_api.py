import base64
from openai import OpenAI
import json
import pdb
import glob
from tqdm import tqdm
import pandas as pd
import os


# Add your OpenAI API key here
api_key = ""
# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

# Path to your local audio file
#local_audio_file_path = "azure/sentence_1.wav"



def get_encoded_audio(filepath):
    with open(filepath, "rb") as audio_file:
        wav_data = audio_file.read()
    encoded_string = base64.b64encode(wav_data).decode('utf-8')
    return encoded_string

def gpt4o_mos(local_audio_file_path):
    # Read and encode the local audio file
    with open(local_audio_file_path, "rb") as audio_file:
        wav_data = audio_file.read()
    encoded_string = base64.b64encode(wav_data).decode('utf-8')

    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview-2024-12-17",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Evaluate the quality of this audio and assign a score between 1.0 and 5.0 (inclusive). Focus on clarity, fidelity, and overall listening experience. The output must be strictly in floating-point format, for example: [[4.5]]."
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]
        }
        ]

    )
    response_text=completion.choices[0].message
    return (response_text.audio.transcript.replace('[', '').replace(']', ''))


def ab_testing(a_local_audio_file_path, b_local_audio_file_path):
    # Read and encode the local audio file
    encoded_audio_a = get_encoded_audio(a_local_audio_file_path)
    encoded_audio_b = get_encoded_audio(b_local_audio_file_path)
    completion = client.chat.completions.create(
    model="gpt-4o-audio-preview-2024-12-17",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert in audio quality assessment. "
                        "Your task is to evaluate two audio files (Audio A and Audio B) and determine which one has better quality. "
                        "Focus specifically on clarity, fidelity, and overall listening experience. "
                        "Make a clear decision based on these criteria, even if the difference is subtle. "
                        "Your response must strictly be one of the following formats: [[Audio A]] or [[Audio B]]. "
                        "If you are unsure, evaluate to the best of your ability and select the file you believe is superior."
                    )
                
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio_a,
                        "format": "wav"
                    }
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio_b,
                        "format": "wav"
                    }
                }
            ]
        }
    ]
    )
    response_text=completion.choices[0].message
    return (response_text.audio.transcript)
