import os
import argparse
import base64
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from gpt4o_audio_api import encode_audio_array_with_resampling

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """You are a helpful assistant. You provide answers to user questions. The question will be in the audio format. Please listen to the question and provide an appropriate response. If users request you to speak in a specific style or tone, please behave accordingly."""

def experiment(
    output_dir,
):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("-----------------------------")

    # load dataset
    dataset = load_dataset("potsawee/chatbot-arena-spoken-style-eval-570")["train"]
    print("len(dataset):", len(dataset))

    ids = [i for i in range(len(dataset))]
    # random.shuffle(ids)

    for i in tqdm(ids):
        x = dataset[i]
        conversation_id = x["id"]

        txt_file = f"{output_dir}/transcript/{conversation_id}.txt"
        wav_file = f"{output_dir}/audio/{conversation_id}.wav"
        # check if the transcript file already exists
        if os.path.exists(txt_file) and os.path.exists(wav_file):
            print(f"Skipping {conversation_id}")
            continue

        question = x['question_refined_wav']
        encoded_audio_question = encode_audio_array_with_resampling(
            question['array'], original_sr=question['sampling_rate'], target_sample_rate=16000)
        
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the user question in the audio format."  
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_question,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=message
        )

        # response = completion.choices[0].message.content ---> this one is empty for audio out      
        
        # save transcript
        transcript = completion.choices[0].message.audio.transcript
        with open(txt_file, "w") as f:
            f.write(transcript)

        # save wav file
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        with open(wav_file, "wb") as f:
            f.write(wav_bytes)
        print("Generated audio:", wav_file)
        print("Transcript:", transcript)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    args = parser.parse_args()
    experiment(args.output_dir)

    # usage: python inference_eval570_gpt.py --output_dir experiments/chatbot-arena-spoken-style-eval-570/gpt4o

if __name__ == "__main__":
    main()