import os
import argparse
import random
import torch
from transformers import AutoModel
from tqdm import tqdm
from datasets import load_dataset
from pydub import AudioSegment
import soundfile as sf

model = AutoModel.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct",
    torch_dtype=torch.float16, 
    trust_remote_code=True
)
model.to("cuda")

system_prompt = """You are a helpful assistant. You provide answers to user questions. The question will be in the audio format. Please listen to the question and provide an appropriate response. If users request you to speak in a specific style or tone, please behave accordingly."""

def experiment(
    output_dir,
    randomize=False,
):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("randomize:", randomize)
    print("type(randomize):", type(randomize))
    print("-----------------------------")

    # load dataset
    dataset = load_dataset("potsawee/chatbot-arena-spoken-style-eval-570")["train"]
    print("len(dataset):", len(dataset))

    ids = [i for i in range(len(dataset))]
    if randomize:
        random.shuffle(ids)

    for i in tqdm(ids):
        x = dataset[i]
        conversation_id = x["id"]
        txt_file = f"{output_dir}/transcript/{conversation_id}.txt"
        wav_file = f"{output_dir}/audio/{conversation_id}.wav"
        # check if the transcript file already exists
        if os.path.exists(txt_file) and os.path.exists(wav_file):
            print(f"Skipping {conversation_id}")
            continue

        # question = x['question_refined_wav']['path'] # {conversation_id}.wav
        path_to_question_wav = f"/data/workspace/ppotsawee/audioLM-as-judge/elevenLabs/refined_questions_kokoro_wav_v1.7/{conversation_id}.wav"

        # check if the sampling rate is 16kHz
        assert AudioSegment.from_wav(path_to_question_wav).frame_rate == 16000
        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": path_to_question_wav,
                    },
                    {"type": "text", "text": "This is the user question in the audio format. Listen and respond to this question."},
                ],
            },
        ]
        output = model.generate(
            conversation=conversation,
            max_new_tokens=500,
            do_sample=True,
            num_beams=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=0.7,
        )

        # save transcript
        transcript = output['text']
        with open(txt_file, "w") as f:
            f.write(transcript)

        # save wav file
        sf.write(wav_file, output["audio"]["array"], output["audio"]["sampling_rate"])

        print("Generated audio:", wav_file)
        print("Transcript:", transcript)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    parser.add_argument("--randomize", action="store_true", help="Randomize the order of the dataset")
    args = parser.parse_args()
    experiment(args.output_dir, args.randomize)

    # usage: python inference_eval570_typhoon2.py --output_dir experiments/chatbot-arena-spoken-style-eval-570/typhoon2 --randomize

if __name__ == "__main__":
    main()