import os
import json
import pandas as pd
import argparse
import random
from glob import glob
from tqdm import tqdm
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """Your task is to refine the following question such that it can be answered in a particular speaking style. You must keep the original meaning of the question, but you can change the wording as needed to make it. I'll provide a few examples for you below.

[Example 1]
Original question: "Write a bedtime story for children"
Target speaking style of answer: "Whispering"
Refined question: "Tell a bedtime story for children in a whispering voice"

[Example 2]
Original question: "What are the benefits of eating fruits?"
Target speaking style of answer: "Indian accent"
Refined question: "What are the benefits of eating fruits, please speak in an Indian accent?"

[Example 3]
Original question: "ok so i missed doomer. what's the next big thing that will make me rich?"
Target speaking style of answer: "Angry"
Refined question: "I missed doomer, what's the next big thing that will make me rich? Imagine you're angry when you answer."

Please try to be creative when refining the question, but don't change the original meaning. You can only add how you expect the answer to be spoken in the refined question. Please get some diversity in your refinement strategy. Also, please note that the question should be in a spoken format, so if the question contains phrase like 'write' or anything that cannot be said, please map them into a spoken word. Also, remember that the speaking style is meant for the answer to the question, and not the refined question itself. You just give me the refined question, without other information. I'll take care of the rest."""

def get_user_prompt(original_question: str, speaking_style: str) -> str:
    line = f"""Original question: "{original_question}"\n"""
    line += f"""Target speaking style of answer: "{speaking_style}" """
    line += f"""Refined question: "{speaking_style}" """
    return line

def main(
    wav_dir: str,
    output_dir: str,
    reset_every: int
):
    print("Output directory:", output_dir)

    # Load the data
    path = "/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english.json"
    with open(path) as f:
        data = json.load(f)

    wav_paths = glob(f"{wav_dir}/*.wav")
    random.shuffle(wav_paths)

    for i, wav_path in tqdm(enumerate(wav_paths)):
        if i % reset_every == 0:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]

        wav_file = wav_path.replace(wav_dir, "")
        wav_file = wav_file.replace(".wav", "")
        style = wav_file.split(".")[-1]
        
        if style == "natural":
            print("Skipping natural style")
            continue

        conversation_id = int(wav_file.split("_")[0])
        refined_question_path = f"{output_dir}/{conversation_id}.refined_question.txt"
        # check if the refined question file already exists
        if os.path.exists(refined_question_path):
            print(f"Skipping {refined_question_path}")
            continue

        data_i = data[conversation_id]
        assert data_i['conversation_a'][0]['content'] == data_i['conversation_b'][0]['content']
        original_question = data_i['conversation_a'][0]['content']

        user_turn = get_user_prompt(original_question, style)
        messages.append({
            "role": "user",
            "content": user_turn
        })

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        response = completion.choices[0].message
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        print("conversation_id:", conversation_id)
        print("Original question:", original_question)
        print("Style:", style)
        print("Refined question:", response.content)
        print("-------------------------------------")
        with open(refined_question_path, "w") as f:
            f.write(response.content)

if __name__ == "__main__":
    # argument parser setup
    parser = argparse.ArgumentParser(description="Generate audio using GPT4o.")
    parser.add_argument("--wav_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--reset_every", type=int, default=50)

    args = parser.parse_args()
    main(args.wav_dir, args.output_dir, args.reset_every)
    # python gpt4o_question_refine.py --wav_dir batch1_generated_gpt_audio/ --output_dir refined_questions --reset_every 50