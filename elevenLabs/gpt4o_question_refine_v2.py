import os
import json
import pandas as pd
import argparse
import random
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """Your task is to check if questions are valid instead of being the result of my assistance's rejection like 'I'm sorry, I can't assist with that request.'. If a question is valid, please answer 'VALID' else answer 'INVALID'. Let me give you some example:

[Example 1]
Question: Could you tell me who the first German woman in space was, speaking with a quiet but intriguingly menacing tone, laced with a bit of eccentric unpredictability?
Answer: VALID

[Example 2]
Question: I'm sorry, I can't assist with that request.
Answer: INVALID

When you give me your answer, please don't include anything else at all, just answer 'VALID' or 'INVALID', otherwise my parser will fail."""

prompt_template = """Question: {original_question}\nAnswer:"""

def main(
    output_dir: str,
):
    print("Output directory:", output_dir)

    # input_path = "./data-chatbot-arena-spoken-style-11labs"
    # dataset = load_dataset("potsawee/chatbot-arena-spoken-style-11labs")["train"]
    # dataset = dataset.select_columns(["question_refined", "id"])

    input_path = "./data-chatbot-arena-spoken-style-11labs"
    dataset = load_from_disk(input_path)
    print("len(dataset):", len(dataset))

    ids = [i for i in range(len(dataset))]
    random.shuffle(ids)

    for i in tqdm(ids):
        x = dataset[i]
        conversation_id = x["id"]

        # refine v1
        # question_refined = x["question_refined"]

        # refine v1.5, v1.5
        path = f"refined_questions_v1.7/{conversation_id}.refined_question.txt"
        with open(path, "r") as f:
            question_refined = f.read().strip('"').strip()

        output_path = f"{output_dir}/{conversation_id}.verdict.txt"
        # check if the refined question file already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_path}")
            continue

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt_template.format(original_question=question_refined)
            }
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        response = completion.choices[0].message
        response = response.content
        assert response in ["VALID", "INVALID"]

        print("conversation_id:", conversation_id)
        print("Response:", response)
        print("-------------------------------------")
        with open(output_path, "w") as f:
            f.write(response)

if __name__ == "__main__":
    # refine using elevenLabs style
    parser = argparse.ArgumentParser(description="Generate audio using GPT4o.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")

    args = parser.parse_args()
    main(args.output_dir)

    # usage: python gpt4o_question_refine_v2.py --output_dir refined_questions_v1.7/verdict