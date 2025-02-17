import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

prompt_template = """I'm working on evaluating paralinguistic characteristics of speech generation. This task focuses on non-lexical vocal attributes such as tone, emotion, prosody (rhythm, stress, intonation), speaking style, and accent, which influence the expressiveness and speaker intent. These features play a crucial role in human communication and interaction quality. I'll provide with 82 instructions (ID=0,1,...,81) to you as follows:

{instructions}

Your task is to help me categorize these instructions into 5 classes. If the instruction contains multiple classes, please select the most dominant class. You also help me come up these classes. Please provide the classification in the following format:

{{
    "class_1": ["id1", "id2", ...],
    "class_2": ["id3", "id4", ...],
    "class_3": ["id5", "id6", ...],
    ...
}}

where "id1", "id2", ... are the IDs of the instructions in the class, and "class_1", "class_2", ... are the nmae of classes (e.g., accent, tones, etc). Please strictly follow the format above. Do not include any additional information. Here are the instructions:
""".strip()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run(
    output_path,  
):
    dataset = load_dataset("potsawee/speecheval-advanced-v1")["train"]
    instruction_dict = {}
    for i in range(len(dataset)):
        instruction_dict[f"id{str(i)}"] = dataset[i]["instruction"]
    instruction_dict_str = json.dumps(instruction_dict, indent=4)

    prompt = prompt_template.format(instructions=instruction_dict_str)
    import ipdb; ipdb.set_trace()   
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
    )
    response = completion.choices[0].message.content.strip()
    import ipdb; ipdb.set_trace()
    with open(output_path, "w") as f:
        f.write(response)
    print("hi")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output file to save the analysis results.",
    )
    args = parser.parse_args()
    run(
        output_path=args.output_path,
    )