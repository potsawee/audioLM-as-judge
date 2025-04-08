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
    "class_1": 
        {{
            "description": "xxx",
            "items": ["id1", "id2", ...]
        }},
    "class_2": 
        {{
            "description": "xxx",
            "items": ["id3", "id4", ...]
        }}
    "class_3": 
        {{
            "description": "xxx",
            "items": ["id5", "id6", ...]
        }}
    ...
}}

where "id1", "id2", ... are the IDs of the instructions in the class, "xxx" is a detailed description of the class, and "class_1", "class_2", ... are the name of classes (e.g., the actual value of "class_1" could be Accent abd Pronunciation Variations). First, provide some planning on how many classes you would like to create and what they are. Then, analyze each instruction one by one. Finally provide the classification in the format above, and note that please use [[Start Final Prediction]] to indicate that you are ready to provide the final classification.:
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
    completion = client.chat.completions.create(
        model="o1-2024-12-17",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        # temperature=0.7,
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