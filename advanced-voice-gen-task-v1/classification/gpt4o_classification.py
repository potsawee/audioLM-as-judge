import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

prompt_template = """I'm working on evaluating paralinguistic characteristics of speech generation. This task focuses on non-lexical vocal attributes such as tone, emotion, prosody (rhythm, stress, intonation), speaking style, and accent, which influence the expressiveness and speaker intent. These features play a crucial role in human communication and interaction quality. I've categorised the paralinguistic characteristics into the following classes:

- "accent_pronunciation": The intended speech response must contain accent features, such as British, American, Singaporean, or Indian accent, or it requires specific pronunciation such as those in tonal languages.
- "emotion_style": The intended speech response must contain emotional state, such as happy, sad, angry, or neutral.
- "prosody_rhythm": The intended speech response must contain prosodic features, such as pitch, stress, and rhythm (speak fast or slow) patterns.
- "sound_effects": The intended speech response must contain sound effects, such as laughter, crying, or immitating natural sounds like animals or objects.
- "language

- "tone": The intended speech response must contain emotional state, such as happy, sad, angry, or neutral.
- "prosody": The intended speech response must contain prosodic features, such as pitch, stress, and rhythm (speak fast or slow) patterns.
- "style": The intended speech response must contain speaking style, such as whispering, shouting, storytelling.
- "accent": The intended speech response must contain accent features, such as British, American, Singaporean, or Indian accent.
- "expressiveness": The intended speech response must contain expressiveness features, such as monotone, lively, or enthusiastic.

I'll provide an speech instruction to you, your task is to classifity the instruction into one of the above classes. If the instruction contains multiple classes, please select the most dominant class. If the instruction does not contain any of the above classes, please select "others". Please provide the classification in the following format:

{{
    "class": "xxx"
}}

where "xxx" is the class you selected. Please strictly follow the format above. Do not include any additional information. Here is the instruction:

Instruction: "{instruction}"
""".strip()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run(
    output_dir,  
):
    dataset = load_dataset("potsawee/speecheval-advanced-v1")["train"]
    categories = ["others", "tone", "prosody", "style", "accent", "expressiveness"]
    count = {category: 0 for category in categories}    
    for i in tqdm(range(len(dataset))):
        output_path = f"{output_dir}/{i}.txt"
        # check if the file already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_path}")
            continue

        instruction = dataset[i]["instruction"]
        prompt = prompt_template.format(instruction=instruction)
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
        response = response.strip('```').strip('json').strip()
        try:
            parsed = json.loads(response)
            category = parsed["class"]
            assert category in categories
        except:
            import ipdb; ipdb.set_trace()
        count[category] += 1
        print(count)
        with open(output_path, "w") as f:
            f.write(category)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output file to save the analysis results.",
    )
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
    )