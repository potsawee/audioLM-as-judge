import os
import json
import openai
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_comparison(text):
    """
    Analyze the input text to determine whether it implies:
    - A is better than B
    - B is better than A
    - They are equally good

    :param text: Input text describing the comparison.
    :return: Analysis result as a string.
    """

    # Prepare the prompt
    prompt = f"""
    Analyze the following text and determine the relationship between A and B. Respond with one of the following:
    - A: 'A is better than B'
    - B: 'B is better than A'
    - C: 'A and B are equally good'

    Text: "{text}"

    You must strictly output only one letter, i.e., A, B, or C following the guideline above. Do not include any additional information.
    """

    for i in range(10):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        response = completion.choices[0].message.content.strip()
        if response in ["A", "B", "C"]:
            break
        else:
            print("failed at attempt", i+1)

    # Extract and return the result
    return response

def run(
    input_path,
    output_path,
):
    # Read input text from file
    data = []
    with open(input_path, "r") as f:
        for line in f:
            x = json.loads(line)
            data.append(x)
    print("len(data):", len(data))

    for i in tqdm(range(len(data))):
        x = data[i] # ['data_path', 'data', 'prompt_text', 'response']
        response = x['response']   

        # Get the analysis result
        processed = analyze_comparison(response)
        x['processed'] = processed  

        # print(i, processed)
        with open(output_path, 'a') as f:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Analyze comparison between A and B from input text.")
    parser.add_argument("--input_path", type=str, help="Path to the input file containing text to analyze.")
    parser.add_argument("--output_path", type=str, help="Path to the output file to save the analysis result.")

    args = parser.parse_args()
    run(args.input_path, args.output_path)