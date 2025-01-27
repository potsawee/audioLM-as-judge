import os
import json
import argparse
from tqdm import tqdm
from gpt4o_audio_api import encode_audio_with_resampling
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# same prompt as SOMOS
prompt_text_2 = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criteria:

1.	Clarity: How clearly the speech is articulated, free from distortion, noise, or artifacts.
2.	Naturalness: The degree to which the speech resembles a natural human voice, including accurate intonation, rhythm, and expressiveness.
3.	Overall Quality: The overall impression of the audio's naturalness and coherence, considering how pleasant and lifelike it sounds.

Follow this step-by-step process for your evaluation:

1.	Listen Carefully: Begin by carefully listening to both Audio A (the first audio) and Audio B (the second audio). Take note of any differences in clarity, fidelity, and overall quality.
2.	Analyze Each Criterion: For each criterion (clarity, naturalness, and overall quality), evaluate how well each audio file performs and provide a brief explanation of your reasoning.
3.	Compare Thoroughly: Summarize the strengths and weaknesses of each audio file based on your analysis.
4.	Decide the Winner: Conclude by determining which audio file is better overall and clearly state your final verdict in this format: [[A]] or [[B]].

Important: Provide a brief summary of your reasoning to justify your decision. Your analysis should be thorough and objective to ensure fairness in the comparison. Your response must start with <explanation> explanation </explanation> and end with <verdict> [[A/B]] </verdict>.

The text for the two audio for you to evaluate is: "###TEXT###"

The two audio snippets to be compared will be provided in the following two user turns."""

def ab_audio_to_message(
    encoded_audio_a, 
    encoded_audio_b, 
    prompt_text,    
):
    message = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is the first audio snippet (audio A)."
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio_a,
                        "format": "wav"
                    }
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is the second audio snippet (audio B)."
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
    return message

def ab_testing(
    encoded_audio_a, 
    encoded_audio_b,
    prompt_text,
):
    message = ab_audio_to_message(encoded_audio_a, encoded_audio_b, prompt_text)
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview-2024-12-17",
        modalities=["text"],
        messages=message
    )
    text_output = completion.choices[0].message.content
    return text_output

def experiment(
    data_path,
    output_path,
    order='ab',
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("order:", order)
    print("-----------------------------")

    with open(data_path) as f:
        data = json.load(f)
    # data = data[:1600]
    # print("----------------------------------------------------------")
    # print("Warning: Only using the first 1600 data points for testing.")
    # print("----------------------------------------------------------")
    print("len(data):", len(data))

    outputs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                x = json.loads(line)
                outputs.append(x)
        num_done = len(outputs)
    else:
        num_done = 0
    print("num_done = {}".format(num_done))

    for i in tqdm(range(num_done, len(data))):
        audio_a, audio_b = data[i]
        assert audio_a["text"] == audio_b["text"]
        text = audio_a["text"]

        encoded_audio_a = encode_audio_with_resampling(audio_a["path"])
        encoded_audio_b = encode_audio_with_resampling(audio_b["path"])

        prompt_text = prompt_text_2
        prompt_text = prompt_text.replace("###TEXT###", text)

        if order == 'ab':
            response = ab_testing(encoded_audio_a, encoded_audio_b, prompt_text)
        elif order == 'ba':
            # BA experiment
            response = ab_testing(encoded_audio_b, encoded_audio_a, prompt_text)
        else:
            raise ValueError(f"Invalid order: {order}")
        
        item = {
            "data_path": data_path,
            "data": data[i],
            "prompt_text": prompt_text,
            "response": response
        }
        print(i, response)
        with open(output_path, 'a') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Run a specific model via gradio_client.")
    parser.add_argument("--data_path", type=str, required=True, help="Specify the model name to run.")
    parser.add_argument("--output_path", type=str, required=True, help="Output Path")
    parser.add_argument("--order", type=str, default='ab', help="Order of the audio files")
    args = parser.parse_args()
    experiment(args.data_path, args.output_path, args.order)
    # python exp1_tmhintqi_gpt_multiturn.py --data_path data/data_tmhintqi_pairwise_diffall.json --output_path experiments/tmhintqi/ab_testing/shuffled_prompt2.jsonl --order ab
    # python exp1_tmhintqi_gpt_multiturn.py --data_path data/data_tmhintqi_pairwise_diffall.json --output_path experiments/tmhintqi/ab_testing/shuffled_prompt2_BA.jsonl --order ba

if __name__ == "__main__":
    main()
