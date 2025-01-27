import os
import json
import argparse
from tqdm import tqdm
from gpt4o_audio_api import encode_audio_with_resampling
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The instruction used was DataWow annotators
"""
Each subject carefully listens to an utterance and give the scores in three aspects as follows.
1. Sound_quality (Noise level): คุณภาพของไฟล์เสียงว่ามีเสียงรบกวนหรือเสียง noise ต่างๆมากน้อยขนาดไหน
2. Silence: การเว้นจังหวะหายใจระหว่างประโยคและระหว่่างคำ
3. Pronunciation: การออกเสียงในแต่ละคำว่าออกเสียงได้ถูกต้องในระดับไหน

All aspects are assessed from 1 to 5 (The higher, the better).

- Number of human subjects used for listening each utterance: 5-8 subjects
- Evaluation Conditions
    1. The subjects have to be born and raised in Bangkok.
    2. The subjects have to be in quiet place to evaluate the speech audio files.
"""

prompt_text_sound = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criterion:

Sound Quality (Noise Level): Assess the overall quality of the audio file by determining the level of background noise or unwanted sound present in the file. Consider how clean and clear the audio is in terms of absence of noise or distortions.

Follow this step-by-step process for your evaluation:
1. Listen Carefully: Begin by carefully listening to both Audio A (the first audio) and Audio B (the second audio). Take note of any differences in noise levels, distortions, or other unwanted artifacts.
2. Analyze the Criterion: Evaluate how well each audio file performs specifically in terms of noise levels. Provide a brief explanation of your reasoning, mentioning any observed differences.
3. Compare Thoroughly: Summarize the strengths and weaknesses of each audio file based on your analysis.
4. Decide the Winner: Conclude by determining which audio file has better sound quality in terms of noise levels and clearly state your final verdict in this format: [[A]] or [[B]].

Important: Provide a brief summary of your reasoning to justify your decision. Your analysis should be thorough and objective to ensure fairness in the comparison. Your response must start with <explanation> explanation </explanation> and end with <verdict> [[A/B]] </verdict>.

The text for the two audio for you to evaluate is: "###TEXT###"

The two audio snippets to be compared will be provided in the following two user turns."""

prompt_text_rhythm = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criterion:

Silence: Assess the use of pauses in the speech, including the naturalness and appropriateness of breathing intervals between sentences and words. Consider whether the pauses contribute to the flow and coherence of the audio.

Follow this step-by-step process for your evaluation:
1. Listen Carefully: Begin by carefully listening to both Audio A (the first audio) and Audio B (the second audio). Take note of any differences in how silence is managed, including pauses and breathing intervals.
2. Analyze the Criterion: Evaluate how well each audio file performs in terms of silence, providing a brief explanation of your reasoning. Consider factors such as timing, placement, and naturalness of pauses.
3. Compare Thoroughly: Summarize the strengths and weaknesses of each audio file based on your analysis of the silence criterion.
4. Decide the Winner: Conclude by determining which audio file is better overall and clearly state your final verdict in this format: [[A]] or [[B]].

Important: Provide a brief summary of your reasoning to justify your decision. Your analysis should be thorough and objective to ensure fairness in the comparison. Your response must start with <explanation> explanation </explanation> and end with <verdict> [[A/B]] </verdict>.

The text for the two audio for you to evaluate is: "###TEXT###"

The two audio snippets to be compared will be provided in the following two user turns."""

prompt_text_pronunciation = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criterion:

Pronunciation: Assess how accurately each word is pronounced, considering correctness, clarity, and adherence to standard pronunciation norms for the given language.

Follow this step-by-step process for your evaluation:
1. Listen Carefully: Begin by carefully listening to both Audio A (the first audio) and Audio B (the second audio). Take note of any differences in pronunciation accuracy and clarity.
2. Analyze the Criterion: Evaluate how well each audio file performs regarding pronunciation. Provide a brief explanation of your reasoning.
3. Compare Thoroughly: Summarize the strengths and weaknesses of each audio file based on your analysis of pronunciation.
4. Decide the Winner: Conclude by determining which audio file has better pronunciation overall and clearly state your final verdict in this format: [[A]] or [[B]].

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
    criterion='sound'
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("order:", order)
    print("criterion:", criterion)
    print("-----------------------------")

    assert criterion in ['sound', 'rhythm', 'pronunciation']

    with open(data_path) as f:
        data = json.load(f)
    # data = data[:1600]
    # print("----------------------------------------------------------")
    # print("Warning: Only using the first 1500 data points for testing.")
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

        if criterion == 'sound':
            prompt_text = prompt_text_sound
        elif criterion == 'rhythm':
            prompt_text = prompt_text_rhythm
        elif criterion == 'pronunciation':
            prompt_text = prompt_text_pronunciation
        else:
            raise ValueError(f"Invalid criterion: {criterion}")

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
    parser.add_argument("--criterion", type=str, default='sound', help="Criterion")
    args = parser.parse_args()
    experiment(args.data_path, args.output_path, args.order, args.criterion)
    # python exp1_thaimos_gpt_multiturn.py --data_path data/data_thaimos_pairwise_diffall.json --output_path experiments/thaimos/ab_testing/shuffled_prompt2_sound.jsonl --order ab --criterion sound
    # python exp1_thaimos_gpt_multiturn.py --data_path data/data_thaimos_pairwise_diffall.json --output_path experiments/thaimos/ab_testing/shuffled_prompt2_rhythm.jsonl --order ab --criterion rhythm
    # python exp1_thaimos_gpt_multiturn.py --data_path data/data_thaimos_pairwise_diffall.json --output_path experiments/thaimos/ab_testing/shuffled_prompt2_pronunciation.jsonl --order ab --criterion pronunciation
if __name__ == "__main__":
    main()
