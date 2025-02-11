import os
import json
import argparse
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from gpt4o_audio_api import encode_audio_array_with_resampling, encode_audio_with_resampling

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/alpaca_eval.txt

task_prompt = """Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:

{
    "rankings":
    [
        {"model": "<model-name>", "rank": "<model-rank>"},
        {"model": "<model-name>", "rank": "<model-rank>"}
    ]
}

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""

def message_builder(
    encoded_audio_question,
    encoded_audio_responseA,
    encoded_audio_responseB,
    task_prompt,
):

    message = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant, that ranks models by the quality of their answers."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I want you to create a leaderboard of different of audio capable large-language models. To do so, I will give you the instructions (in an audio format) given to the models, and the responses of two models (also in an audio format). Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the instruction (in the audio format)."  
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio_question,
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
                    # GPT4o is quite overfitting to 'model_1'. When swapped to model_a, it didn't work
                    "text": "This is model_1 output in the audio format."  
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio_responseA,
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
                    # GPT4o is quite overfitting to 'model_2'. When swapped to model_b, it didn't work
                    "text": "This is model_2 output in the audio format."  
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio_responseB,
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
                    "text": task_prompt
                }
            ]
        }
    ]
    return message

def experiment(
    model_output_dir,
    judge_output_dir,
    randomize=False,
):
    print("-----------------------------")
    print("model_output_dir:", model_output_dir)
    print("judge_output_dir:", judge_output_dir)
    print("randomize:", randomize)
    print("type(randomize):", type(randomize))
    print("-----------------------------")

    # Load the dataset
    dataset = load_dataset("potsawee/chatbot-arena-spoken-style-eval-570")["train"]
    print("len(dataset):", len(dataset))

    ids = [i for i in range(len(dataset))]
    if randomize:
        random.shuffle(ids)

    winner_ref, winner_can = 0, 0
    for i in tqdm(ids):
        x = dataset[i]
        conversation_id = x["id"]
        judge_output_path = f"{judge_output_dir}/{conversation_id}.txt"
        ref_model_type = x["model_type"]
        # check if judge_output_path already exists
        if os.path.exists(judge_output_path):
            print(f"Skipping {judge_output_path}")
            continue

        # question
        question = x['question_refined_wav']
        encoded_audio_question = encode_audio_array_with_resampling(
            question['array'], original_sr=question['sampling_rate'], target_sample_rate=16000)

        # ref model
        ref_model = x["assistant_wav"]
        # candidate model
        candidate_wav_path = f"{model_output_dir}/{conversation_id}.wav"

        if ref_model_type == "model_a":
            encoded_audio_responseA = encode_audio_array_with_resampling(
                ref_model['array'], original_sr=ref_model['sampling_rate'], target_sample_rate=16000)
            encoded_audio_responseB = encode_audio_with_resampling(candidate_wav_path, target_sample_rate=16000)
        elif ref_model_type == "model_b":
            encoded_audio_responseB = encode_audio_array_with_resampling(
                ref_model['array'], original_sr=ref_model['sampling_rate'], target_sample_rate=16000)
            encoded_audio_responseA = encode_audio_with_resampling(candidate_wav_path, target_sample_rate=16000)
        else:
            raise ValueError("ref_model_type must be 'model_a' or 'model_b'")

        try:
            message = message_builder(encoded_audio_question,encoded_audio_responseA, encoded_audio_responseB, task_prompt)
            # import ipdb; ipdb.set_trace()
            # Send request to GPT-4o for A/B testing
            completion = client.chat.completions.create(
                model="gpt-4o-audio-preview-2024-12-17",
                modalities=["text"],
                messages=message
            )
            # Extract and return the A/B testing decision from the response
            response = completion.choices[0].message.content
            response = response.strip("```").strip("json").strip("python").strip()
            item = {
                "id": conversation_id,
                "original_id": x['original_id'],
                "response": response,
                "ref_model_type": ref_model_type,
            }
            print(i, response)

            # calculate winner ratio
            parsed = json.loads(response)
            yy = parsed['rankings'][0]
            if yy['model'] == 'model_1':
                if yy['rank'] == 1:
                    if ref_model_type == 'model_a':
                        winner_ref += 1
                    else:
                        winner_can += 1
                else:
                    if ref_model_type == 'model_a':
                        winner_can += 1
                    else:
                        winner_ref += 1
            elif yy['model'] == 'model_2':
                if yy['rank'] == 1:
                    if ref_model_type == 'model_b':
                        winner_ref += 1
                    else:
                        winner_can += 1
                else:
                    if ref_model_type == 'model_b':
                        winner_can += 1
                    else:
                        winner_ref += 1
            print("candidate winner percentage: {:.2f}%".format(winner_can / (winner_can + winner_ref) * 100))
            with open(judge_output_path, 'w') as f:
                json.dump(item, f, indent=4)
        except Exception as e:
            print("i:", i)
            print("conversation_id:", conversation_id)
            print("error:", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_dir", type=str, required=True)
    parser.add_argument("--judge_output_dir", type=str, required=True)
    parser.add_argument("--randomize", type=bool, default=False)
    args = parser.parse_args()

    if args.randomize == "True":
        args.randomize = True
    elif args.randomize == "False":
        args.randomize = False
    experiment(args.model_output_dir, args.judge_output_dir, args.randomize)

    # usage: python evaluate_eval570.py --model_output_dir experiments/chatbot-arena-spoken-style-eval-570/gpt4o/audio/ --judge_output_dir judge_outputs/eval570_judge_gpt4o_candidate_gpt4o --randomize True

    # usage: python evaluate_eval570.py --model_output_dir experiments/chatbot-arena-spoken-style-eval-570/gpt4o/transcript_kokoro_wav --judge_output_dir judge_outputs/eval570_judge_gpt4o_candidate_4okokoro --randomize True

if __name__ == "__main__":
    main()