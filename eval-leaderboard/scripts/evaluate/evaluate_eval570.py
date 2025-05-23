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

task_prompt = """Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format. 

After providing your explaination, please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks. You must generate the response following this format exactly:

[Explaination]
your explaination

[Rankings]
{
    "rankings":
    [
        {"model": "<model-name>", "rank": "<model-rank>"},
        {"model": "<model-name>", "rank": "<model-rank>"}
    ]
}

Your response after [Rankings] must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""

main_instruction = """
I want you to act as an evaluator of speech outputs produced by different audio-capable large language models (Audio LLMs). You will be provided with:
1.	A user's instructions (in audio format) specifying what the response should contain and any desired speaking style or tone (e.g., whispering, angry, cheerful, etc.).
2.	The audio responses of two models generated according to the user's instructions.

Your task is to compare and rank the models based on the following evaluation criteria:
1.	Content Quality (Semantics):
- How well does the content of the response satisfy the user's instructions in terms of meaning, coherence, relevance, and accuracy?
2.	Voice Quality (Speaking Style):
- How well does the model's generated voice match the style, tone, and delivery specified by the user's instructions?

Rank the models as follows:
- Provide a leaderboard of the evaluated models, placing the model with the best balance of content and voice quality at the top.
- Explain your ranking with brief reasoning for each model (mentioning any notable strengths or weaknesses in content and style).""".strip()

def message_builder(
    encoded_audio_question,
    encoded_audio_responseA,
    encoded_audio_responseB,
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
                    "text": main_instruction,
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

    mapping = {
        'model_a': 'model_1',
        'model_b': 'model_2',
    }

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
            message = message_builder(encoded_audio_question,encoded_audio_responseA, encoded_audio_responseB)
            # import ipdb; ipdb.set_trace()
            # Send request to GPT-4o for A/B testing
            completion = client.chat.completions.create(
                model="gpt-4o-audio-preview-2024-12-17",
                modalities=["text"],
                messages=message
            )
            # Extract and return the A/B testing decision from the response
            response = completion.choices[0].message.content
            # response = response.strip("```").strip("json").strip("python").strip()
            item = {
                "id": conversation_id,
                "original_id": x['original_id'],
                "response": response,
                "ref_model_type": ref_model_type,
            }
            print(i, response)

            # calculate winner ratio
            verdict = response.split("[Rankings]")[-1].strip("```").strip("json").strip("python").strip()
            parsed = json.loads(verdict)
            yy = parsed['rankings'][0]

            ref_model_type_mapped = mapping[ref_model_type]

            if int(yy['rank']) == 1:
                if yy['model'] == 'model_1' and ref_model_type_mapped == 'model_1':
                    winner_ref += 1
                elif yy['model'] == 'model_1' and ref_model_type_mapped == 'model_2':
                    winner_can += 1
                elif yy['model'] == 'model_2' and ref_model_type_mapped == 'model_1':
                    winner_can += 1
                elif yy['model'] == 'model_2' and ref_model_type_mapped == 'model_2':
                    winner_ref += 1
                else:
                    raise Exception()
            else:
                raise Exception()

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

    # usage: python evaluate_eval570.py --model_output_dir experiments/chatbot-arena-spoken-style-eval-570/gpt4o/audio/ --judge_output_dir judge_outputs/eval570_judge_gpt4o_candidate_gpt4o --randomize False

    # usage: python evaluate_eval570.py --model_output_dir experiments/chatbot-arena-spoken-style-eval-570/gpt4o/transcript_kokoro_wav --judge_output_dir judge_outputs/eval570_judge_gpt4o_candidate_4okokoro --randomize True

    # usage: python evaluate_eval570.py --model_output_dir experiments/chatbot-arena-spoken-style-eval-570/typhoon2/audio/ --judge_output_dir judge_outputs/eval570_judge_gpt4o_candidate_typhoon2 --randomize False

    # usage: python evaluate_eval570.py --model_output_dir experiments/chatbot-arena-spoken-style-eval-570/typhoon2/transcript_kokoro_wav/ --judge_output_dir judge_outputs/eval570_judge_gpt4o_candidate_typ2kokoro --randomize True


if __name__ == "__main__":
    main()