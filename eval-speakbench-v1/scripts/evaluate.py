import os
import json
import argparse
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from scripts.gpt4o_audio_api import encode_audio_array_with_resampling

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/alpaca_eval.txt

# prompt1
main_instruction = """
I want you to act as an evaluator of audio outputs produced by different audio-capable large language models (Audio LLMs). You will be provided with:
1. A user's instruction specifying what the response should contain and any desired speaking style, tone, emotion, pacing, and rhythm.
2. The audio responses from two models, generated according to the user's instruction.

Please note that typically you can compare and rank the models based on these evaluation criteria:
1. Semantics:
- Does the content accurately and coherently fulfill the user's request?

2. Paralingusitics:
- How well does the generated speech match the requested tone, emotion, speaking style, pacing, rhythm, and expressiveness?
- Does the model produce natural and dynamic speech, or does it sound robotic, monotonous, or lacking in prosody?
- Does it exhibit realistic variations in intonation, stress, and articulation according to the user's intent?

Important Consideration:
- Do not favor verbalized descriptions of tone or speaking style over actual tonal expression. A model that simply states "I am speaking angrily" but delivers a flat voice should be ranked lower than a model that correctly expresses a angry tone in speech.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Model comparison stage will be provided in the subsequent steps. Please follow the instructions carefully and evaluate the models objectively.
""".strip()


task_prompt = """Please perform 'voice quality evaluation' focusing on the voice characteristics of generated speech responses. Check the voice quality of the first response (model_1) and the second response (model_2). Analyze the voice quality of each response based on the following criteria: How well does the generated speech match the requested tone, emotion, speaking style, pacing, prosody? Does it exhibit realistic variations in intonation, stress, and articulation according to the user's intent? Please perform evaluation step by step as follows:

[Step 1 - Analyze the voice characteristics to be examined]
Analyze the key voice characteristics that the user requested in the instruction. This could be anything from tone, emotion, speaking style, pacing, rhythm, prosody, etc.

[Step 2 - Check the first response (model_1)] 
Check the first response (model_1) on the key characteristics identified in Step 2.1? Also check whether the model verbalize of the voice characteristics without exhibiting the actual characteristics in the speech. If this happens, the model might try to trick you into thinking that it can exhibit that speech characteristics, please don't fall into it and penalize that model in the voice quality evaluation. Note that performing verbalization 'in addition to' performing the characteristics is acceptable. Please be very careful about this and provide an explanation.

[Step 3 - Check the second response (model_2)] 
Check the second response (model_2) on the key characteristics identified in Step 2.1? Also check whether the model verbalize of the voice characteristics without exhibiting the actual characteristics in the speech. If this happens, the model might try to trick you into thinking that it can exhibit that speech characteristics, please don't fall into it and penalize that model in the voice quality evaluation. Note that performing verbalization 'in addition to' performing the characteristics is acceptable. Please be very careful about this and provide an explanation.

[Step 4 - Final Step]
After providing your explaination, please compare the models by the quality of their answers. Do not allow the length of the responses to influence your evaluation, and lengthy but unnecessarily details could distract users from the main task. Then return your verdict about which model is the winner (model_1, model_2, tie). You must generate the response following this format exactly:

[Verdict]
{
    "verdict": "<verdict>"
}

There are four possible values for the verdict:
- "model_1": Model 1 is better
- "model_2": Model 2 is better
- "tie": Both models are equally good or bad

Plesae use "tie" sparingly, and only when you absoultely cannot choose the winner.

Your response after [Verdict] must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the verdict that the majority of humans would give."""


def message_builder_multiturn(
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

def message_builder_singleturn(
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
                },
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
                },
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
                },
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
                },
                {
                    "type": "text",
                    "text": task_prompt
                }
            ]
        }
    ]
    return message


def message_builder(
    encoded_audio_question,
    encoded_audio_responseA,
    encoded_audio_responseB,
    message_type
):
    if message_type == "multiturn":
        message = message_builder_multiturn(
            encoded_audio_question,
            encoded_audio_responseA,
            encoded_audio_responseB,
        )
    elif message_type == "singleturn":
        message = message_builder_singleturn(
            encoded_audio_question,
            encoded_audio_responseA,
            encoded_audio_responseB,
        )
    else:
        raise NotImplementedError(f"message_type: {message_type} is not implemented")

    return message

def experiment(
    message_type,
    output_dir,
    randomize=False,
):
    print("-----------------------------")
    print("message_type:", message_type)
    print("output_dir:", output_dir)
    print("randomize:", randomize)
    print("-----------------------------")


    # Load the dataset
    dataset = load_dataset("potsawee/speakbench-v1-label")["train"]
    print("len(dataset):", len(dataset))

    ids = [i for i in range(len(dataset))]
    if randomize:
        random.shuffle(ids)

    label_mapping = {
        'a': 'model_1',
        'b': 'model_2',
        'both_good': 'tie',
        'both_bad': 'tie',
    }
    count_correct, count_incorrect = 0, 0

    for i in tqdm(ids):
        row = dataset[i]
        row_id = row["idx"]
        output_path = f"{output_dir}/{row_id}.txt"

        # check if judge_output_path already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_path}")
            continue

        
        # question
        question = row['instruction']
        encoded_audio_question = encode_audio_array_with_resampling(
            question['array'], original_sr=question['sampling_rate'], target_sample_rate=16000)

        # responseA
        audio_a = row['audio_a']
        encoded_audio_responseA = encode_audio_array_with_resampling(
            audio_a['array'], original_sr=audio_a['sampling_rate'], target_sample_rate=16000)
        
        # responseB
        audio_b = row['audio_b']
        encoded_audio_responseB = encode_audio_array_with_resampling(
            audio_b['array'], original_sr=audio_b['sampling_rate'], target_sample_rate=16000)
    

        try:
            message = message_builder(encoded_audio_question,encoded_audio_responseA, encoded_audio_responseB, message_type)

            # import ipdb; ipdb.set_trace()
            # Send request to GPT-4o for A/B testing
            completion = client.chat.completions.create(
                model="gpt-4o-audio-preview-2024-12-17",
                modalities=["text"],
                messages=message,
                temperature=0.0,
            )
            # Extract and return the A/B testing decision from the response
            response = completion.choices[0].message.content
            # response = response.strip("```").strip("json").strip("python").strip()
            item = {
                "id": row_id,
                "response": response,
            }
            print(i, response)

            verdict = response.split("[Verdict]")[-1].strip("```").strip("json").strip("python").strip()
            parsed = json.loads(verdict)

            if parsed['verdict'] not in ['model_1', 'model_2', 'tie']:
                print("Invalid verdict:", parsed['verdict'])
                continue
            
            label_gt = label_mapping[row['label']]
            if parsed['verdict'] == label_gt:
                count_correct += 1
            else:
                count_incorrect += 1

            print("------------------------------------------------")
            print("output_path:", output_path)
            print("Accuracy: {:.2f}%".format(count_correct / (count_correct + count_incorrect) * 100))
            print("-----------------------------------------------")
            with open(output_path, 'w') as f:
                json.dump(item, f, indent=4)
        except json.JSONDecodeError as e:
            print("i:", i)
            print("row_id:", row_id)
            print("error:", e)
    print("Finished")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message_type", type=str, required=True, help="Type of message to use")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--randomize", action="store_true", help="Randomize the dataset order")
    args = parser.parse_args()
    experiment(
        message_type=args.message_type,
        output_dir=args.output_dir,
        randomize=args.randomize,
    )

if __name__ == "__main__":
    main()

    # usage: python -m scripts.evaluate --message_type multiturn  --output_dir experiments/gpt4o-audio/standard_cot/multiturn
    # usage: python -m scripts.evaluate --message_type singleturn  --output_dir experiments/gpt4o-audio/standard_cot/singleturn