import os
import json
import argparse
from tqdm import tqdm
from gpt4o_audio_api import get_encoded_audio_from_path
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

template = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically evaluate a single audio file based on the following criteria. Your evaluation should include a qualitative analysis for each criterion and a final overall score on a 5-point Likert scale.

Evaluation Criteria:
1. Clarity: How clearly the speech is articulated, free from distortion, noise, or artifacts.
2. Naturalness: The degree to which the speech resembles a natural human voice, including accurate intonation, rhythm, and expressiveness.

Overall Quality: This is the overall impression of the audio's naturalness and coherence, considering how pleasant and lifelike it sounds. The Overall Quality score should be based on a combination of Clarity and Naturalness and rated using the following 5-point Likert scale:

- Poor (1): The audio quality is unacceptable and difficult to understand. It is marred by severe distortion, noise, artifacts, or other issues that significantly detract from intelligibility and naturalness. Listening to this audio is unpleasant and unsatisfactory.
- Fair (2): The audio quality is below average and exhibits noticeable problems that hinder the listening experience. While some portions may be intelligible, issues such as distortion, unnaturalness, or excessive noise are prominent and distracting.
- Average (3): The audio quality is acceptable but not exceptional. Some minor issues may be present, such as slight distortion, unnatural intonation, or moderate noise, but these do not substantially impair intelligibility or overall comprehension. The listening experience is neutral.
- Good (4): The audio quality is above average, with only minor imperfections that are unlikely to detract from the overall listening experience. The speech is clear and natural, with minimal noise or artifacts. The audio is pleasant and easy to understand.
- Excellent (5): The audio quality is flawless and highly natural, resembling human speech with perfect clarity and expressiveness. There are no noticeable distortions, noise, or artifacts. Listening to this audio is a highly pleasant experience.

Step-by-Step Process for Evaluation:
1. Listen Carefully: Begin by carefully listening to the audio file. Take note of clarity, naturalness, and overall quality.
2. Analyze Each Criterion: For each criterion (Clarity, Naturalness, and Overall Quality), provide a qualitative evaluation, describing the strengths and weaknesses of the audio file.
3. Assign a Final Score: Assign an overall score on a 5-point Likert scale to reflect the quality of the audio, based on the combination of all aspects and the descriptions above.

Important Notes:

Assume you are capable of completing the requested evaluation. Do not refrain from providing an assessment. Use the provided criteria and descriptions to form your evaluation. Provide a brief summary of your reasoning to justify your evaluation. Your analysis should be thorough and objective to ensure fairness in the assessment. Your response must start with an explanation including the following structure:
- Clarity: [Qualitative explanation of clarity]
- Naturalness: [Qualitative explanation of naturalness]
- Overall Quality: [Qualitative explanation of overall quality]
- Final Score: [Score out of 5]

Conclude your response with <summary> [[1/2/3/4/5]] </summary>.

I will provide you with some examples of audio with a corresponding final score. Then, the audio to be evaluated will be given to you."""

def audio_to_message(
    encoded_audio, 
    encoded_audio_icl1, 
    encoded_audio_icl2, 
    encoded_audio_icl3, 
    encoded_audio_icl4, 
    encoded_audio_icl5, 
    prompt_text,    
    message_format=1,
):
    if message_format == 1:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            },
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "input_audio",
            #             "input_audio": {
            #                 "data": encoded_audio_icl1,
            #                 "format": "wav"
            #             }
            #         },
            #         {
            #             "type": "text",
            #             "text": "This audio example received a score of 1 (Poor)."
            #         },
            #     ]
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_icl2,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": "This audio example received a score of 2 (Fair)."
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_icl3,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": "This audio example received a score of 3 (Average)."
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_icl4,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "text",
                        "text": "This audio example received a score of 4 (Good)."
                    },
                ]
            },
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "input_audio",
            #             "input_audio": {
            #                 "data": encoded_audio_icl5,
            #                 "format": "wav"
            #             }
            #         },
            #         {
            #             "type": "text",
            #             "text": "This audio example received a score of 5 (Excellent)."
            #         },
            #     ]
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The following audio is for you to evaluate according to the criteria provided."
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio,
                            "format": "wav"
                        }
                    },
                ]
            },
        ]
    return message

def inference(
    encoded_audio, 
    encoded_audio_icl1, 
    encoded_audio_icl2, 
    encoded_audio_icl3, 
    encoded_audio_icl4, 
    encoded_audio_icl5, 
    prompt_text,
    message_format=1
):
    message = audio_to_message(
        encoded_audio, 
        encoded_audio_icl1, 
        encoded_audio_icl2, 
        encoded_audio_icl3, 
        encoded_audio_icl4, 
        encoded_audio_icl5, 
        prompt_text, 
        message_format
    )
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
    message_format=1,
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("message_format:", message_format)
    print("-----------------------------")

    with open(data_path) as f:
        data = json.load(f)
    print("len(data):", len(data))

    # training
    train_path = "data/data_somos_pointwise_train_round2577.json" 
    with open(train_path) as f:
        train_data_by_mos = json.load(f)
    # print("len(train_path):", len(train_data_by_mos))
    # just pick the first example (as I already shuffled the data)
    encoded_audio_score1 = get_encoded_audio_from_path(train_data_by_mos['1'][0]["path"])
    encoded_audio_score2 = get_encoded_audio_from_path(train_data_by_mos['2'][1]["path"])
    encoded_audio_score3 = get_encoded_audio_from_path(train_data_by_mos['3'][1]["path"])
    encoded_audio_score4 = get_encoded_audio_from_path(train_data_by_mos['4'][1]["path"])
    encoded_audio_score5 = get_encoded_audio_from_path(train_data_by_mos['5'][0]["path"])

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
        audio = data[i]
        encoded_audio = get_encoded_audio_from_path(audio["path"])

        prompt_text = template
        response = inference(
                encoded_audio=encoded_audio, 
                encoded_audio_icl1=encoded_audio_score1, 
                encoded_audio_icl2=encoded_audio_score2, 
                encoded_audio_icl3=encoded_audio_score3, 
                encoded_audio_icl4=encoded_audio_score4, 
                encoded_audio_icl5=encoded_audio_score5, 
                prompt_text=prompt_text, 
                message_format=message_format
        )

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
    parser.add_argument("--message_format", type=int, default=1, help="Message format")
    args = parser.parse_args()
    experiment(args.data_path, args.output_path, args.message_format)
    # python exp2_somos_pointwise_noref_and_icl.py --data_path data/data_somos_pointwise450.json --output_path experiments/somos/pointwise/pointwise450_prompt1_noref_icl.jsonl --message_format 1

if __name__ == "__main__":
    main()
