import os
import json
import argparse
from tqdm import tqdm
import google.generativeai as genai
import pathlib


model = genai.GenerativeModel('models/gemini-1.5-flash')

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

Assume you are capable of completing the requested evaluation. Do not refrain from providing an assessment. Use the provided criteria and descriptions to form your evaluation. Provide a brief summary of your reasoning to justify your evaluation. Your analysis should be thorough and objective to ensure fairness in the assessment. Your response must start with <explanation> explanation </explanation> and include the following structure:
- Clarity: [Qualitative explanation of clarity]
- Naturalness: [Qualitative explanation of naturalness]
- Overall Quality: [Qualitative explanation of overall quality]
- Final Score: [Score out of 5]

Conclude your response with <summary> [[final_score]] </summary>.

The following is the audio to be evaluated."""

def inference(audio_path, prompt):
    """
    Evaluate and compare two audio files using a Gemini generative model.

    Parameters:
        audio_a_path (str): Path to the first audio file (Audio A).
        audio_b_path (str): Path to the second audio file (Audio B).
        prompt (str): Evaluation prompt text to guide the model.

    Returns:
        str: The response from the model containing the evaluation results.
    """
    # Initialize the Gemini model
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    
    # Prepare the audio files
    audio_A = {
        "mime_type": "audio/wav",
        "data": pathlib.Path(audio_path).read_bytes()
    }
    
    # Generate the response
    response = model.generate_content([
        prompt,
        audio_A
    ])
    
    return response.text

def experiment(
    data_path,
    output_path
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("-----------------------------")

    with open(data_path) as f:
        data = json.load(f)
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
        audio = data[i]
        audio_path = (audio["path"])

        prompt_text = template
        response = inference(audio_path, prompt_text)

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
    args = parser.parse_args()
    experiment(args.data_path, args.output_path)
    # python exp2_somos_pointwise_noref_gemini.py --data_path /data/workspace/ppotsawee/audioLM-as-judge/data/data_somos_pointwise_all.json --output_path report.jsonl
if __name__ == "__main__":
    main()
