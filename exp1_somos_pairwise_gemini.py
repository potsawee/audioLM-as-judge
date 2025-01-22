import os
import json
import argparse
from tqdm import tqdm
import google.generativeai as genai
import pathlib

def ab_testing(audio_a_path, audio_b_path, prompt):
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
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    
    # Prepare the audio files
    audio_A = {
        "mime_type": "audio/wav",
        "data": pathlib.Path(audio_a_path).read_bytes()
    }
    audio_B = {
        "mime_type": "audio/wav",
        "data": pathlib.Path(audio_b_path).read_bytes()
    }
    
    # Generate the response
    response = model.generate_content([
        prompt,
        audio_A,
        audio_B
    ])
    
    return response.text

def experiment(
    data_path,
    output_path,
    prompt_text,
    order='ab'
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("order:", order)

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
        audio_a, audio_b = data[i]
        assert audio_a["text"] == audio_b["text"]
        text = audio_a["text"]
        audio_a_path = (audio_a["path"])
        audio_b_path = (audio_b["path"])

        prompt_text = prompt_text.replace("###TEXT###", text)

        if order == 'ab':
            response = ab_testing(audio_a_path, audio_b_path, prompt_text)
        elif order == 'ba':
            # BA experiment
            response = ab_testing(audio_b_path, audio_a_path, prompt_text)
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


    prompt_text_1 = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criteria:

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

    The two audio for you to evaluate are the following."""

    prompt = prompt_text_1

    experiment(args.data_path, args.output_path, prompt, args.order)
    # SOMOS Experiment
    # python exp1_somos_pairwise_gemini.py --data_path /data/workspace/ppotsawee/audioLM-as-judge/data/data_somos_pairwise_diff15.json --output_path ./somos_pairwise_diff15.jsonl --order ab
if __name__ == "__main__":
    main()
