import os
import io
import json
import argparse
import numpy as np
import base64
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from pydub import AudioSegment


def encode_audio_array_with_resampling(audio_array: np.ndarray, 
                                       original_sr: int, 
                                       target_sample_rate: int = 16000) -> str:
    """
    Encode audio (provided as a NumPy array of float64) as base64 WAV, resampling if necessary.
    
    :param audio_array: NumPy array of audio samples. 
                       (float64, typically in [-1, 1])
    :param original_sr: The original sampling rate of the audio.
    :param target_sample_rate: Desired sampling rate for the output.
    :return: Base64-encoded string representing the WAV audio, or None if error.
    """
    try:
        # Ensure data is in [-1, 1] to avoid int16 overflow
        audio_array = np.clip(audio_array, -1.0, 1.0)

        # Convert float in [-1, 1] to 16-bit int. For example: 0.5 -> 16383
        audio_int16 = (audio_array * 32767.0).astype(np.int16)
        
        # Create an AudioSegment from the raw int16 data
        # Assuming the audio is mono; if stereo, set channels=2
        audio_segment = AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=2,      # 16 bits = 2 bytes
            frame_rate=original_sr,
            channels=1
        )

        # Resample if needed
        if original_sr != target_sample_rate:
            print(f"Resampling from {original_sr} Hz to {target_sample_rate} Hz")
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)

        # Export to WAV in-memory
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()

        # Encode to Base64
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return encoded_audio

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider this main criterion:

Style Aspect: Evaluate the responses based on the style and tone of the assistants on how well it aligns with the user instruction. The user may request a specific style or tone in the assistant's reponse. You should not take into account the content of the responses, only the style and tone.

Please compare the two responses on the criterion. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better. Please be decisive, and choose 'A' or 'B'. Note that the user question and the responses of the assistants will be provided to you in the audio format. You should evaluate the responses based on the user question and not on the responses of the other assistant.

After providing your explanation, please provide your final verdict in the following format:

[Verdict]: [[X]] where X can be A or B (you must choose one!)"""

def experiment(
    dataset, # potsawee/chatbot-arena-spoken-style-samecontent, 
    output_path,
):
    print("-----------------------------")
    print("dataset:", dataset)
    print("output_path:", output_path)
    print("-----------------------------")

    # Load the dataset
    data = load_dataset(dataset)['train']
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
        # question
        question = data[i]['question_refined_wav']
        encoded_audio_question = encode_audio_array_with_resampling(
            question['array'], original_sr=question['sampling_rate'], target_sample_rate=16000)

        assistant_a = data[i]['assistant_a_wav']
        encoded_audio_responseA = encode_audio_array_with_resampling(
            assistant_a['array'], original_sr=assistant_a['sampling_rate'], target_sample_rate=16000)

        assistant_b = data[i]['assistant_b_wav']
        encoded_audio_responseB = encode_audio_array_with_resampling(
            assistant_b['array'], original_sr=assistant_b['sampling_rate'], target_sample_rate=16000)
        
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the user question in the audio format."  
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
                        "text": "This is the assistant A's response in the audio format."  
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
                        "text": "This is the assistant B's response in the audio format."  
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_responseB,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
        # Send request to GPT-4o for A/B testing
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text"],
            messages=message
        )
        # Extract and return the A/B testing decision from the response
        response = completion.choices[0].message.content
        item = {
            "id": i,
            "original_id": data[i]['original_id'],
            "response": response
        }
        print(i, response)
        print("winner_style:", data[i]['winner_style'])
        with open(output_path, 'a') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Run a specific model via gradio_client.")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output Path")
    args = parser.parse_args()
    experiment(args.dataset, args.output_path)
    # usage: python exp1_chatbotarenastyle_gpt_audio_audio_styleonly.py --dataset potsawee/chatbot-arena-spoken-style-samecontent --output_path experiments/chatbot-arena-style-654/exp1_audio_audio_gpt4o.styleonly.samecontent.jsonl

    # usage: python exp1_chatbotarenastyle_gpt_audio_audio_styleonly.py --dataset potsawee/chatbot-arena-spoken-style-11labs-samecontent --output_path experiments/chatbot-arena-style-11labs-632/exp1_audio_audio_gpt4o.styleonly.samecontent.jsonl

    # usage: python exp1_chatbotarenastyle_gpt_audio_audio_styleonly.py --dataset potsawee/chatbot-arena-spoken-style-11labs --output_path experiments/chatbot-arena-style-11labs-632/exp1_audio_audio_gpt4o.styleonly.jsonl

if __name__ == "__main__":
    main()