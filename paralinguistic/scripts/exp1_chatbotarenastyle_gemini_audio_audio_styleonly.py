import os
import io
import json
import argparse
from tqdm import tqdm
import google.generativeai as genai
import numpy as np
from pydub import AudioSegment
from datasets import load_dataset

def convert_array_to_16kHz_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a float64 NumPy audio array (in [-1.0, 1.0] range) to a
    16 kHz WAV byte string.

    Args:
        audio_array (np.ndarray): Audio data in float64 format.
                                  Shape can be (num_samples,) for mono
                                  or (num_samples, num_channels) for multi-channel.
        sample_rate (int): Original sample rate of the audio.

    Returns:
        bytes: WAV data at 16 kHz as a byte string.
    """

    # If your float data isn't necessarily in [-1.0, 1.0], you may need a
    # different scaling strategy. For standard audio, we assume [-1, 1].
    # 1) Clamp to [-1,1]
    audio_array = np.clip(audio_array, -1.0, 1.0)

    # 2) Scale float64 to 16-bit integer range
    audio_array_int16 = (audio_array * 32767.0).astype(np.int16)

    # Determine number of channels
    if audio_array_int16.ndim == 1:
        channels = 1
    elif audio_array_int16.ndim == 2:
        channels = audio_array_int16.shape[1]
    else:
        raise ValueError("audio_array must be 1D (mono) or 2D (multi-channel).")

    # Create AudioSegment from raw PCM data
    sample_width = 2  # 16 bits = 2 bytes
    audio_segment = AudioSegment(
        data=audio_array_int16.tobytes(),
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=channels
    )

    # Resample to 16 kHz if necessary
    if audio_segment.frame_rate != 16000:
        print(f"Resampling from {audio_segment.frame_rate} Hz to 16 kHz...")
        audio_segment = audio_segment.set_frame_rate(16000)

    # Export to an in-memory buffer in WAV format
    output_buffer = io.BytesIO()
    audio_segment.export(output_buffer, format="wav")
    output_buffer.seek(0)

    # Return the WAV bytes (same as if you'd done pathlib.Path(...).read_bytes())
    return output_buffer.read()

model_name = "gemini-1.5-flash"
model = genai.GenerativeModel(f'models/{model_name}')

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
    print("output_path:", output_path)
    print("-----------------------------")

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
        assistant_a = data[i]['assistant_a_wav']
        assistant_b = data[i]['assistant_b_wav']
        
        # Generate the response
        response = model.generate_content([
            system_prompt,
            "This is the user question in the audio format.",
            {
                "mime_type": "audio/wav",
                "data": convert_array_to_16kHz_bytes(question['array'], question['sampling_rate'])
            },
            "This is the assistant A's response in the audio format.",
            {
                "mime_type": "audio/wav",
                "data": convert_array_to_16kHz_bytes(assistant_a['array'], assistant_a['sampling_rate'])
            },
            "This is the assistant B's response in the audio format.",
            {
                "mime_type": "audio/wav",
                "data": convert_array_to_16kHz_bytes(assistant_b['array'], assistant_b['sampling_rate'])
            },
        ])
   
        try:
            response = response.text
        except ValueError as e:
            response = "ValueError: " + str(e) + "\n" + "[[D]]"

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

    # usage: python exp1_chatbotarenastyle_gemini_audio_audio_styleonly.py --dataset potsawee/chatbot-arena-spoken-style  --output_path experiments/chatbot-arena-style-654/exp1_audio_audio_gemini15flash.styleonly.jsonl

    # usage: python exp1_chatbotarenastyle_gemini_audio_audio_styleonly.py --dataset potsawee/chatbot-arena-spoken-style-samecontent --output_path experiments/chatbot-arena-style-654/exp1_audio_audio_gemini15flash.styleonly.samecontent.jsonl

    # usage: python exp1_chatbotarenastyle_gemini_audio_audio_styleonly.py --dataset potsawee/chatbot-arena-spoken-style-11labs-samecontent --output_path experiments/chatbot-arena-style-11labs-632/exp1_audio_audio_gemini15flash.styleonly.samecontent.jsonl

    # usage: python exp1_chatbotarenastyle_gemini_audio_audio_styleonly.py --dataset potsawee/chatbot-arena-spoken-style-11labs --output_path experiments/chatbot-arena-style-11labs-632/exp1_audio_audio_gemini15flash.styleonly.jsonl


if __name__ == "__main__":
    main()