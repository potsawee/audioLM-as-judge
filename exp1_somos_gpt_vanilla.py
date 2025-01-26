import os
import json
import argparse
from tqdm import tqdm
from gpt4o_audio_api import get_encoded_audio_from_path
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_text_1 = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criteria:

1.	Clarity: How clearly the speech is articulated, free from distortion, noise, or artifacts.
2.	Naturalness: The degree to which the speech resembles a natural human voice, including accurate intonation, rhythm, and expressiveness.
3.	Overall Quality: The overall impression of the audio's naturalness and coherence, considering how pleasant and lifelike it sounds.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please give your verdict following this format <verdict> [[A/B]] </verdict>.

The text for the two audio for you to evaluate is: "###TEXT###"

The two audio for you to evaluate are the following."""

def ab_audio_to_message(
    encoded_audio_a, 
    encoded_audio_b, 
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
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_a,
                            "format": "wav"
                        }
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
    message_format=1
):
    # Send request to GPT-4o for A/B testing
    message = ab_audio_to_message(encoded_audio_a, encoded_audio_b, prompt_text, message_format)
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview-2024-12-17",
        # modalities=["text", "audio"],
        modalities=["text"],
        audio={"voice": "alloy", "format": "wav"},
        messages=message
    )
    # Extract and return the A/B testing decision from the response
    # text_output = completion.choices[0].message.audio.transcript.strip()
    text_output = completion.choices[0].message.content
    return text_output

def experiment(
    data_path,
    output_path,
    order='ab'
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("order:", order)
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
        audio_a, audio_b = data[i]
        assert audio_a["text"] == audio_b["text"]
        text = audio_a["text"]
        encoded_audio_a = get_encoded_audio_from_path(audio_a["path"])
        encoded_audio_b = get_encoded_audio_from_path(audio_b["path"])

        prompt_text = prompt_text_1.replace("###TEXT###", text)
        if order == 'ab':
            response = ab_testing(encoded_audio_a, encoded_audio_b, prompt_text, message_format=1)
        elif order == 'ba':
            # BA experiment
            response = ab_testing(encoded_audio_b, encoded_audio_a, prompt_text, message_format=1)
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

if __name__ == "__main__":
    main()
