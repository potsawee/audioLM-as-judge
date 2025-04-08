import os
import json
import argparse
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel
import soundfile as sf
import librosa
from hashlib import md5

prompt_text_2 = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criteria:

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

The two audio snippets to be compared will be provided in the following two user turns."""

def ab_audio_to_conversation(
    audio_a_path, 
    audio_b_path, 
    prompt_text,    
    message_format=1,
):
    # check if audio_a_path is 16 kHz
    audio_a, sr_a = sf.read(audio_a_path)
    if sr_a != 16000:
        audio_a = librosa.resample(audio_a, orig_sr=sr_a, target_sr=16000, res_type="fft")
        array_hash_a = str(md5(audio_a.tostring()).hexdigest())
        tmp_path_a = f"tmp/{array_hash_a}.wav"
        sf.write(tmp_path_a, audio_a, 16000, format='wav')
        audio_a_path = tmp_path_a
        print(f"converted audio_a_path from {sr_a} to 16 kHz:", audio_a_path)
    # check if audio_b_path is 16 kHz
    audio_b, sr_b = sf.read(audio_b_path)
    if sr_b != 16000:
        audio_b = librosa.resample(audio_b, orig_sr=sr_b, target_sr=16000, res_type="fft")
        array_hash_b = str(md5(audio_b.tostring()).hexdigest())
        tmp_path_b = f"tmp/{array_hash_b}.wav"
        sf.write(tmp_path_b, audio_b, 16000, format='wav')
        audio_b_path = tmp_path_b
        print(f"converted audio_b_path from {sr_b} to 16 kHz:", audio_b_path)

    if message_format == 1:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "audio",
                        "audio_url": audio_a_path,
                    },
                    {
                        "type": "audio",
                        "audio_url": audio_b_path,
                    }
                ]
            }
        ]
    elif message_format == 2:
        conversation = [
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
                        "type": "audio",
                        "audio_url": audio_a_path,
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
                        "type": "audio",
                        "audio_url": audio_b_path,
                    }
                ]
            }
        ]
    else:
        raise ValueError(f"Invalid message_format: {message_format}")
    
    return conversation

model = AutoModel.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct",
    torch_dtype=torch.float16, 
    trust_remote_code=True
)
model.to("cuda")
model.eval()

def experiment(
    data_path,
    output_path,
    order='ab',
    message_format=1
):
    print("-----------------------------")
    print("data_path:", data_path)
    print("output_path:", output_path)
    print("order:", order)
    print("message_format:", message_format)
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

        # if message_format == 2:
        prompt_text = prompt_text_2
        # else:
            # raise ValueError(f"Invalid message_format: {message_format}")

        prompt_text = prompt_text.replace("###TEXT###", text)

        if order == 'ab':
            conversation = ab_audio_to_conversation(
                            audio_a["path"], 
                            audio_b["path"], 
                            prompt_text=prompt_text,    
                            message_format=message_format
            )
        elif order == 'ba':
            conversation = ab_audio_to_conversation(
                            audio_b["path"], 
                            audio_a["path"], 
                            prompt_text=prompt_text,    
                            message_format=message_format
            )      
        else:
            raise ValueError(f"Invalid order: {order}")
         
        x = model.generate(
            conversation=conversation,
            max_new_tokens=500,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )
        response = x['text']
        # x => x['text'] (text), x['audio'] (numpy array)
        
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
    parser.add_argument("--message_format", type=int, default=1, help="Message format")
    args = parser.parse_args()
    experiment(args.data_path, args.output_path, args.order, args.message_format)

    # [x] python exp1_somos_typhoon2_multiturn.py --data_path data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_typhoon2_prompt2.txt --order ab --message_format 2
    # [x] python exp1_somos_typhoon2_multiturn.py --data_path data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_typhoon2_prompt2_format1.txt --order ab --message_format 1
    # [x] python exp1_somos_typhoon2_multiturn.py --data_path data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_typhoon2_prompt2_BA.txt --order ba --message_format 2
    # [x] python exp1_somos_typhoon2_multiturn.py --data_path data/data_somos_pairwise_diffall.json --output_path experiments/somos/ab_testing/diffall_typhoon2_prompt2_format1_BA.txt --order ba --message_format 1
if __name__ == "__main__":
    main()
