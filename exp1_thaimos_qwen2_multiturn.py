import os
import json
import argparse
import torch
import librosa
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

prompt_text_pronunciation = """You are an expert in audio quality assessment specializing in synthesized speech evaluation. Your task is to critically compare two audio files, the first audio (Audio A) and the second audio (Audio B), will be provided after this instruction. The evaluation is based on the following criterion:

Pronunciation: Assess how accurately each word is pronounced, considering correctness, clarity, and adherence to standard pronunciation norms for the given language.

Follow this step-by-step process for your evaluation:
1. Listen Carefully: Begin by carefully listening to both Audio A (the first audio) and Audio B (the second audio). Take note of any differences in pronunciation accuracy and clarity.
2. Analyze the Criterion: Evaluate how well each audio file performs regarding pronunciation. Provide a brief explanation of your reasoning.
3. Compare Thoroughly: Summarize the strengths and weaknesses of each audio file based on your analysis of pronunciation.
4. Decide the Winner: Conclude by determining which audio file has better pronunciation overall and clearly state your final verdict in this format: [[A]] or [[B]].

Important: Provide a brief summary of your reasoning to justify your decision. Your analysis should be thorough and objective to ensure fairness in the comparison. Your response must start with <explanation> explanation </explanation> and end with <verdict> [[A/B]] </verdict>.

The text for the two audio for you to evaluate is: "###TEXT###"

The two audio snippets to be compared will be provided in the following two user turns."""

def ab_audio_to_conversation(
    audio_a_path, 
    audio_b_path, 
    prompt_text,    
    message_format=1,
):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
)

@torch.no_grad()
def run_inference(conversation):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        ele['audio_url'], 
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )

    inputs = processor(text=text, audios=audios, 
                       return_tensors="pt", padding=True,
                       sampling_rate=processor.feature_extractor.sampling_rate
                       ).to("cuda")

    generate_ids = model.generate(**inputs, max_length=6000, do_sample=False)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

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
        prompt_text = prompt_text_pronunciation
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
         

        response = run_inference(conversation)

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

    # ThaiMOS & Qwen2
    # python exp1_thaimos_qwen2_multiturn.py --data_path data/data_thaimos_pairwise_diffall.json --output_path experiments/thaimos/ab_testing/shuffled_qwen2_prompt1.txt --order ab --message_format 1

if __name__ == "__main__":
    main()
