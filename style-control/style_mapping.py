
import argparse
import os
import re
import json
import random
import anthropic
from tqdm import tqdm
from datasets import load_dataset

prompt_template = """
Please help me find a conversation that would be the most suitable for each speaking style. I will provide you with a list of conversations where each conversation contains user question and two assistant responses. Please assign the conversation that would be the most suitable for each speaking style. I will provide you with ###NUMBER### conversations, and please assign the most suitable conversation to each speaking style, and please try to assign one conversation to each speaking style. One conversation can be used at most once. The possible speaking styles are as follows:

speaking_styles = [
    "amazed", "angry", "annoyed", "anxious", "calm", "concerned", "confused", 
    "curious", "disappointed", "disgusted", "doubtful", "embarrassed", 
    "encouraging", "excited", "fast", "frustrated", "happy", "hesitant", 
    "hurt", "joking", "laughs", "loud", "nervous", "proud", "quiet", "sad", 
    "sarcastic", "scared", "secretive", "serious", "shocked", "shy", "slow", 
    "surprised", "suspicious", "terrified", "upset", "urgent", "whispering", 
    "confident", "cowboy", "dramatic", "hippie", "hyperactive", "child", 
    "medieval", "knight", "nervous", "politician", "robot", "sarcastic", 
    "comedian", "shy", "teenager", "snobbish", "villain",
    "cockney_accent", "french_accent", "german_accent", "indian_accent",
    "italian_accent", "japanese_accent", "russian_accent", "scottish_accent", 
    "spanish_accent", "singaporean_accent",
]

For example a conversation that suits the "whispering" style could be:
{
    "id": conversation_id,
    "user": "Write a bedtime story for my 5 years old daughter.",
    "assistant": "Sure! Here is a story about a little ..."
}

These are conversations that would be suitable for the "whispering" style:
###list_of_conversations###

Please assign the most suitable conversation to each speaking style in a JSON format as follows:
{
    "speaking_style_1": "conversation_id,
    "speaking_style_2": "conversation_id,
    ...
}

Also if a conversation is too short or too long, please do not use that conversation for any speaking style.

The output will be parsed as a JSON object where the key is the speaking style and the value is the conversation id, so please do not include anything else in your response.
""".strip()

def calling_claude(
    client,
    prompt: str,
) -> str:
    # Create the message request
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=5000,
        temperature=0.2,
        system="You are an helpful assistant that helps me find a conversation that would be the most suitable for each speaking style.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def main(
    input_path: str, # jsonl
    output_path: str, # jsonl
    start_id: int = 0,
    end_id: int = 200,
    num_try: int = 3,
):
    output_path_with_id = os.path.join(output_path, f"{start_id}-{end_id}.json")
    if os.path.exists(output_path_with_id):
        print("File already exists, skipping", output_path_with_id)
        return

    # Initialize the Claude client
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    client = anthropic.Client(api_key=api_key)

    with open(input_path, "r") as f:
        data = json.load(f)

    conversations = []
    for i in range(start_id, end_id):
        conversations.append({
            "id": i,
            # "question_id": data[i]["question_id"],
            # "model_a": data[i]["model_a"],
            # "model_b": data[i]["model_b"],
            # "content_winner": data[i]["winner"],
            "user": data[i]["conversation_a"][0]["content"],
            "assistant_a": data[i]["conversation_a"][1]["content"],
            "assistant_b": data[i]["conversation_b"][1]["content"],
        })
    conversation_as_string = ""
    for conversation in conversations:
        conversation_as_string += f"[Conversation ID = {conversation['id']}]\n\n"
        conversation_as_string += f"### User: {conversation['user']}\n"
        conversation_as_string += f"### Assistant A: {conversation['assistant_a']}\n\n"
        conversation_as_string += f"### Assistant B: {conversation['assistant_b']}\n\n"

    number = end_id - start_id
    assert number > 0
    number = str(number)
    
    prompt = prompt_template.replace("###NUMBER###", number).replace("###list_of_conversations###", conversation_as_string)

    for j in range(num_try):
        # try it num_try times
        response = calling_claude(client, prompt)
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            print("Error in JSON decoding, trying again...")
            continue
    print("Wrote output path:", output_path_with_id)
    with open(output_path_with_id, "w") as f:
        json.dump(response_json, f, ensure_ascii=False)


if __name__ == "__main__":
    # argument parser setup
    parser = argparse.ArgumentParser(description="Rate the question using Claude.")
    parser.add_argument("--input_path", type=str, help="Path to the input json file.")
    parser.add_argument("--output_path", type=str, help="Path to the output json file.")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size.")
    args = parser.parse_args()

    intervals = []
    for i in range(0, 7800, args.chunk_size):
        intervals.append((i, i+args.chunk_size))
    random.shuffle(intervals)

    for i, interval in enumerate(intervals):
        print(f"{i}-th experimenting from", interval[0], "to", interval[1])
        main(args.input_path, args.output_path, interval[0], interval[1])

    # example usage:
    # python style_mapping.py --input_path /data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english.json --output_path styles/ --chunk_size 200
    # nohup python style_mapping.py --input_path /data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english.json --output_path styles/ --chunk_size 200 >> nohup.txt &