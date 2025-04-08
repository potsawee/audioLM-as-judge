import os
import re
import json
import openai
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
I want to create questions where the tone or speaking style influences the type of response. Your task is to generate users' questions that can be spoken in a specific tone or style, and then provide two responses:

1. Response to Question spoken in the Target Tone: A response that directly aligns with the tone or speaking style of the question.
2. Response to Question spoken in a different (natural or opposite) Tone: First you need to identify what an opposite speaking style could be for the same question. Then, if the question is spoken in this different tone or style, this is the response that would be appropriate. 

I'll provide the target speaking style or tone for the question, and you will generate both the question (user) and the response (assistant) appropriate with the target speaking style of the user and another response as well as the hypothetical style as described above. Ensure that both responses remain realistic and contextually appropriate.

[Example 1]
Target Speaking Style: Angry
Question: "Why didn’t you tell me about the meeting earlier?"

{
    "conversation_a": {
        "user": "Why didn’t you tell me about the meeting earlier?",
        "tone_of_user": "Angry",
        "assistant": "I’m really sorry. I should have been more proactive. It won’t happen again—I’ll make sure you’re updated next time immediately."
    },
    "conversation_b": {
        "user": "Why didn’t you tell me about the meeting earlier?",
        "tone_of_user": "Normal",
        "assistant": "Oh, I thought you were already aware! My bad. I’ll double-check with you next time to avoid any confusion."
    }
}

[Example 2]
Speaking Style: Amused
Question: "Oh, so you really thought that plan was going to work, huh?"

{
    "conversation_a": {
        "user": "Oh, so you really thought that plan was going to work, huh?",
        "tone_of_user": "Amused",
        "assistant": "Haha, yeah, I had high hopes! It sounded much better in my head, though.",
    },
    "conversation_b": {
        "user": "Oh, so you really thought that plan was going to work, huh?",
        "tone_of_user": "Annoyed",
        "assistant": "I know, I know. It was a long shot, but I had to try something. I’ll come up with a better idea next time.",
    }
}

[Example 3]
Speaking Style: Confused
Question: "Wait, how did we end up with this as the final decision?"

{
    "conversation_a": {
        "user": "Wait, how did we end up with this as the final decision?",
        "tone_of_user": "Confused",
        "assistant": "I’m not even sure! I thought we were going in a completely different direction.",
    },
    "conversation_b": {
        "user": "Wait, how did we end up with this as the final decision?",
        "tone_of_user": "Happy",
        "assistant": "I know, right? It’s amazing how everything fell into place perfectly!",
    }
}

Please provide the question and the two responses in the JSON format as shown above. Your response will be parsed using json.loads() to extract the question and responses. Please ensure that your response is in valid JSON format. 
""".strip()

prompt_template = """Target Speaking Style or the tone of the question: '{style}'. Please generate a question as well as responses. Try being creative and generate conversations that are different from the examples for previous questions."""

def parse_relaxed_json(input_string):
    """
    Parses a JSON-like string that might include extra formatting such as:
      - Code block markers (```json ... ```)
      - Trailing commas in objects/arrays

    Returns a Python object corresponding to the JSON.
    """
    # Step 1: Remove code block markers if present.
    # This regex matches content inside ```json ... ```
    code_block_pattern = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL)
    match = code_block_pattern.search(input_string)
    if match:
        input_string = match.group(1)

    # Step 2: Remove trailing commas before a closing } or ]
    # This will replace commas followed by whitespace and then a } or ] with just the closing brace/bracket.
    cleaned = re.sub(r',\s*(?=[}\]])', '', input_string)

    # Step 3 (Optional): If your entire JSON string is enclosed in single quotes,
    # you might need to remove them and ensure the JSON uses double quotes.
    stripped = cleaned.strip()
    if (stripped.startswith("'") and stripped.endswith("'")) or \
       (stripped.startswith('"""') and stripped.endswith('"""')):
        # Remove the outer quotes and convert single quotes to double quotes.
        cleaned = stripped[1:-1].replace("'", '"')

    # Step 4: Parse the cleaned JSON string.
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

def run(
    style,
    num_questions,
    output_path,

):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        }
    ]
    num_try = 5
    generated_conversations = []
    for i in tqdm(range(num_questions)):
        prompt = prompt_template.format(style=style)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.9,
        )
        for j in range(num_try):
            try:
                response = completion.choices[0].message.content
                response = response.strip("```").strip("json").strip()
                parsed = parse_relaxed_json(response)
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(f"try: {j}, error: {e}")
                continue
            break
        print("response:", response)
        generated_conversations.append(parsed)
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response
                    }
                ]
            }
        )
    # Save the generated questions and responses to a JSON file
    with open(output_path, 'w') as f:
        json.dump(generated_conversations, f, indent=4)
    print("Generated questions and responses saved to:", output_path)

if __name__ == "__main__":
    speaking_styles = [
        "amazed", "angry", "annoyed", "anxious", "concerned", "confused", 
        "curious", "disappointed", "disgusted", "embarrassed", 
        "encouraging", "excited", "frustrated", "happy", "hesitant", 
        "hurt", "joking", "laughs", "nervous", "sad", "sarcastic", 
        "scared", "secretive", "serious", "shocked", "shy", "surprised", 
        "suspicious", "terrified", "upset", "urgent", "whispering", "confident", 
        "dramatic", "hippie", "hyperactive", "nervous", "sarcastic", "shy",
    ]
    num_questions = 10
    output_dir = "generated_conversations/"
    for style in speaking_styles:
        output_path = os.path.join(output_dir, f"{style}.json")
        # check if the output file already exists
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping...")
            continue
        run(style, num_questions, output_path)
