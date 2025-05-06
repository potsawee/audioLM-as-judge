import os
import io
import json
import argparse
from tqdm import tqdm
import google.generativeai as genai
import google
from pydub import AudioSegment

model_name = "gemini-2.0-flash"
model = genai.GenerativeModel(f'models/{model_name}')

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie (which can be equally good and equally bad). Note that the user question will be provided to you in the audio format, while the responses of the two assistants will be provided to you in the text format. You should evaluate the responses based on the user question and not on the responses of the other assistant. You should also not consider the quality of the audio or the voice of the assistants. You should only consider the content of the responses.""" 

prompt_template_question = """This is the user question in the audio format."""

prompt_template_response = """[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"""

def convert_to_16kHz_bytes(audio_path):
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Check if resampling is necessary
    if audio.frame_rate != 16000:
        print(f"Resampling from {audio.frame_rate} Hz to 16 kHz...")
        audio = audio.set_frame_rate(16000)
    else:
        # print("Audio is already 16 kHz.")
        pass

    # Export the audio to an in-memory buffer in WAV format
    output_buffer = io.BytesIO()
    audio.export(output_buffer, format="wav")
    output_buffer.seek(0)  # Reset the buffer to the beginning

    # Return the binary data equivalent to pathlib.Path().read_bytes()
    return output_buffer.read()

def experiment(
    data_path,
    output_path,
    order='ab',
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
        if order == 'ab':
            conversation_a, conversation_b = data[i]['conversation_a'], data[i]['conversation_b']
        elif order == 'ba':
            conversation_a, conversation_b = data[i]['conversation_b'], data[i]['conversation_a']
        else:
            raise ValueError("order must be 'ab' or 'ba'")
        assert conversation_a[0]['content'] == conversation_b[0]['content']

        question_wav_path = f"/data/workspace/ppotsawee/audioLM-as-judge-new/kokoroTTS/wav/{data[i]['question_id']}-user.wav"

        prompt_response = prompt_template_response.format(answer_a=conversation_a[1]['content'], answer_b=conversation_b[1]['content'])  

        # Generate the response
        response = model.generate_content([
            system_prompt,
            prompt_template_question,
            {
                "mime_type": "audio/wav",
                "data": convert_to_16kHz_bytes(question_wav_path)
            },
            prompt_response
        ])

        try:
            response = response.text
        except ValueError as e:
            response = "ValueError: " + str(e) + "\n" + "[[D]]"

        item = {
            "data_path": data_path,
            "i": i,
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

    # usage: python exp1_chatbotarena_gemini_audio_text.py --data_path chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-text-gemini1.5flash.jsonl --order ab
    # usage: python exp1_chatbotarena_gemini_audio_text.py --data_path chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-text-gemini1.5flash_BA.jsonl --order ba

    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_text --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-text-gemini2.0flash.jsonl --order ab
    # usage: python -m scripts.exp1_chatbotarena_gemini_audio_text --data_path data/chatbot-arena-spoken-1turn-english-difference-voices.json --output_path experiments/chatbot-arena-7824/audio-text-gemini2.0flash_BA.jsonl --order ba


if __name__ == "__main__":
    main()