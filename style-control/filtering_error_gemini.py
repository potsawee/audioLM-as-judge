import os
import io
import json
import argparse
from tqdm import tqdm
import google.generativeai as genai
import google
from glob import glob
from pydub import AudioSegment

model_name = "gemini-1.5-flash"
model = genai.GenerativeModel(f'models/{model_name}')

system_prompt = """You are a helpful assistant who helps me listen to a recording, and determine if it says exactly the words in the reference text.""" 

prompt_template = """This is the reference text that the speaker was meant to speak:\n\n{reference_text}\n\nPlease listen to the recording and determine if it says exactly the words in the reference text. Provide some explanation and your understanding first, then provide your judgement. If it does follow, please respond with "[[Yes]]". If it does not, please respond with "[[No]]"."""


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
    data_dir
):
    print("-----------------------------")
    print("data_dir:", data_dir)
    print("-----------------------------")

    # Load the data
    path = "/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english.json"
    with open(path) as f:
        data = json.load(f)

    # Wav paths
    wav_paths = sorted(glob(f"{data_dir}/*.wav"))

    for i, wav_path in tqdm(enumerate(wav_paths)):
        import ipdb; ipdb.set_trace()
        judge_text_path = wav_path + ".judge.txt"
        # check if the judge text file already exists
        if os.path.exists(judge_text_path):
            print(f"Skipping {judge_text_path}")
            continue
        
        conversation_id = int(wav_path.replace(data_dir, "").split("_")[0])

        data_i = data[conversation_id]

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
    parser = argparse.ArgumentParser(description="Filtering Error with Gemini")
    parser.add_argument("--data_dir", type=str, required=True, help="Data Path")
    args = parser.parse_args()
    experiment(args.data_dir)


if __name__ == "__main__":
    main()